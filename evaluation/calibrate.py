"""Fit and report per-head calibration (plan task 3.3).

Produces logits from the trained heads, hands them to
`evaluation/calibration.py`, and writes three things:

* `docs/assets/calibration/report.json` - every head, fitted and rejected alike
* `docs/assets/calibration/*.svg` - reliability diagrams before and after
* `configs/calibration.json` - the registry the runtime actually reads

Only heads whose fit is **accepted** reach the registry. A rejected head stays
uncalibrated at runtime and keeps reporting the `ece_after = -1.0` sentinel,
which is the correct outcome: shipping a temperature fitted on 14 points
would be worse than shipping none, and silently doing so is exactly the class
of mistake this project has already had to correct twice.

Usage:
    python evaluation/calibrate.py --heads landcover intent change_mask \
        --ben-data data/ben_full --levir-index data/levircd/index.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.calibration import (  # noqa: E402
    CalibrationCurve,
    CalibrationReport,
    Bin,
    calibrate_head,
    reliability_svg,
    write_report,
)

REGISTRY_PATH = Path("configs/calibration.json")

# The report and the diagrams are report deliverables and belong in version
# control; `artifacts/` is gitignored runtime output, so they would vanish
# from a fresh clone if they were written there.
REPORT_DIR = Path("docs/assets/calibration")

# The cached logits are large and reproducible, so they stay in the ignored
# tree. Nothing but a re-fit reads them.
CACHE_DIR = Path("artifacts/calibration/logits")

# Registry keys are the task IDs the executor plans, so a lookup at answer
# time is `registry["heads"][plan.tasks[0]]` with no translation table.
HEAD_TASK_ID = {
    "landcover": "SINGLE_LANDCOVER",
    "change_mask": "TEMPORAL_CHANGE_MAP",
    "intent": "_router_intent",  # not a task; the Tier-1 router's own head
}

# Per-head fitting setup. Every multi-label head is fitted with BOTH methods:
# temperature is what task 3.3 names, affine is the two-parameter fallback,
# and running both means the choice between them is made on held-out ECE
# rather than on which one was tried first.
HEAD_SPEC: dict[str, dict] = {
    "landcover": {
        "mode": "multilabel",
        "dataset": "BigEarthNet-19",
        "methods": ["temperature", "affine"],
    },
    "change_mask": {
        "mode": "multilabel",
        "dataset": "LEVIR-CD",
        "methods": ["temperature", "affine"],
    },
    "intent": {
        # Affine scaling is a binary/multi-label construction; the multiclass
        # analogue is vector scaling, which is not what 3.3 asks for.
        "mode": "multiclass",
        "dataset": "CLEAN_HOLDOUT",
        "methods": ["temperature"],
    },
}


# --- Logit producers --------------------------------------------------------


def landcover_logits(data_dir: Path, checkpoint: Path, dim: int, batch_size: int):
    """Track A land-cover head over the official BigEarthNet test shard."""
    import torch

    from training.track_a_full import (
        BEN_GSD_M,
        ShardedBigEarthNet,
        batches,
        compute_stats,
    )
    from evaluation.splits.multires import load_model

    test_paths = sorted(Path(x) for x in glob.glob(str(data_dir / "*test*.hdf5")))
    if not test_paths:
        raise SystemExit(f"no test shards in {data_dir}")
    train_paths = sorted(Path(x) for x in glob.glob(str(data_dir / "*train*.hdf5")))

    # Normalisation statistics must come from the same place training took
    # them, or the logits are produced under inputs the model never saw.
    raw = ShardedBigEarthNet(train_paths or test_paths)
    stats = compute_stats(raw)
    raw.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, latest, has_gsd = load_model(checkpoint, torch, device, dim)
    dataset = ShardedBigEarthNet(test_paths, stats)

    logits, labels = [], []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for x, y in batches(dataset, batch_size, rng, shuffle=False):
            mask = np.ones((x.shape[0], x.shape[1]), dtype="float32")
            out = model(
                torch.from_numpy(x).to(device),
                torch.from_numpy(mask).to(device),
                torch.full((x.shape[0],), BEN_GSD_M, device=device),
            )
            logits.append(out.cpu().numpy())
            labels.append(y)
    dataset.close()

    return (
        np.concatenate(logits),
        np.concatenate(labels),
        f"BigEarthNet official test shard ({', '.join(p.name for p in test_paths)}), "
        f"all 12 bands at native 10 m, checkpoint {latest.name} "
        f"(gsd_conditioning={has_gsd}). Uniformly 10 m, so this split measures "
        f"calibration at native resolution ONLY and cannot show whether "
        f"confidence stays honest as resolution coarsens - that needs the "
        f"multi-resolution split.",
    )


def intent_logits():
    """Tier-1 router head over the clean, never-tuned-on holdout."""
    from satquery.controller.intent import IntentClassifier
    from satquery.synth.holdout import CLEAN_HOLDOUT

    clf = IntentClassifier()
    texts = [t for t, _ in CLEAN_HOLDOUT]
    truth = [label for _, label in CLEAN_HOLDOUT]

    # decision_function is the pre-softmax score; for a binary problem
    # sklearn returns one column, which is not the shape this expects.
    scores = clf.pipeline.decision_function(texts)
    scores = np.asarray(scores, dtype="float64")
    if scores.ndim == 1:
        raise SystemExit("intent head is binary; multiclass calibration expects >2")

    classes = list(clf.classes_)
    labels = np.array([classes.index(t) for t in truth], dtype="int64")
    return (
        scores,
        labels,
        f"CLEAN_HOLDOUT from satquery/synth/holdout.py, n={len(texts)}, "
        f"hand-written and never used to tune the templates. The synthetic "
        f"bank's own held-out split was NOT used: it measures template "
        f"memorisation, so a temperature fitted there would be calibrated to "
        f"a distribution the router never meets.",
    )


def change_mask_logits(
    index_path: Path, checkpoint: Path, dim: int, batch_size: int,
    pixels_per_image: int, limit: int | None,
):
    """Change head over LEVIR-CD test, subsampled by pixel within each image.

    Pixels inside one 256x256 tile are heavily correlated, so the fit/eval
    split must be made across IMAGES, not across pixels. Sampling a fixed
    number of pixels per image and keeping the image as the row does exactly
    that - `calibrate_head` then splits on axis 0, which is the image axis.
    """
    import torch

    from training.common.checkpointing import find_latest_checkpoint, load_checkpoint
    from training.train_change_mask import LevirCD, batches, build_model

    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = index["splits"].get("test", [])
    if limit:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"no test rows in {index_path}")

    latest = find_latest_checkpoint(checkpoint)
    if latest is None:
        raise SystemExit(f"no checkpoint in {checkpoint}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # As in evaluation/splits/multires.py: the checkpoint says which
    # architecture wrote it. Absent means v1, so v1 checkpoints are unchanged.
    from training.common.checkpointing import safe_torch_load

    extra = safe_torch_load(latest).get("extra") or {}
    model = build_model(extra.get("dim", dim), arch=extra.get("arch", "v1"))
    load_checkpoint(latest, model, map_location="cpu")
    model = model.to(device).eval()

    dataset = LevirCD(rows)
    rng = np.random.default_rng(0)
    sampler = np.random.default_rng(7)
    logits, labels = [], []
    with torch.no_grad():
        for a, b, m in batches(dataset, batch_size, rng, shuffle=False):
            out = model(
                torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)
            ).cpu().numpy()
            flat_logit = out.reshape(out.shape[0], -1)
            flat_label = m.reshape(m.shape[0], -1)
            idx = sampler.choice(
                flat_logit.shape[1], size=pixels_per_image, replace=False
            )
            logits.append(flat_logit[:, idx])
            labels.append(flat_label[:, idx])

    return (
        np.concatenate(logits),
        np.concatenate(labels),
        f"LEVIR-CD official test split, {len(rows)} tiles, {pixels_per_image} "
        f"pixels sampled per tile, checkpoint {latest.name}. Split for "
        f"fitting is by TILE, not by pixel, because pixels within a tile are "
        f"spatially correlated and a pixel-wise split would leak. The head was "
        f"trained with pos_weight=10.1, so its logits carry a deliberate "
        f"positive-class offset that temperature scaling alone cannot remove.",
    )


def produce_logits(head: str, args):
    """Dispatch to the right logit producer for `head`."""
    if head == "landcover":
        return landcover_logits(
            args.ben_data, args.track_a_ckpt, args.dim, args.batch_size
        )
    if head == "intent":
        return intent_logits()
    return change_mask_logits(
        args.levir_index, args.change_ckpt, args.change_dim,
        args.batch_size, args.pixels_per_image, args.limit_change,
    )


# --- Driver -----------------------------------------------------------------


def _curve_from_dict(d: dict) -> CalibrationCurve:
    bins = [Bin(**b) for b in d["bins"]]
    return CalibrationCurve(**{k: v for k, v in d.items() if k != "bins"}, bins=bins)


def emit_diagrams(report: CalibrationReport, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for stage in ("before", "after"):
        curve = _curve_from_dict(getattr(report, stage))
        title = f"{report.head} - {stage} ({report.method})"
        path = out_dir / f"{report.head}_{report.method}_{stage}.svg"
        path.write_text(reliability_svg(curve, title), encoding="utf-8")
        written.append(str(path))
    return written


def build_registry(reports: list[CalibrationReport]) -> dict:
    """Registry from reports, one entry per head.

    A head may be fitted with more than one method. Only the accepted one
    with the lowest held-out ECE ships; the others stay in report.json as the
    evidence for that choice.
    """
    heads, rejected = {}, {}
    for r in reports:
        entry = {
            "method": r.method,
            "T": r.fit.get("T", r.fit.get("T_equivalent", 1.0)),
            "a": r.fit.get("a"),
            "b": r.fit.get("b"),
            "ece_before": r.before["ece"],
            "ece_after": r.after["ece"],
            "n_fit": r.n_fit,
            "n_eval": r.n_eval,
            "dataset": r.dataset,
            "split_note": r.split_note,
        }
        key = HEAD_TASK_ID.get(r.head, r.head)
        if r.accepted:
            incumbent = heads.get(key)
            if incumbent is None or entry["ece_after"] < incumbent["ece_after"]:
                heads[key] = entry
        elif key not in heads:
            # Keep the rejection that is most informative: the first one, or
            # any later one that at least improved ECE more.
            incumbent = rejected.get(key)
            if incumbent is None or entry["ece_after"] < incumbent["ece_after"]:
                rejected[key] = {**entry, "rejection_reason": r.rejection_reason}
    # A head that ended up calibrated by one method is not "rejected" because
    # another method failed on it.
    rejected = {k: v for k, v in rejected.items() if k not in heads}
    return {
        "version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "heads": heads,
        "rejected": rejected,
        "note": (
            "Heads under `rejected` are deliberately left uncalibrated at "
            "runtime and keep reporting ece_after = -1.0. A fitted parameter "
            "that did not survive its own held-out check is worse than none."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heads", nargs="+", default=["landcover", "intent"],
                   choices=sorted(HEAD_TASK_ID))
    p.add_argument("--ben-data", type=Path, default=Path("data/ben_full"))
    p.add_argument("--track-a-ckpt", type=Path,
                   default=Path("checkpoints/track_a_full_base"))
    p.add_argument("--levir-index", type=Path, default=Path("data/levircd/index.json"))
    p.add_argument("--change-ckpt", type=Path, default=Path("checkpoints/change_mask"))
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--change-dim", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--pixels-per-image", type=int, default=1024)
    p.add_argument("--limit-change", type=int)
    p.add_argument("--bins", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    p.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    p.add_argument("--no-write-registry", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                   help="cache raw logits here so a re-fit needs no GPU")
    p.add_argument("--refresh-cache", action="store_true",
                   help="recompute logits even if a cache file exists")
    args = p.parse_args()

    reports: list[CalibrationReport] = []

    for head in args.heads:
        print(f"\n=== {head} ===", flush=True)
        spec = HEAD_SPEC[head]
        # Producing logits needs torch, a GPU and the datasets on disk;
        # re-fitting them needs neither. Caching keeps a re-fit cheap enough
        # that trying a second method is never a reason not to.
        cached = args.cache_dir / f"{head}.npz" if args.cache_dir else None
        if cached is not None and cached.exists() and not args.refresh_cache:
            blob = np.load(cached, allow_pickle=False)
            logits, labels, note = blob["logits"], blob["labels"], str(blob["note"])
            print(f"loaded cached logits from {cached}", flush=True)
        else:
            logits, labels, note = produce_logits(head, args)
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cached, logits=logits, labels=labels, note=np.array(note)
                )
                print(f"cached logits to {cached}", flush=True)

        print(f"logits {logits.shape}  labels {labels.shape}", flush=True)

        for method in spec["methods"]:
            report = calibrate_head(
                logits, labels, head=head, mode=spec["mode"],
                dataset=spec["dataset"], split_note=note, method=method,
                n_bins=args.bins, seed=args.seed,
            )
            print("  " + report.summary(), flush=True)
            for path in emit_diagrams(report, args.out_dir):
                print(f"  wrote {path}")
            reports.append(report)

    report_path = args.out_dir / "report.json"
    write_report(reports, report_path)
    print(f"\nWrote {report_path}")

    if not args.no_write_registry:
        registry = build_registry(reports)
        args.registry.parent.mkdir(parents=True, exist_ok=True)
        args.registry.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        print(f"Wrote {args.registry}  "
              f"({len(registry['heads'])} calibrated, "
              f"{len(registry['rejected'])} rejected)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
