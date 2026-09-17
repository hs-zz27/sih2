"""Did the Phase 5 retrain actually help? One table, per tool.

This is the question the whole campaign exists to answer, and it is easy to
answer dishonestly. Three rules are built in rather than left to whoever
writes the report:

**1. The v1 number is quoted, never recomputed.** Every baseline below is the
figure published in `docs/model-cards.md`, with the metric key it was read
from. Re-running a v1 evaluation to "check" it would produce a number from a
different environment on a different day, and `docs/code-freeze.md` makes a
published number immutable precisely so the comparison has a fixed point.

**2. The headline metric is fixed in advance.** Picking the metric after
seeing the results is how a worse model gets reported as an improvement -
there is always some number that went up. The metric named here is the one
the v1 card led with.

**3. A regression is printed as a regression.** No "comparable to", no
omitting a row. Two of these are expected to regress and the report is
stronger for saying so plainly:

* `optsar_fusion` v1 already measured `complementarity_gain` at **-0.0064** -
  the fused head was *worse* than optical alone, so the fusion was decoration.
  If v2 does not fix that, that is the finding.
* `change_vqa_scratch_v2` is a from-scratch model on ~1,600 SECOND pairs and
  is expected to lose to the pretrained arm. It is run to measure the claim
  rather than assume it twice.

Usage:
    python evaluation/compare_v2.py
    python evaluation/compare_v2.py --json --out docs/assets/v1_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Comparison:
    """One tool's headline metric, its published baseline, and where to look."""

    run_id: str
    tool: str
    metric: str
    label: str
    v1_value: float | None
    v1_source: str
    higher_is_better: bool = True
    published_range: str = ""
    note: str = ""


# Baselines are transcribed from docs/model-cards.md and the metrics.json each
# v1 run wrote. `v1_value is None` means the tool had no v1 number to beat.
COMPARISONS: tuple[Comparison, ...] = (
    Comparison(
        run_id="track_a", tool="landcover_v1",
        metric="map_all_bands", label="mAP (12 bands)",
        v1_value=0.2853654937481183,
        v1_source="checkpoints/track_a_full_base/metrics.json",
        published_range="~0.65-0.85",
        note="v1 was dim-64 on 30k of ~590k patches",
    ),
    Comparison(
        run_id="track_a", tool="landcover_v1",
        metric="retention", label="4-band retention",
        v1_value=0.9015117550444813,
        v1_source="checkpoints/track_a_full_base/metrics.json",
        note="Cartosat has 4 of 12 bands; this must not regress",
    ),
    Comparison(
        run_id="grounding", tool="grounding_v1",
        metric="acc@0.5", label="Acc@0.5",
        v1_value=0.07624890446976336,
        v1_source="checkpoints/grounding/metrics.json",
        published_range="~0.70-0.80",
        note="v1 pooled before regressing the box - the defect v2 removes",
    ),
    Comparison(
        run_id="grounding", tool="grounding_v1",
        metric="miou", label="mIoU",
        v1_value=0.14048209367087824,
        v1_source="checkpoints/grounding/metrics.json",
    ),
    Comparison(
        run_id="change_mask", tool="change_mask_v1",
        metric="f1", label="F1 (change class)",
        v1_value=0.5597365719863596,
        v1_source="checkpoints/change_mask/metrics.json",
        note="v1 was 4 epochs at 0.05M params",
    ),
    Comparison(
        run_id="change_mask", tool="change_mask_v1",
        metric="iou", label="IoU (change class)",
        v1_value=0.3886348574151661,
        v1_source="checkpoints/change_mask/metrics.json",
    ),
    Comparison(
        run_id="caption", tool="caption_v1",
        metric="bleu4_sentence_mean", label="BLEU-4",
        v1_value=0.24460787515482577,
        v1_source="checkpoints/caption/metrics.json",
    ),
    Comparison(
        run_id="change_caption", tool="change_caption_v1",
        metric="bleu4_changed", label="BLEU-4 (changed half)",
        v1_value=0.3063,
        v1_source="checkpoints/change_caption/metrics.json",
        note="the meaningful figure - v1's card says quote this, never the "
             "aggregate, which the trivially-unchanged half inflates to ~0.57. "
             "The split had been dropped from the evaluator and was reinstated "
             "on 2026-09-12; the v2 checkpoint was re-scored under it",
    ),
    Comparison(
        run_id="change_caption", tool="change_caption_v1",
        metric="bleu4_aggregate", label="BLEU-4 (aggregate)",
        v1_value=0.5686,
        v1_source="checkpoints/change_caption/metrics.json",
        note="inflated; shown only so the two rows can be read together",
    ),
    Comparison(
        run_id="optsar_fusion", tool="optsar_fusion_v1",
        metric="complementarity_gain", label="fused - best single",
        v1_value=-0.006376118942820641,
        v1_source="checkpoints/optsar_fusion/metrics.json",
        note="NEGATIVE in v1: the fused head was worse than optical alone, so "
             "the fusion added nothing. This is the number that decides "
             "whether the PS-mandatory fusion claim survives",
    ),
    Comparison(
        run_id="optsar_fusion", tool="optsar_fusion_v1",
        metric="fused", label="fused mAP",
        v1_value=0.7714183422364739,
        v1_source="checkpoints/optsar_fusion/metrics.json",
    ),
    Comparison(
        run_id="change_vqa", tool="change_vqa_v1",
        metric="miou_change_classes", label="mIoU (change classes)",
        v1_value=0.263639293535969,
        v1_source="checkpoints/change_vqa/metrics.json",
        note="pretrained-stem arm; SECOND weights remain unpublishable",
    ),
    Comparison(
        run_id="change_vqa_scratch_v2", tool="change_vqa_v1",
        metric="miou_change_classes", label="mIoU (change classes)",
        v1_value=0.263639293535969,
        v1_source="checkpoints/change_vqa/metrics.json",
        note="ABLATION: from-scratch on ~1,600 pairs, expected to lose to the "
             "pretrained arm",
    ),

    # --- Pretrained-backbone arms (Phase 5b) -------------------------------
    #
    # Not in configs/campaign.yaml: these were run after the campaign closed,
    # once the "no outbound network" premise that excluded pretrained weights
    # turned out to be false on this machine. `ckpt_dir` is resolved by the
    # fallback path, `checkpoints/v2/<run_id>`.
    Comparison(
        run_id="grounding_pre", tool="grounding_v1",
        metric="acc@0.5", label="Acc@0.5 (pretrained)",
        v1_value=0.07624890446976336,
        v1_source="checkpoints/grounding/metrics.json",
        published_range="~0.70-0.80",
        note="ImageNet ResNet-50 backbone; compare also to v2 from-scratch 0.1262",
    ),
    Comparison(
        run_id="grounding_pre", tool="grounding_v1",
        metric="miou", label="mIoU (pretrained)",
        v1_value=0.14048209367087824,
        v1_source="checkpoints/grounding/metrics.json",
    ),
    Comparison(
        run_id="caption_pre", tool="caption_v1",
        metric="bleu4_sentence_mean", label="BLEU-4 (pretrained)",
        v1_value=0.24460787515482577,
        v1_source="checkpoints/caption/metrics.json",
        note="the from-scratch v2 arm REGRESSED to 0.2255; pretraining "
             "recovers it and passes v1",
    ),
)


def ckpt_dirs_from_config() -> dict[str, str]:
    """run id -> the directory that run actually writes to.

    They are not the same thing: `change_vqa_scratch_v2` writes to
    `checkpoints/v2/change_vqa_scratch`. Assuming `ckpt_root/<run_id>` made
    that run's result invisible, and the table reported it as `pending` long
    after it had finished - a silent omission, which is the failure mode this
    file is supposed to prevent.
    """
    try:
        import yaml

        config = yaml.safe_load(
            (REPO_ROOT / "configs/campaign.yaml").read_text(encoding="utf-8")
        )
    except (OSError, ImportError):
        return {}
    return {r["id"]: r["ckpt_dir"] for r in config.get("runs", []) if r.get("ckpt_dir")}


# Runs that deliberately do not appear in `configs/campaign.yaml`, because
# they were done after the campaign closed. Listed rather than silently
# tolerated: an unknown run id is otherwise a typo that hides a result.
POST_CAMPAIGN_RUNS = frozenset({"grounding_pre", "caption_pre"})

_CKPT_DIRS = ckpt_dirs_from_config()


def load_metrics(ckpt_root: Path, run_id: str) -> dict | None:
    configured = _CKPT_DIRS.get(run_id)
    if configured:
        path = REPO_ROOT / configured / "metrics.json"
        if path.is_file():
            return _read(path)
    path = ckpt_root / run_id / "metrics.json"
    if not path.is_file():
        return None
    return _read(path)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  ! {path}: unreadable ({exc})", file=sys.stderr)
        return None


def verdict(v1: float | None, v2: float | None, higher_is_better: bool) -> str:
    if v2 is None:
        return "pending"
    if v1 is None:
        return "new"
    if v2 == v1:
        return "unchanged"
    better = (v2 > v1) if higher_is_better else (v2 < v1)
    return "BETTER" if better else "WORSE"


def build_rows(ckpt_root: Path) -> list[dict]:
    cache: dict[str, dict | None] = {}
    rows = []
    for entry in COMPARISONS:
        if entry.run_id not in cache:
            cache[entry.run_id] = load_metrics(ckpt_root, entry.run_id)
        metrics = cache[entry.run_id]
        v2 = None
        if metrics is not None:
            value = metrics.get(entry.metric)
            v2 = float(value) if isinstance(value, (int, float)) else None
        rows.append({
            "run": entry.run_id,
            "tool": entry.tool,
            "metric": entry.metric,
            "label": entry.label,
            "v1": entry.v1_value,
            "v1_source": entry.v1_source,
            "v2": v2,
            "delta": None if (v2 is None or entry.v1_value is None)
                     else v2 - entry.v1_value,
            "verdict": verdict(entry.v1_value, v2, entry.higher_is_better),
            "published_range": entry.published_range,
            "note": entry.note,
        })
    return rows


def render(rows: list[dict]) -> str:
    out = [
        "Phase 5: v2 against the published v1 baselines",
        "",
        "v1 numbers are quoted from docs/model-cards.md, never recomputed - a",
        "published number is the fixed point the comparison needs.",
        "",
        f"{'run':<22} {'metric':<24} {'v1':>9} {'v2':>9} {'delta':>9}  verdict",
        "-" * 88,
    ]
    pending = 0
    for row in rows:
        v1 = "-" if row["v1"] is None else f"{row['v1']:.4f}"
        if row["v2"] is None:
            pending += 1
            v2 = delta = "-"
        else:
            v2 = f"{row['v2']:.4f}"
            delta = f"{row['delta']:+.4f}" if row["delta"] is not None else "-"
        out.append(f"{row['run']:<22} {row['label']:<24} {v1:>9} {v2:>9} "
                   f"{delta:>9}  {row['verdict']}")
        if row["published_range"]:
            out.append(f"{'':<22} published elsewhere: {row['published_range']}")

    done = len(rows) - pending
    out += ["", f"{done}/{len(rows)} metrics measured."]

    regressions = [r for r in rows if r["verdict"] == "WORSE"]
    if regressions:
        out += ["", "REGRESSIONS - these go in the report as regressions:"]
        for row in regressions:
            out.append(f"  {row['run']}.{row['label']}: "
                       f"{row['v1']:.4f} -> {row['v2']:.4f}")
            if row["note"]:
                out.append(f"      {row['note']}")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description="Compare Phase 5 v2 results to v1.")
    p.add_argument("--checkpoints", type=Path,
                   default=REPO_ROOT / "checkpoints" / "v2",
                   help="root holding <run_id>/metrics.json")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--out", type=Path, help="also write JSON here")
    args = p.parse_args()

    rows = build_rows(args.checkpoints)
    print(json.dumps(rows, indent=2) if args.json else render(rows))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
