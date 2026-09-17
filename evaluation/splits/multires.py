"""Multi-resolution evaluation split.

Task 2.1 left a question its own test set could not answer. Multi-resolution
augmentation scored slightly *worse* than the baseline, but the official
BigEarthNet test shard is entirely 10 m - so it measures only the native
resolution, which is exactly the condition the augmentation trades away in
exchange for coarse-resolution robustness. Reporting that as "GSD
conditioning does not help" would be reading a result the experiment did not
contain.

This builds the missing evaluation: the same test patches at several
simulated resolutions, using the same block-average degradation as training
so the effective GSD label is honest. A model is then scored at each, giving:

* **native mAP** - what the old test measured
* **degraded mAP** at 20/30/40 m - what the augmentation was trained for
* **degradation slope** - how fast performance falls with coarsening, which
  is the property that actually matters for a 1.6 m target sensor

The degradation is simulated, not real. It removes high-frequency detail a
coarser sensor would not resolve, but it does not reproduce a different
sensor's optics, radiometry or noise. It answers "is this model robust to
losing spatial detail", not "does this model work on Cartosat" - the
cross-sensor test remains the only thing that answers the latter.

Usage:
    python evaluation/splits/multires.py --data data/ben_full \
        --checkpoints checkpoints/track_a_full_base checkpoints/track_a_full_multires
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.track_a_full import (  # noqa: E402
    BEN_GSD_M,
    CARTOSAT_IDX_12,
    ShardedBigEarthNet,
    batches,
    build_model,
    compute_stats,
    degrade_resolution,
)
from training.track_a_encoder import mean_average_precision  # noqa: E402

# Effective resolutions to evaluate, as multiples of BigEarthNet's 10 m.
FACTORS = (1, 2, 3, 4)


def evaluate_at(model, dataset, torch, batch_size, device, factor, keep=None):
    """mAP with inputs degraded to `factor` x the native GSD."""
    model.eval()
    rng = np.random.default_rng(0)
    scores, targets = [], []
    gsd = BEN_GSD_M * factor

    with torch.no_grad():
        for x, y in batches(dataset, batch_size, rng, shuffle=False):
            x = degrade_resolution(x, factor)
            mask = np.zeros((x.shape[0], x.shape[1]), dtype="float32")
            if keep is None:
                mask[:] = 1.0
            else:
                mask[:, keep] = 1.0
            out = model(
                torch.from_numpy(x).to(device),
                torch.from_numpy(mask).to(device),
                torch.full((x.shape[0],), gsd, device=device),
            )
            scores.append(torch.sigmoid(out).cpu().numpy())
            targets.append(y)

    value, _ = mean_average_precision(np.concatenate(scores), np.concatenate(targets))
    return value


def degradation_slope(by_factor: dict[int, float]) -> float:
    """Least-squares slope of mAP against log2(factor).

    A slope near zero means performance is resolution-robust; a steep
    negative slope means the model depends on detail it will not have at the
    target sensor. This is the number the resolution work should move.
    """
    xs = np.log2(np.array(sorted(by_factor), dtype="float64"))
    ys = np.array([by_factor[int(2 ** x)] for x in xs], dtype="float64")
    if len(xs) < 2:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def load_model(checkpoint: Path, torch, device, dim: int):
    from training.common.checkpointing import (
        find_latest_checkpoint,
        load_checkpoint,
        safe_torch_load,
    )

    latest = find_latest_checkpoint(checkpoint)
    if latest is None:
        raise SystemExit(f"no checkpoint in {checkpoint}")
    payload = safe_torch_load(latest)
    # GSD conditioning presence must match the checkpoint, or load_state_dict
    # silently leaves the FiLM layers randomly initialised.
    has_gsd = any(k.startswith("gsd_mlp.") for k in payload["model_state_dict"])
    # `arch` and `dim` travel with the weights since Phase 5. A checkpoint
    # without them is v1 at the caller's dim, which is every checkpoint
    # written before then - so nothing published changes.
    extra = payload.get("extra") or {}
    model = build_model(dim=extra.get("dim", dim), gsd_conditioning=has_gsd,
                        arch=extra.get("arch", "v1"))
    load_checkpoint(latest, model, map_location="cpu")
    return model.to(device).eval(), latest, has_gsd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_paths = sorted(Path(x) for x in glob.glob(str(args.data / "*test*.hdf5")))
    if not test_paths:
        print(f"no test shards in {args.data}", file=sys.stderr)
        return 1

    train_paths = sorted(Path(x) for x in glob.glob(str(args.data / "*train*.hdf5")))
    raw = ShardedBigEarthNet(train_paths or test_paths)
    stats = compute_stats(raw)
    raw.close()

    dataset = ShardedBigEarthNet(test_paths, stats)
    if args.limit:
        dataset.total = min(dataset.total, args.limit)
    print(f"test patches: {len(dataset)}  |  device: {device}")

    results: dict = {}
    for checkpoint in args.checkpoints:
        model, latest, has_gsd = load_model(checkpoint, torch, device, args.dim)
        name = checkpoint.name
        print(f"\n=== {name} (gsd_conditioning={has_gsd}) ===")

        all_bands, cartosat = {}, {}
        for factor in FACTORS:
            all_bands[factor] = evaluate_at(
                model, dataset, torch, args.batch_size, device, factor
            )
            cartosat[factor] = evaluate_at(
                model, dataset, torch, args.batch_size, device, factor,
                keep=CARTOSAT_IDX_12,
            )
            print(
                f"  {BEN_GSD_M*factor:5.0f} m  all-bands {all_bands[factor]:.4f}"
                f"   cartosat-4 {cartosat[factor]:.4f}"
            )

        slope = degradation_slope(all_bands)
        print(f"  degradation slope (mAP per doubling of GSD): {slope:+.4f}")
        results[name] = {
            "checkpoint": str(latest),
            "gsd_conditioning": has_gsd,
            "map_by_gsd_m": {str(BEN_GSD_M * f): all_bands[f] for f in FACTORS},
            "map_cartosat_by_gsd_m": {str(BEN_GSD_M * f): cartosat[f] for f in FACTORS},
            "degradation_slope": slope,
        }

    dataset.close()

    if len(results) == 2:
        names = list(results)
        a, b = results[names[0]], results[names[1]]
        print(
            f"\nslope {names[0]} {a['degradation_slope']:+.4f} vs "
            f"{names[1]} {b['degradation_slope']:+.4f}  "
            f"(less negative is more resolution-robust)"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
