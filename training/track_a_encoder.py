"""Track A v0: band-agnostic encoder + land-cover head (plan task 1.10).

This is the component the project's central claim rests on. The evaluation
sensor (Cartosat-2E MX) carries **4 VNIR bands and no SWIR** - confirmed from
real product metadata, docs/verification.md item 6 - while the training data
(Sentinel-2) carries 10. A model that hard-codes 10 input channels simply
cannot run on the target sensor.

The mechanism, per docs/03 section 3: **random band dropout during training**
(p ~ 0.3 per band, always retaining at least 3). Three design choices make it
work:

1. **A shared per-band stem.** Every band goes through the same 1-channel
   convolution, so the network never learns a fixed channel layout.
2. **A learned band-identity embedding.** Without it a random subset is
   ambiguous - the model could not tell NIR from SWIR1 and would have to
   average over the confusion. With it, "band 7 is NIR" is explicit.
3. **Masked mean pooling across bands.** Absent bands contribute nothing
   rather than contributing zeros, so the representation's scale does not
   collapse when bands are missing.

Together these mean inference with an arbitrary band subset is the same
operation as inference with all of them, not a degraded special case.

Usage:
    python training/track_a_encoder.py --index data/bigearthnet_14k/index.json \
        --ckpt-dir checkpoints/track_a --epochs 3
    python training/track_a_encoder.py --index ... --eval-only --ablation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.checkpointing import (  # noqa: E402
    TrainingState,
    maybe_resume,
    save_checkpoint,
    set_seed,
    write_run_metadata,
)
from training.common.paths import index_path  # noqa: E402

# reBEN 10-band order (60 m atmospheric bands dropped, per docs/03 section 4.1).
BAND_NAMES = [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
]
CANONICAL = {
    "B02": "BLUE", "B03": "GREEN", "B04": "RED", "B08": "NIR",
    "B11": "SWIR1", "B12": "SWIR2",
}

# The target sensor: Cartosat-2E MX is 4-band VNIR, no SWIR, no red edge.
CARTOSAT_BANDS = ["B02", "B03", "B04", "B08"]
CARTOSAT_INDICES = [BAND_NAMES.index(b) for b in CARTOSAT_BANDS]

# Sentinel-2 L2A surface reflectance is scaled by 10000.
REFLECTANCE_SCALE = 10000.0

N_CLASSES = 19
MIN_BANDS_KEPT = 3


def load_index(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class BigEarthNetS2:
    """Minimal dataset: 10-band S2 patch -> multi-hot label vector."""

    def __init__(self, rows: list[dict], band_stats: tuple | None = None):
        self.rows = rows
        self.band_stats = band_stats

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import rasterio

        row = self.rows[i]
        with rasterio.open(index_path(row["s2"])) as src:
            arr = src.read().astype("float32")  # (C, H, W)

        arr /= REFLECTANCE_SCALE
        if self.band_stats is not None:
            mean, std = self.band_stats
            arr = (arr - mean[:, None, None]) / std[:, None, None]

        target = np.zeros(N_CLASSES, dtype="float32")
        for c in row["labels"]:
            target[c] = 1.0
        return arr, target


def compute_band_stats(rows: list[dict], sample: int = 400, seed: int = 0):
    """Per-band mean/std over a sample of the training split."""
    import rasterio

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(rows), size=min(sample, len(rows)), replace=False)
    acc = []
    for i in picks:
        with rasterio.open(index_path(rows[int(i)]["s2"])) as src:
            acc.append(src.read().astype("float32").reshape(src.count, -1))
    stacked = np.concatenate(acc, axis=1) / REFLECTANCE_SCALE
    mean = stacked.mean(axis=1)
    std = stacked.std(axis=1)
    std[std < 1e-6] = 1.0
    return mean.astype("float32"), std.astype("float32")


def band_dropout_mask(batch: int, n_bands: int, p: float, rng: np.random.Generator):
    """Random per-band presence mask, always retaining >= MIN_BANDS_KEPT.

    The guarantee matters: a sample with zero bands has no signal at all and
    would inject a meaningless gradient.
    """
    keep = rng.random((batch, n_bands)) > p
    for i in range(batch):
        if keep[i].sum() < MIN_BANDS_KEPT:
            idx = rng.choice(n_bands, size=MIN_BANDS_KEPT, replace=False)
            keep[i] = False
            keep[i, idx] = True
    return keep.astype("float32")


def build_model(n_bands: int = 10, dim: int = 64):
    import torch
    import torch.nn as nn

    class BandAgnosticEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Shared across bands: the network never sees a fixed channel layout.
            self.stem = nn.Sequential(
                nn.Conv2d(1, dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            )
            # Learned band identity. Without this a subset is ambiguous.
            self.band_embed = nn.Parameter(torch.zeros(n_bands, dim))
            self.trunk = nn.Sequential(
                nn.Conv2d(dim, dim, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dim * 2), nn.ReLU(inplace=True),
                nn.Conv2d(dim * 2, dim * 4, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dim * 4), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(dim * 4, N_CLASSES)

        def forward(self, x, mask):
            b, c, h, w = x.shape
            # Zero the absent bands before the stem so BatchNorm statistics are
            # not polluted by values the model is meant not to see.
            x = x * mask[:, :, None, None]
            z = self.stem(x.reshape(b * c, 1, h, w)).reshape(b, c, -1, h, w)
            z = z + self.band_embed[None, :, :, None, None]

            m = mask[:, :, None, None, None]
            # Masked mean: absent bands contribute nothing, and the divisor is
            # the number of present bands, so scale is preserved.
            z = (z * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

            return self.head(self.trunk(z).flatten(1))

    return BandAgnosticEncoder()


def average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    """Average precision for one class (area under precision-recall)."""
    if targets.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)
    t = targets[order]
    tp = np.cumsum(t)
    precision = tp / np.arange(1, len(t) + 1)
    return float((precision * t).sum() / t.sum())


def mean_average_precision(scores: np.ndarray, targets: np.ndarray) -> tuple[float, list]:
    per_class = [average_precision(scores[:, c], targets[:, c]) for c in range(N_CLASSES)]
    valid = [v for v in per_class if not np.isnan(v)]
    return (float(np.mean(valid)) if valid else float("nan")), per_class


def iterate_batches(dataset, batch_size, rng, shuffle=True):
    order = rng.permutation(len(dataset)) if shuffle else np.arange(len(dataset))
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        xs, ys = zip(*(dataset[int(i)] for i in idx))
        yield np.stack(xs), np.stack(ys)


def evaluate(model, dataset, torch, batch_size, device, keep_indices=None):
    """Evaluate, optionally with only a subset of bands present.

    `keep_indices=None` means all bands. Passing CARTOSAT_INDICES simulates the
    real evaluation sensor, which is the ablation the plan asks for.
    """
    model.eval()
    rng = np.random.default_rng(0)
    all_scores, all_targets = [], []

    with torch.no_grad():
        for x, y in iterate_batches(dataset, batch_size, rng, shuffle=False):
            xb = torch.from_numpy(x).to(device)
            mask = np.zeros((x.shape[0], x.shape[1]), dtype="float32")
            if keep_indices is None:
                mask[:] = 1.0
            else:
                mask[:, keep_indices] = 1.0
            mb = torch.from_numpy(mask).to(device)
            logits = model(xb, mb)
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(y)

    return mean_average_precision(
        np.concatenate(all_scores), np.concatenate(all_targets)
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/track_a"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--band-dropout", type=float, default=0.3)
    p.add_argument("--limit-train", type=int)
    p.add_argument("--limit-eval", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument(
        "--ablation", action="store_true",
        help="also report mAP with only Cartosat's 4 VNIR bands present",
    )
    p.add_argument("--no-band-dropout", action="store_true",
                   help="train without band dropout, for the ablation baseline")
    return p


def main() -> int:
    args = build_parser().parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("torch is required: pip install torch", file=sys.stderr)
        return 1

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    index = load_index(args.index)
    train_rows = index["splits"]["train"]
    test_rows = index["splits"].get("test", [])
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_eval:
        test_rows = test_rows[: args.limit_eval]

    print("computing band statistics...")
    stats = compute_band_stats(train_rows)
    train_ds = BigEarthNetS2(train_rows, stats)
    test_ds = BigEarthNetS2(test_rows, stats)
    print(f"train {len(train_ds)} | test {len(test_ds)}")

    model = build_model(n_bands=len(BAND_NAMES), dim=args.dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    state, _ = maybe_resume(args.ckpt_dir, model, optimizer, enabled=args.resume)

    if not args.eval_only:
        write_run_metadata(
            args.ckpt_dir,
            {
                "task": "track_a_encoder_v0",
                "bands": BAND_NAMES,
                "band_dropout": 0.0 if args.no_band_dropout else args.band_dropout,
                "n_train": len(train_ds),
                "epochs": args.epochs,
                "lr": args.lr,
                "dim": args.dim,
                "seed": args.seed,
                "n_params": n_params,
            },
        )

        rng = np.random.default_rng(args.seed)
        step = state.step
        started = time.time()

        for epoch in range(state.epoch, args.epochs):
            model.train()
            running, seen = 0.0, 0
            for x, y in iterate_batches(train_ds, args.batch_size, rng):
                xb = torch.from_numpy(x).to(device)
                yb = torch.from_numpy(y).to(device)

                if args.no_band_dropout:
                    mask = np.ones((x.shape[0], x.shape[1]), dtype="float32")
                else:
                    mask = band_dropout_mask(
                        x.shape[0], x.shape[1], args.band_dropout, rng
                    )
                mb = torch.from_numpy(mask).to(device)

                loss = criterion(model(xb, mb), yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running += loss.item() * x.shape[0]
                seen += x.shape[0]
                step += 1

                if step % args.save_every == 0:
                    state.step, state.epoch = step, epoch
                    save_checkpoint(args.ckpt_dir, step, model, optimizer, state=state)

            print(
                f"epoch {epoch+1}/{args.epochs}  loss {running/max(seen,1):.4f}  "
                f"({time.time()-started:.0f}s)",
                flush=True,
            )

        state.step, state.epoch = step, args.epochs
        save_checkpoint(args.ckpt_dir, step, model, optimizer, state=state)

    # --- Evaluation -------------------------------------------------------
    print("\nEvaluating on the official test split...")
    full_map, per_class = evaluate(model, test_ds, torch, args.batch_size, device)
    print(f"  all {len(BAND_NAMES)} bands      : mAP {full_map:.4f}")

    results = {"map_all_bands": full_map}

    if args.ablation:
        # The headline claim: a model trained on 10 bands must still work when
        # only Cartosat's 4 VNIR bands are available.
        cartosat_map, _ = evaluate(
            model, test_ds, torch, args.batch_size, device,
            keep_indices=CARTOSAT_INDICES,
        )
        retention = cartosat_map / full_map if full_map else float("nan")
        print(
            f"  Cartosat 4-band VNIR : mAP {cartosat_map:.4f} "
            f"({retention*100:.1f}% of full-band performance)"
        )
        print(f"    bands used: {[CANONICAL.get(b, b) for b in CARTOSAT_BANDS]}")
        results["map_cartosat_4band"] = cartosat_map
        results["retention"] = retention

    print("\n  per-class AP:")
    for name, ap in sorted(
        zip(index["classes"], per_class, strict=True),
        key=lambda kv: -(kv[1] if not np.isnan(kv[1]) else -1),
    ):
        print(f"    {ap:.4f}  {name[:60]}")

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    (args.ckpt_dir / "metrics.json").write_text(
        json.dumps({**results, "per_class_ap": dict(zip(index["classes"], per_class,
                                                        strict=True))}, indent=2),
        encoding="utf-8",
    )
    print(f"\nMetrics written to {args.ckpt_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
