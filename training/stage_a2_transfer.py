"""Stage A2: resolution bridge on WHU-OPT-SAR ~5 m (plan task 2.2).

WHY THIS IS THE PRIORITY

The cross-sensor test measured where Track A actually fails. On the real
Cartosat product, agreement with the physics engine held for water (+0.69)
and built-up (+0.73) but **collapsed for vegetation, from +0.476 at 10 m to
-0.135 at native 1.6 m**. Band adaptation was comparatively cheap. The
dominant gap is resolution, not bands - so bridging 10 m -> 5 m -> 1.6 m is
the highest-value adaptation available, which is what Stage A2 does.

APPROACH

The Track A encoder is loaded and its head replaced: BigEarthNet's 19 classes
give way to WHU-OPT-SAR's 7. The encoder itself carries over, which is the
whole point - we are adapting learned features to a finer scale, not training
a new model.

Segmentation masks are reduced to per-tile multi-label presence vectors.
Stage A2 exists to move the *encoder* to 5 m features; adding a segmentation
decoder would train a different thing and confound the result.

Band dropout is kept on, so the adapted encoder stays usable on Cartosat's
4-band VNIR.

Usage:
    python training/stage_a2_transfer.py --index data/whu_opt_sar/index.json \
        --init checkpoints/track_a_dropout --ckpt-dir checkpoints/stage_a2 --epochs 3
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
    TrainingState, find_latest_checkpoint, load_checkpoint, maybe_resume,
    safe_torch_load, save_checkpoint, set_seed, write_run_metadata,
)
from training.common.paths import index_path  # noqa: E402
from training.track_a_encoder import (  # noqa: E402
    BAND_NAMES, CARTOSAT_INDICES, band_dropout_mask, build_model,
    iterate_batches, mean_average_precision,
)

# WHU-OPT-SAR optical is 4-band (R, G, B, NIR). Those map onto the encoder's
# 10 canonical slots so the learned band embedding still sees RED where RED
# belongs - the same discipline the Cartosat path uses.
WHU_BAND_ORDER = ["RED", "GREEN", "BLUE", "NIR"]
WHU_TO_SLOT = {
    "BLUE": BAND_NAMES.index("B02"),
    "GREEN": BAND_NAMES.index("B03"),
    "RED": BAND_NAMES.index("B04"),
    "NIR": BAND_NAMES.index("B08"),
}
N_WHU_CLASSES = 8
PATCH = 120


def load_index(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class WHUOptSar:
    """Optical tile -> multi-label class presence vector."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import rasterio

        row = self.rows[i]
        with rasterio.open(index_path(row["optical"])) as src:
            arr = src.read(
                out_shape=(src.count, PATCH, PATCH), masked=True
            ).astype("float32")
        arr = np.ma.filled(arr, np.nan)

        cube = np.zeros((len(BAND_NAMES), PATCH, PATCH), dtype="float32")
        present = np.zeros(len(BAND_NAMES), dtype="float32")
        for band_index, name in enumerate(WHU_BAND_ORDER[: arr.shape[0]]):
            slot = WHU_TO_SLOT[name]
            band = arr[band_index]
            finite = band[np.isfinite(band)]
            if finite.size:
                mean, std = float(finite.mean()), float(finite.std()) or 1.0
                band = (band - mean) / std
            cube[slot] = np.nan_to_num(band)
            present[slot] = 1.0

        with rasterio.open(index_path(row["label"])) as src:
            mask = src.read(1, out_shape=(PATCH, PATCH))
        target = np.zeros(N_WHU_CLASSES, dtype="float32")
        for value in np.unique(mask):
            if 0 <= int(value) < N_WHU_CLASSES:
                target[int(value)] = 1.0

        return cube, target, present


def batches(dataset, size, rng, shuffle=True):
    order = rng.permutation(len(dataset)) if shuffle else np.arange(len(dataset))
    for start in range(0, len(order), size):
        idx = order[start : start + size]
        items = [dataset[int(i)] for i in idx]
        xs, ys, ps = zip(*items)
        yield np.stack(xs), np.stack(ys), np.stack(ps)


def load_pretrained_encoder(init: Path, torch, device, dim: int):
    """Load Track A weights, keeping the encoder and discarding the old head.

    The head is 19-way for BigEarthNet and 8-way here, so it cannot transfer.
    Loading with strict=False and reporting exactly which tensors were skipped
    keeps that explicit - a silent partial load is how people end up training
    a randomly-initialised encoder and reporting it as fine-tuning.
    """
    model = build_model(n_bands=len(BAND_NAMES), dim=dim)
    latest = find_latest_checkpoint(init)
    if latest is None:
        print(f"WARNING: no checkpoint in {init}; training from scratch")
        return model.to(device), None

    payload = safe_torch_load(latest)
    state = payload["model_state_dict"]
    encoder_state = {k: v for k, v in state.items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    print(f"loaded encoder from {latest}")
    print(f"  reinitialised (head): {sorted(missing)}")
    if unexpected:
        print(f"  ignored: {sorted(unexpected)}")
    return model.to(device), latest


def replace_head(model, torch, dim: int, n_classes: int):
    import torch.nn as nn

    model.head = nn.Linear(dim * 4, n_classes)
    return model


def evaluate(model, dataset, torch, batch_size, device, cartosat_only=False):
    model.eval()
    rng = np.random.default_rng(0)
    scores, targets = [], []
    with torch.no_grad():
        for x, y, present in batches(dataset, batch_size, rng, shuffle=False):
            mask = present.copy()
            if cartosat_only:
                keep = np.zeros_like(mask)
                keep[:, CARTOSAT_INDICES] = 1.0
                mask = mask * keep
            xb = torch.from_numpy(x).to(device)
            mb = torch.from_numpy(mask).to(device)
            scores.append(torch.sigmoid(model(xb, mb)).cpu().numpy())
            targets.append(y)
    return mean_average_precision_n(
        np.concatenate(scores), np.concatenate(targets)
    )


def mean_average_precision_n(scores, targets):
    """mAP over however many classes this dataset has."""
    from training.track_a_encoder import average_precision

    per_class = [
        average_precision(scores[:, c], targets[:, c]) for c in range(scores.shape[1])
    ]
    valid = [v for v in per_class if not np.isnan(v)]
    return (float(np.mean(valid)) if valid else float("nan")), per_class


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--init", type=Path, help="Track A checkpoint to start from")
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/stage_a2"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--band-dropout", type=float, default=0.3)
    p.add_argument("--limit-train", type=int)
    p.add_argument("--limit-eval", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--freeze-encoder", action="store_true",
        help="train only the new head; separates adaptation from forgetting",
    )
    args = p.parse_args()

    import torch
    import torch.nn as nn

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    index = load_index(args.index)
    train_rows = index["splits"]["train"]
    val_rows = index["splits"].get("validation", [])
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_eval:
        val_rows = val_rows[: args.limit_eval]

    train_ds, val_ds = WHUOptSar(train_rows), WHUOptSar(val_rows)
    print(f"train {len(train_ds)} | validation {len(val_ds)}")
    print(f"split note: {index.get('split_method', 'unspecified')}")

    model, source = load_pretrained_encoder(args.init or Path("."), torch, device, args.dim)
    model = replace_head(model, torch, args.dim, N_WHU_CLASSES).to(device)

    if args.freeze_encoder:
        # Only the head learns. If cross-sensor agreement holds here but
        # collapses with a fully fine-tuned encoder, the cause is
        # catastrophic forgetting rather than the bridge itself.
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("head.")
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"frozen encoder: training {sum(p.numel() for p in trainable)} params only")
    else:
        trainable = list(model.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    state, _ = maybe_resume(args.ckpt_dir, model, optimizer, enabled=args.resume)

    write_run_metadata(args.ckpt_dir, {
        "task": "stage_a2_resolution_bridge",
        "initialised_from": str(source) if source else None,
        "n_train": len(train_ds),
        "epochs": args.epochs,
        "lr": args.lr,
        "band_dropout": args.band_dropout,
        "freeze_encoder": args.freeze_encoder,
        "classes": index["classes"],
        "split_method": index.get("split_method"),
    })

    rng = np.random.default_rng(args.seed)
    step = state.step
    started = time.time()

    for epoch in range(state.epoch, args.epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y, present in batches(train_ds, args.batch_size, rng):
            # Dropout is applied *within* the bands this sensor actually has.
            dropout = band_dropout_mask(x.shape[0], x.shape[1], args.band_dropout, rng)
            mask = present * dropout
            # Never leave a sample with nothing present.
            empty = mask.sum(axis=1) == 0
            mask[empty] = present[empty]

            xb = torch.from_numpy(x).to(device)
            yb = torch.from_numpy(y).to(device)
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

        print(f"epoch {epoch+1}/{args.epochs}  loss {running/max(seen,1):.4f}  "
              f"({time.time()-started:.0f}s)", flush=True)

    state.step, state.epoch = step, args.epochs
    save_checkpoint(args.ckpt_dir, step, model, optimizer, state=state)

    if val_ds:
        full, _ = evaluate(model, val_ds, torch, args.batch_size, device)
        cart, _ = evaluate(model, val_ds, torch, args.batch_size, device, cartosat_only=True)
        print(f"\nWHU-OPT-SAR validation mAP : {full:.4f}")
        print(f"  restricted to Cartosat bands: {cart:.4f}")
        (args.ckpt_dir / "metrics.json").write_text(
            json.dumps({"map_all_bands": full, "map_cartosat_bands": cart}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
