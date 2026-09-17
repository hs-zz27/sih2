"""Optical-SAR fusion in triad mode (plan task 2.3).

The PS requires extracting **complementary** information from a co-registered
optical + SAR pair. As docs/01 section 5.2 argues, a good fused number does
not demonstrate complementarity - a triad does. So one model carries three
heads over shared streams:

    A. optical-only    B. SAR-only    C. fused (cross-attention)

and the complementarity score is `metric(C) - max(metric(A), metric(B))`.
That is the number the PS actually asks for, and it costs three cheap forward
passes rather than three models.

Training all three heads jointly is deliberate. Separately trained models
would differ in capacity and optimisation luck, so a "gain" could be an
artefact of one being better tuned. Sharing the encoders means the only
difference between the three predictions is *which modality they see*.

Trained on WHU-OPT-SAR (~5 m, co-registered optical + SAR + land-cover).

Usage:
    python training/train_optsar_fusion.py --index data/whu_opt_sar/index.json \
        --ckpt-dir checkpoints/optsar_fusion --epochs 4
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
    TrainingState, maybe_resume, save_checkpoint, set_seed, write_run_metadata,
)
from training.common.eval_only import (  # noqa: E402
    add_eval_only_args, epochs_for, resume_or_load_for_eval,
    save_checkpoint_unless_eval, write_metrics, write_run_metadata_unless_eval,
)
from training.common.paths import index_path  # noqa: E402
from training.stage_a2_transfer import N_WHU_CLASSES  # noqa: E402
from training.track_a_encoder import average_precision  # noqa: E402

PATCH = 120


def build_model(dim: int = 32, arch: str = "v1"):
    if arch == "v2":
        from training.v2.architectures import build_optsar_fusion

        return build_optsar_fusion(dim=dim, n_classes=N_WHU_CLASSES,
                                   n_optical=4, n_sar=1)

    import torch
    import torch.nn as nn

    def stream(cin):
        return nn.Sequential(
            nn.Conv2d(cin, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim * 2), nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim * 2), nn.ReLU(inplace=True),
        )

    class FusionTriad(nn.Module):
        """Two streams, cross-attention fusion, three heads."""

        def __init__(self) -> None:
            super().__init__()
            self.optical = stream(4)   # R,G,B,NIR
            self.sar = stream(1)       # single-channel backscatter
            width = dim * 2

            # Cross-attention: each modality attends over the other's spatial
            # features, so fusion is a learned correspondence rather than a
            # concatenation that a 1x1 conv could trivially ignore.
            self.attn = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
            self.pool = nn.AdaptiveAvgPool2d(1)

            self.head_optical = nn.Linear(width, N_WHU_CLASSES)
            self.head_sar = nn.Linear(width, N_WHU_CLASSES)
            self.head_fused = nn.Linear(width * 2, N_WHU_CLASSES)

        def forward(self, optical, sar):
            fo = self.optical(optical)
            fs = self.sar(sar)
            b, c, h, w = fo.shape

            so = fo.flatten(2).transpose(1, 2)   # (B, HW, C)
            ss = fs.flatten(2).transpose(1, 2)
            # Optical queries SAR: "where in the radar does this pixel's
            # optical context find support?"
            attended, _ = self.attn(so, ss, ss)

            vo = self.pool(fo).flatten(1)
            vs = self.pool(fs).flatten(1)
            vf = torch.cat([vo, attended.mean(dim=1)], dim=1)

            return (
                self.head_optical(vo),
                self.head_sar(vs),
                self.head_fused(vf),
            )

    return FusionTriad()


class WHUPair:
    """Optical + SAR + multi-label target. Requires both modalities."""

    def __init__(self, rows: list[dict]):
        self.rows = [r for r in rows if r.get("sar")]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import rasterio

        row = self.rows[i]

        def read(path, count=None):
            with rasterio.open(path) as src:
                arr = src.read(
                    out_shape=(src.count, PATCH, PATCH), masked=True
                ).astype("float32")
            arr = np.ma.filled(arr, np.nan)
            out = []
            for band in arr:
                finite = band[np.isfinite(band)]
                if finite.size:
                    mean, std = float(finite.mean()), float(finite.std()) or 1.0
                    band = (band - mean) / std
                out.append(np.nan_to_num(band))
            stacked = np.stack(out)
            if count is not None:
                if stacked.shape[0] > count:
                    stacked = stacked[:count]
                while stacked.shape[0] < count:
                    stacked = np.concatenate([stacked, stacked[-1:]])
            return stacked

        optical = read(index_path(row["optical"]), 4)
        sar = read(index_path(row["sar"]), 1)

        with rasterio.open(index_path(row["label"])) as src:
            mask = src.read(1, out_shape=(PATCH, PATCH))
        target = np.zeros(N_WHU_CLASSES, dtype="float32")
        for value in np.unique(mask):
            if 0 <= int(value) < N_WHU_CLASSES:
                target[int(value)] = 1.0

        return optical, sar, target


def batches(dataset, size, rng, shuffle=True):
    order = rng.permutation(len(dataset)) if shuffle else np.arange(len(dataset))
    for start in range(0, len(order), size):
        idx = order[start : start + size]
        o, s, t = zip(*(dataset[int(i)] for i in idx))
        yield np.stack(o), np.stack(s), np.stack(t)


def map_score(scores, targets) -> float:
    per_class = [
        average_precision(scores[:, c], targets[:, c]) for c in range(scores.shape[1])
    ]
    valid = [v for v in per_class if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def evaluate_triad(model, dataset, torch, batch_size, device) -> dict:
    """mAP for each arm plus the complementarity gain."""
    model.eval()
    rng = np.random.default_rng(0)
    acc = {"optical": [], "sar": [], "fused": []}
    targets = []
    with torch.no_grad():
        for o, s, t in batches(dataset, batch_size, rng, shuffle=False):
            ob = torch.from_numpy(o).to(device)
            sb = torch.from_numpy(s).to(device)
            lo, ls, lf = model(ob, sb)
            acc["optical"].append(torch.sigmoid(lo).cpu().numpy())
            acc["sar"].append(torch.sigmoid(ls).cpu().numpy())
            acc["fused"].append(torch.sigmoid(lf).cpu().numpy())
            targets.append(t)

    y = np.concatenate(targets)
    result = {k: map_score(np.concatenate(v), y) for k, v in acc.items()}
    best_single = max(result["optical"], result["sar"])
    result["best_single"] = best_single
    result["complementarity_gain"] = result["fused"] - best_single
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/optsar_fusion"))
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--limit-train", type=int)
    p.add_argument("--limit-eval", type=int)
    p.add_argument(
        "--arch", choices=["v1", "v2"], default="v1",
        help="v1 = the published architecture; v2 = training/v2/architectures.py",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    add_eval_only_args(p)
    args = p.parse_args()

    import torch
    import torch.nn as nn

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    index = json.loads(args.index.read_text(encoding="utf-8"))
    train_rows = index["splits"]["train"]
    val_rows = index["splits"].get("validation", [])
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_eval:
        val_rows = val_rows[: args.limit_eval]

    train_ds, val_ds = WHUPair(train_rows), WHUPair(val_rows)
    print(f"train {len(train_ds)} | validation {len(val_ds)} (paired only)")

    model = build_model(args.dim, arch=args.arch).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    state = resume_or_load_for_eval(args, model, optimizer)

    write_run_metadata_unless_eval(args, {
        "task": "optsar_fusion_triad", "n_train": len(train_ds),
        "epochs": args.epochs, "lr": args.lr, "dim": args.dim,
        "classes": index["classes"], "split_method": index.get("split_method"),
    })

    rng = np.random.default_rng(args.seed)
    step = state.step
    started = time.time()

    for epoch in range(state.epoch, epochs_for(args)):
        model.train()
        running, seen = 0.0, 0
        for o, s, t in batches(train_ds, args.batch_size, rng):
            ob = torch.from_numpy(o).to(device)
            sb = torch.from_numpy(s).to(device)
            tb = torch.from_numpy(t).to(device)

            lo, ls, lf = model(ob, sb)
            # All three arms are trained jointly on the same batch, so a
            # measured gain cannot be an artefact of one arm being better
            # tuned than another.
            loss = criterion(lo, tb) + criterion(ls, tb) + criterion(lf, tb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * o.shape[0]
            seen += o.shape[0]
            step += 1
            if step % args.save_every == 0:
                state.step, state.epoch = step, epoch
                save_checkpoint_unless_eval(
                    args, step, model, optimizer, state=state,
                    extra={"arch": args.arch, "dim": args.dim},
                )

        print(f"epoch {epoch+1}/{args.epochs}  loss {running/max(seen,1):.4f}  "
              f"({time.time()-started:.0f}s)", flush=True)

    state.step, state.epoch = step, args.epochs
    save_checkpoint_unless_eval(
        args, step, model, optimizer, state=state,
        extra={"arch": args.arch, "dim": args.dim},
    )

    if val_ds:
        m = evaluate_triad(model, val_ds, torch, args.batch_size, device)
        print("\nWHU-OPT-SAR validation, triad mAP:")
        print(f"  A optical-only : {m['optical']:.4f}")
        print(f"  B SAR-only     : {m['sar']:.4f}")
        print(f"  C fused        : {m['fused']:.4f}")
        print(f"  complementarity gain (C - best single): {m['complementarity_gain']:+.4f}")
        write_metrics(args, m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
