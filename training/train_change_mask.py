"""Change-mask model: a compact siamese detector (plan task 2.4).

The plan names Change-Agent/LEVIR-MCI as first choice with **TinyCD** as the
fallback, and notes TinyCD "trains in ~1 h" at ~0.3M parameters. Change-Agent
weights are still unverified (verification item 4), and the plan is explicit
that an unverified dependency must not sit on the critical path - so the
fallback is what gets built.

Architecture: a shared encoder applied to both dates, differenced, then
decoded to a per-pixel change logit. Sharing the encoder is the point - two
independent encoders can drift so that identical input produces different
features, which appears as change that is not there.

Trained on LEVIR-CD (building change, 256px, official splits).

Usage:
    python training/train_change_mask.py --index data/levircd/index.json \
        --ckpt-dir checkpoints/change_mask --epochs 4
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

PATCH = 256


def build_model(dim: int = 16, arch: str = "v1"):
    if arch == "v2":
        from training.v2.architectures import build_change_mask

        return build_change_mask(dim=dim)

    import torch
    import torch.nn as nn

    def block(cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    class TinyChangeDetector(nn.Module):
        """Siamese encoder, absolute-difference fusion, upsampling decoder."""

        def __init__(self) -> None:
            super().__init__()
            self.enc1 = block(3, dim)
            self.enc2 = block(dim, dim * 2, stride=2)
            self.enc3 = block(dim * 2, dim * 4, stride=2)

            self.dec = nn.Sequential(
                block(dim * 4, dim * 2),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                block(dim * 2, dim),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                block(dim, dim),
            )
            self.head = nn.Conv2d(dim, 1, 1)

        def encode(self, x):
            return self.enc3(self.enc2(self.enc1(x)))

        def forward(self, a, b):
            # The SAME encoder sees both dates. Two separate encoders could
            # drift apart and report change where the imagery is identical.
            fa, fb = self.encode(a), self.encode(b)
            # Absolute difference is symmetric: swapping the dates flips the
            # sign of the change, not its magnitude.
            return self.head(self.dec(torch.abs(fa - fb)))

    return TinyChangeDetector()


class LevirCD:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        from PIL import Image

        row = self.rows[i]
        a = np.asarray(Image.open(index_path(row["a"])).convert("RGB"), dtype="float32") / 255.0
        b = np.asarray(Image.open(index_path(row["b"])).convert("RGB"), dtype="float32") / 255.0
        m = np.asarray(Image.open(index_path(row["label"])).convert("L"), dtype="float32")
        m = (m > 127).astype("float32")
        return a.transpose(2, 0, 1), b.transpose(2, 0, 1), m[None]


def batches(dataset, size, rng, shuffle=True):
    order = rng.permutation(len(dataset)) if shuffle else np.arange(len(dataset))
    for start in range(0, len(order), size):
        idx = order[start : start + size]
        a, b, m = zip(*(dataset[int(i)] for i in idx))
        yield np.stack(a), np.stack(b), np.stack(m)


def f1_iou(pred: np.ndarray, truth: np.ndarray, threshold: float = 0.5):
    """Change-class F1 and IoU.

    Scored on the CHANGE class only. LEVIR-CD is heavily imbalanced - most
    pixels are unchanged - so overall pixel accuracy would sit near 0.98 for
    a model that predicts "nothing changed" everywhere.
    """
    p = pred >= threshold
    t = truth >= 0.5
    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"f1": f1, "iou": iou, "precision": precision, "recall": recall}


def evaluate(model, dataset, torch, batch_size, device, threshold=0.5):
    model.eval()
    rng = np.random.default_rng(0)
    preds, truths = [], []
    with torch.no_grad():
        for a, b, m in batches(dataset, batch_size, rng, shuffle=False):
            logits = model(
                torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)
            )
            preds.append(torch.sigmoid(logits).cpu().numpy())
            truths.append(m)
    return f1_iou(np.concatenate(preds), np.concatenate(truths), threshold)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/change_mask"))
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--limit-train", type=int)
    p.add_argument("--limit-eval", type=int)
    p.add_argument(
        "--arch", choices=["v1", "v2"], default="v1",
        help="v1 = the published architecture; v2 = training/v2/architectures.py",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=100)
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
    test_rows = index["splits"].get("test", [])
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_eval:
        test_rows = test_rows[: args.limit_eval]

    train_ds, test_ds = LevirCD(train_rows), LevirCD(test_rows)
    print(f"train {len(train_ds)} | test {len(test_ds)}")

    model = build_model(args.dim, arch=args.arch).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e6:.3f}M")

    # Change pixels are a small minority, so an unweighted BCE converges to
    # predicting "no change" everywhere. The positive class is upweighted by
    # its measured inverse frequency.
    sample = np.concatenate([train_ds[i][2] for i in range(min(64, len(train_ds)))])
    positive_rate = float(sample.mean())
    pos_weight = torch.tensor(
        [(1 - positive_rate) / positive_rate if positive_rate > 0 else 1.0]
    ).to(device)
    print(f"change-pixel rate {positive_rate:.4f} -> pos_weight {float(pos_weight):.1f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    state = resume_or_load_for_eval(args, model, optimizer)

    write_run_metadata_unless_eval(args, {
        "task": "change_mask_tinycd", "n_train": len(train_ds),
        "epochs": args.epochs, "lr": args.lr, "dim": args.dim,
        "n_params": n_params, "pos_weight": float(pos_weight),
    })

    rng = np.random.default_rng(args.seed)
    step = state.step
    started = time.time()

    for epoch in range(state.epoch, epochs_for(args)):
        model.train()
        running, seen = 0.0, 0
        for a, b, m in batches(train_ds, args.batch_size, rng):
            ab = torch.from_numpy(a).to(device)
            bb = torch.from_numpy(b).to(device)
            mb = torch.from_numpy(m).to(device)
            loss = criterion(model(ab, bb), mb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * a.shape[0]
            seen += a.shape[0]
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

    if test_ds:
        metrics = evaluate(model, test_ds, torch, args.batch_size, device)
        print("\nLEVIR-CD test (change class only):")
        for k, v in metrics.items():
            print(f"  {k:<10} {v:.4f}")
        write_metrics(args, metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
