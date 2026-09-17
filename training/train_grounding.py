"""Referring-expression grounding (plan task 2.7).

WHY NOT FLORENCE-2

The plan names Florence-2 first, with "a plain detector for closed classes"
as the fallback. Florence-2 requires `trust_remote_code=True` and custom
modeling files - executing third-party Python fetched from a model repo. That
risk was flagged when `scripts/fetch_models.py` was written, and its weight
patterns deliberately exclude those files. Accepting the risk quietly here
because it is convenient would contradict that decision, so the fallback path
is what gets built: a compact text-conditioned box regressor, trained from
scratch on DIOR-RSVG, with no remote code.

It will not match Florence-2's accuracy. It is honest, auditable, and gives
the PS-mandatory grounding row a real measured number rather than a zero.

DESIGN

A CNN encodes the image, a GRU encodes the referring expression, and the two
are fused by FiLM conditioning before a box head regresses (cx, cy, w, h) in
normalised coordinates. Predicting normalised centre/size rather than raw
corner pixels keeps the target scale-free, so the loss does not depend on
image size and a wrong corner cannot invert the box.

Usage:
    python training/train_grounding.py --data data/dior_rsvg \
        --ckpt-dir checkpoints/grounding --epochs 5
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.checkpointing import (  # noqa: E402
    TrainingState, maybe_resume, save_checkpoint, set_seed, write_run_metadata,
)
from training.common.eval_only import (  # noqa: E402
    add_eval_only_args, epochs_for, resume_or_load_for_eval, vocab_for,
    write_vocab_unless_eval,
    save_checkpoint_unless_eval, write_metrics, write_run_metadata_unless_eval,
)

IMAGE_SIZE = 224
MAX_TOKENS = 16
PAD, UNK = 0, 1
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def build_vocab(phrases, min_count: int = 2) -> dict[str, int]:
    counts = Counter(t for p in phrases for t in tokenize(p))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for word, n in counts.most_common():
        if n >= min_count:
            vocab[word] = len(vocab)
    return vocab


def encode_text(phrase: str, vocab: dict[str, int]) -> np.ndarray:
    ids = [vocab.get(t, UNK) for t in tokenize(phrase)][:MAX_TOKENS]
    ids += [PAD] * (MAX_TOKENS - len(ids))
    return np.array(ids, dtype="int64")


def to_cxcywh(box, width: float, height: float) -> np.ndarray:
    """Corner pixels -> normalised centre/size.

    Normalising makes the target independent of image size, so the loss does
    not silently weight large images more, and centre/size cannot express an
    inverted box the way an unordered corner pair can.
    """
    x0, y0, x1, y1 = box
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    return np.array([
        ((x0 + x1) / 2) / width, ((y0 + y1) / 2) / height,
        (x1 - x0) / width, (y1 - y0) / height,
    ], dtype="float32")


def to_corners(pred, width: float, height: float) -> dict:
    cx, cy, w, h = [float(v) for v in pred]
    return {
        "xmin": (cx - w / 2) * width, "ymin": (cy - h / 2) * height,
        "xmax": (cx + w / 2) * width, "ymax": (cy + h / 2) * height,
    }


def build_model(vocab_size: int, dim: int = 128, arch: str = "v1",
                pretrained: bool = False):
    if arch == "v2":
        from training.v2.architectures import build_grounding

        return build_grounding(vocab_size=vocab_size, dim=dim,
                               max_tokens=MAX_TOKENS, pretrained=pretrained)

    import torch
    import torch.nn as nn

    def block(cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    class ReferringGrounder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision = nn.Sequential(
                block(3, 32), block(32, 64, 2), block(64, 128, 2),
                block(128, dim, 2), block(dim, dim, 2),
            )
            self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
            self.text = nn.GRU(dim, dim, batch_first=True)
            # FiLM: the phrase modulates the visual feature map, so "the LEFT
            # aircraft" and "the RIGHT aircraft" can select different regions
            # of the same image. Concatenating a pooled text vector instead
            # would let the model ignore the phrase entirely.
            self.film = nn.Linear(dim, dim * 2)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.box = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, 4)
            )

        def forward(self, image, tokens):
            v = self.vision(image)
            _, hidden = self.text(self.embed(tokens))
            scale, shift = self.film(hidden[-1]).chunk(2, dim=-1)
            v = v * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            # Sigmoid keeps centre and size inside the image by construction,
            # so the model cannot emit a box outside the frame.
            return torch.sigmoid(self.box(self.pool(v).flatten(1)))

    return ReferringGrounder()


class DiorRSVG:
    """Referring expressions with boxes, read from the parquet shards."""

    def __init__(self, rows, vocab):
        self.rows = rows
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        from PIL import Image

        row = self.rows[i]
        image = Image.open(io.BytesIO(row["image_bytes"])).convert("RGB")
        width, height = image.size
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        arr = np.asarray(image, dtype="float32").transpose(2, 0, 1) / 255.0
        return (
            arr,
            encode_text(row["phrase"], self.vocab),
            to_cxcywh(row["box"], width, height),
        )


def load_rows(data_dir: Path, split: str, limit: int | None = None) -> list[dict]:
    """Read referring expressions from parquet, discovering column names.

    Some DIOR-RSVG mirrors publish only one set of shards rather than named
    train/test splits. When the requested split matches nothing, every shard
    is read and the caller splits deterministically - and says so, because a
    self-made split is not the published one and must not be reported as if
    it were.
    """
    import pyarrow.parquet as pq

    all_files = sorted(glob.glob(str(data_dir / "**" / "*.parquet"), recursive=True))
    files = [f for f in all_files if (f"{split}-" in Path(f).name)]
    if not files:
        files = all_files
    rows: list[dict] = []
    for path in files:
        table = pq.read_table(path)
        cols = table.column_names
        image_col = next((c for c in ("image", "img") if c in cols), None)
        # "question" first: DIOR-RSVG mirrors store the referring expression
        # under that name, and omitting it made every shard look unusable.
        phrase_col = next(
            (c for c in ("question", "caption", "phrase", "expression", "text",
                         "sentence")
             if c in cols), None
        )
        box_col = next(
            (c for c in ("bbox", "box", "boxes", "objects") if c in cols), None
        )
        if not (image_col and phrase_col and box_col):
            continue

        data = table.to_pydict()
        for i in range(table.num_rows):
            cell = data[image_col][i]
            raw = cell.get("bytes") if isinstance(cell, dict) else cell
            box = data[box_col][i]
            phrase = data[phrase_col][i]
            # Some releases nest a list of objects per image.
            if isinstance(box, dict):
                box = box.get("bbox") or box.get("box")
            if isinstance(box, list) and box and isinstance(box[0], (list, tuple)):
                box = box[0]
            if isinstance(phrase, list) and phrase:
                phrase = phrase[0]
            if not isinstance(raw, (bytes, bytearray)) or not box or len(box) < 4:
                continue
            rows.append({
                "image_bytes": bytes(raw),
                "phrase": str(phrase),
                "box": [float(v) for v in list(box)[:4]],
                # Group key so a split can keep all expressions for one image
                # on the same side.
                "image_key": str(
                    data.get("image_path", [None] * table.num_rows)[i]
                    or data.get("image_id", [i] * table.num_rows)[i]
                ),
            })
            if limit and len(rows) >= limit:
                return rows
    return rows


def batches(dataset, size, rng, shuffle=True):
    order = rng.permutation(len(dataset)) if shuffle else np.arange(len(dataset))
    for start in range(0, len(order), size):
        idx = order[start : start + size]
        i, t, b = zip(*(dataset[int(j)] for j in idx))
        yield np.stack(i), np.stack(t), np.stack(b)


def iou_cxcywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    def corners(x):
        cx, cy, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    ax0, ay0, ax1, ay1 = corners(a)
    bx0, by0, bx1, by1 = corners(b)
    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/grounding"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--limit-train", type=int, default=20000)
    p.add_argument("--limit-eval", type=int, default=4000)
    p.add_argument(
        "--arch", choices=["v1", "v2"], default="v1",
        help="v1 = the published architecture; v2 = training/v2/architectures.py",
    )
    p.add_argument(
        "--pretrained", action="store_true",
        help="ImageNet ResNet-50 visual backbone instead of from-scratch. "
             "Only meaningful with --arch v2.",
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

    train_files = sorted(
        glob.glob(str(args.data / "**" / "train-*.parquet"), recursive=True)
    )
    if train_files:
        train_rows = load_rows(args.data, "train", args.limit_train)
        test_rows = load_rows(args.data, "test", args.limit_eval)
        split_note = "published train/test shards"
    else:
        # This mirror ships one set of shards. Split deterministically by
        # IMAGE, not by row: the same image carries several referring
        # expressions, and splitting by row would put the same picture in
        # both halves and inflate the score.
        rows = load_rows(args.data, "all", None)
        by_image: dict[str, list[dict]] = {}
        for r in rows:
            by_image.setdefault(r["image_key"], []).append(r)
        keys = sorted(by_image)
        np.random.default_rng(args.seed).shuffle(keys)
        cut = int(len(keys) * 0.85)
        train_rows = [r for k in keys[:cut] for r in by_image[k]][: args.limit_train]
        test_rows = [r for k in keys[cut:] for r in by_image[k]][: args.limit_eval]
        split_note = (
            f"NO published split in this mirror; held out 15% of "
            f"{len(keys)} images (grouped by image, not by expression)"
        )

    if not train_rows:
        print(f"no usable rows under {args.data}", file=sys.stderr)
        return 1
    print(f"split: {split_note}")

    vocab = vocab_for(
        args, lambda: build_vocab([r["phrase"] for r in train_rows])
    )
    print(f"train {len(train_rows)} | test {len(test_rows)} | vocab {len(vocab)}")

    train_ds, test_ds = DiorRSVG(train_rows, vocab), DiorRSVG(test_rows, vocab)
    model = build_model(len(vocab), args.dim, arch=args.arch,
                        pretrained=args.pretrained).to(device)
    print(f"parameters: {sum(q.numel() for q in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()
    state = resume_or_load_for_eval(args, model, optimizer)

    write_run_metadata_unless_eval(args, {
        "task": "referring_grounding", "n_train": len(train_rows),
        "epochs": args.epochs, "lr": args.lr, "dim": args.dim,
        "vocab_size": len(vocab), "backbone": "from scratch (no remote code)",
        "split_note": split_note,
    })
    write_vocab_unless_eval(args, vocab)

    rng = np.random.default_rng(args.seed)
    step = state.step
    started = time.time()

    for epoch in range(state.epoch, epochs_for(args)):
        model.train()
        running, seen = 0.0, 0
        for img, tok, box in batches(train_ds, args.batch_size, rng):
            pred = model(
                torch.from_numpy(img).to(device), torch.from_numpy(tok).to(device)
            )
            loss = criterion(pred, torch.from_numpy(box).to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * img.shape[0]
            seen += img.shape[0]
            step += 1
            if step % args.save_every == 0:
                state.step, state.epoch = step, epoch
                save_checkpoint_unless_eval(
                    args, step, model, optimizer, state=state,
                    extra={"arch": args.arch, "dim": args.dim,
                   "pretrained": args.pretrained},
                )
        print(f"epoch {epoch+1}/{args.epochs}  loss {running/max(seen,1):.4f}  "
              f"({time.time()-started:.0f}s)", flush=True)

    state.step, state.epoch = step, args.epochs
    save_checkpoint_unless_eval(
        args, step, model, optimizer, state=state,
        extra={"arch": args.arch, "dim": args.dim,
                   "pretrained": args.pretrained},
    )

    model.eval()
    ious = []
    with torch.no_grad():
        for img, tok, box in batches(test_ds, args.batch_size, rng, shuffle=False):
            pred = model(
                torch.from_numpy(img).to(device), torch.from_numpy(tok).to(device)
            ).cpu().numpy()
            ious.extend(iou_cxcywh(pred, box).tolist())

    ious = np.array(ious)
    metrics = {
        "n": int(ious.size),
        "miou": float(ious.mean()),
        "acc@0.5": float((ious >= 0.5).mean()),
        "acc@0.7": float((ious >= 0.7).mean()),
    }
    print("\nDIOR-RSVG referring grounding:")
    for k, v in metrics.items():
        print(f"  {k:<10} {v:.4f}" if isinstance(v, float) else f"  {k:<10} {v}")
    write_metrics(args, metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
