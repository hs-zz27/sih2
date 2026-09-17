"""Stage A3: high-resolution transfer (plan task 3.2).

Stage A2 bridged the Track A encoder from BigEarthNet's 10 m to WHU-OPT-SAR's
~5 m, because the cross-sensor test showed resolution - not bands - was the
dominant gap against a 1.6 m Cartosat product. A3 is the last leg of that
bridge: ~5 m to sub-metre.

## The plan offered high-res SAR or optical-only; this is optical-only

Task 3.2 reads "SpaceNet 6 / Umbra, or optical-only with the limitation
documented". **This is the optical-only arm, and the limitation is the
headline finding, not a footnote:**

> Stage A3 adapts the encoder to fine spatial detail. It does NOT adapt it to
> high-resolution SAR. The Cartosat path is optical, so the bridge is
> complete for the optical half; the EOS-04 / RISAT path still has no
> high-resolution SAR training data at any stage, and no result here should
> be read as evidence about it.

Verification item 8 (SpaceNet 6 / Umbra / Capella accessible and licensed)
remains open, and this run does not close it.

## Source: DIOR, via the DIOR-RSVG mirror already on disk

DIOR images are 800x800 at 0.5-30 m GSD - a genuine order-of-magnitude step
down from WHU's 5 m, and the finest optical imagery available locally that is
not the held-out Bhoonidhi set. (Per docs/03 section 4.3 the Cartosat and
EOS-04 products are the cross-sensor generalisation set and are never trained
on. Using them here would destroy the only honest out-of-distribution
measurement the project has.)

## The confound, stated because it changes how the number reads

WHU-OPT-SAR labels are **land-cover classes**; DIOR labels are **object
categories** (aircraft, ship, storage tank...). So A3 changes both the
resolution and the label semantics at once, and an mAP here is not comparable
to A2's 0.7759 in any direction.

What it CAN measure, and what the report should say, is whether the encoder's
features survive the resolution change: a frozen-encoder probe against a
fine-tuned encoder on the same DIOR target. If the frozen probe does nearly as
well, the features already transfer and A3 buys little; if fine-tuning helps
substantially, the 5 m features genuinely did not cover sub-metre detail.
That comparison is internally controlled - same data, same head, same split -
so the semantic change cancels out of it.

Usage:
    python training/stage_a3_transfer.py --data data/dior_rsvg/data \
        --init checkpoints/stage_a2 --ckpt-dir checkpoints/stage_a3 --epochs 2
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.checkpointing import (  # noqa: E402
    find_latest_checkpoint,
    maybe_resume,
    save_checkpoint,
    set_seed,
    write_run_metadata,
)
from training.stage_a2_transfer import (  # noqa: E402
    load_pretrained_encoder,
    replace_head,
)
from training.track_a_encoder import (  # noqa: E402
    BAND_NAMES,
    CARTOSAT_INDICES,
    band_dropout_mask,
    mean_average_precision,
)

PATCH = 120

# DIOR object categories, derived from the referring expressions rather than
# assumed: the mirror ships no category field, so the vocabulary is built from
# the data and recorded in run_metadata.json.
CATEGORY_PATTERNS = [
    "airplane", "airport", "baseball", "basketball", "bridge", "chimney",
    "dam", "expressway", "golf", "ground track field", "harbor", "overpass",
    "ship", "stadium", "storage tank", "tennis", "train station", "vehicle",
    "windmill", "toll station",
]

# RGB -> the encoder's canonical band slots, so the learned band embedding
# still sees RED where RED belongs. DIOR is 3-band optical with no NIR, which
# is itself informative: the band-dropout training is what makes this legal.
RGB_TO_SLOT = {
    "RED": BAND_NAMES.index("B04"),
    "GREEN": BAND_NAMES.index("B03"),
    "BLUE": BAND_NAMES.index("B02"),
}


def categories_of(expression: str) -> list[int]:
    lowered = str(expression).lower()
    return [i for i, name in enumerate(CATEGORY_PATTERNS) if name in lowered]


class DiorParquet:
    """DIOR images with multi-label object-category presence targets.

    Rows are grouped by image: DIOR-RSVG carries several referring
    expressions per picture, so the presence vector is the union over every
    expression for that image. Grouping is also what makes the split honest -
    the same picture must not appear on both sides, which is the correction
    task 2.7 already had to make for the grounding split.
    """

    def __init__(self, files: list[Path], limit: int | None = None):
        import pandas as pd

        frames = []
        for path in files:
            frames.append(pd.read_parquet(path))
        table = pd.concat(frames, ignore_index=True)

        text_col = next(
            (c for c in table.columns if c.lower() in
             {"question", "caption", "expression", "sentence", "text", "phrase"}),
            None,
        )
        image_col = next(
            (c for c in table.columns if c.lower() in {"image", "img", "picture"}),
            None,
        )
        id_col = next(
            (c for c in table.columns if c.lower() in {"image_id", "id"}), None
        )
        if text_col is None or image_col is None:
            raise SystemExit(
                f"could not find image/text columns in {list(table.columns)}"
            )

        # Grouped by image_id where the mirror provides one. Grouping by raw
        # image bytes also works but hashes a few hundred KB per row for no
        # benefit.
        grouped: dict[object, set[int]] = {}
        blobs: dict[object, bytes] = {}
        order: list[object] = []
        ids = table[id_col] if id_col else range(len(table))
        for key, image, text in zip(ids, table[image_col], table[text_col],
                                    strict=False):
            if key not in grouped:
                if limit and len(order) >= limit:
                    continue
                grouped[key] = set()
                blobs[key] = image["bytes"] if isinstance(image, dict) else image
                order.append(key)
            grouped[key].update(categories_of(text))

        self.keys = order
        self.targets = grouped
        self.blobs = blobs

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int):
        import io

        from PIL import Image

        key = self.keys[i]
        image = (
            Image.open(io.BytesIO(self.blobs[key]))
            .convert("RGB")
            .resize((PATCH, PATCH))
        )
        arr = np.asarray(image, dtype="float32").transpose(2, 0, 1) / 255.0

        cube = np.zeros((len(BAND_NAMES), PATCH, PATCH), dtype="float32")
        present = np.zeros(len(BAND_NAMES), dtype="float32")
        for band_index, name in enumerate(("RED", "GREEN", "BLUE")):
            slot = RGB_TO_SLOT[name]
            band = arr[band_index]
            mean, std = float(band.mean()), float(band.std()) or 1.0
            cube[slot] = (band - mean) / std
            present[slot] = 1.0

        target = np.zeros(len(CATEGORY_PATTERNS), dtype="float32")
        for index in self.targets[key]:
            target[index] = 1.0
        return cube, target, present


def batches(dataset, size, rng, shuffle=True):
    order = rng.permutation(len(dataset)) if shuffle else np.arange(len(dataset))
    for start in range(0, len(order), size):
        idx = order[start : start + size]
        xs, ys, ps = zip(*(dataset[int(i)] for i in idx))
        yield np.stack(xs), np.stack(ys), np.stack(ps)


def evaluate(model, dataset, torch, batch_size, device, keep=None):
    model.eval()
    rng = np.random.default_rng(0)
    scores, targets = [], []
    with torch.no_grad():
        for x, y, present in batches(dataset, batch_size, rng, shuffle=False):
            mask = present.copy()
            if keep is not None:
                restricted = np.zeros_like(mask)
                restricted[:, keep] = 1.0
                mask = mask * restricted
            out = model(
                torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device)
            )
            scores.append(torch.sigmoid(out).cpu().numpy())
            targets.append(y)
    value, _ = mean_average_precision(
        np.concatenate(scores), np.concatenate(targets)
    )
    return value


def run_arm(
    model, train_ds, test_ds, torch, args, device, freeze_encoder: bool, label: str
):
    """Train one arm and return its test mAP.

    `freeze_encoder=True` is the probe: only the head learns, so the result
    measures what the Stage A2 features already encode about sub-metre
    imagery without any adaptation.
    """
    import torch.nn as nn

    if freeze_encoder:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("head.")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        running = seen = 0
        for x, y, present in batches(train_ds, args.batch_size, rng):
            mask = present * band_dropout_mask(
                x.shape[0], x.shape[1], args.band_dropout, rng
            )
            out = model(
                torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device)
            )
            loss = criterion(out, torch.from_numpy(y).to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.shape[0]
            seen += x.shape[0]
        print(
            f"  [{label}] epoch {epoch + 1}/{args.epochs}  "
            f"loss {running / max(seen, 1):.4f}  ({time.time() - started:.0f}s)",
            flush=True,
        )

    return evaluate(model, test_ds, torch, args.batch_size, device)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/dior_rsvg/data"))
    p.add_argument("--init", type=Path, default=Path("checkpoints/stage_a2"))
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/stage_a3"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--band-dropout", type=float, default=0.3)
    p.add_argument("--limit", type=int, default=3000)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import torch

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    files = [Path(f) for f in sorted(glob.glob(str(args.data / "*.parquet")))]
    if not files:
        raise SystemExit(f"no parquet files in {args.data}")
    print(f"reading {len(files)} parquet file(s)")

    dataset = DiorParquet(files, limit=args.limit)
    print(f"{len(dataset)} unique images")

    # Split by image index, which is already grouped by picture.
    cut = int(len(dataset) * (1 - args.test_fraction))
    train_ds, test_ds = _Subset(dataset, range(cut)), _Subset(
        dataset, range(cut, len(dataset))
    )
    print(f"train {len(train_ds)} | test {len(test_ds)}")

    label_counts = Counter()
    for i in range(len(dataset)):
        for index in dataset.targets[dataset.keys[i]]:
            label_counts[CATEGORY_PATTERNS[index]] += 1
    print(f"categories seen: {dict(label_counts.most_common(8))}")

    results = {}
    for label, freeze in (("frozen_probe", True), ("finetuned", False)):
        model, source = load_pretrained_encoder(
            args.init, torch, device, args.dim
        )
        model = replace_head(model, torch, args.dim, len(CATEGORY_PATTERNS))
        model = model.to(device)
        results[label] = run_arm(
            model, train_ds, test_ds, torch, args, device, freeze, label
        )
        print(f"  [{label}] test mAP {results[label]:.4f}")
        if not freeze:
            args.ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(args.ckpt_dir, args.epochs, model, None)

    delta = results["finetuned"] - results["frozen_probe"]
    metrics = {
        "frozen_probe_map": results["frozen_probe"],
        "finetuned_map": results["finetuned"],
        "adaptation_gain": delta,
    }
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    (args.ckpt_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    write_run_metadata(args.ckpt_dir, {
        "task": "stage_a3_highres_transfer",
        "initialised_from": str(source) if source else None,
        "source_dataset": "DIOR (via the DIOR-RSVG mirror)",
        "n_images": len(dataset),
        "epochs": args.epochs,
        "lr": args.lr,
        "band_dropout": args.band_dropout,
        "categories": CATEGORY_PATTERNS,
        "modality": "optical only",
        "limitation": (
            "This is the optical-only arm of task 3.2. It adapts the encoder "
            "to fine spatial detail; it does NOT adapt it to high-resolution "
            "SAR. Verification item 8 (SpaceNet 6 / Umbra / Capella) is still "
            "open and this run does not close it. No result here is evidence "
            "about the EOS-04 / RISAT path."
        ),
        "confound": (
            "WHU-OPT-SAR labels are land-cover classes and DIOR labels are "
            "object categories, so resolution AND label semantics change at "
            "once. The frozen-probe vs fine-tuned comparison is internally "
            "controlled - same data, head and split - so the semantic change "
            "cancels out of the GAIN even though neither absolute mAP is "
            "comparable to Stage A2's 0.7759."
        ),
        "split_method": (
            "grouped by image and taken in order; DIOR-RSVG carries several "
            "expressions per picture, so an expression-level split would put "
            "the same image on both sides - the same correction task 2.7 made"
        ),
        "seed": args.seed,
    })

    print(f"\nfrozen probe : {results['frozen_probe']:.4f}")
    print(f"fine-tuned   : {results['finetuned']:.4f}")
    print(f"adaptation gain: {delta:+.4f}")
    print(
        "\nA positive gain means the Stage A2 (5 m) features did not already "
        "cover sub-metre detail and the adaptation is doing real work. A gain "
        "near zero means the features transferred as they were and A3 buys "
        "little."
    )
    return 0


class _Subset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.dataset[self.indices[i]]


if __name__ == "__main__":
    sys.exit(main())
