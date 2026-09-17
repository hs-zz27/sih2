"""Build a Track A index from the BigEarthNet subset (plan task 1.10).

Pairs each S2 patch with its multi-label ground truth and its **official**
split. The official split matters more than it looks: BigEarthNet patches are
tiled from larger Sentinel scenes, so adjacent patches are near-duplicates and
a random split puts neighbours on both sides, inflating validation accuracy by
a wide margin (docs/03 section 4.3). We never resplit.

Output: one JSON index with train/validation/test lists of
    {"patch_id": ..., "s2": <path>, "s1": <path|null>, "labels": [...]}

Usage:
    python training/prepare/bigearthnet.py \
        --src data/bigearthnet_14k/extracted/BEN_14k \
        --metadata data/bigearthnet_14k/metadata.parquet \
        --out data/bigearthnet_14k/index.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# BigEarthNet v2 19-class nomenclature, fixed order so a trained head always
# maps the same index to the same class.
CLASSES = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}

SPLITS = ("train", "validation", "test")


def build_index(src: Path, metadata: Path) -> dict:
    """Match the on-disk patches to their labels via the metadata parquet."""
    import pyarrow.parquet as pq

    table = pq.read_table(metadata).to_pydict()
    labels_by_patch = dict(zip(table["patch_id"], table["labels"], strict=True))

    # S1 is keyed by its own product name, so map S2 patch id -> S1 name.
    s1_by_patch = dict(zip(table["patch_id"], table["s1_name"], strict=True))

    index: dict = {"classes": CLASSES, "splits": {}}
    missing_labels = 0

    for split in SPLITS:
        s2_dir = src / "BigEarthNet-S2" / split
        s1_dir = src / "BigEarthNet-S1" / split
        if not s2_dir.is_dir():
            continue

        s1_files = {p.stem: p for p in s1_dir.glob("*.tif")} if s1_dir.is_dir() else {}

        rows = []
        for s2_path in sorted(s2_dir.glob("*.tif")):
            patch_id = s2_path.stem
            labels = labels_by_patch.get(patch_id)
            if not labels:
                missing_labels += 1
                continue

            s1_name = s1_by_patch.get(patch_id)
            s1_path = s1_files.get(s1_name) if s1_name else None

            rows.append(
                {
                    "patch_id": patch_id,
                    "s2": str(s2_path),
                    "s1": str(s1_path) if s1_path else None,
                    "labels": [CLASS_INDEX[x] for x in labels if x in CLASS_INDEX],
                }
            )
        index["splits"][split] = rows

    index["missing_labels"] = missing_labels
    return index


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if not args.src.is_dir():
        print(f"source not found: {args.src}", file=sys.stderr)
        return 1

    index = build_index(args.src, args.metadata)
    counts = {k: len(v) for k, v in index["splits"].items()}
    if not any(counts.values()):
        print("No patches indexed - check --src points at the BEN_14k directory",
              file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(index), encoding="utf-8")

    print(f"Wrote {args.out}")
    for split, n in counts.items():
        with_s1 = sum(1 for r in index["splits"][split] if r["s1"])
        print(f"  {split:<11} {n:>6} patches ({with_s1} with paired S1)")
    if index["missing_labels"]:
        print(f"  skipped {index['missing_labels']} patches with no label entry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
