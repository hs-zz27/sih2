"""Index WHU-OPT-SAR for the Stage A2 resolution bridge (task 2.2).

WHU-OPT-SAR is ~5 m co-registered optical + SAR with land-cover labels. It
sits between Sentinel-2 (10 m, where Track A was trained) and Cartosat-2E
(1.6 m, the evaluation sensor), which is exactly the gap the cross-sensor test
identified as the dominant transfer problem: vegetation agreement collapsed
from +0.476 at 10 m to -0.135 at native 1.6 m, while band adaptation cost far
less. Stage A2 was a plan assumption; that measurement made it the priority.

Layouts differ between mirrors, so discovery is by directory name rather than
a hardcoded tree, and the script reports what it found instead of failing
silently on an unexpected structure.

Usage:
    python training/prepare/whu_opt_sar.py --src data/whu_opt_sar/extracted \
        --out data/whu_opt_sar/index.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys
from pathlib import Path

# WHU-OPT-SAR's published 7-class land-cover nomenclature (background = 0).
CLASSES = [
    "background", "farmland", "city", "village", "water", "forest", "road",
    "others",
]

OPTICAL_HINTS = ("optical", "opt", "rgb", "image")
SAR_HINTS = ("sar",)
LABEL_HINTS = ("lbl", "label", "mask", "gt", "annotation")

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg"}

TILE_PX = 512
# WHU-OPT-SAR label rasters encode the 8 classes as multiples of ten
# (0, 10, ... 70), not 0..7. Dividing is required; using the raw value would
# put every class outside the head's index range.
LABEL_STRIDE = 10
_TILE_RE = re.compile(r"^(.+)_(\d+)_(\d+)$")


def extract_from_archives(
    tile_zip, label_zip, dest, limit=None,
):
    """Build paired optical/SAR/label tiles from the two published archives.

    The 512px partition mirror ships optical and SAR but no labels; the
    full-scene mirror ships labels but not the partition. Tile names encode
    their position (SCENE_ROW_COL), so each label tile is cropped from the
    full-scene mask at row*512, col*512. Verified against scene
    NH50E007004: a 5555x3704 label yields exactly the 11x8 tile grid the
    partition uses.
    """
    import zipfile

    import numpy as np
    import rasterio
    dest = pathlib.Path(dest)
    for kind in ("opt", "sar", "lbl"):
        (dest / kind).mkdir(parents=True, exist_ok=True)

    tz = zipfile.ZipFile(tile_zip)
    lz = zipfile.ZipFile(label_zip)
    label_members = {
        pathlib.Path(n).stem: n for n in lz.namelist() if n.endswith(".tif")
    }

    tiles = sorted(
        n for n in tz.namelist()
        if n.startswith("opt/") and n.endswith(".tif") and "__MACOSX" not in n
    )
    if limit:
        tiles = tiles[:limit]

    written, skipped = 0, 0
    label_cache: dict[str, "np.ndarray"] = {}

    for member in tiles:
        stem = pathlib.Path(member).stem
        match = _TILE_RE.match(stem)
        if not match:
            skipped += 1
            continue
        scene, row, col = match.group(1), int(match.group(2)), int(match.group(3))
        if scene not in label_members:
            skipped += 1
            continue

        if scene not in label_cache:
            lz.extract(label_members[scene], dest / "_labels")
            with rasterio.open(dest / "_labels" / label_members[scene]) as src:
                label_cache[scene] = src.read(1)

        full = label_cache[scene]
        y, x = row * TILE_PX, col * TILE_PX
        crop = full[y : y + TILE_PX, x : x + TILE_PX]
        if crop.shape != (TILE_PX, TILE_PX):
            # Edge tile: the partition drops these, so the label has no
            # matching full window. Skipping keeps pairs exact.
            skipped += 1
            continue

        tz.extract(member, dest / "_tiles")
        (dest / "opt" / f"{stem}.tif").write_bytes(
            (dest / "_tiles" / member).read_bytes()
        )
        sar_member = member.replace("opt/", "sar/")
        if sar_member in tz.namelist():
            tz.extract(sar_member, dest / "_tiles")
            (dest / "sar" / f"{stem}.tif").write_bytes(
                (dest / "_tiles" / sar_member).read_bytes()
            )

        classes = (crop // LABEL_STRIDE).astype("uint8")
        with rasterio.open(
            dest / "lbl" / f"{stem}.tif", "w", driver="GTiff",
            height=TILE_PX, width=TILE_PX, count=1, dtype="uint8",
        ) as dst:
            dst.write(classes, 1)
        written += 1

    import shutil
    shutil.rmtree(dest / "_tiles", ignore_errors=True)
    shutil.rmtree(dest / "_labels", ignore_errors=True)
    return written, skipped


def _classify_dir(path: Path) -> str | None:
    name = path.name.lower()
    # Labels first: a directory called "sar_label" is a label directory.
    if any(h in name for h in LABEL_HINTS):
        return "label"
    if any(h in name for h in SAR_HINTS):
        return "sar"
    if any(h in name for h in OPTICAL_HINTS):
        return "optical"
    return None


def discover(src: Path) -> dict[str, dict[str, Path]]:
    """Map stem -> {optical, sar, label} across whatever layout is present."""
    found: dict[str, dict[str, Path]] = {}
    for directory in sorted(p for p in src.rglob("*") if p.is_dir()):
        kind = _classify_dir(directory)
        if kind is None:
            continue
        for image in directory.iterdir():
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                found.setdefault(image.stem, {})[kind] = image
    return found


def build_index(src: Path, val_fraction: float, seed: int) -> dict:
    found = discover(src)
    complete = {
        stem: parts for stem, parts in found.items()
        if "optical" in parts and "label" in parts
    }

    # A geographic split is not possible without tile coordinates, so a
    # deterministic random split is used and labelled as such. WHU-OPT-SAR
    # tiles come from a small number of large scenes, so neighbouring tiles
    # are correlated and this split is optimistic - stated rather than hidden.
    rng = random.Random(seed)
    stems = sorted(complete)
    rng.shuffle(stems)
    cut = int(len(stems) * val_fraction)
    splits = {"validation": stems[:cut], "train": stems[cut:]}

    return {
        "classes": CLASSES,
        "split_method": (
            "deterministic random by tile; NOT geographic. WHU-OPT-SAR tiles "
            "derive from a few large scenes, so adjacent tiles are correlated "
            "and validation scores here are optimistic."
        ),
        "splits": {
            name: [
                {
                    "id": stem,
                    "optical": str(complete[stem]["optical"]),
                    "sar": str(complete[stem]["sar"]) if "sar" in complete[stem] else None,
                    "label": str(complete[stem]["label"]),
                }
                for stem in members
            ]
            for name, members in splits.items()
        },
        "n_with_sar": sum(1 for s in complete.values() if "sar" in s),
        "n_incomplete": len(found) - len(complete),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tile-zip", type=Path, help="512px partition archive")
    p.add_argument("--label-zip", type=Path, help="full-scene label archive")
    p.add_argument("--limit", type=int, help="cap tiles when extracting")
    args = p.parse_args()

    if args.tile_zip and args.label_zip:
        written, skipped = extract_from_archives(
            args.tile_zip, args.label_zip, args.src, args.limit
        )
        print(f"extracted {written} paired tiles ({skipped} skipped)")

    if not args.src.is_dir():
        print(f"source not found: {args.src}", file=sys.stderr)
        return 1

    index = build_index(args.src, args.val_fraction, args.seed)
    counts = {k: len(v) for k, v in index["splits"].items()}

    if not any(counts.values()):
        print("No optical/label pairs found. Directories seen:", file=sys.stderr)
        for d in sorted({p.parent.name for p in args.src.rglob("*")
                         if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES})[:20]:
            print(f"    {d}", file=sys.stderr)
        print("Extend OPTICAL_HINTS / SAR_HINTS / LABEL_HINTS to match.",
              file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(index), encoding="utf-8")
    print(f"Wrote {args.out}")
    for name, n in counts.items():
        print(f"  {name:<11} {n:>6} tiles")
    print(f"  with paired SAR : {index['n_with_sar']}")
    if index["n_incomplete"]:
        print(f"  incomplete (skipped): {index['n_incomplete']}")
    print(f"  split: {index['split_method']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
