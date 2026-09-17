"""Validate an installed gazetteer directory (see docs/gazetteer.md).

The gazetteer is operator-installed third-party data, so the failure mode to
guard against is a directory that *looks* right and silently reports nothing:
a raster with no `nodata` set, a legend whose keys do not match the codes in
the raster, a layer with no attribution. Each of those degrades answers
quietly rather than raising, which is correct at query time and unhelpful at
install time. This script is the loud version.

It reads only local files and never touches the network.

Usage:
    python scripts/check_gazetteer.py --dest /path/to/gazetteer
    python scripts/check_gazetteer.py --dest /path/to/gazetteer --at 54.9 21.08
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satquery.geo.gazetteer import LAYERS  # noqa: E402

# How many distinct codes to sample out of the raster when checking that the
# legend covers what is actually in the data. Reading the whole of a global
# 1 km raster to list its codes is minutes of I/O for a sanity check.
_SAMPLE_STRIDE = 37


def _check_layer(directory: Path, name: str) -> list[str]:
    """Problems with one layer, as human-readable lines. Empty means good."""
    import numpy as np
    import rasterio

    raster = directory / f"{name}.tif"
    legend_path = directory / f"{name}.json"
    problems: list[str] = []

    with rasterio.open(raster) as src:
        crs = str(src.crs) if src.crs else None
        if crs is None or "4326" not in crs:
            problems.append(
                f"CRS is {crs!r}; the module samples in degrees and expects "
                f"EPSG:4326"
            )
        if src.count != 1:
            problems.append(f"{src.count} bands; expected a single band")
        if src.nodata is None:
            problems.append(
                "no nodata value in the header - unmapped cells will only be "
                "recognised if they happen to be 0 (see docs/gazetteer.md)"
            )
        # Decimated read: enough to see which codes exist without loading a
        # global raster at full resolution.
        sample = src.read(
            1,
            out_shape=(
                max(1, src.height // _SAMPLE_STRIDE),
                max(1, src.width // _SAMPLE_STRIDE),
            ),
        )
        blank = src.nodata if src.nodata is not None else 0
        codes = {int(v) for v in np.unique(sample) if int(v) != blank}

    if not legend_path.exists():
        problems.append(f"no {name}.json - every code will be unnamed")
        return problems

    try:
        payload = json.loads(legend_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"{name}.json is unreadable ({exc}) - every code will be unnamed")
        return problems

    labels = {int(k) for k in (payload.get("labels") or {})}
    if not labels:
        problems.append(f"{name}.json has no labels - every code will be unnamed")
    unlabelled = sorted(codes - labels)
    if unlabelled:
        problems.append(
            f"codes present in the raster but absent from the legend, so they "
            f"will never be reported: {unlabelled[:12]}"
            + (" ..." if len(unlabelled) > 12 else "")
        )
    if not payload.get("attribution"):
        problems.append(
            "no attribution - Trace.data_sources will not credit this layer"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest", default=os.getenv("SATQUERY_GAZETTEER"),
        help="gazetteer directory (default: $SATQUERY_GAZETTEER)",
    )
    parser.add_argument(
        "--at", nargs=2, type=float, metavar=("LAT", "LON"),
        help="also run a live lookup at this coordinate",
    )
    args = parser.parse_args(argv)

    if not args.dest:
        print("no directory given and SATQUERY_GAZETTEER is not set", file=sys.stderr)
        return 2
    directory = Path(args.dest)
    if not directory.is_dir():
        print(f"not a directory: {directory}", file=sys.stderr)
        return 2

    found = [name for name in LAYERS if (directory / f"{name}.tif").exists()]
    if not found:
        print(
            f"no layer rasters in {directory} (looked for "
            f"{', '.join(f'{n}.tif' for n in LAYERS)})",
            file=sys.stderr,
        )
        return 1

    failed = False
    for name in found:
        problems = _check_layer(directory, name)
        if problems:
            failed = True
            print(f"{name}: {len(problems)} problem(s)")
            for line in problems:
                print(f"  - {line}")
        else:
            print(f"{name}: ok")
    for name in LAYERS:
        if name not in found:
            print(f"{name}: not installed (optional)")

    if args.at:
        # Import here so a directory that fails validation still reports its
        # problems rather than dying on the lookup.
        os.environ["SATQUERY_GAZETTEER"] = str(directory)
        from satquery.geo import lookup

        place = lookup(args.at[0], args.at[1])
        print(f"\nlookup at {args.at[0]}, {args.at[1]}:")
        print(f"  country: {place.country or '-'}")
        print(f"  climate: {place.climate or '-'}"
              + (f" ({place.climate_description})" if place.climate_description else ""))
        if place.ambiguous:
            print(f"  ambiguous (window disagreed): {', '.join(sorted(place.ambiguous))}")
        print(f"  sources: {', '.join(place.sources) or '-'}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
