"""Inspect a Bhoonidhi (or any) raster product and report what the week-0
verification gate needs.

Specifically targets items 5 and 6 of `docs/verification.md`:
  * item 6 - Cartosat-2S MX band composition (4-band VNIR, or is SWIR present?)
  * item 5 - which RISAT / which frequency (C-band vs X-band) and polarizations

Usage:
    python scripts/inspect_product.py <path-to-product-file-or-directory>
"""

import argparse
import sys
from pathlib import Path

import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RASTER_SUFFIXES = {".tif", ".tiff", ".img", ".jp2", ".h5", ".hdf", ".nc", ".ntf"}

# Tag keys worth surfacing verbatim — these are where sensor, frequency,
# polarization and wavelength information usually hides in vendor metadata.
INTERESTING_TAG_HINTS = (
    "band",
    "freq",
    "polar",
    "wavelen",
    "sensor",
    "satellite",
    "mission",
    "instrument",
    "product",
    "resolution",
    "gsd",
    "incidence",
    "look",
    "orbit",
    "date",
    "time",
)


def find_rasters(path: Path) -> list[Path]:
    """Return raster files at `path` (the file itself, or those under a dir)."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in RASTER_SUFFIXES
    )


def interesting_tags(tags: dict) -> dict:
    """Filter a tag dict down to keys that hint at sensor/band/frequency info."""
    return {
        k: v
        for k, v in tags.items()
        if any(hint in k.lower() for hint in INTERESTING_TAG_HINTS)
    }


def describe_gsd(src) -> str:
    """Human-readable ground sample distance from the dataset transform."""
    t = src.transform
    x_res, y_res = abs(t.a), abs(t.e)
    if src.crs is not None and src.crs.is_geographic:
        # Degrees are not directly comparable to metres; flag rather than convert.
        return f"{x_res:.8f} x {y_res:.8f} degrees (geographic CRS — not metres)"
    return f"{x_res:.4f} x {y_res:.4f} (CRS units, typically metres)"


def inspect(path: Path, all_tags: bool = False) -> None:
    print("=" * 78)
    print(f"FILE: {path}")
    print("=" * 78)

    try:
        src = rasterio.open(path)
    except Exception as e:
        print(f"  !! could not open with rasterio/GDAL: {e}")
        print()
        return

    with src:
        print(f"  driver       : {src.driver}")
        print(f"  size         : {src.width} x {src.height} px")
        print(f"  band count   : {src.count}")
        print(f"  dtypes       : {', '.join(sorted(set(src.dtypes)))}")
        print(f"  CRS          : {src.crs}")
        print(f"  pixel size   : {describe_gsd(src)}")
        print(f"  nodata       : {src.nodata}")
        print(f"  bounds       : {src.bounds}")

        subs = src.subdatasets
        if subs:
            print(f"\n  subdatasets ({len(subs)}) — re-run this script on each:")
            for s in subs:
                print(f"    - {s}")

        print("\n  per-band detail:")
        for i in range(1, src.count + 1):
            desc = src.descriptions[i - 1]
            cinterp = src.colorinterp[i - 1].name
            print(f"    band {i}: description={desc!r} color_interp={cinterp}")
            btags = src.tags(i)
            for k, v in interesting_tags(btags).items():
                print(f"             {k} = {v}")

        ds_tags = src.tags()
        shown = interesting_tags(ds_tags)
        if shown:
            print("\n  dataset metadata (filtered to sensor/band/frequency hints):")
            for k, v in sorted(shown.items()):
                print(f"    {k} = {v}")
        hidden = len(ds_tags) - len(shown)
        if hidden > 0:
            print(f"    ... plus {hidden} other tag(s); use --all-tags to see them")

        if all_tags and ds_tags:
            print("\n  ALL dataset metadata:")
            for k, v in sorted(ds_tags.items()):
                print(f"    {k} = {v}")

        print("\n  verification notes:")
        print(f"    - item 6 (Cartosat SWIR): this product has {src.count} band(s).")
        if src.count == 4:
            print("      4 bands is consistent with the assumed VNIR (B,G,R,NIR),")
            print("      i.e. NO SWIR. Confirm against the band descriptions and")
            print("      the vendor metadata file shipped alongside the raster.")
        elif src.count == 1:
            print("      1 band - likely a PAN product, or one band of a multi-file")
            print("      product. If a vendor layout was recognised above, trust that")
            print("      band count instead of this per-file one.")
        elif src.count > 4:
            print("      MORE than 4 bands - SWIR may be present. If so, the MNDWI/NDBI")
            print("      index paths can be enabled (pure upside per docs/03 section 6).")
        print("    - item 5 (RISAT frequency/mode): look for frequency, polarization")
        print("      and look-count keys in the metadata above. If absent here, they")
        print("      are usually in the product's XML/BAND_META sidecar file, not the")
        print("      raster header - check the sibling files listed by this script.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="product file or directory")
    parser.add_argument(
        "--all-tags", action="store_true", help="print every metadata tag, unfiltered"
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Path does not exist: {args.path}")
        return 1

    # Prefer the product resolver: real vendor products split one logical
    # image across many files (Cartosat MX BAND1..4.tif, EOS-04
    # scene_<POL>/imagery_<POL>.tif). Inspecting those files individually
    # reports four 1-band "PAN" images instead of one 4-band MX product,
    # which is exactly the wrong answer for verification item 6.
    try:
        from satquery.ingest.product import discover

        layout = discover(args.path)
        if layout.kind != "single_file":
            print(f"Recognised vendor product: {layout.kind}")
            print(f"  band files : {len(layout.band_files)}")
            print(f"  band names : {layout.band_names}")
            meta = {k: v for k, v in layout.metadata.items() if k != "raw"}
            for k, v in meta.items():
                print(f"  {k:<24}: {v}")
            print()
    except Exception as exc:  # noqa: BLE001 - inspection must never hard-fail
        print(f"(product detection unavailable: {type(exc).__name__}: {exc})\n")

    rasters = find_rasters(args.path)
    if not rasters:
        print(f"No raster files found under {args.path}")
        print(f"Looked for suffixes: {', '.join(sorted(RASTER_SUFFIXES))}")
        return 1

    print(f"Found {len(rasters)} raster file(s) under {args.path}\n")
    for r in rasters:
        inspect(r, all_tags=args.all_tags)

    if args.path.is_dir():
        sidecars = sorted(
            p.name
            for p in args.path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".xml", ".txt", ".met", ".hdr"}
        )
        if sidecars:
            print("Sidecar metadata files present (read these for item 5 / item 6):")
            for s in sidecars:
                print(f"  - {s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
