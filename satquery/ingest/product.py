"""Vendor product layouts: many files, one logical image.

Real ISRO products do not ship as a single multi-band GeoTIFF. Verified
against actual Bhoonidhi downloads on 2026-08-29:

* **Cartosat-2E MX** (`5132611`): `BAND1.tif` .. `BAND4.tif`, four separate
  single-band files, plus `BAND_META.txt` giving `NoOfBands=4`,
  `BandNumbers=1234`, `BitsPerPixel=11`.
* **EOS-04 SAR** (FRS-1 `226981731`, MRS `226981721`): one directory per
  polarisation - `scene_HH/imagery_HH.tif`, `scene_VV/imagery_VV.tif`, ... -
  plus `BAND_META.txt` and a `product.xml` carrying
  `radarCenterFrequency = 5.40e09` (C-band) and the look counts.

Reading `BAND1.tif` on its own yields a 1-band image, which the modality
inference correctly but uselessly calls PAN. This module assembles the
scattered files into one logical dataset **by writing a GDAL VRT**, so
`read_image` and `read_canonical_band` keep working on a single path and the
frozen `ImageMeta` contract does not have to change.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import rasterio

# Cartosat MX band order. BAND_META gives BandNumbers=1234 but not what each
# band *is*; 4-band VNIR order is B1=Blue, B2=Green, B3=Red, B4=NIR, which the
# saturation radiances corroborate (B1 highest, consistent with blue).
CARTOSAT_MX_BANDS = ["BLUE", "GREEN", "RED", "NIR"]

_BAND_FILE_RE = re.compile(r"^BAND(\d+)\.tif$", re.IGNORECASE)
# Georeferenced products: imagery_HH.tif. SLC ScanSAR products additionally
# split each polarisation across sub-swath beams: imagery_HH_b0.tif ...
_POL_FILE_RE = re.compile(
    r"imagery_(HH|HV|VH|VV|RH|RV)(?:_b(\d+))?\.tif$", re.IGNORECASE
)

# GDAL dtype names keyed by numpy dtype, for the VRT header.
_VRT_DTYPE = {
    "uint8": "Byte", "uint16": "UInt16", "int16": "Int16",
    "uint32": "UInt32", "int32": "Int32",
    "float32": "Float32", "float64": "Float64",
}


@dataclass
class ProductLayout:
    """A multi-file product assembled into one logical dataset."""

    kind: str                      # cartosat_mx | eos04_sar | single_file
    root: Path
    band_files: list[Path] = field(default_factory=list)
    band_names: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def parse_band_meta(path: Path) -> dict:
    """Parse a Bhoonidhi `BAND_META.txt` (plain `Key=Value` lines)."""
    out: dict = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key:
            out[key] = value
    return out


def parse_product_xml(path: Path) -> dict:
    """Pull radar parameters out of an EOS-04 `product.xml`.

    Only the fields the verifier and the trace actually need; the file is
    large and most of it is irrelevant here.
    """
    out: dict = {}
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError):
        return out

    wanted = {
        "radarCenterFrequency": "radar_centre_frequency_hz",
        "pulseBandwidth": "pulse_bandwidth_hz",
        "incidenceAngle": "incidence_angle_deg",
    }
    for element in tree.iter():
        tag = element.tag.split("}")[-1]  # strip any namespace
        if tag in wanted and element.text:
            try:
                out[wanted[tag]] = float(element.text.strip())
            except ValueError:
                out[wanted[tag]] = element.text.strip()
    return out


def _sar_metadata(meta: dict, xml: dict) -> dict:
    """Normalise SAR metadata into the keys the pipeline reasons about."""
    out: dict = {}
    freq_hz = xml.get("radar_centre_frequency_hz")
    if isinstance(freq_hz, float):
        ghz = freq_hz / 1e9
        out["radar_frequency_ghz"] = round(ghz, 4)
        # Standard radar band letters. C-band spans 4-8 GHz; EOS-04 sits at
        # 5.40 GHz, essentially Sentinel-1's 5.405 GHz.
        if 4.0 <= ghz < 8.0:
            out["radar_band"] = "C"
        elif 8.0 <= ghz < 12.0:
            out["radar_band"] = "X"
        elif 2.0 <= ghz < 4.0:
            out["radar_band"] = "S"
        elif 1.0 <= ghz < 2.0:
            out["radar_band"] = "L"

    for src, dst, cast in (
        ("RangeLooks", "range_looks", float),
        ("AzimuthLooks", "azimuth_looks", float),
        ("IncidenceAngle", "incidence_angle_deg", float),
        ("NoOfPolarizations", "n_polarisations", int),
        ("ImagingMode", "imaging_mode", str),
        ("OutputPixelSpacing", "output_pixel_spacing_m", float),
    ):
        if src in meta:
            try:
                out[dst] = cast(meta[src].strip())
            except (ValueError, AttributeError):
                out[dst] = meta[src]

    looks = out.get("range_looks"), out.get("azimuth_looks")
    if all(isinstance(v, float) for v in looks):
        out["equivalent_looks"] = looks[0] * looks[1]
    return out


def discover(path: str | Path) -> ProductLayout:
    """Identify what kind of product `path` is.

    A file is taken at face value. A directory is searched for the known
    multi-file layouts, and falls back to `single_file` on the first raster
    found so an unrecognised vendor layout degrades rather than failing.
    """
    path = Path(path)
    if path.is_file():
        return ProductLayout(kind="single_file", root=path, band_files=[path])

    if not path.is_dir():
        raise FileNotFoundError(f"no such product: {path}")

    # Bhoonidhi archives nest one level (5132611.zip -> 5132611/).
    root = path
    entries = [p for p in root.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        root = entries[0]

    meta = parse_band_meta(root / "BAND_META.txt")

    # --- Cartosat MX: BAND1.tif .. BANDn.tif ---------------------------
    band_files = sorted(
        (p for p in root.iterdir() if p.is_file() and _BAND_FILE_RE.match(p.name)),
        key=lambda p: int(_BAND_FILE_RE.match(p.name).group(1)),  # type: ignore[union-attr]
    )
    if band_files:
        n = len(band_files)
        names = (
            list(CARTOSAT_MX_BANDS[:n]
                 ) if n <= len(CARTOSAT_MX_BANDS) else [f"BAND_{i+1}" for i in range(n)]
        )
        return ProductLayout(
            kind="cartosat_mx", root=root, band_files=band_files,
            band_names=names,
            metadata={
                "satellite": meta.get("SatID"),
                "sensor": meta.get("Sensor"),
                "acquisition_date": meta.get("DateOfPass"),
                "bits_per_pixel": _as_int(meta.get("BitsPerPixel")),
                "processing_level": meta.get("ProcessingLevel"),
                "product_type": meta.get("ProdType"),
                "raw": meta,
            },
        )

    # --- EOS-04 SAR: scene_<POL>/imagery_<POL>.tif ----------------------
    pol_files = sorted(
        (p for p in root.rglob("imagery_*.tif") if _POL_FILE_RE.search(p.name)),
        key=lambda p: p.name,
    )
    if pol_files:
        matches = [_POL_FILE_RE.search(p.name) for p in pol_files]
        beams = {m.group(2) for m in matches if m and m.group(2) is not None}
        xml = parse_product_xml(root / "product.xml")
        level = (meta.get("ProcessingLevel") or "").strip().upper()

        if beams:
            # ScanSAR SLC: each polarisation is split across sub-swath beams
            # with different look angles and slant-range geometry. These are
            # NOT bands - stacking them would silently fabricate a raster
            # whose pixels do not correspond. Merging them is real SAR
            # processing (beam mosaicking + geocoding), which docs/01
            # deliberately keeps out of scope. So the product is identified
            # and reported, not guessed at.
            by_pol: dict[str, list[Path]] = {}
            for path_i, m in zip(pol_files, matches):
                if m:
                    by_pol.setdefault(m.group(1).upper(), []).append(path_i)
            representative = [sorted(v)[0] for v in by_pol.values()]
            return ProductLayout(
                kind="eos04_sar_slc",
                root=root,
                band_files=representative,
                band_names=sorted(by_pol),
                metadata={
                    "satellite": meta.get("SatID"),
                    "sensor": meta.get("Sensor"),
                    "acquisition_date": meta.get("DateOfPass"),
                    "polarisations": sorted(by_pol),
                    "processing_level": level or "SLC",
                    "n_beams": len(beams),
                    "map_projection": meta.get("MapProjection"),
                    "requires_geocoding": True,
                    "unsupported_reason": (
                        f"{level or 'SLC'} slant-range ScanSAR product with "
                        f"{len(beams)} sub-swath beams per polarisation and "
                        "MapProjection=NA. Beam mosaicking and geocoding are "
                        "required before analysis; only the first beam of each "
                        "polarisation is exposed, for inspection only."
                    ),
                    **_sar_metadata(meta, xml),
                    "raw": meta,
                },
            )

        pols = [
            m.group(1).upper() for m in matches if m  # type: ignore[union-attr]
        ]
        return ProductLayout(
            kind="eos04_sar", root=root, band_files=pol_files, band_names=pols,
            metadata={
                "satellite": meta.get("SatID"),
                "sensor": meta.get("Sensor"),
                "acquisition_date": meta.get("DateOfPass"),
                "bits_per_pixel": _as_int(meta.get("BitsPerSample")),
                "polarisations": pols,
                **_sar_metadata(meta, xml),
                "raw": meta,
            },
        )

    # --- Fallback: any raster in the directory --------------------------
    rasters = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".img", ".jp2"}
    )
    if not rasters:
        raise FileNotFoundError(f"no raster files found under {path}")
    return ProductLayout(
        kind="single_file", root=root, band_files=[rasters[0]],
        metadata={"note": "unrecognised vendor layout; used the first raster found"},
    )


def _as_int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def build_vrt(layout: ProductLayout, out_path: Path | None = None) -> Path:
    """Stack a layout's band files into a GDAL VRT and return its path.

    A VRT is the right tool here: it is a plain-XML virtual dataset that
    presents N single-band files as one N-band raster, with no pixel copying
    and no extra disk beyond a few KB. Downstream code opens it with rasterio
    exactly like any other raster, so nothing else has to know that the
    product was ever split across files.
    """
    if len(layout.band_files) == 1:
        return layout.band_files[0]

    out_path = out_path or (layout.root / f"_satquery_{layout.kind}.vrt")

    with rasterio.open(layout.band_files[0]) as src:
        width, height = src.width, src.height
        dtype = src.dtypes[0]
        crs_wkt = src.crs.to_wkt() if src.crs else ""
        transform = src.transform
        nodata = src.nodata

    gdal_dtype = _VRT_DTYPE.get(dtype, "Float32")
    geo = (
        f"{transform.c}, {transform.a}, {transform.b}, "
        f"{transform.f}, {transform.d}, {transform.e}"
    )

    parts = [
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
        f"  <SRS>{_xml_escape(crs_wkt)}</SRS>",
        f"  <GeoTransform>{geo}</GeoTransform>",
    ]

    names = layout.band_names or [f"BAND_{i+1}" for i in range(len(layout.band_files))]
    for i, (band_file, name) in enumerate(zip(layout.band_files, names), start=1):
        # relativeToVRT=0 with a resolved path: the VRT lives beside the data,
        # but an absolute path keeps it valid if the VRT is opened from
        # elsewhere.
        src_path = _xml_escape(str(band_file.resolve()))
        nodata_line = (
            f"    <NoDataValue>{nodata}</NoDataValue>\n" if nodata is not None else ""
        )
        parts.append(
            f'  <VRTRasterBand dataType="{gdal_dtype}" band="{i}">\n'
            f"    <Description>{_xml_escape(name)}</Description>\n"
            f"{nodata_line}"
            f'    <SimpleSource>\n'
            f'      <SourceFilename relativeToVRT="0">{src_path}</SourceFilename>\n'
            f"      <SourceBand>1</SourceBand>\n"
            f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
            f'      <DstRect xOff="0" yOff="0" xSize="{width}" ySize="{height}"/>\n'
            f"    </SimpleSource>\n"
            f"  </VRTRasterBand>"
        )
    parts.append("</VRTDataset>")

    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def resolve(path: str | Path) -> tuple[Path, ProductLayout]:
    """Turn any product path into (openable raster path, layout).

    This is the function ingest calls. For a plain file it is a no-op; for a
    multi-file vendor product it returns the VRT that unifies the bands.
    """
    layout = discover(path)
    return build_vrt(layout), layout
