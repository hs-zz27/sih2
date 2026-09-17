"""Raster reader: turns a file on disk into a validated `ImageMeta`.

Everything here is metadata-driven with statistical fallbacks. No sensor is
assumed, because the evaluation set (Cartosat/RISAT) differs from the training
set (Sentinel-1/2) in band layout, GSD and dtype.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from satquery.contracts.input_manifest import ImageMeta

from .modality import detect_polarisations, harmonise_bands, infer_modality
from .product import resolve as resolve_product

# Metadata keys that commonly carry an acquisition timestamp.
_DATE_KEYS = (
    "ACQUISITION_DATE",
    "ACQUISITION_DATETIME",
    "DATE_ACQUIRED",
    "IMAGING_DATE",
    "TIFFTAG_DATETIME",
    "DATETIME",
    "START_TIME",
)

_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d",
    "%d-%b-%Y %H:%M:%S",
    "%d-%b-%Y",
)

# How many pixels to sample for statistics. Full reads are wasteful on the
# 8000x8000 scenes docs/04 expects, and a sample is sufficient for dtype
# range, nodata fraction and the backscatter heuristic.
_SAMPLE_MAX = 512


def parse_acquisition_dt(tags: dict) -> datetime | None:
    """Best-effort acquisition timestamp from vendor metadata."""
    for key in _DATE_KEYS:
        for tag_key, raw in tags.items():
            if tag_key.upper() != key:
                continue
            value = str(raw).strip()
            for fmt in _DATE_FORMATS:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
    return None


def estimate_effective_bits(sample: np.ndarray, dtype: str) -> int:
    """Actual used bit depth, which often differs from the container dtype.

    12-bit sensor data is routinely delivered in a uint16 container. Knowing
    the real depth matters for normalisation - scaling by 65535 when the data
    only spans 0-4095 crushes contrast.
    """
    if dtype.startswith("float"):
        return 32 if dtype == "float32" else 64
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return int(np.dtype(dtype).itemsize * 8)
    peak = float(finite.max())
    if peak <= 0:
        return int(np.dtype(dtype).itemsize * 8)
    bits = int(np.ceil(np.log2(peak + 1)))
    return max(1, min(bits, int(np.dtype(dtype).itemsize * 8)))


def _read_sample(src) -> np.ndarray:
    """Read a decimated sample of band 1 for statistics."""
    out_h = min(src.height, _SAMPLE_MAX)
    out_w = min(src.width, _SAMPLE_MAX)
    arr = src.read(1, out_shape=(out_h, out_w), masked=True)
    return np.ma.filled(arr.astype("float64"), np.nan)


# Cloud is bright in every optical band and spectrally flat - which is what
# distinguishes it from bright ground (roofs, sand, snow aside), whose bands
# differ. Both conditions are relative to the scene's own brightest pixels,
# so a dark scene and a bright one are judged on the same terms.
_CLOUD_BRIGHTNESS = 0.85     # fraction of each band's 99.9th percentile
_CLOUD_MAX_CV = 0.15         # coefficient of variation across bands
_CLOUD_MAX_BANDS = 4         # visible + NIR is enough; SWIR adds nothing here


def _read_band_sample(src, indexes: list[int]) -> np.ndarray:
    """A decimated sample of the given 1-based bands, (len, H, W), NaN for nodata.

    Takes explicit indexes rather than "the first N": an alpha channel that
    `photometric_bands` dropped is still present in the file, and reading
    1..N would hand the estimator a transparency mask as if it were a band.
    """
    out_h = min(src.height, _SAMPLE_MAX)
    out_w = min(src.width, _SAMPLE_MAX)
    arr = src.read(indexes, out_shape=(len(indexes), out_h, out_w), masked=True)
    return np.ma.filled(arr.astype("float64"), np.nan)


def estimate_cloud_pct(bands: np.ndarray) -> float | None:
    """Percentage of pixels that look like cloud, or None if it cannot be judged.

    WHY THIS EXISTS NOW

    `cloud_pct` has been in the manifest contract since Phase 1, annotated
    "requires a cloud mask; Phase 2 work", and was never populated or read.
    The demo bundle's abstention beat - a 63% clouded scene that "must
    abstain" - passed for two reasons neither of which was cloud: under CI the
    beat's expectation is `answered_or_abstained` and cannot fail, and with
    the learned tools enabled the land-cover head vetoed the answer with a
    0.0 that meant "no claim". When that veto was fixed on 2026-09-12 the
    scene was captioned at 0.88 confidence as "a white building near a road
    lake". Nothing in the pipeline had ever looked at the cloud.

    This is deliberately a coarse estimator, not a cloud mask: bright in every
    band AND spectrally flat, both relative to the scene's own brightest
    pixels. It will count snow and some very bright roofs, and it will miss
    thin cirrus. Its job is the WARN/FAIL check in `checks.py` - "is most of
    this scene unusable?" - not per-pixel masking, and it is honest about
    that in its name.
    """
    if bands.ndim != 3 or bands.shape[0] < 3:
        return None
    finite = np.all(np.isfinite(bands), axis=0)
    if finite.sum() < 100:
        return None
    pixels = bands[:, finite]                                   # (B, N)
    top = np.nanpercentile(pixels, 99.9, axis=1)                # per band
    if np.any(top <= 0):
        return None
    scaled = pixels / top[:, None]
    bright = np.all(scaled >= _CLOUD_BRIGHTNESS, axis=0)
    mean = scaled.mean(axis=0)
    flat = (scaled.std(axis=0) / np.maximum(mean, 1e-9)) <= _CLOUD_MAX_CV
    return float(100.0 * np.mean(bright & flat))


def _gsd_metres(src) -> float:
    """Ground sample distance in metres, converting from degrees if needed.

    For an ungeoreferenced raster - a plain PNG or JPEG - GDAL hands back the
    identity transform, so this returns 1.0. That is a placeholder, not a
    measurement, and `ImageMeta.georeferenced` is what tells the trace and the
    tools apart; see the note on that field.
    """
    x_res = abs(src.transform.a)
    if src.crs is not None and src.crs.is_geographic:
        # Approximate: 1 degree of latitude ~= 111320 m. Good enough for a
        # manifest field used for ordering-of-magnitude decisions; projected
        # products (the normal case) are exact.
        return float(x_res * 111_320.0)
    return float(x_res)


def read_canonical_band(meta: ImageMeta, band: str) -> np.ndarray:
    """Read one canonical band (e.g. "RED") from an image as float64.

    Raises KeyError if the band is not present - callers must check
    `index_availability` first rather than relying on an exception.
    """
    if band not in meta.bands:
        raise KeyError(
            f"band {band!r} not present in {meta.path.name}; has {meta.bands}"
        )
    idx = meta.bands.index(band) + 1  # rasterio bands are 1-indexed
    with rasterio.open(meta.path) as src:
        arr = src.read(idx, masked=True)
    return np.ma.filled(arr.astype("float64"), np.nan)


# GDAL's per-band photometric declaration, when it makes one. PNG and JPEG
# report ("red", "green", "blue"[, "alpha"]); the vendor GeoTIFFs report
# "undefined" for every band, so this refines the ordinary-image case without
# disturbing the products where positional convention is the only signal.
_PHOTOMETRIC = {"red": "RED", "green": "GREEN", "blue": "BLUE"}


def _tag_bounds(tags: dict, key: str) -> tuple[float, float, float, float] | None:
    """Read a four-number bounds tag written by the AOI crop, if present.

    A malformed tag is treated as absent: a crop whose provenance cannot be
    parsed should not fail the run, it should simply not claim anything.
    """
    raw = tags.get(key)
    if not raw:
        return None
    try:
        west, south, east, north = (float(v) for v in json.loads(raw))
    except Exception:  # noqa: BLE001 - an unreadable tag is not a failure
        return None
    return (west, south, east, north)


def wgs84_bounds(src) -> tuple[float, float, float, float] | None:
    """(west, south, east, north) in EPSG:4326, or None if unmeasurable.

    The system already held every ingredient for this - a CRS and an affine
    transform - and surfaced none of it, so "where is this?" was answered
    with "satellite imagery does not carry that information" on a GeoTIFF
    that carried exactly that. Measured from the file, never inferred.

    None rather than a guess in all three cases where the file does not
    actually say where it is: no CRS (a PNG or JPEG cannot carry one), the
    identity transform, or a projection that will not transform. The identity
    case is the subtle one - GDAL hands it back for any ungeoreferenced
    raster, and a GeoTIFF written without a geotransform gets it too, so the
    bounds would come out as pixel indices dressed as coordinates. That is
    worse than silence, because they look plausible.

    `densify_pts=21` matches `report/evidence_pack.raster_footprint`: a
    reprojected rectangle has curved edges, and sampling the sides keeps the
    envelope from cutting the corners off the real footprint.
    """
    if src.crs is None or src.transform.is_identity:
        return None
    try:
        west, south, east, north = transform_bounds(
            src.crs, "EPSG:4326", *src.bounds, densify_pts=21
        )
    except Exception:  # noqa: BLE001 - an unprojectable CRS is not a failure
        return None
    if not all(map(math.isfinite, (west, south, east, north))):
        return None
    return (round(west, 6), round(south, 6), round(east, 6), round(north, 6))


def photometric_bands(src) -> tuple[list[int], list[str | None], bool]:
    """(1-based band indices to keep, their names, whether alpha was dropped).

    Two bugs came from ignoring this, both on the ordinary PNG/JPEG path that
    PS-26167 opened up, and both measured rather than supposed:

    * **Alpha read as NIR.** A 4-channel RGBA PNG - a screenshot, or any
      export with transparency - has band 4 declared `alpha`. Band-count
      inference called it MSI and the positional fallback named the fourth
      band NIR, so `index_availability` advertised **ndvi and ndwi as
      computable from an opacity channel**. That is exactly the false
      capability claim the ingest checks exist to prevent.

    * **Red and blue swapped.** GDAL reports band 1 of a PNG as `red`, but
      with no descriptions the fallback assumed the GeoTIFF convention
      [BLUE, GREEN, RED]. `to_rgb_preview` then looked "RED" up at index 3
      and handed the model a channel-reversed image. Measured on a pure-red
      PNG: every preview channel came back flat, rendering mid-grey.

    Returns indices rather than mutating anything, so the caller keeps the
    mapping from canonical name to raster band index intact.
    """
    try:
        interp = [ci.name.lower() for ci in src.colorinterp]
    except Exception:  # noqa: BLE001 - a driver that declines is not a failure
        return list(range(1, src.count + 1)), [None] * src.count, False

    keep = [i for i, name in enumerate(interp) if name != "alpha"]
    dropped = len(keep) != len(interp)
    # Only trust the names when the container actually declared them; an
    # all-"undefined" raster must keep falling through to the conventional
    # ordering, which is what the vendor products rely on.
    named = any(interp[i] in _PHOTOMETRIC for i in keep)
    names = [_PHOTOMETRIC.get(interp[i]) if named else None for i in keep]
    return [i + 1 for i in keep], names, dropped


def read_image(
    path: str | Path,
    role: Literal["single", "optical", "sar", "t1", "t2"] = "single",
) -> ImageMeta:
    """Open a raster and build its `ImageMeta`. Raises on unreadable files."""
    # Vendor products ship one file per band (Cartosat MX: BAND1..4.tif;
    # EOS-04: scene_<POL>/imagery_<POL>.tif). resolve_product() unifies those
    # into a single openable path via a VRT, and hands back the vendor
    # metadata, which carries the band/polarisation identities and the radar
    # frequency that the raster headers do not.
    original = Path(path)
    path, layout = resolve_product(original)

    with rasterio.open(path) as src:
        tags = dict(src.tags())
        descriptions = list(src.descriptions)

        # Drop any alpha channel and take the photometric names the container
        # declared, before anything infers a modality from the band count.
        keep_indices, photometric, dropped_alpha = photometric_bands(src)
        descriptions = [descriptions[i - 1] for i in keep_indices]
        for position, name in enumerate(photometric):
            if name and not descriptions[position]:
                descriptions[position] = name
        band_count = len(keep_indices)

        # Vendor band/polarisation names beat the raster header, which for
        # these products is empty.
        if layout.band_names and len(layout.band_names) == band_count:
            descriptions = list(layout.band_names)

        # Surface vendor metadata as tags so modality inference can see the
        # satellite, sensor and polarisations it would otherwise miss.
        for key in ("satellite", "sensor", "radar_band", "imaging_mode"):
            value = layout.metadata.get(key)
            if value:
                tags[key.upper()] = str(value)
        if layout.metadata.get("polarisations"):
            tags["POLARISATION"] = " ".join(layout.metadata["polarisations"])
        sample = _read_sample(src)

        modality, evidence = infer_modality(
            band_count=band_count,
            dtype=src.dtypes[0],
            tags=tags,
            band_descriptions=descriptions,
            sample=sample,
        )

        # Dropping a channel changes what every downstream index can be
        # computed from, so it is disclosed rather than done silently.
        if dropped_alpha:
            evidence["alpha_band_dropped"] = True
            evidence["signals"] = list(evidence.get("signals", [])) + [
                "alpha_channel_excluded_not_spectral"
            ]

        # Carry vendor-level limitations into the evidence dict so the
        # checks can name the real reason a product is unusable, rather than
        # only reporting the downstream symptom (a missing CRS).
        for key in ("requires_geocoding", "unsupported_reason", "processing_level",
                    "radar_frequency_ghz", "radar_band", "n_beams"):
            if layout.metadata.get(key) is not None:
                evidence[key] = layout.metadata[key]

        bands, band_presence = harmonise_bands(descriptions, modality)
        pols = detect_polarisations(tags, descriptions)

        finite = sample[np.isfinite(sample)]
        nodata_pct = float(100.0 * (1.0 - finite.size / sample.size)) if sample.size else 0.0

        # Only for optical: SAR has no notion of cloud, and estimating it on
        # backscatter would report speckle as weather.
        cloud_pct = None
        if modality in ("OPTICAL", "MSI") and band_count >= 3:
            cloud_pct = estimate_cloud_pct(
                _read_band_sample(src, list(keep_indices[:_CLOUD_MAX_BANDS]))
            )
            if cloud_pct is not None:
                cloud_pct = round(cloud_pct, 2)

        sensor_guess = (
            tags.get("SATELLITE")
            or tags.get("SENSOR")
            or tags.get("MISSION")
            or tags.get("PLATFORM")
        )

        return ImageMeta(
            role=role,
            path=path,
            modality=modality,
            modality_evidence=evidence,
            crs=str(src.crs) if src.crs else "UNKNOWN",
            gsd_m=_gsd_metres(src),
            width=src.width,
            height=src.height,
            bands=bands,
            band_presence=band_presence,
            dtype=src.dtypes[0],
            effective_bits=estimate_effective_bits(sample, src.dtypes[0]),
            acquisition_dt=parse_acquisition_dt(tags),
            nodata_pct=round(nodata_pct, 4),
            cloud_pct=cloud_pct,
            sensor_guess=str(sensor_guess) if sensor_guess else None,
            polarisations=pols or None,
            # Equivalent number of looks, from the vendor's RangeLooks x
            # AzimuthLooks. Confirmed present in real EOS-04 metadata
            # (verification item 5), so it is no longer a placeholder.
            look_count_est=layout.metadata.get("equivalent_looks"),
            container_format=src.driver,
            georeferenced=src.crs is not None,
            lonlat_bounds=wgs84_bounds(src),
            # Written into the GeoTIFF by the API when an area was selected on
            # the map. Read back here rather than threaded through the
            # controller, so the provenance travels with the pixels and cannot
            # be separated from them by a later copy.
            aoi_applied=_tag_bounds(tags, "SATQUERY_AOI"),
            source_lonlat_bounds=_tag_bounds(tags, "SATQUERY_SOURCE_BOUNDS"),
            crs_is_projected=bool(src.crs is not None and src.crs.is_projected),
        )
