"""Synthetic scene builders, shared by tests and evaluation scripts.

Real Cartosat/RISAT products are not in the repo (see docs/verification.md),
so both the test suite and the Phase 3 evaluation scripts - the adversarial
routing suite (3.8), the soak test (3.11) and fault injection (3.13) - need
rasters with known properties. These were previously defined inside
`tests/conftest.py`, which meant an evaluation script could only reach them by
importing from the test tree.

They live here so there is **one definition**. The seeds and parameters are
unchanged from the original fixtures, because the golden traces in
`tests/golden_traces/` are byte-compared against traces derived from these
exact scenes: changing a seed here silently invalidates ten goldens.

NOTE: deliberately no shared module-level RNG. A single generator whose state
advances across builders makes raster content depend on how many other
builders ran first, which silently breaks the golden-trace comparisons when
the suite runs in a different order or subset. Every builder seeds its own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def write_raster(
    path,
    array: np.ndarray,
    *,
    band_names: list[str] | None = None,
    crs: str = "EPSG:32643",
    gsd: float = 10.0,
    origin: tuple[float, float] = (500000.0, 2000000.0),
    tags: dict | None = None,
    dtype: str | None = None,
    nodata: float | None = None,
):
    """Write a (bands, h, w) array to a GeoTIFF with the given metadata.

    `nodata` is optional and defaults to unset, which is what every golden
    scene uses. It exists for the confidence stress harness (task 3.4), which
    needs a raster the ingest nodata check will actually fire on - writing
    zeros without declaring them nodata produces a flat scene, not a nodata
    scene, and measures the wrong component.
    """
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    count, height, width = array.shape
    dtype = dtype or array.dtype.name
    transform = from_origin(origin[0], origin[1], gsd, gsd)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        **({"nodata": nodata} if nodata is not None else {}),
    ) as dst:
        dst.write(array.astype(dtype))
        if band_names:
            dst.descriptions = tuple(band_names)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})
    return path


def structured_scene(h: int, w: int, seed: int = 0) -> np.ndarray:
    """A scene with real spatial structure (edges), not pure noise.

    Co-registration needs actual features to lock onto; white noise has no
    stable correlation peak.
    """
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w), dtype="float64")
    # A few bright rectangles act as field/building boundaries.
    for _ in range(6):
        y0, x0 = rng.integers(0, h - h // 4), rng.integers(0, w - w // 4)
        dy, dx = rng.integers(h // 8, h // 4), rng.integers(w // 8, w // 4)
        base[y0 : y0 + dy, x0 : x0 + dx] += rng.uniform(0.4, 1.0)
    base += rng.normal(0, 0.02, size=(h, w))
    return base


def _optical_bands(scene: np.ndarray, swir: bool) -> np.ndarray:
    layers = [
        scene * 800 + 200,   # BLUE
        scene * 900 + 250,   # GREEN
        scene * 700 + 180,   # RED
        scene * 2200 + 400,  # NIR - vegetation bright
    ]
    if swir:
        layers += [scene * 1100 + 300, scene * 800 + 220]
    return np.stack(layers).astype("uint16")


def build_msi_4band(path: Path):
    """4-band VNIR optical, the assumed Cartosat-2S MX layout (no SWIR)."""
    scene = structured_scene(128, 128, seed=1)
    return write_raster(
        path,
        _optical_bands(scene, swir=False),
        band_names=["BLUE", "GREEN", "RED", "NIR"],
        gsd=1.6,
        tags={"SATELLITE": "CARTOSAT-2S", "SENSOR": "MX"},
    )


def build_rgb_3band(path: Path):
    """3-band RGB optical: no NIR and no SWIR, so no standard index applies.

    The prescribed public benchmarks ship exactly this - RSVQA and VRSBench
    are plain RGB rasters - and it is the configuration in which a claimed
    NDWI substitution is a claim about a band that is not there.
    """
    scene = structured_scene(128, 128, seed=1)
    return write_raster(
        path,
        np.stack([scene * 800 + 200, scene * 900 + 250, scene * 700 + 180]).astype(
            "uint16"
        ),
        band_names=["BLUE", "GREEN", "RED"],
        gsd=1.6,
        tags={"SATELLITE": "RGB-ONLY"},
    )


def build_msi_6band(path: Path):
    """6-band optical including SWIR - enables MNDWI and NDBI."""
    scene = structured_scene(128, 128, seed=2)
    return write_raster(
        path,
        _optical_bands(scene, swir=True),
        band_names=["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
        gsd=10.0,
        tags={"SATELLITE": "SENTINEL-2", "ACQUISITION_DATE": "2026-03-01"},
    )


def build_msi_6band_t2(path: Path):
    """A second, later acquisition of the same area - bitemporal partner."""
    scene = structured_scene(128, 128, seed=2)
    # Simulate change: one region loses vegetation.
    scene[20:60, 20:60] *= 0.3
    return write_raster(
        path,
        _optical_bands(scene, swir=True),
        band_names=["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
        gsd=10.0,
        tags={"SATELLITE": "SENTINEL-2", "ACQUISITION_DATE": "2026-06-01"},
    )


def build_sar_dualpol(path: Path):
    """2-band C-band SAR, VV+VH, float32 - EOS-04/Sentinel-1 shaped."""
    h = w = 128
    scene = structured_scene(h, w, seed=3)
    # Gamma-distributed speckle on top of structure: right-skewed, non-negative.
    speckle = np.random.default_rng(303).gamma(shape=2.0, scale=0.5, size=(h, w))
    vv = np.clip((scene + 0.2) * speckle, 1e-4, None)
    vh = np.clip((scene + 0.1) * speckle * 0.4, 1e-4, None)
    return write_raster(
        path,
        np.stack([vv, vh]).astype("float32"),
        band_names=["VV", "VH"],
        gsd=10.0,
        tags={"SATELLITE": "EOS-04", "SENSOR": "SAR", "POLARISATION": "VV VH"},
        dtype="float32",
    )


def build_pan_1band(path: Path):
    """Single-band panchromatic optical."""
    scene = structured_scene(128, 128, seed=4)
    return write_raster(
        path,
        (scene * 3000 + 500).astype("uint16"),
        band_names=["PAN"],
        gsd=0.65,
        tags={"SATELLITE": "CARTOSAT-2S", "SENSOR": "PAN"},
    )


def build_tiny_raster(path: Path):
    """Below the minimum dimension - must trigger a blocking check failure."""
    data = np.random.default_rng(404).random((1, 8, 8)).astype("float32")
    return write_raster(path, data, dtype="float32")


def build_no_crs_raster(path: Path):
    """Missing CRS - must trigger a blocking check failure."""
    array = (np.random.default_rng(505).random((3, 64, 64)) * 1000).astype("uint16")
    with rasterio.open(
        path, "w", driver="GTiff", height=64, width=64, count=3, dtype="uint16"
    ) as dst:
        dst.write(array)
    return path


BUILDERS = {
    "msi_4band": build_msi_4band,
    "msi_6band": build_msi_6band,
    "msi_6band_t2": build_msi_6band_t2,
    "sar_dualpol": build_sar_dualpol,
    "pan_1band": build_pan_1band,
    "tiny_raster": build_tiny_raster,
    "no_crs_raster": build_no_crs_raster,
}


def build_configurations(directory: Path) -> dict[str, list[Path]]:
    """The three input configurations the capability matrix distinguishes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    single = build_msi_6band(directory / "msi_6band.tif")
    second = build_msi_6band_t2(directory / "msi_6band_t2.tif")
    sar = build_sar_dualpol(directory / "sar_dualpol.tif")
    return {
        "SINGLE": [single],
        "CROSSMODAL": [single, sar],
        "BITEMPORAL": [single, second],
    }
