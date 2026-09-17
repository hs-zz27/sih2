from enum import Enum
from typing import Literal
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel

class IngestMode(str, Enum):
    OPERATIONAL = "operational"
    BENCHMARK = "benchmark"

class ImageMeta(BaseModel):
    role: Literal["single", "optical", "sar", "t1", "t2"]
    path: Path
    modality: Literal["OPTICAL", "MSI", "PAN", "SAR"]
    modality_evidence: dict
    crs: str
    gsd_m: float
    width: int
    height: int
    bands: list[str]
    band_presence: list[bool]
    dtype: str
    effective_bits: int
    acquisition_dt: datetime | None
    nodata_pct: float
    cloud_pct: float | None
    sensor_guess: str | None
    polarisations: list[str] | None
    look_count_est: float | None
    # GDAL driver of the container the pixels came out of, and whether
    # that container actually carried georeferencing. A PNG or JPEG
    # cannot carry a CRS at all, which is a different fact from a
    # GeoTIFF that was written without one - the first is a format
    # limitation to disclose, the second is a defective product to
    # reject. Defaulted so existing constructions stay valid.
    container_format: str | None = None
    georeferenced: bool = True
    # Set when the run was given a crop of a larger scene rather than the
    # scene itself. `aoi_applied` is the requested box and
    # `source_lonlat_bounds` the extent of the file it came out of, both in
    # EPSG:4326. Reporting on a crop as though the whole scene had been
    # examined is the kind of quiet overclaim the manifest exists to prevent,
    # so both travel with the image. None on an ordinary upload.
    aoi_applied: tuple[float, float, float, float] | None = None
    source_lonlat_bounds: tuple[float, float, float, float] | None = None
    # Geographic extent in EPSG:4326 as (west, south, east, north), when
    # the raster carries enough georeferencing to compute one. This is
    # measured from the file's own CRS and transform - it is not a guess,
    # and it is None rather than zeroed when it cannot be measured.
    #
    # This one field is the only stored footprint; the centre and the ground
    # extent below are derived from it and from `gsd_m`. Two branches added
    # this measurement independently - `lonlat_bounds` here and a
    # `bounds_wgs84` / `centroid_wgs84` pair on the answer-composition branch -
    # and carrying both would have put the same number on the manifest twice.
    # This name won because the running API image and the frontend map already
    # read it.
    lonlat_bounds: tuple[float, float, float, float] | None = None
    # Whether the CRS measures in metres. `gsd_m` is exact for a projected
    # CRS and an approximation for a geographic one (see `_gsd_metres`, which
    # converts degrees at a flat 111320 m and is therefore wrong by the
    # cosine of the latitude in x). Ground extent is only reported for the
    # exact case; the approximation is fine for ordering-of-magnitude routing
    # decisions but not for a number shown to a reader as a distance.
    crs_is_projected: bool = False

    @property
    def centroid_latlon(self) -> tuple[float, float] | None:
        """Centre of the footprint as (latitude, longitude).

        Latitude first - the order a reader says them in, and deliberately
        the opposite of `lonlat_bounds`, which is in the (west, south, east,
        north) order every GIS tool expects. Derived rather than stored so
        the two can never disagree.
        """
        if self.lonlat_bounds is None:
            return None
        west, south, east, north = self.lonlat_bounds
        return ((south + north) / 2.0, (west + east) / 2.0)

    @property
    def ground_extent_m(self) -> tuple[float, float] | None:
        """Scene width and height on the ground, in metres.

        Only defined where `gsd_m` is a real measurement in metres.
        """
        if not self.georeferenced or not self.crs_is_projected:
            return None
        return (self.width * self.gsd_m, self.height * self.gsd_m)

class CheckResult(BaseModel):
    name: str
    status: Literal["PASS", "WARN", "FAIL"]
    value: float | str | bool | None = None
    threshold: float | str | None = None
    message: str

class CoregReport(BaseModel):
    method: Literal["phase_correlation", "gradient_phase_correlation", "mutual_information"]
    shift_px: tuple[float, float]
    shift_m: tuple[float, float]
    residual_px: float
    applied_correction: bool

class TilingReport(BaseModel):
    applied: bool
    level1_tiles: int | None = None
    retrieved_tiles: int | None = None
    retrieval_reason: str | None = None

class InputManifest(BaseModel):
    run_id: str
    ingest_mode: IngestMode
    benchmark: str | None = None
    images: list[ImageMeta]
    config: Literal["SINGLE", "CROSSMODAL_PAIR", "BITEMPORAL_PAIR"]
    checks: list[CheckResult]
    coreg: CoregReport | None = None
    tiling: TilingReport | None = None
    artifacts: dict[str, Path]
    blocking_failures: list[str]
    index_availability: dict[str, bool]
