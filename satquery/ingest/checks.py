"""Input validation checks.

Each check returns a `CheckResult`. FAIL statuses become blocking failures and
force the planner toward CLARIFY_OR_ABSTAIN rather than producing an answer
the physics cannot support (docs/02). WARN statuses are surfaced in the trace
and feed the input-quality confidence component.
"""

from __future__ import annotations

from satquery.contracts.input_manifest import CheckResult, ImageMeta

# Thresholds. Kept here rather than inline so configs/thresholds.yaml can
# override them later without hunting through logic.
NODATA_WARN_PCT = 20.0
NODATA_FAIL_PCT = 80.0
GSD_RATIO_WARN = 4.0
GSD_RATIO_FAIL = 20.0
MIN_DIMENSION_PX = 32

# Below this the two images cannot be describing the same place at all, and
# any joint answer is about two different scenes. The per-task requirement is
# stricter and lives in the capability matrix (`min_overlap_pct`); this is the
# floor at which the pair itself is invalid rather than merely poor.
OVERLAP_FAIL_PCT = 10.0


def _pass(name: str, message: str, value=None) -> CheckResult:
    return CheckResult(name=name, status="PASS", value=value, message=message)


def _warn(name: str, message: str, value=None, threshold=None) -> CheckResult:
    return CheckResult(
        name=name, status="WARN", value=value, threshold=threshold, message=message
    )


def _fail(name: str, message: str, value=None, threshold=None) -> CheckResult:
    return CheckResult(
        name=name, status="FAIL", value=value, threshold=threshold, message=message
    )


# Container formats that cannot carry a CRS at all. A missing CRS in one of
# these is a property of the format, not a defect in the product.
NON_GEOSPATIAL_FORMATS = {"PNG", "JPEG", "JPEG2000", "GIF", "BMP", "WEBP"}


def check_crs_present(img: ImageMeta, benchmark: bool = False) -> CheckResult:
    """Georeferencing check, relaxed for benchmark inputs.

    The problem statement admits PNG and JPEG "only for the prescribed public
    benchmark datasets", and those are ungeoreferenced by construction - RSVQA
    and VRSBench ship plain rasters with no CRS. Failing them blocked every
    prescribed benchmark image from entering the pipeline at all, which is a
    mandatory-scope gap, not a strictness setting.

    In benchmark mode a missing CRS is a WARN: the answer is still valid, but
    nothing downstream can georeference an output or co-register a pair, and
    the trace has to say so rather than let a reader assume the mask is
    placeable. In operational mode it stays a hard failure.
    """
    if img.crs and img.crs != "UNKNOWN":
        return _pass("crs_present", f"{img.role}: CRS {img.crs}", img.crs)
    if (img.container_format or "").upper() in NON_GEOSPATIAL_FORMATS:
        # A PNG or JPEG the user uploaded to ask an ordinary visual question.
        # Failing it here refused the whole query, which is the wrong trade:
        # the vision model can answer "is this urban or rural?" from pixels
        # alone. What is NOT available is anything geospatial - the outputs
        # cannot be georeferenced, GSD and sensor are unknown, and the index
        # engine has no calibrated bands to work from - so this WARNs and
        # says so rather than passing silently. The distinction from the
        # branch below is deliberate: a GeoTIFF with no CRS is a defective
        # product and still fails.
        return _warn(
            "crs_present",
            f"{img.role}: {img.container_format} carries no geospatial "
            "metadata - visual questions can be answered from the pixels, but "
            "CRS, GSD and sensor are unknown, outputs cannot be georeferenced, "
            "and no geospatial index is available",
            img.crs,
        )
    if benchmark:
        return _warn(
            "crs_present",
            f"{img.role}: no CRS - accepted because this is a benchmark input, "
            "but outputs cannot be georeferenced or co-registered",
            img.crs,
        )
    return _fail(
        "crs_present",
        f"{img.role}: no CRS - cannot georeference outputs or co-register",
        img.crs,
    )


# Cloud cover, from the coarse estimator in ingest/reader.py. FAIL is a
# blocking failure - the run abstains with `input_validation` - so it is set
# where the scene is more cloud than not. WARN is where a caption or a count
# starts describing cloud as terrain.
CLOUD_WARN_PCT = 20.0
CLOUD_FAIL_PCT = 50.0


def check_cloud_cover(img: ImageMeta) -> CheckResult | None:
    """Is the scene usable at all, or mostly cloud?

    None for SAR and for anything the estimator declined to judge, rather
    than a PASS: a check that did not run must not read as a check that
    passed. Before this existed the pipeline had never looked at cloud - see
    `estimate_cloud_pct` for the 0.88-confidence caption of a 63% clouded
    scene that showed it.
    """
    v = img.cloud_pct
    if v is None:
        return None
    if v >= CLOUD_FAIL_PCT:
        return _fail(
            "cloud_cover",
            f"{img.role}: ~{v:.0f}% of the scene looks like cloud - too little "
            "ground is visible to describe",
            v,
            CLOUD_FAIL_PCT,
        )
    if v >= CLOUD_WARN_PCT:
        return _warn(
            "cloud_cover",
            f"{img.role}: ~{v:.0f}% of the scene looks like cloud - answers "
            "cover the clear part only",
            v,
            CLOUD_WARN_PCT,
        )
    return _pass("cloud_cover", f"{img.role}: ~{v:.0f}% cloud", v)


def check_nodata(img: ImageMeta) -> CheckResult:
    v = img.nodata_pct
    if v >= NODATA_FAIL_PCT:
        return _fail(
            "nodata_fraction",
            f"{img.role}: {v:.1f}% nodata - image is mostly empty",
            v,
            NODATA_FAIL_PCT,
        )
    if v >= NODATA_WARN_PCT:
        return _warn(
            "nodata_fraction",
            f"{img.role}: {v:.1f}% nodata - answers may cover only part of the scene",
            v,
            NODATA_WARN_PCT,
        )
    return _pass("nodata_fraction", f"{img.role}: {v:.1f}% nodata", v)


def check_dimensions(img: ImageMeta) -> CheckResult:
    smallest = min(img.width, img.height)
    if smallest < MIN_DIMENSION_PX:
        return _fail(
            "min_dimension",
            f"{img.role}: {img.width}x{img.height} is too small to analyse",
            smallest,
            MIN_DIMENSION_PX,
        )
    return _pass("min_dimension", f"{img.role}: {img.width}x{img.height}", smallest)


def check_crs_match(a: ImageMeta, b: ImageMeta) -> CheckResult:
    if a.crs == b.crs:
        return _pass("crs_match", f"both images in {a.crs}", a.crs)
    return _warn(
        "crs_match",
        f"CRS mismatch ({a.crs} vs {b.crs}) - reprojection required before co-registration",
        f"{a.crs}|{b.crs}",
    )


def check_gsd_ratio(a: ImageMeta, b: ImageMeta) -> CheckResult:
    lo, hi = sorted((a.gsd_m, b.gsd_m))
    if lo <= 0:
        return _fail("gsd_ratio", "non-positive GSD reported", lo)
    ratio = hi / lo
    if ratio >= GSD_RATIO_FAIL:
        return _fail(
            "gsd_ratio",
            f"GSD ratio {ratio:.1f}x is too large to compare meaningfully",
            round(ratio, 3),
            GSD_RATIO_FAIL,
        )
    if ratio >= GSD_RATIO_WARN:
        return _warn(
            "gsd_ratio",
            f"GSD ratio {ratio:.1f}x - the coarser image limits achievable detail",
            round(ratio, 3),
            GSD_RATIO_WARN,
        )
    return _pass("gsd_ratio", f"GSD ratio {ratio:.2f}x", round(ratio, 3))


def footprint_overlap_pct(a: ImageMeta, b: ImageMeta) -> float | None:
    """Percentage of the smaller footprint covered by the larger.

    Measured in the reference image's CRS. Returns None when either image is
    ungeoreferenced, because "no overlap" and "overlap unknown" are different
    answers and only the first is a defect in the pair.

    Ratio-of-the-smaller rather than intersection-over-union: a 1.6 m Cartosat
    scene inside a much larger EOS-04 ScanSAR swath is a *valid* pair whose IoU
    is tiny. What matters is whether the smaller image is covered.
    """
    import rasterio
    from rasterio.warp import transform_bounds

    try:
        with rasterio.open(a.path) as src_a, rasterio.open(b.path) as src_b:
            if src_a.crs is None or src_b.crs is None:
                return None
            bounds_a = src_a.bounds
            bounds_b = (
                src_b.bounds
                if src_b.crs == src_a.crs
                else transform_bounds(src_b.crs, src_a.crs, *src_b.bounds)
            )
    except Exception:  # noqa: BLE001 - an unreadable raster is another check's job
        return None

    left = max(bounds_a[0], bounds_b[0])
    bottom = max(bounds_a[1], bounds_b[1])
    right = min(bounds_a[2], bounds_b[2])
    top = min(bounds_a[3], bounds_b[3])
    if right <= left or top <= bottom:
        return 0.0

    intersection = (right - left) * (top - bottom)
    area_a = (bounds_a[2] - bounds_a[0]) * (bounds_a[3] - bounds_a[1])
    area_b = (bounds_b[2] - bounds_b[0]) * (bounds_b[3] - bounds_b[1])
    smaller = min(area_a, area_b)
    if smaller <= 0:
        return None
    return min(100.0, intersection / smaller * 100.0)


def check_footprint_overlap(a: ImageMeta, b: ImageMeta) -> CheckResult:
    """Do the two images describe the same place?

    Added 2026-08-30 (limitation L16). The capability matrix had declared
    `min_overlap_pct` since Phase 0 and **nothing read it**, so an optical and
    a SAR scene 60 km apart were fused into a single confident answer. The
    matrix's per-task threshold is enforced by the router; this check supplies
    the measurement it gates on, and fails outright below OVERLAP_FAIL_PCT
    where no task could accept the pair anyway.
    """
    overlap = footprint_overlap_pct(a, b)
    if overlap is None:
        return _warn(
            "footprint_overlap",
            "footprint overlap could not be measured - an image is "
            "ungeoreferenced, so co-registration cannot be verified",
        )
    if overlap < OVERLAP_FAIL_PCT:
        return _fail(
            "footprint_overlap",
            f"footprint overlap {overlap:.0f}% - the images do not cover the "
            f"same area, so no joint answer is meaningful",
            round(overlap, 2),
            OVERLAP_FAIL_PCT,
        )
    return _pass(
        "footprint_overlap", f"footprint overlap {overlap:.0f}%", round(overlap, 2)
    )


def check_crossmodal_pairing(images: list[ImageMeta]) -> CheckResult:
    mods = {img.modality for img in images}
    has_sar = "SAR" in mods
    has_optical = bool(mods & {"OPTICAL", "MSI", "PAN"})
    if has_sar and has_optical:
        return _pass("crossmodal_pairing", "one optical and one SAR image present")
    return _fail(
        "crossmodal_pairing",
        f"cross-modal analysis needs one optical and one SAR image; got {sorted(mods)}",
        ",".join(sorted(mods)),
    )


def check_temporal_order(a: ImageMeta, b: ImageMeta) -> CheckResult:
    if a.acquisition_dt is None or b.acquisition_dt is None:
        return _warn(
            "temporal_order",
            "acquisition dates missing - t1/t2 order taken from input order",
        )
    if a.acquisition_dt <= b.acquisition_dt:
        return _pass(
            "temporal_order",
            f"t1 {a.acquisition_dt.date()} precedes t2 {b.acquisition_dt.date()}",
        )
    return _warn(
        "temporal_order",
        f"t1 {a.acquisition_dt.date()} is AFTER t2 {b.acquisition_dt.date()} - "
        "change direction will be reported reversed unless inputs are swapped",
    )


def check_geocoding(img: ImageMeta) -> CheckResult | None:
    """Flag products that are not geocoded, naming the real reason.

    An SLC slant-range ScanSAR product genuinely cannot produce georeferenced
    output without beam mosaicking and geocoding first. Reporting only
    "no CRS" would be true but would send someone hunting for a projection
    that was never there.
    """
    if not img.modality_evidence.get("requires_geocoding"):
        return None
    return _fail(
        "geocoding_required",
        f"{img.role}: "
        + str(img.modality_evidence.get("unsupported_reason", "product is not geocoded")),
        img.modality_evidence.get("processing_level"),
    )


def run_checks(
    images: list[ImageMeta], config: str, benchmark: bool = False
) -> list[CheckResult]:
    """Run every check applicable to this input configuration."""
    results: list[CheckResult] = []
    for img in images:
        geocoding = check_geocoding(img)
        if geocoding is not None:
            results.append(geocoding)
        results.append(check_crs_present(img, benchmark=benchmark))
        results.append(check_nodata(img))
        cloud = check_cloud_cover(img)
        if cloud is not None:
            results.append(cloud)
        results.append(check_dimensions(img))

    if len(images) == 2:
        a, b = images
        results.append(check_crs_match(a, b))
        results.append(check_gsd_ratio(a, b))
        results.append(check_footprint_overlap(a, b))
        if config == "CROSSMODAL_PAIR":
            results.append(check_crossmodal_pairing(images))
        elif config == "BITEMPORAL_PAIR":
            results.append(check_temporal_order(a, b))

    return results


def blocking_failures(checks: list[CheckResult]) -> list[str]:
    """Names of checks that must stop the pipeline producing a real answer."""
    return [c.name for c in checks if c.status == "FAIL"]
