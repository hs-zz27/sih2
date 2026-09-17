"""Layer 0 ingest pipeline: files on disk -> validated `InputManifest`.

This is the entry point the controller calls. It infers the input
configuration, reads every image, runs the checks, co-registers pairs, and
reports which spectral indices are computable so the planner can route around
missing bands instead of failing.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from satquery.contracts.input_manifest import (
    CheckResult,
    ImageMeta,
    IngestMode,
    InputManifest,
    TilingReport,
)

from .checks import blocking_failures, run_checks
from .coreg import coregister
from .modality import CANONICAL_BANDS, index_availability
from .tiling import DEFAULT_TILE_PX, needs_tiling, plan_tiles
from .reader import read_image

OPTICAL_MODALITIES = {"OPTICAL", "MSI", "PAN"}

# Tiling thresholds now live in satquery/ingest/tiling.py, which implements
# the pyramid rather than only reporting the need for one (task 2.10).


def infer_config(images: list[ImageMeta]) -> str:
    """SINGLE / CROSSMODAL_PAIR / BITEMPORAL_PAIR from the images themselves."""
    if len(images) == 1:
        return "SINGLE"
    if len(images) != 2:
        raise ValueError(f"expected 1 or 2 images, got {len(images)}")

    mods = [img.modality for img in images]
    has_sar = "SAR" in mods
    has_optical = any(m in OPTICAL_MODALITIES for m in mods)
    if has_sar and has_optical:
        return "CROSSMODAL_PAIR"
    return "BITEMPORAL_PAIR"


def assign_roles(images: list[ImageMeta], config: str) -> list[ImageMeta]:
    """Set each image's role to match the inferred configuration."""
    if config == "SINGLE":
        return [images[0].model_copy(update={"role": "single"})]

    if config == "CROSSMODAL_PAIR":
        out = []
        for img in images:
            role = "sar" if img.modality == "SAR" else "optical"
            out.append(img.model_copy(update={"role": role}))
        # Keep optical first for a stable reference image in co-registration.
        out.sort(key=lambda i: i.role != "optical")
        return out

    # BITEMPORAL_PAIR: order by acquisition date when both are known,
    # otherwise trust the caller's input order.
    ordered = list(images)
    if all(i.acquisition_dt is not None for i in ordered):
        ordered.sort(key=lambda i: i.acquisition_dt)
    return [
        ordered[0].model_copy(update={"role": "t1"}),
        ordered[1].model_copy(update={"role": "t2"}),
    ]


def build_tiling_report(images: list[ImageMeta]) -> TilingReport:
    """Report the tile plan for the largest input image.

    Retrieval itself is query-dependent, so the manifest records the plan and
    the executor narrows it per query via `tiling.select_tiles`.
    """
    largest = max(images, key=lambda i: i.width * i.height)
    if not needs_tiling(largest.width, largest.height):
        return TilingReport(applied=False)

    tiles = plan_tiles(largest.width, largest.height, DEFAULT_TILE_PX)
    return TilingReport(
        applied=True,
        level1_tiles=len(tiles),
        retrieved_tiles=None,  # set per query when retrieval runs
        retrieval_reason=(
            f"scene is {largest.width}x{largest.height}; covered by "
            f"{len(tiles)} tiles of {DEFAULT_TILE_PX}px. Statistics are "
            "accumulated tile by tile, so peak memory scales with the tile "
            "size rather than the scene."
        ),
    )


def merged_index_availability(images: list[ImageMeta]) -> dict[str, bool]:
    """Union band presence across images, then derive index availability."""
    n = len(images[0].band_presence)
    merged = [any(img.band_presence[i] for img in images) for i in range(n)]
    modalities = [img.modality for img in images]
    pols: list[str] = []
    for img in images:
        for p in img.polarisations or []:
            if p not in pols:
                pols.append(p)
    return index_availability(merged, modalities, pols)


def _unreadable_manifest(
    run_id: str,
    mode: IngestMode,
    benchmark: str | None,
    failure: CheckResult,
) -> InputManifest:
    """A manifest carrying nothing but the reason it is empty.

    `config="SINGLE"` is a placeholder with no images behind it; the router
    short-circuits to CLARIFY_OR_ABSTAIN on `blocking_failures` before the
    configuration is used for anything, so it never selects a task.
    """
    return InputManifest(
        run_id=run_id,
        ingest_mode=mode,
        benchmark=benchmark,
        images=[],
        config="SINGLE",
        checks=[failure],
        coreg=None,
        tiling=None,
        artifacts={},
        blocking_failures=[failure.name],
        # No bands were read, so nothing is computable. Built from the
        # canonical band list rather than an empty one so this stays correct
        # if a band is ever added.
        index_availability=index_availability(
            [False] * len(CANONICAL_BANDS), [], []
        ),
    )


def ingest(
    paths: list[str | Path],
    mode: IngestMode = IngestMode.OPERATIONAL,
    benchmark: str | None = None,
    run_id: str | None = None,
) -> InputManifest:
    """Build an `InputManifest` from one or two raster paths.

    Unreadable inputs and an empty input list are reported as *blocking check
    failures*, not exceptions (task 3.13). Ingest is the first thing a user's
    file touches, so it is where a corrupt upload, a zero-byte file or a
    text file arrives - and raising there put a `RasterioIOError` traceback
    in front of whoever called the API. A manifest that says "this file could
    not be opened" flows through the same abstention path as every other bad
    input, which is the behaviour the rest of the system already handles.
    """
    if len(paths) > 2:
        raise ValueError(f"at most two images are supported, got {len(paths)}")

    run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"

    if not paths:
        return _unreadable_manifest(
            run_id, mode, benchmark,
            CheckResult(
                name="inputs_present", status="FAIL", value=0,
                message="no input images were supplied",
            ),
        )

    images = []
    for path in paths:
        try:
            images.append(read_image(path))
        except Exception as exc:  # noqa: BLE001 - degradation, not a crash
            # The exception type is part of the message because "not
            # recognized as being in a supported file format" and "failed to
            # read directory at offset N" point at different problems: the
            # wrong kind of file versus a truncated upload.
            return _unreadable_manifest(
                run_id, mode, benchmark,
                CheckResult(
                    name="file_readable", status="FAIL", value=str(path),
                    message=(
                        f"{Path(path).name} could not be opened as a raster "
                        f"({type(exc).__name__}); it may be corrupt, "
                        f"truncated, or not an image file"
                    ),
                ),
            )

    config = infer_config(images)
    images = assign_roles(images, config)

    # Benchmark inputs (RSVQA, VRSBench) are ungeoreferenced PNG/JPEG by
    # construction; the problem statement admits them for exactly those
    # datasets, so the CRS check relaxes to a WARN in this mode.
    checks = run_checks(images, config, benchmark=mode == IngestMode.BENCHMARK)
    failures = blocking_failures(checks)

    coreg = None
    if len(images) == 2 and not failures:
        # Co-registration reads pixels; skip it when the inputs are already
        # known-bad, so a broken file surfaces as a check failure rather than
        # an exception from deep inside the correlator.
        coreg = coregister(images[0], images[1])

    return InputManifest(
        run_id=run_id,
        ingest_mode=mode,
        benchmark=benchmark,
        images=images,
        config=config,
        checks=checks,
        coreg=coreg,
        tiling=build_tiling_report(images),
        artifacts={},
        blocking_failures=failures,
        index_availability=merged_index_availability(images),
    )
