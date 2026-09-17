"""Shared synthetic raster fixtures.

Real Cartosat/RISAT products are not in the repo (see docs/verification.md),
so tests build synthetic rasters with known properties. Every fixture states
what it is meant to represent so a failure is interpretable.

The builders themselves live in `evaluation/scenes.py` so the Phase 3
evaluation scripts - adversarial routing (3.8), the soak test (3.11) and fault
injection (3.13) - use the same definitions rather than importing from the
test tree or growing a second copy that drifts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The repo root holds `training/`, `evaluation/` and `scripts/`, which are
# operational code rather than installed runtime packages. Putting the root on
# sys.path lets the tests import them without publishing a generically-named
# `scripts` package in the installed distribution, where it could collide with
# another project's.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest  # noqa: E402

from evaluation.scenes import (  # noqa: E402
    build_msi_4band,
    build_msi_6band,
    build_msi_6band_t2,
    build_no_crs_raster,
    build_pan_1band,
    build_rgb_3band,
    build_sar_dualpol,
    build_tiny_raster,
    structured_scene,
    write_raster,
)

# Re-exported: several test modules build one-off rasters with these.
__all__ = ["write_raster", "structured_scene"]

_structured_scene = structured_scene


@pytest.fixture(scope="session", autouse=True)
def no_automatic_artifact_prune():
    """Stop the suite from pruning the developer's real `artifacts/` tree.

    `satquery ask` and `satquery eval` prune old `artifacts/run_*` directories
    after a run, and several tests invoke those entry points for real from the
    repository root. The first suite run after retention landed reclaimed
    **12.29 GB** as a side effect. Nothing protected was touched - all 78 named
    evidence directories survived, which is the guarantee working exactly as
    designed - but a test run that silently deletes a developer's output is a
    surprise, and this project has already paid for one of those.

    Only the *implicit* prune is disabled. `prune_run_artifacts` itself still
    runs, so `tests/test_retention.py` exercises the real thing against
    `tmp_path`, and `satquery prune` typed by a human still prunes.
    """
    from satquery.controller.retention import ENV_NO_AUTO_PRUNE

    previous = os.environ.get(ENV_NO_AUTO_PRUNE)
    os.environ[ENV_NO_AUTO_PRUNE] = "1"
    yield
    if previous is None:
        os.environ.pop(ENV_NO_AUTO_PRUNE, None)
    else:
        os.environ[ENV_NO_AUTO_PRUNE] = previous


@pytest.fixture
def msi_4band(tmp_path):
    """4-band VNIR optical, the assumed Cartosat-2S MX layout (no SWIR)."""
    return build_msi_4band(tmp_path / "msi_4band.tif")


@pytest.fixture
def rgb_3band(tmp_path):
    return build_rgb_3band(tmp_path / "rgb_3band.tif")


@pytest.fixture
def msi_6band(tmp_path):
    """6-band optical including SWIR - enables MNDWI and NDBI."""
    return build_msi_6band(tmp_path / "msi_6band.tif")


@pytest.fixture
def sar_dualpol(tmp_path):
    """2-band C-band SAR, VV+VH, float32 - EOS-04/Sentinel-1 shaped."""
    return build_sar_dualpol(tmp_path / "sar_dualpol.tif")


@pytest.fixture
def pan_1band(tmp_path):
    """Single-band panchromatic optical."""
    return build_pan_1band(tmp_path / "pan.tif")


@pytest.fixture
def msi_6band_t2(tmp_path):
    """A second, later acquisition of the same area - bitemporal partner."""
    return build_msi_6band_t2(tmp_path / "msi_6band_t2.tif")


@pytest.fixture
def tiny_raster(tmp_path):
    """Below the minimum dimension - must trigger a blocking check failure."""
    return build_tiny_raster(tmp_path / "tiny.tif")


@pytest.fixture
def no_crs_raster(tmp_path):
    """Missing CRS - must trigger a blocking check failure."""
    return build_no_crs_raster(tmp_path / "nocrs.tif")


def pytest_addoption(parser):
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Regenerate golden trace files instead of comparing against them.",
    )
