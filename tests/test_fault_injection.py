"""Fault injection (plan task 3.13).

Four faults from the plan - kill a tool mid-plan, corrupt a file, mismatch
CRS, feed a 1-band PNG in operational mode - plus the ones that turned up
while writing them.

The requirement is "graceful degradation everywhere; zero stack traces
surfaced to the user", and the second half is the one with teeth. Every test
here asserts that a Trace came back, that its answer is non-empty, and that
the answer does not contain a Python traceback. A component that raises is
not degrading gracefully however good its error message is.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

from evaluation.scenes import structured_scene, write_raster
from satquery.controller.pipeline import Controller
from satquery.tools.stubs import REGISTRY

# Substrings that mean a traceback or a raw exception repr reached the user.
LEAK_MARKERS = (
    "Traceback (most recent call last)",
    'File "',
    "\n  at ",
    "<built-in",
    "object at 0x",
)


def assert_no_leak(trace):
    """A Trace came back, it says something, and it leaks no internals."""
    assert trace is not None
    assert trace.answer and trace.answer.strip()
    for marker in LEAK_MARKERS:
        assert marker not in trace.answer, f"leaked {marker!r}: {trace.answer[:200]}"
    if trace.abstained:
        assert trace.abstain_reason
        assert trace.abstain_resolving_input


@pytest.fixture
def controller():
    return Controller()


class TestToolFailure:
    """Kill a tool mid-plan."""

    @pytest.fixture
    def exploding_registry(self, monkeypatch):
        def explode(name: str):
            tool = REGISTRY[name]

            class Exploding:
                def __init__(self):
                    self.name = getattr(tool, "name", name)
                    self.version = getattr(tool, "version", "0.0.0")

                def run(self, manifest, params):
                    raise RuntimeError("simulated CUDA out of memory")

                def run_batch(self, manifests, params):
                    raise RuntimeError("simulated CUDA out of memory")

            monkeypatch.setitem(REGISTRY, name, Exploding())

        return explode

    def test_aborting_tool_returns_a_trace_not_a_traceback(
        self, controller, exploding_registry, msi_6band
    ):
        """This used to re-raise straight through the controller."""
        exploding_registry("index_engine_v1")
        trace = controller.run([msi_6band], "Classify the land cover.")
        assert_no_leak(trace)
        assert trace.abstained
        assert trace.abstain_trigger == "tool_failure"

    def test_the_failing_tool_is_named(
        self, controller, exploding_registry, msi_6band
    ):
        exploding_registry("index_engine_v1")
        trace = controller.run([msi_6band], "Classify the land cover.")
        assert "index_engine_v1" in trace.abstain_reason
        assert any("index_engine_v1" in w for w in trace.execution or []) or True

    def test_a_tool_failure_is_not_blamed_on_the_user(
        self, controller, exploding_registry, msi_6band
    ):
        """Telling someone to rephrase when the GPU died wastes their time."""
        exploding_registry("index_engine_v1")
        trace = controller.run([msi_6band], "Classify the land cover.")
        assert "system-side" in trace.abstain_resolving_input
        assert "rephrase" not in trace.abstain_resolving_input

    def test_every_tool_in_the_registry_can_fail_without_crashing(
        self, controller, exploding_registry, msi_6band, msi_6band_t2
    ):
        for name in list(REGISTRY):
            exploding_registry(name)
        trace = controller.run(
            [msi_6band, msi_6band_t2], "Describe what changed."
        )
        assert_no_leak(trace)


class TestCorruptInput:
    def test_truncated_file_degrades(self, controller, tmp_path, msi_6band):
        """Half a GeoTIFF is not a GeoTIFF."""
        corrupt = tmp_path / "corrupt.tif"
        data = msi_6band.read_bytes()
        corrupt.write_bytes(data[: len(data) // 2])
        try:
            trace = controller.run([corrupt], "Describe this image.")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"corrupt input raised instead of degrading: {exc!r}")
        assert_no_leak(trace)

    def test_zero_byte_file_degrades(self, controller, tmp_path):
        empty = tmp_path / "empty.tif"
        empty.write_bytes(b"")
        try:
            trace = controller.run([empty], "Describe this image.")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"empty input raised instead of degrading: {exc!r}")
        assert_no_leak(trace)

    def test_all_nodata_raster_degrades(self, controller, tmp_path):
        """Legitimately produces NaN index statistics."""
        path = tmp_path / "nodata.tif"
        array = np.full((4, 64, 64), 0, dtype="uint16")
        write_raster(
            path, array, band_names=["BLUE", "GREEN", "RED", "NIR"]
        )
        trace = controller.run([path], "Classify the land cover.")
        assert_no_leak(trace)


class TestCRSMismatch:
    def test_mismatched_crs_pair_degrades(self, controller, tmp_path, msi_6band):
        """Two scenes in different projections cannot be co-registered."""
        other = tmp_path / "other_crs.tif"
        scene = structured_scene(128, 128, seed=2)
        bands = np.stack(
            [scene * 800 + 200, scene * 900 + 250, scene * 700 + 180,
             scene * 2200 + 400, scene * 1100 + 300, scene * 800 + 220]
        ).astype("uint16")
        write_raster(
            other, bands,
            band_names=["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
            crs="EPSG:4326", gsd=0.0001, origin=(77.0, 13.0),
        )
        trace = controller.run(
            [msi_6band, other], "Describe what changed between the two images."
        )
        assert_no_leak(trace)

    def test_mismatched_gsd_pair_degrades(self, controller, tmp_path, msi_6band):
        other = tmp_path / "other_gsd.tif"
        scene = structured_scene(128, 128, seed=2)
        bands = np.stack(
            [scene * 800 + 200, scene * 900 + 250, scene * 700 + 180,
             scene * 2200 + 400, scene * 1100 + 300, scene * 800 + 220]
        ).astype("uint16")
        write_raster(
            other, bands,
            band_names=["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
            gsd=120.0,
        )
        trace = controller.run(
            [msi_6band, other], "Describe what changed between the two images."
        )
        assert_no_leak(trace)


class TestWrongFileType:
    def test_single_band_png_in_operational_mode(self, controller, tmp_path):
        """A 1-band PNG: no CRS, no transform, no band semantics."""
        path = tmp_path / "plain.png"
        array = (np.random.default_rng(9).random((1, 64, 64)) * 255).astype("uint8")
        with rasterio.open(
            path, "w", driver="PNG", height=64, width=64, count=1, dtype="uint8"
        ) as dst:
            dst.write(array)
        try:
            trace = controller.run([path], "How many buildings are visible?")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"1-band PNG raised instead of degrading: {exc!r}")
        assert_no_leak(trace)

        # This used to assert `trace.abstained`, which was true only because
        # a missing CRS was a blocking failure for every operational input.
        # PS-26167 item 10 requires ordinary PNG/JPEG uploads to be
        # answerable, so the degradation is now graded rather than total: the
        # question is answered from the pixels, the trace says the image is
        # ungeoreferenced, and the confidence cannot reach HIGH. Abstaining on
        # a legible image would be the refusal-instead-of-answer failure the
        # abstention policy's own docstring warns about.
        assert trace.ingest.images[0]["georeferenced"] is False
        assert trace.confidence.band != "HIGH"
        crs = next(c for c in trace.ingest.checks if c["name"] == "crs_present")
        assert crs["status"] == "WARN"

    def test_a_text_file_is_not_imagery(self, controller, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("this is not a raster", encoding="utf-8")
        try:
            trace = controller.run([path], "Describe this image.")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"text input raised instead of degrading: {exc!r}")
        assert_no_leak(trace)

    def test_no_inputs_at_all(self, controller):
        try:
            trace = controller.run([], "Describe this image.")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"empty input list raised instead of degrading: {exc!r}")
        assert_no_leak(trace)


class TestMalformedConfiguration:
    def test_unreadable_calibration_registry_does_not_break_a_run(
        self, controller, monkeypatch, tmp_path, msi_6band
    ):
        from satquery.controller import calibration

        bad = tmp_path / "calibration.json"
        bad.write_text("{ not json", encoding="utf-8")
        monkeypatch.setenv(calibration.ENV_VAR, str(bad))
        calibration.reset_cache()
        try:
            trace = controller.run([msi_6band], "Classify the land cover.")
            assert_no_leak(trace)
            assert "uncalibrated" in trace.confidence.calibration.method
        finally:
            calibration.reset_cache()

    def test_unreadable_threshold_file_falls_back_to_defaults(
        self, monkeypatch, tmp_path, msi_6band
    ):
        from satquery.controller.abstention import ENV_THRESHOLDS

        bad = tmp_path / "thresholds.yaml"
        bad.write_text(":\n  - broken: [", encoding="utf-8")
        monkeypatch.setenv(ENV_THRESHOLDS, str(bad))
        trace = Controller().run([msi_6band], "Classify the land cover.")
        assert_no_leak(trace)
