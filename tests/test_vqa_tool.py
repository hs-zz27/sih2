"""Tests for RGB preview rendering and the adapter-backed VQA tool.

The tool's generation path needs a GPU and a trained adapter, so it is
activated only when SATQUERY_VQA_BASE and SATQUERY_VQA_ADAPTER are set. What
is tested here without a GPU is everything that decides *what the model sees*
and *whether the real tool is used at all* - which is where the silent,
damaging failures live.
"""

from __future__ import annotations

import numpy as np
import pytest

from satquery.ingest import ingest
from satquery.tools.imaging import to_rgb_preview
from satquery.tools.rs_vqa import is_available
from satquery.tools.stubs import REGISTRY, RSVQAStub


class TestRGBPreview:
    def test_uses_canonical_bands_not_positional(self, msi_6band):
        """Band 1 is blue on Cartosat but HH on EOS-04.

        Selecting positionally would render a false-colour image and silently
        change what the model is being asked about.
        """
        meta = ingest([msi_6band]).images[0]
        _, prov = to_rgb_preview(meta)
        assert prov["band_selection"] == "canonical_rgb"
        assert prov["bands_shown"] == ["RED", "GREEN", "BLUE"]

    def test_falls_back_when_no_canonical_rgb(self, sar_dualpol):
        """SAR has no RGB; say so rather than inventing colour."""
        meta = ingest([sar_dualpol]).images[0]
        _, prov = to_rgb_preview(meta)
        assert prov["band_selection"] in {"first_three_bands", "single_band_greyscale"}

    def test_single_band_becomes_greyscale(self, pan_1band):
        meta = ingest([pan_1band]).images[0]
        image, prov = to_rgb_preview(meta)
        assert prov["band_selection"] == "single_band_greyscale"
        assert image.mode == "RGB"

    def test_output_is_rgb_8bit(self, msi_6band):
        meta = ingest([msi_6band]).images[0]
        image, _ = to_rgb_preview(meta)
        assert image.mode == "RGB"
        assert np.asarray(image).dtype == np.uint8

    def test_downsamples_large_scenes(self, msi_6band):
        meta = ingest([msi_6band]).images[0]
        image, prov = to_rgb_preview(meta, max_edge=64)
        assert max(image.size) <= 64
        assert prov["downsample_factor"] > 1

    def test_small_image_not_upscaled(self, msi_6band):
        meta = ingest([msi_6band]).images[0]
        image, prov = to_rgb_preview(meta, max_edge=4096)
        assert image.size == (meta.width, meta.height)
        assert prov["downsample_factor"] == 1.0

    def test_percentile_stretch_uses_full_range(self, msi_6band):
        """11-bit data in a uint16 container must not render near-black.

        Dividing by the dtype maximum instead of stretching would leave the
        model answering questions about a dark image.
        """
        meta = ingest([msi_6band]).images[0]
        arr = np.asarray(to_rgb_preview(meta)[0])
        assert arr.max() > 200, "stretch failed: image is too dark"
        assert arr.min() < 60, "stretch failed: image has no dark tones"

    def test_flat_band_becomes_mid_grey_not_noise(self, tmp_path):
        from tests.conftest import write_raster

        path = write_raster(
            tmp_path / "flat.tif",
            np.full((3, 64, 64), 1000, dtype="uint16"),
            band_names=["RED", "GREEN", "BLUE"],
        )
        meta = ingest([path]).images[0]
        arr = np.asarray(to_rgb_preview(meta)[0])
        assert arr.min() == arr.max() == 128

    def test_provenance_records_source_size(self, msi_6band):
        meta = ingest([msi_6band]).images[0]
        _, prov = to_rgb_preview(meta)
        assert prov["source_size"] == [meta.width, meta.height]
        assert "percentile" in prov["stretch"]


class TestAvailabilityGating:
    """The real tool must never activate half-configured."""

    def test_unavailable_without_env_vars(self, monkeypatch):
        monkeypatch.delenv("SATQUERY_VQA_BASE", raising=False)
        monkeypatch.delenv("SATQUERY_VQA_ADAPTER", raising=False)
        ok, reason = is_available()
        assert ok is False
        assert "not both set" in reason

    def test_unavailable_when_base_missing(self, monkeypatch, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        monkeypatch.setenv("SATQUERY_VQA_BASE", str(tmp_path / "nope"))
        monkeypatch.setenv("SATQUERY_VQA_ADAPTER", str(adapter))
        ok, reason = is_available()
        assert ok is False
        assert "base model not found" in reason

    def test_unavailable_when_adapter_missing(self, monkeypatch, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        monkeypatch.setenv("SATQUERY_VQA_BASE", str(base))
        monkeypatch.setenv("SATQUERY_VQA_ADAPTER", str(tmp_path / "nope"))
        ok, reason = is_available()
        assert ok is False
        assert "adapter not found" in reason

    def test_registry_falls_back_to_stub_when_unconfigured(self):
        """CI and GPU-less machines must get the stub, not a broken model."""
        if is_available()[0]:
            pytest.skip("real adapter is configured in this environment")
        assert isinstance(REGISTRY["rs_vqa_v1"], RSVQAStub)


class TestQueryInjection:
    def test_query_reaches_the_tool_without_entering_the_plan(self, msi_6band):
        """The question is runtime data, not a matrix-governed parameter.

        It must reach the tool but must not appear in the plan's params,
        which are what the validator checks for legality.
        """
        from satquery.controller.pipeline import Controller

        seen: dict = {}

        class Spy(RSVQAStub):
            def run(self, manifest, params):
                seen.update(params)
                return super().run(manifest, params)

        controller = Controller()
        original = REGISTRY["rs_vqa_v1"]
        REGISTRY["rs_vqa_v1"] = Spy()
        try:
            trace = controller.run([msi_6band], "How many buildings are visible?")
        finally:
            REGISTRY["rs_vqa_v1"] = original

        assert seen.get("_query") == "How many buildings are visible?"
        step = next(s for s in trace.execution if s.tool == "rs_vqa_v1")
        assert "_query" not in step.params, "query leaked into the audited plan params"
