"""End-to-end tests for the real `index_engine_v1` tool (plan task 1.2)."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

from satquery.ingest import ingest
from satquery.tools.index_engine import IndexEngine
from satquery.tools.stubs import REGISTRY


@pytest.fixture
def engine():
    return IndexEngine()


def run_on(engine, paths, tmp_path, **params):
    manifest = ingest(paths)
    params.setdefault("output_dir", str(tmp_path / "artifacts"))
    return manifest, engine.run(manifest, params)


class TestRegistryWiring:
    def test_registry_uses_the_real_engine_not_the_stub(self):
        assert isinstance(REGISTRY["index_engine_v1"], IndexEngine)

    def test_reports_deterministic_confidence(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path)
        assert result.confidence == 1.0
        assert result.confidence_method == "deterministic"
        assert result.version == "1.0.0"


class TestSixBandOptical:
    def test_computes_all_four_optical_indices(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path)
        indices = result.payload.data["indices"]
        assert {"ndvi", "ndwi", "mndwi", "ndbi"} <= set(indices)

    def test_every_index_has_stats_and_a_threshold(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path)
        for name, entry in result.payload.data["indices"].items():
            if name == "cov":
                continue
            assert "stats" in entry, name
            assert "threshold" in entry, name
            assert entry["threshold_method"] in {"otsu", "gmm", "fixed_prior"}

    def test_index_values_within_physical_range(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path)
        for name in ("ndvi", "ndwi", "mndwi", "ndbi"):
            stats = result.payload.data["indices"][name]["stats"]
            assert stats["min"] >= -1.0
            assert stats["max"] <= 1.0

    def test_no_substitution_needed_when_swir_present(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path)
        subs = " ".join(result.payload.data["substitutions"])
        assert "MNDWI unavailable" not in subs
        assert "NDBI unavailable" not in subs


class TestSwirFreePath:
    """A 4-band VNIR product - the assumed Cartosat-2S MX case."""

    def test_swir_indices_absent(self, engine, msi_4band, tmp_path):
        _, result = run_on(engine, [msi_4band], tmp_path)
        indices = result.payload.data["indices"]
        assert "mndwi" not in indices
        assert "ndbi" not in indices

    def test_vnir_indices_still_computed(self, engine, msi_4band, tmp_path):
        _, result = run_on(engine, [msi_4band], tmp_path)
        assert {"ndvi", "ndwi"} <= set(result.payload.data["indices"])

    def test_builtup_proxy_substituted_for_ndbi(self, engine, msi_4band, tmp_path):
        _, result = run_on(engine, [msi_4band], tmp_path)
        assert "builtup_proxy" in result.payload.data["indices"]

    def test_substitutions_are_named_explicitly(self, engine, msi_4band, tmp_path):
        _, result = run_on(engine, [msi_4band], tmp_path)
        subs = " ".join(result.payload.data["substitutions"])
        assert "MNDWI unavailable" in subs
        assert "NDBI unavailable" in subs

    def test_proxy_lowers_reliability_via_warning(self, engine, msi_4band, tmp_path):
        _, result = run_on(engine, [msi_4band], tmp_path)
        assert any("SWIR-free proxy" in w for w in result.warnings)


class TestRgbOnlyNoNirNoSwir:
    """3-band RGB: no NIR, no SWIR1, so no standard optical index applies.

    The defect this class exists for: the MNDWI branch appended
    "MNDWI unavailable (no SWIR1); NDWI used as the water index" whenever
    MNDWI was unavailable - including when NDWI was unavailable too. An
    RGB-only input therefore reported that NDWI had been used as the water
    index while computing no water index at all, and because the executor
    turns every substitution into a verification conflict, the false claim
    reached the trace twice.
    """

    def test_no_optical_index_is_available(self, engine, rgb_3band, tmp_path):
        manifest, _ = run_on(engine, [rgb_3band], tmp_path)
        avail = manifest.index_availability
        assert avail["ndvi"] is False
        assert avail["ndwi"] is False
        assert avail["mndwi"] is False
        assert avail["ndbi"] is False

    def test_no_index_is_actually_computed(self, engine, rgb_3band, tmp_path):
        _, result = run_on(engine, [rgb_3band], tmp_path)
        assert result.payload.data["indices"] == {}

    def test_does_not_claim_ndwi_was_used(self, engine, rgb_3band, tmp_path):
        _, result = run_on(engine, [rgb_3band], tmp_path)
        subs = result.payload.data["substitutions"]
        assert subs == [], f"substitution claimed with nothing computed: {subs}"
        assert not any("NDWI used" in s for s in subs)

    def test_says_explicitly_that_no_water_index_was_computed(
        self, engine, rgb_3band, tmp_path
    ):
        # Requirement: report the absence, rather than either claiming a
        # substitution or saying nothing.
        _, result = run_on(engine, [rgb_3band], tmp_path)
        assert any(
            "no water index computed" in w for w in result.warnings
        ), result.warnings

    def test_says_explicitly_that_no_builtup_index_was_computed(
        self, engine, rgb_3band, tmp_path
    ):
        _, result = run_on(engine, [rgb_3band], tmp_path)
        assert any(
            "no built-up index computed" in w for w in result.warnings
        ), result.warnings

    def test_glcm_texture_still_computed_on_a_single_band(
        self, engine, rgb_3band, tmp_path
    ):
        # Availability says texture is computable on any band; this asserts
        # the claim is true rather than only declared.
        manifest, result = run_on(engine, [rgb_3band], tmp_path)
        assert manifest.index_availability["glcm_texture"] is True
        assert result.payload.data["glcm"]


class TestAvailabilityMatchesExecution:
    """The invariant, across every band configuration.

    Availability, the indices actually computed, and the substitutions claimed
    must describe the same run. Each of the three could drift from the others
    independently, and the RGB defect was exactly that drift.
    """

    WATER = ("ndwi", "mndwi")

    @pytest.fixture(params=["rgb_3band", "msi_4band", "msi_6band", "pan_1band"])
    def any_optical(self, request):
        return request.getfixturevalue(request.param)

    def test_every_computed_index_was_declared_available(
        self, engine, any_optical, tmp_path
    ):
        manifest, result = run_on(engine, [any_optical], tmp_path)
        avail = manifest.index_availability
        for name in result.payload.data["indices"]:
            if name == "builtup_proxy":
                continue  # a substitute, not a declared index
            assert avail.get(name) is True, f"{name} computed but not available"

    def test_every_available_optical_index_was_computed(
        self, engine, any_optical, tmp_path
    ):
        manifest, result = run_on(engine, [any_optical], tmp_path)
        computed = result.payload.data["indices"]
        for name in ("ndvi", "ndwi", "mndwi", "ndbi"):
            if manifest.index_availability.get(name):
                assert name in computed, f"{name} available but not computed"

    def test_substitutions_only_name_indices_that_ran(
        self, engine, any_optical, tmp_path
    ):
        _, result = run_on(engine, [any_optical], tmp_path)
        data = result.payload.data
        for sub in data["substitutions"]:
            if "NDWI used as the water index" in sub:
                assert "ndwi" in data["indices"], (
                    "claimed NDWI as the substitute without computing it"
                )
            if "built-up estimated from" in sub:
                assert "builtup_proxy" in data["indices"], (
                    "claimed a built-up proxy without computing it"
                )

    def test_a_water_substitution_implies_a_water_index_exists(
        self, engine, any_optical, tmp_path
    ):
        _, result = run_on(engine, [any_optical], tmp_path)
        data = result.payload.data
        if any("water index" in s for s in data["substitutions"]):
            assert any(w in data["indices"] for w in self.WATER)

    def test_no_water_index_means_no_water_substitution_and_a_warning(
        self, engine, any_optical, tmp_path
    ):
        manifest, result = run_on(engine, [any_optical], tmp_path)
        avail = manifest.index_availability
        if not any(avail.get(w) for w in self.WATER):
            assert not any(
                "water index" in s for s in result.payload.data["substitutions"]
            )
            assert any("no water index computed" in w for w in result.warnings)


class TestRgbTraceConsistency:
    """The same invariant one level up, in the trace a judge reads."""

    def test_trace_carries_no_false_substitution_conflict(self, rgb_3band):
        from satquery.controller.pipeline import Controller

        # This query routes through index_engine_v1; "Highlight the water
        # body" does not - it goes straight to grounding_v1, which would make
        # this test pass without exercising the engine at all.
        trace = Controller().run(
            [rgb_3band], "What land cover types are present in this image?"
        )
        assert "index_engine_v1" in [s.tool for s in trace.execution]
        conflicts = trace.verification.conflicts
        assert not any("NDWI used as the water index" in c for c in conflicts), (
            f"false substitution reached the verifier: {conflicts}"
        )

    def test_trace_availability_matches_the_executed_indices(self, rgb_3band):
        from satquery.controller.pipeline import Controller

        trace = Controller().run(
            [rgb_3band], "What land cover types are present in this image?"
        )
        assert "index_engine_v1" in [s.tool for s in trace.execution]
        avail = trace.ingest.index_availability
        assert avail["ndvi"] is False
        assert avail["ndwi"] is False
        assert avail["mndwi"] is False
        assert avail["ndbi"] is False
        for step in trace.execution:
            if step.tool == "index_engine_v1":
                assert step.outputs.get("indices", {}) == {}


class TestSAR:
    def test_sigma0_and_ratio_computed(self, engine, sar_dualpol, tmp_path):
        _, result = run_on(engine, [sar_dualpol], tmp_path)
        indices = result.payload.data["indices"]
        assert "sigma0_vv" in indices
        assert "vh_vv_ratio_db" in indices

    def test_cov_computed_for_sar(self, engine, sar_dualpol, tmp_path):
        _, result = run_on(engine, [sar_dualpol], tmp_path)
        assert "cov" in result.payload.data["indices"]

    def test_cross_pol_ratio_is_negative_for_typical_backscatter(
        self, engine, sar_dualpol, tmp_path
    ):
        """VH is weaker than VV for most surfaces, so the ratio is below 0 dB."""
        _, result = run_on(engine, [sar_dualpol], tmp_path)
        mean = result.payload.data["indices"]["vh_vv_ratio_db"]["stats"]["mean"]
        assert mean < 0


class TestCrossModal:
    def test_optical_and_sar_indices_both_present(
        self, engine, msi_4band, sar_dualpol, tmp_path
    ):
        _, result = run_on(engine, [msi_4band, sar_dualpol], tmp_path)
        indices = result.payload.data["indices"]
        assert "ndvi" in indices          # from optical
        assert "sigma0_vv" in indices     # from SAR

    def test_sar_strengthens_the_swir_free_builtup_proxy(
        self, engine, msi_4band, sar_dualpol, tmp_path
    ):
        """With SAR available the proxy must say it used the sigma0 term."""
        _, result = run_on(engine, [msi_4band, sar_dualpol], tmp_path)
        subs = " ".join(result.payload.data["substitutions"])
        assert "sar_sigma0" in subs


class TestArtifacts:
    def test_writes_readable_cogs(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path)
        assert result.artifacts
        for art in result.artifacts:
            assert art.path.exists(), art.key
            with rasterio.open(art.path) as src:
                assert src.count == 1
                assert src.dtypes[0] == "float32"

    def test_cogs_are_georeferenced_to_the_source(self, engine, msi_6band, tmp_path):
        manifest, result = run_on(engine, [msi_6band], tmp_path)
        ndvi_art = next(a for a in result.artifacts if a.key == "ndvi")
        with rasterio.open(ndvi_art.path) as out, rasterio.open(msi_6band) as src:
            assert out.crs == src.crs
            assert out.transform == src.transform
            assert (out.height, out.width) == (src.height, src.width)

    def test_cog_driver_used(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path)
        with rasterio.open(result.artifacts[0].path) as src:
            assert src.driver in {"COG", "GTiff"}  # COG reads back as GTiff

    def test_artifact_values_match_recomputed_index(self, engine, msi_6band, tmp_path):
        """The written raster must actually contain the index, not a placeholder."""
        from satquery.ingest.reader import read_canonical_band
        from satquery.verify import ndvi as ndvi_fn

        manifest, result = run_on(engine, [msi_6band], tmp_path)
        art = next(a for a in result.artifacts if a.key == "ndvi")
        with rasterio.open(art.path) as src:
            written = src.read(1)
        img = manifest.images[0]
        expected = ndvi_fn(
            read_canonical_band(img, "RED"), read_canonical_band(img, "NIR")
        )
        assert np.allclose(written, expected.astype("float32"), equal_nan=True)

    def test_write_artifacts_can_be_disabled(self, engine, msi_6band, tmp_path):
        _, result = run_on(engine, [msi_6band], tmp_path, write_artifacts=False)
        assert result.artifacts == []
        assert result.payload.data["indices"]  # stats still computed


class TestDegradation:
    def test_pan_only_input_computes_no_spectral_indices(
        self, engine, pan_1band, tmp_path
    ):
        _, result = run_on(engine, [pan_1band], tmp_path)
        indices = result.payload.data["indices"]
        assert "ndvi" not in indices
        assert any("no indices computable" in w for w in result.warnings)

    def test_glcm_still_computed_for_pan(self, engine, pan_1band, tmp_path):
        """Texture needs only one band, so it must survive where indices cannot."""
        _, result = run_on(engine, [pan_1band], tmp_path)
        assert result.payload.data["glcm"]

    def test_does_not_raise_on_degenerate_input(self, engine, tiny_raster, tmp_path):
        manifest = ingest([tiny_raster])
        result = engine.run(manifest, {"output_dir": str(tmp_path)})
        assert result.tool == "index_engine"

    def test_batch_matches_individual_runs(self, engine, msi_6band, tmp_path):
        manifest = ingest([msi_6band])
        params = {"output_dir": str(tmp_path / "a"), "write_artifacts": False}
        batch = engine.run_batch([manifest, manifest], params)
        assert len(batch) == 2
        assert (
            batch[0].payload.data["indices"]["ndvi"]["stats"]
            == batch[1].payload.data["indices"]["ndvi"]["stats"]
        )
