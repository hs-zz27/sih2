"""Regression tests against real Bhoonidhi products (plan task 1.1).

Task 1.1's acceptance criterion is an `InputManifest` produced correctly for
real Cartosat and RISAT files, which synthetic fixtures cannot demonstrate.
Real data exposed the gap these tests now guard: vendor products ship one file
per band, and reading a single file yields a 1-band image that modality
inference calls PAN.

The products are gitignored (large, and held out as the cross-sensor
generalisation set per docs/03 section 4.3), so every test skips when they are
absent. Values asserted here were read from the vendor metadata on 2026-08-29
and are recorded in docs/verification.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from satquery.ingest import ingest
from satquery.ingest.product import discover

ROOT = Path("data/bhoonidhi")

CARTOSAT = ROOT / "cartosat2s_mx_5132611"
EOS04_FRS1 = ROOT / "eos04_frs1_226981731"
EOS04_MRS = ROOT / "eos04_mrs_p1_226981721"
EOS04_SLC = ROOT / "eos04_mrs_p2_247111021"

requires_cartosat = pytest.mark.skipif(
    not CARTOSAT.exists(), reason="Cartosat sample not downloaded"
)
requires_frs1 = pytest.mark.skipif(
    not EOS04_FRS1.exists(), reason="EOS-04 FRS-1 sample not downloaded"
)
requires_mrs = pytest.mark.skipif(
    not EOS04_MRS.exists(), reason="EOS-04 MRS sample not downloaded"
)
requires_slc = pytest.mark.skipif(
    not EOS04_SLC.exists(), reason="EOS-04 SLC sample not downloaded"
)


@requires_cartosat
class TestCartosatMX:
    """Cartosat-2E MX: BAND1..BAND4.tif, verification item 6."""

    def test_four_band_files_assembled_into_one_image(self):
        layout = discover(CARTOSAT)
        assert layout.kind == "cartosat_mx"
        assert len(layout.band_files) == 4

    def test_bands_named_vnir_no_swir(self):
        """Item 6: 4-band VNIR, no SWIR."""
        meta = ingest([CARTOSAT]).images[0]
        assert meta.bands == ["BLUE", "GREEN", "RED", "NIR"]
        assert "SWIR1" not in meta.bands
        assert "SWIR2" not in meta.bands

    def test_modality_is_msi_not_pan(self):
        """The bug real data exposed: one file per band read as a PAN image."""
        assert ingest([CARTOSAT]).images[0].modality == "MSI"

    def test_swir_indices_unavailable_vnir_available(self):
        avail = ingest([CARTOSAT]).index_availability
        assert avail["ndvi"] is True
        assert avail["ndwi"] is True
        assert avail["mndwi"] is False   # needs SWIR1
        assert avail["ndbi"] is False    # needs SWIR1

    def test_gsd_matches_vendor_metadata(self):
        """BAND_META says PixelSpacing 1.6 m."""
        assert ingest([CARTOSAT]).images[0].gsd_m == pytest.approx(1.6, abs=0.01)

    def test_effective_bits_matches_vendor_bits_per_pixel(self):
        """BAND_META says BitsPerPixel=11 inside a uint16 container."""
        meta = ingest([CARTOSAT]).images[0]
        assert meta.dtype == "uint16"
        assert meta.effective_bits == 11

    def test_sensor_identified(self):
        assert ingest([CARTOSAT]).images[0].sensor_guess == "CARTOSAT-2E"

    def test_large_scene_flagged_for_tiling(self):
        """7687x7640 exceeds the tile-pyramid trigger."""
        tiling = ingest([CARTOSAT]).tiling
        assert tiling.level1_tiles is not None
        assert tiling.level1_tiles > 1

    def test_no_blocking_failures(self):
        assert ingest([CARTOSAT]).blocking_failures == []


@requires_mrs
class TestEOS04MRS:
    """EOS-04 MRS: verification item 5."""

    def test_detected_as_sar(self):
        assert ingest([EOS04_MRS]).images[0].modality == "SAR"

    def test_dual_polarisation(self):
        meta = ingest([EOS04_MRS]).images[0]
        assert set(meta.polarisations) == {"HH", "HV"}

    def test_c_band_frequency_from_product_xml(self):
        """Item 5: radarCenterFrequency 5.40e09 Hz = C-band."""
        layout = discover(EOS04_MRS)
        assert layout.metadata["radar_band"] == "C"
        assert layout.metadata["radar_frequency_ghz"] == pytest.approx(5.40, abs=0.01)

    def test_frequency_matches_sentinel1_closely(self):
        """The de-risking claim: within ~0.1% of Sentinel-1's 5.405 GHz."""
        ghz = discover(EOS04_MRS).metadata["radar_frequency_ghz"]
        assert abs(ghz - 5.405) / 5.405 < 0.005

    def test_look_count_populated_from_metadata(self):
        """RangeLooks=2 x AzimuthLooks=1."""
        assert ingest([EOS04_MRS]).images[0].look_count_est == pytest.approx(2.0)

    def test_sar_indices_available(self):
        avail = ingest([EOS04_MRS]).index_availability
        assert avail["sigma0"] is True
        assert avail["cov"] is True

    def test_gsd_matches_vendor_pixel_spacing(self):
        """BAND_META says OutputPixelSpacing 18.0 m."""
        assert ingest([EOS04_MRS]).images[0].gsd_m == pytest.approx(18.0, abs=0.1)


@requires_frs1
class TestEOS04FRS1:
    def test_quad_polarisation_assembled(self):
        meta = ingest([EOS04_FRS1]).images[0]
        assert set(meta.polarisations) == {"HH", "HV", "VH", "VV"}

    def test_cross_pol_ratio_available_with_quad_pol(self):
        assert ingest([EOS04_FRS1]).index_availability["vh_vv_ratio"] is True

    def test_finer_gsd_than_mrs(self):
        """FRS-1 is the fine-resolution mode; it must be sharper than MRS."""
        assert ingest([EOS04_FRS1]).images[0].gsd_m < 10.0


@requires_slc
class TestEOS04SLCRejection:
    """An SLC slant-range product must be refused with a real reason."""

    def test_detected_as_slc_with_beams(self):
        layout = discover(EOS04_SLC)
        assert layout.kind == "eos04_sar_slc"
        assert layout.metadata["n_beams"] == 8
        assert layout.metadata["requires_geocoding"] is True

    def test_still_identified_as_sar(self):
        assert ingest([EOS04_SLC]).images[0].modality == "SAR"

    def test_blocks_with_geocoding_reason_not_just_missing_crs(self):
        """'no CRS' is true but unhelpful; the real reason must be named."""
        manifest = ingest([EOS04_SLC])
        assert "geocoding_required" in manifest.blocking_failures
        message = next(
            c.message for c in manifest.checks if c.name == "geocoding_required"
        )
        assert "slant-range" in message.lower()
        assert "geocod" in message.lower()

    def test_pipeline_abstains_rather_than_answering(self):
        from satquery.controller.pipeline import Controller

        trace = Controller().run([EOS04_SLC], "What land cover is here?")
        assert trace.abstained is True
        assert trace.abstain_reason


@requires_cartosat
class TestRealEndToEnd:
    """The Phase 1 exit path, on real target-sensor imagery."""

    def test_swir_free_path_exercised_on_real_cartosat(self):
        from satquery.controller.pipeline import Controller

        trace = Controller().run(
            [CARTOSAT], "What is the land cover in this scene?",
            tool_params={},
        )
        assert trace.abstained is False
        assert trace.routing.selected_task == "SINGLE_LANDCOVER"
        # NDBI is impossible without SWIR, so the documented proxy must be used
        # and the substitution must be visible in the trace.
        assert trace.verification.built_up_path == "swir_free_proxy"
        assert any("NDBI unavailable" in c for c in trace.verification.conflicts)
        assert trace.answer
