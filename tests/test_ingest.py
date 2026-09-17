"""Tests for Layer 0 ingest (plan task 1.1)."""

from __future__ import annotations

import pytest

from satquery.contracts.input_manifest import IngestMode, InputManifest
from satquery.ingest import (
    harmonise_bands,
    index_availability,
    infer_config,
    infer_modality,
    ingest,
    read_image,
)


class TestModalityInference:
    def test_msi_from_band_names(self, msi_6band):
        meta = read_image(msi_6band)
        assert meta.modality == "MSI"
        assert "optical_band_names" in " ".join(meta.modality_evidence["signals"])

    def test_four_band_vnir_is_msi(self, msi_4band):
        meta = read_image(msi_4band)
        assert meta.modality == "MSI"
        assert meta.bands == ["BLUE", "GREEN", "RED", "NIR"]

    def test_sar_from_polarisation_tags(self, sar_dualpol):
        meta = read_image(sar_dualpol)
        assert meta.modality == "SAR"
        assert set(meta.polarisations) == {"VV", "VH"}

    def test_pan_single_band(self, pan_1band):
        meta = read_image(pan_1band)
        assert meta.modality == "PAN"

    def test_single_float_backscatter_detected_without_tags(self):
        """A bare single float band with SAR statistics is called SAR."""
        import numpy as np

        rng = np.random.default_rng(7)
        speckle = rng.gamma(shape=1.0, scale=1.0, size=(64, 64))
        modality, ev = infer_modality(
            band_count=1, dtype="float32", tags={}, band_descriptions=[None],
            sample=speckle,
        )
        assert modality == "SAR"
        assert "statistical_backscatter_match" in ev["signals"]

    def test_symmetric_single_band_is_pan_not_sar(self):
        import numpy as np

        rng = np.random.default_rng(8)
        gaussian = rng.normal(500, 50, size=(64, 64))
        modality, _ = infer_modality(
            band_count=1, dtype="float32", tags={}, band_descriptions=[None],
            sample=gaussian,
        )
        assert modality == "PAN"


class TestBandHarmonisation:
    def test_named_bands_map_to_canonical(self):
        names, presence = harmonise_bands(
            ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"], "MSI"
        )
        assert names == ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
        assert presence == [True] * 6

    def test_sentinel_style_band_codes(self):
        names, _ = harmonise_bands(["B02", "B03", "B04", "B08"], "MSI")
        assert names == ["BLUE", "GREEN", "RED", "NIR"]

    def test_swir1_not_swallowed_by_generic_swir_rule(self):
        names, _ = harmonise_bands(["SWIR2", "SWIR1"], "MSI")
        assert names == ["SWIR2", "SWIR1"]

    def test_unnamed_four_band_assumes_conventional_order(self):
        names, presence = harmonise_bands([None, None, None, None], "MSI")
        assert names == ["BLUE", "GREEN", "RED", "NIR"]
        assert presence == [True, True, True, True, False, False]


class TestIndexAvailability:
    def test_four_band_vnir_has_no_swir_indices(self):
        avail = index_availability(
            [True, True, True, True, False, False], ["MSI"], []
        )
        assert avail["ndvi"] is True
        assert avail["ndwi"] is True
        assert avail["mndwi"] is False  # needs SWIR1
        assert avail["ndbi"] is False   # needs SWIR1

    def test_six_band_enables_swir_indices(self):
        avail = index_availability([True] * 6, ["MSI"], [])
        assert avail["mndwi"] is True
        assert avail["ndbi"] is True

    def test_dual_pol_sar_enables_ratio(self):
        avail = index_availability([False] * 6, ["SAR"], ["VV", "VH"])
        assert avail["sigma0"] is True
        assert avail["vh_vv_ratio"] is True

    def test_single_pol_sar_has_no_ratio(self):
        avail = index_availability([False] * 6, ["SAR"], ["VV"])
        assert avail["sigma0"] is True
        assert avail["vh_vv_ratio"] is False


class TestConfigInference:
    def test_single_image(self, msi_6band):
        manifest = ingest([msi_6band])
        assert manifest.config == "SINGLE"
        assert manifest.images[0].role == "single"

    def test_optical_plus_sar_is_crossmodal(self, msi_6band, sar_dualpol):
        manifest = ingest([msi_6band, sar_dualpol])
        assert manifest.config == "CROSSMODAL_PAIR"
        assert {i.role for i in manifest.images} == {"optical", "sar"}

    def test_two_optical_is_bitemporal(self, msi_6band, msi_6band_t2):
        manifest = ingest([msi_6band, msi_6band_t2])
        assert manifest.config == "BITEMPORAL_PAIR"
        assert [i.role for i in manifest.images] == ["t1", "t2"]

    def test_bitemporal_ordered_by_acquisition_date(self, msi_6band, msi_6band_t2):
        """Inputs given latest-first must still come out t1-then-t2."""
        manifest = ingest([msi_6band_t2, msi_6band])
        t1, t2 = manifest.images
        assert t1.acquisition_dt < t2.acquisition_dt

    def test_sar_listed_first_still_puts_optical_first(self, sar_dualpol, msi_6band):
        manifest = ingest([sar_dualpol, msi_6band])
        assert manifest.images[0].role == "optical"

    def test_three_images_rejected(self, msi_6band):
        with pytest.raises(ValueError, match="at most two"):
            ingest([msi_6band, msi_6band, msi_6band])

    def test_empty_input_becomes_a_blocking_failure_not_an_exception(self):
        """Changed in task 3.13, deliberately.

        This used to raise `ValueError("at least one image path is required")`,
        which reached the caller as a traceback. An empty or unreadable input
        is a user-facing condition, so it now returns a manifest whose single
        check has FAILed - the same path every other bad input takes, which
        the router and the abstention policy already handle.

        `>2 images` still raises: the API rejects that with a 400 before
        ingest is reached, so hitting it means a caller ignored the contract.
        """
        manifest = ingest([])
        assert manifest.blocking_failures == ["inputs_present"]
        assert manifest.images == []
        assert "no input images" in manifest.checks[0].message


class TestChecks:
    def test_clean_image_has_no_blocking_failures(self, msi_6band):
        manifest = ingest([msi_6band])
        assert manifest.blocking_failures == []

    def test_missing_crs_blocks(self, no_crs_raster):
        manifest = ingest([no_crs_raster])
        assert "crs_present" in manifest.blocking_failures

    def test_tiny_image_blocks(self, tiny_raster):
        manifest = ingest([tiny_raster])
        assert "min_dimension" in manifest.blocking_failures

    def test_crossmodal_pairing_check_runs(self, msi_6band, sar_dualpol):
        manifest = ingest([msi_6band, sar_dualpol])
        names = {c.name for c in manifest.checks}
        assert "crossmodal_pairing" in names
        assert all(c.status != "FAIL" for c in manifest.checks)

    def test_gsd_ratio_warns_on_large_mismatch(self, msi_4band, msi_6band):
        """1.6m vs 10m is a 6.25x ratio - above the warn threshold."""
        manifest = ingest([msi_4band, msi_6band])
        gsd = next(c for c in manifest.checks if c.name == "gsd_ratio")
        assert gsd.status == "WARN"
        assert gsd.value == pytest.approx(6.25, rel=0.01)


class TestCoregistration:
    def test_same_modality_uses_phase_correlation(self, msi_6band, msi_6band_t2):
        manifest = ingest([msi_6band, msi_6band_t2])
        assert manifest.coreg is not None
        assert manifest.coreg.method == "phase_correlation"

    def test_optical_sar_uses_gradient_phase(self, msi_6band, sar_dualpol):
        manifest = ingest([msi_6band, sar_dualpol])
        assert manifest.coreg is not None
        assert manifest.coreg.method == "gradient_phase_correlation"

    def test_no_coreg_for_single_image(self, msi_6band):
        assert ingest([msi_6band]).coreg is None

    def test_coreg_skipped_when_inputs_already_failing(self, tiny_raster, msi_6band):
        manifest = ingest([tiny_raster, msi_6band])
        assert manifest.blocking_failures
        assert manifest.coreg is None


class TestManifestShape:
    def test_manifest_validates_and_roundtrips(self, msi_6band, sar_dualpol):
        manifest = ingest([msi_6band, sar_dualpol])
        restored = InputManifest.model_validate_json(manifest.model_dump_json())
        assert restored.run_id == manifest.run_id
        assert restored.config == manifest.config

    def test_run_id_is_unique_per_call(self, msi_6band):
        assert ingest([msi_6band]).run_id != ingest([msi_6band]).run_id

    def test_explicit_run_id_respected(self, msi_6band):
        assert ingest([msi_6band], run_id="fixed_id").run_id == "fixed_id"

    def test_effective_bits_below_container_width(self, msi_6band):
        """uint16 container holding ~12-bit data must report the real depth."""
        meta = ingest([msi_6band]).images[0]
        assert meta.dtype == "uint16"
        assert meta.effective_bits < 16

    def test_gsd_read_from_transform(self, msi_4band):
        assert ingest([msi_4band]).images[0].gsd_m == pytest.approx(1.6)

    def test_index_availability_merged_across_pair(self, msi_6band, sar_dualpol):
        avail = ingest([msi_6band, sar_dualpol]).index_availability
        assert avail["ndvi"] is True      # from the optical image
        assert avail["sigma0"] is True    # from the SAR image
        assert avail["vh_vv_ratio"] is True

    def test_small_scene_not_flagged_for_tiling(self, msi_6band):
        assert ingest([msi_6band]).tiling.applied is False
        assert ingest([msi_6band]).tiling.level1_tiles is None


class TestBenchmarkFormats:
    """PNG/JPEG for the prescribed benchmarks (problem statement, input scope).

    The PS admits PNG and JPEG "only for the prescribed public benchmark
    datasets". Those are ungeoreferenced by construction - RSVQA and VRSBench
    ship plain rasters - and the CRS check failed them in every mode, so no
    prescribed benchmark image could enter the pipeline at all.
    """

    def _png(self, tmp_path):
        import numpy as np
        import rasterio

        path = tmp_path / "benchmark.png"
        array = (np.random.default_rng(3).random((3, 128, 128)) * 255).astype("uint8")
        with rasterio.open(
            path, "w", driver="PNG", height=128, width=128, count=3, dtype="uint8"
        ) as dst:
            dst.write(array)
        return path

    def test_benchmark_mode_accepts_an_ungeoreferenced_png(self, tmp_path):
        manifest = ingest([self._png(tmp_path)], mode=IngestMode.BENCHMARK)
        assert manifest.blocking_failures == []

    def test_the_limitation_is_recorded_not_hidden(self, tmp_path):
        """Accepting it must not imply the outputs are placeable."""
        manifest = ingest([self._png(tmp_path)], mode=IngestMode.BENCHMARK)
        crs = next(c for c in manifest.checks if c.name == "crs_present")
        assert crs.status == "WARN"
        assert "cannot be georeferenced" in crs.message

    def test_operational_mode_accepts_a_png_and_discloses_the_limits(
        self, tmp_path
    ):
        """PS-26167 item 10: an ordinary PNG upload must be answerable.

        This used to assert the opposite - a PNG in operational mode was a
        blocking failure, so every normal user upload that was not a GeoTIFF
        refused the query outright. The requirement is that PNG/JPEG work for
        normal image interaction, so the check now WARNs and names exactly
        what is unavailable rather than refusing.
        """
        manifest = ingest([self._png(tmp_path)], mode=IngestMode.OPERATIONAL)
        assert manifest.blocking_failures == []
        crs = next(c for c in manifest.checks if c.name == "crs_present")
        assert crs.status == "WARN"
        assert "no geospatial metadata" in crs.message

    def test_an_ungeoreferenced_geotiff_is_still_refused(self, no_crs_raster):
        """The relaxation is about the container, not about CRS in general.

        A GeoTIFF without a CRS is a defective product - the format carries
        georeferencing and this one does not - and must keep failing.
        """
        manifest = ingest([no_crs_raster], mode=IngestMode.OPERATIONAL)
        assert "crs_present" in manifest.blocking_failures

    def test_a_benchmark_png_answers_end_to_end(self, tmp_path):
        from satquery.controller.pipeline import Controller

        trace = Controller().run(
            [self._png(tmp_path)],
            "How many buildings are visible?",
            mode=IngestMode.BENCHMARK,
            benchmark="rsvqa_lr",
        )
        assert trace.abstained is False
        assert trace.routing.selected_task == "SINGLE_VQA"
        assert trace.answer


class TestFootprintOverlap:
    """The gate that was declared for three phases and enforced by nothing.

    `configs/capability_matrix.yaml` has carried `min_overlap_pct` since Phase
    0. `RequiresSchema` declared only `config` with `extra="allow"`, so the
    value was parsed and read by nothing, and `Router.legal_tasks` gated on
    configuration alone. The measured consequence (2026-08-30): an optical and
    a SAR scene written 60 km apart routed to `XMODAL_JOINT_EXTRACT`, answered,
    and raised no failing check - the system fused two different places into
    one confident answer.
    """

    def _offset(self, tmp_path, source, easting, northing, names):
        import rasterio
        from rasterio.transform import from_origin

        with rasterio.open(source) as src:
            data, profile = src.read(), src.profile
        profile.update(transform=from_origin(easting, northing, 10.0, 10.0))
        out = tmp_path / f"offset_{easting:.0f}.tif"
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data)
            dst.descriptions = names
        return out

    def test_identical_footprints_are_full_overlap(self, msi_6band, sar_dualpol):
        from satquery.ingest.checks import footprint_overlap_pct
        from satquery.ingest import ingest

        manifest = ingest([msi_6band, sar_dualpol])
        assert footprint_overlap_pct(*manifest.images) == pytest.approx(100.0)

    def test_disjoint_footprints_fail(self, tmp_path, msi_6band, sar_dualpol):
        from satquery.ingest import ingest

        far = self._offset(tmp_path, sar_dualpol, 560000.0, 1940000.0, ("VV", "VH"))
        manifest = ingest([msi_6band, far])
        overlap = next(c for c in manifest.checks if c.name == "footprint_overlap")
        assert overlap.status == "FAIL"
        assert overlap.value == 0.0

    def test_a_disjoint_pair_cannot_reach_the_crossmodal_task(
        self, tmp_path, msi_6band, sar_dualpol
    ):
        """The regression that matters: not the check, but the gate on it."""
        from satquery.controller.matrix_loader import load_matrix
        from satquery.controller.router import Router
        from satquery.ingest import ingest

        far = self._offset(tmp_path, sar_dualpol, 560000.0, 1940000.0, ("VV", "VH"))
        manifest = ingest([msi_6band, far])
        router = Router(load_matrix("configs/capability_matrix.yaml"))

        assert "XMODAL_JOINT_EXTRACT" not in router.legal_tasks(manifest)
        unmet = router.unmet_requirements("XMODAL_JOINT_EXTRACT", manifest)
        assert unmet and "overlap" in unmet[0]

    def test_a_valid_pair_still_reaches_the_crossmodal_task(
        self, msi_6band, sar_dualpol
    ):
        """The gate must not cost the legitimate path.

        This is why `max_coreg_shift_px` is deliberately not enforced: on this
        very pair - identical footprints, same CRS, same GSD - the cross-modal
        phase correlation reports ~38 px against the matrix's 2.0 px limit.
        """
        from satquery.controller.matrix_loader import load_matrix
        from satquery.controller.router import Router
        from satquery.ingest import ingest

        manifest = ingest([msi_6band, sar_dualpol])
        router = Router(load_matrix("configs/capability_matrix.yaml"))
        assert "XMODAL_JOINT_EXTRACT" in router.legal_tasks(manifest)

    def test_clarify_or_abstain_is_never_gated_away(
        self, tmp_path, msi_6band, sar_dualpol
    ):
        """Gating the destination-of-last-resort would leave nothing to route
        to, which turns a graceful refusal into a crash."""
        from satquery.controller.matrix_loader import load_matrix
        from satquery.controller.router import Router
        from satquery.ingest import ingest

        far = self._offset(tmp_path, sar_dualpol, 560000.0, 1940000.0, ("VV", "VH"))
        manifest = ingest([msi_6band, far])
        router = Router(load_matrix("configs/capability_matrix.yaml"))
        assert "CLARIFY_OR_ABSTAIN" in router.legal_tasks(manifest)

    def test_an_ungeoreferenced_pair_warns_rather_than_failing(self, tmp_path):
        """"Overlap unknown" and "no overlap" are different answers, and only
        the second is a defect in the pair. Benchmark inputs are
        ungeoreferenced by construction."""
        from evaluation.scenes import build_no_crs_raster
        from satquery.ingest import ingest
        from satquery.ingest.checks import (
            check_footprint_overlap,
            footprint_overlap_pct,
        )

        image = ingest([build_no_crs_raster(tmp_path / "a.tif")]).images[0]
        assert footprint_overlap_pct(image, image) is None
        assert check_footprint_overlap(image, image).status == "WARN"


class TestSceneFootprint:
    """`lonlat_bounds` and the centre derived from it.

    The answer path had no idea where a scene was: ImageMeta carried `crs`
    and `gsd_m` and nothing else, so "describe this image" could never say
    where the image is. The footprint is transformed once at ingest, and it
    is `None` in every case where the file does not actually say.
    """

    def test_a_projected_geotiff_gets_a_footprint(self, msi_6band):
        from satquery.ingest import ingest

        image = ingest([msi_6band]).images[0]
        west, south, east, north = image.lonlat_bounds
        assert west < east and south < north
        latitude, longitude = image.centroid_latlon
        # Centre lies inside its own envelope, and is (lat, lon) - the order
        # a reader says them in, which is the opposite of the bounds order.
        assert south <= latitude <= north
        assert west <= longitude <= east

    def test_the_ground_extent_is_pixels_times_gsd(self, msi_6band):
        from satquery.ingest import ingest

        image = ingest([msi_6band]).images[0]
        assert image.crs_is_projected
        assert image.ground_extent_m == (
            image.width * image.gsd_m,
            image.height * image.gsd_m,
        )

    def test_an_ungeoreferenced_container_has_no_footprint(self, tmp_path):
        """A PNG cannot carry a CRS, and GDAL hands back the identity
        transform. Bounds derived from that are pixel indices dressed as
        coordinates - plausible-looking and wrong."""
        import numpy as np
        import rasterio
        from satquery.ingest import ingest

        path = tmp_path / "scene.png"
        with rasterio.open(
            path, "w", driver="PNG", width=64, height=64, count=3, dtype="uint8"
        ) as dst:
            dst.write(np.full((3, 64, 64), 128, dtype="uint8"))

        image = ingest([path]).images[0]
        assert image.georeferenced is False
        assert image.lonlat_bounds is None
        assert image.centroid_latlon is None
        assert image.ground_extent_m is None

    def test_a_geotiff_without_a_crs_has_no_footprint(self, no_crs_raster):
        from satquery.ingest import ingest

        image = ingest([no_crs_raster]).images[0]
        assert image.lonlat_bounds is None
        assert image.centroid_latlon is None

    def test_ground_extent_is_withheld_for_a_geographic_crs(self, msi_6band):
        """`gsd_m` converts degrees at a flat 111320 m, which is wrong by
        the cosine of the latitude in x. That is fine for ordering-of-
        magnitude routing and not fine as a distance shown to a reader."""
        from satquery.ingest import ingest

        image = ingest([msi_6band]).images[0]
        geographic = image.model_copy(update={"crs_is_projected": False})
        assert geographic.ground_extent_m is None


# --- Cloud cover (2026-09-12) -------------------------------------------------


class TestCloudCover:
    """`cloud_pct` was in the contract from Phase 1 and never populated or read.

    The demo's abstention beat - a 63% clouded scene - passed because its
    expectation could not fail, and with the learned tools on it was captioned
    at 0.88 confidence. This is the coarse estimator that makes the check real.
    """

    @staticmethod
    def _scene(cloud_fraction: float, bands: int = 4, seed: int = 0):
        import numpy as np

        rng = np.random.default_rng(seed)
        base = rng.uniform(0.05, 0.5, (256, 256))
        # Independent per-band variation, as real ground has. Bands that are
        # exact multiples of one base make EVERY pixel spectrally flat, which
        # is the one property the estimator uses to tell cloud from ground -
        # a first version of this scene was built that way and reported 17%
        # cloud on a scene with none.
        arr = np.stack([base * (0.7 + 0.1 * i) * rng.uniform(0.6, 1.4, base.shape)
                        for i in range(bands)])
        mask = np.zeros((256, 256), dtype=bool)
        while mask.mean() < cloud_fraction:
            cy, cx = rng.integers(0, 256, size=2)
            r = int(rng.integers(30, 70))
            yy, xx = np.ogrid[:256, :256]
            mask |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        if cloud_fraction > 0:
            arr[:, mask] = float(arr.max()) * 1.6      # flat and bright everywhere
        return arr, float(mask.mean())

    def test_recovers_a_known_cloud_fraction(self):
        from satquery.ingest.reader import estimate_cloud_pct

        arr, truth = self._scene(0.6)
        est = estimate_cloud_pct(arr)
        assert est is not None
        assert abs(est / 100.0 - truth) < 0.12, (est, truth)

    def test_a_clear_scene_is_near_zero(self):
        from satquery.ingest.reader import estimate_cloud_pct

        arr, _ = self._scene(0.0)
        assert estimate_cloud_pct(arr) < 5.0

    def test_bright_but_spectrally_varied_ground_is_not_cloud(self):
        """A bright roof with a real spectral signature must not count."""
        import numpy as np

        from satquery.ingest.reader import estimate_cloud_pct

        arr, _ = self._scene(0.0)
        top = float(arr.max())
        # Bright, but the bands DIFFER - the flatness test is what excludes it.
        arr[:, 50:150, 50:150] = np.array([1.0, 0.6, 0.35, 0.9])[:, None, None] * top * 1.6
        assert estimate_cloud_pct(arr) < 5.0

    def test_declines_rather_than_guessing_on_too_few_bands(self):
        import numpy as np

        from satquery.ingest.reader import estimate_cloud_pct

        assert estimate_cloud_pct(np.random.rand(2, 64, 64)) is None
        assert estimate_cloud_pct(np.random.rand(64, 64)) is None

    def test_check_thresholds(self):
        from satquery.ingest.checks import (
            CLOUD_FAIL_PCT, CLOUD_WARN_PCT, check_cloud_cover,
        )

        def meta(pct):
            from types import SimpleNamespace

            return SimpleNamespace(cloud_pct=pct, role="image_0")

        assert check_cloud_cover(meta(None)) is None, "SAR must not be judged"
        assert check_cloud_cover(meta(5.0)).status == "PASS"
        assert check_cloud_cover(meta(CLOUD_WARN_PCT)).status == "WARN"
        assert check_cloud_cover(meta(CLOUD_FAIL_PCT)).status == "FAIL"

    def test_a_failing_cloud_check_blocks_the_run(self):
        """FAIL must reach `blocking_failures`, which is what makes it abstain."""
        from types import SimpleNamespace

        from satquery.ingest.checks import blocking_failures, check_cloud_cover

        fail = check_cloud_cover(SimpleNamespace(cloud_pct=71.3, role="image_0"))
        assert "cloud_cover" in blocking_failures([fail])
