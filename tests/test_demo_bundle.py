"""The curated demo bundle (plan task 4.1).

The PS's second deliverable is *"Codes and models including test and
demonstration"*, so the demonstration's inputs are a deliverable rather than
a convenience. These tests keep the bundle buildable and keep its beats
honest: a bundle that is generated once and never re-checked is one that
breaks between the last rehearsal and the venue.

The heavy `--verify` pass runs every input through the real controller and is
not repeated here - it needs the real Bhoonidhi products and takes minutes.
What is tested is that the bundle *builds*, that it covers the configurations
the demo script needs, and that the two rejection beats are what the script
claims they are.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.make_demo_bundle import (
    build_benchmark_png,
    build_bundle,
    build_clouded_optical,
    build_incompatible_pair,
)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    return build_bundle(tmp_path_factory.mktemp("demo"))


class TestBundleShape:
    def test_it_builds_and_covers_every_scripted_beat(self, bundle):
        keys = {i.key for i in bundle}
        assert keys == {
            "incompatible_pair", "png_operational", "single_optical",
            "single_sar", "crossmodal_pair", "bitemporal_pair",
            "change_what_and_where", "clouded_optical", "large_scene",
        }

    def test_every_input_exists_on_disk(self, bundle):
        for item in bundle:
            for image in item.images:
                assert image.exists(), f"{item.key}: {image} missing"

    def test_single_and_pair_configurations_are_both_present(self, bundle):
        counts = {len(i.images) for i in bundle}
        assert counts == {1, 2}

    def test_a_synthetic_substitution_is_recorded_not_hidden(self, bundle):
        """When a real product is absent the bundle degrades - and says so."""
        for item in bundle:
            if not item.real and item.key in {"single_optical", "single_sar", "large_scene"}:
                assert item.notes, f"{item.key} substituted silently"


class TestRejectionBeats:
    def test_the_incompatible_pair_is_optical_plus_sar(self, tmp_path):
        """Two optical scenes would be read as bi-temporal, and the
        cross-modal task excluded on configuration - which tests the wrong
        gate entirely."""
        import rasterio

        optical, sar = build_incompatible_pair(tmp_path)
        with rasterio.open(optical) as src:
            assert src.count == 6
        with rasterio.open(sar) as src:
            assert set(src.descriptions) == {"VV", "VH"}

    def test_the_incompatible_pair_footprints_are_disjoint(self, tmp_path):
        import rasterio
        from rasterio.coords import disjoint_bounds

        optical, sar = build_incompatible_pair(tmp_path)
        with rasterio.open(optical) as a, rasterio.open(sar) as b:
            assert disjoint_bounds(a.bounds, b.bounds), (
                "the rejection beat needs footprints that cannot overlap"
            )

    def test_the_benchmark_png_carries_no_crs(self, tmp_path):
        """This is why operational mode refuses it, and why BENCHMARK mode
        has to relax `crs_present` to a WARN for CDVQA to run at all."""
        import rasterio

        png = build_benchmark_png(tmp_path / "t.png")
        with rasterio.open(png) as src:
            assert src.crs is None


class TestCloudedScene:
    def test_the_cloud_fraction_is_controlled_and_recorded(self, tmp_path):
        """The abstention beat states a cloud percentage out loud, so the
        scene must carry a known one rather than a hoped-for one."""
        import rasterio

        path = build_clouded_optical(tmp_path / "cloudy.tif", cloud_fraction=0.63)
        with rasterio.open(path) as src:
            recorded = float(src.tags()["cloud_fraction"])
            assert recorded >= 0.63
            band = src.read(1)
        # Cloud is written as the brightest, spectrally flat value.
        assert (band >= band.max() * 0.99).mean() >= 0.6
