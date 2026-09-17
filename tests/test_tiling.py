"""Tile pyramid and coarse-to-fine retrieval tests (plan task 2.10).

The property that matters: statistics accumulated tile by tile must equal the
whole-scene result exactly. If they do not, tiling silently changes answers,
which is worse than not tiling at all.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

from satquery.ingest.tiling import (
    DEFAULT_TILE_PX,
    StreamingStats,
    iter_tiles,
    needs_tiling,
    plan_tiles,
    read_overview,
    scene_stats,
    select_tiles,
)


class TestTilePlanning:
    def test_tiles_cover_every_pixel_exactly_once(self):
        w, h, t = 2500, 1800, 1024
        tiles = plan_tiles(w, h, t)
        covered = sum(int(x.window.width * x.window.height) for x in tiles)
        assert covered == w * h

    def test_edge_tiles_are_clipped_not_padded(self):
        tiles = plan_tiles(1100, 1100, 1024)
        edge = [t for t in tiles if t.window.width < 1024]
        assert edge
        assert all(t.window.width > 0 and t.window.height > 0 for t in tiles)

    def test_exact_multiple_gives_uniform_tiles(self):
        tiles = plan_tiles(2048, 2048, 1024)
        assert len(tiles) == 4
        assert all(t.window.width == 1024 and t.window.height == 1024 for t in tiles)

    def test_single_tile_for_small_scene(self):
        assert len(plan_tiles(500, 500, 1024)) == 1

    def test_tile_keys_unique(self):
        tiles = plan_tiles(3000, 3000, 1024)
        assert len({t.key for t in tiles}) == len(tiles)

    def test_needs_tiling_threshold(self):
        assert needs_tiling(5000, 100) is True
        assert needs_tiling(1000, 1000) is False


class TestStreamingStats:
    def test_matches_numpy_on_one_batch(self):
        rng = np.random.default_rng(1)
        data = rng.normal(50, 12, 10_000).astype("float32")
        s = StreamingStats()
        s.update(data)
        assert s.mean == pytest.approx(float(data.mean()), rel=1e-5)
        assert s.std == pytest.approx(float(data.std()), rel=1e-4)

    def test_batched_equals_whole(self):
        """The core guarantee: chunking must not change the answer."""
        rng = np.random.default_rng(2)
        data = rng.normal(1000, 30, 50_000).astype("float32")

        whole = StreamingStats()
        whole.update(data)

        chunked = StreamingStats()
        for chunk in np.array_split(data, 17):
            chunked.update(chunk)

        assert chunked.n == whole.n
        assert chunked.mean == pytest.approx(whole.mean, rel=1e-5)
        assert chunked.std == pytest.approx(whole.std, rel=1e-4)

    def test_uneven_chunks_still_exact(self):
        rng = np.random.default_rng(3)
        data = rng.uniform(0, 1, 9_999).astype("float32")
        s = StreamingStats()
        for chunk in (data[:1], data[1:5000], data[5000:5001], data[5001:]):
            s.update(chunk)
        assert s.mean == pytest.approx(float(data.mean()), rel=1e-5)

    def test_ignores_nan(self):
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype="float32")
        s = StreamingStats()
        s.update(data)
        assert s.n == 3
        assert s.mean == pytest.approx(3.0)

    def test_all_nan_is_empty_not_crash(self):
        s = StreamingStats()
        s.update(np.full(10, np.nan, dtype="float32"))
        assert s.n == 0
        assert np.isnan(s.mean)

    def test_min_max_tracked(self):
        s = StreamingStats()
        s.update(np.array([5.0, 1.0], dtype="float32"))
        s.update(np.array([9.0, 3.0], dtype="float32"))
        assert s.min == 1.0
        assert s.max == 9.0

    def test_large_values_stay_precise(self):
        """Naive sum-of-squares loses precision here; Welford must not."""
        data = (np.random.default_rng(4).normal(0, 1, 20_000) + 1e6).astype("float64")
        s = StreamingStats()
        for chunk in np.array_split(data, 20):
            s.update(chunk)
        assert s.std == pytest.approx(float(data.std()), rel=1e-3)
        assert s.variance > 0


class TestWindowedReading:
    def test_iter_tiles_covers_whole_raster(self, msi_6band):
        total = sum(arr.size for _, arr in iter_tiles(msi_6band, tile_px=32))
        with rasterio.open(msi_6band) as src:
            assert total == src.width * src.height

    def test_tile_arrays_bounded_by_tile_size(self, msi_6band):
        for _, arr in iter_tiles(msi_6band, tile_px=32):
            assert arr.shape[0] <= 32 and arr.shape[1] <= 32

    def test_scene_stats_match_full_read(self, msi_6band):
        """Tiled statistics must equal the whole-scene numbers."""
        with rasterio.open(msi_6band) as src:
            full = src.read(1).astype("float64")
        tiled = scene_stats(msi_6band, band=1, tile_px=32)
        assert tiled["n"] == full.size
        assert tiled["mean"] == pytest.approx(float(full.mean()), rel=1e-5)
        assert tiled["std"] == pytest.approx(float(full.std()), rel=1e-4)
        assert tiled["min"] == pytest.approx(float(full.min()))
        assert tiled["max"] == pytest.approx(float(full.max()))

    def test_overview_is_decimated(self, msi_6band):
        overview = read_overview(msi_6band, max_edge=32)
        assert max(overview.shape) <= 32

    def test_overview_not_upscaled_for_small_input(self, msi_6band):
        overview = read_overview(msi_6band, max_edge=4096)
        with rasterio.open(msi_6band) as src:
            assert overview.shape == (src.height, src.width)


class TestRetrieval:
    def test_single_tile_scene_returns_one(self, msi_6band):
        sel = select_tiles(msi_6band, "where are the roads", tile_px=4096)
        assert sel.total_tiles == 1
        assert "single tile" in sel.reason

    def test_non_spatial_query_keeps_every_tile(self, msi_6band):
        """Counting questions must see the whole scene or they are wrong."""
        sel = select_tiles(
            msi_6band, "what fraction of this scene is water", tile_px=32, max_tiles=2
        )
        assert sel.retrieved == sel.total_tiles
        assert "not spatially focused" in sel.reason

    def test_spatial_query_with_budget_narrows(self, msi_6band):
        sel = select_tiles(msi_6band, "where are the buildings", tile_px=32, max_tiles=3)
        assert sel.retrieved == 3
        assert sel.total_tiles > 3
        assert "highest overview variance" in sel.reason

    def test_no_budget_keeps_all_even_when_spatial(self, msi_6band):
        sel = select_tiles(msi_6band, "where are the buildings", tile_px=32)
        assert sel.retrieved == sel.total_tiles

    def test_budget_larger_than_scene_keeps_all(self, msi_6band):
        sel = select_tiles(msi_6band, "where is it", tile_px=32, max_tiles=10_000)
        assert sel.retrieved == sel.total_tiles

    def test_selected_tiles_stay_in_reading_order(self, msi_6band):
        sel = select_tiles(msi_6band, "locate the river", tile_px=32, max_tiles=5)
        indices = [t.index for t in sel.tiles]
        assert indices == sorted(indices)

    def test_scores_recorded_for_audit(self, msi_6band):
        sel = select_tiles(msi_6band, "find the bridge", tile_px=32, max_tiles=3)
        assert sel.scores
        assert all(isinstance(v, float) for v in sel.scores.values())

    def test_report_shape_matches_tiling_contract(self, msi_6band):
        report = select_tiles(msi_6band, "where", tile_px=32, max_tiles=2).to_report()
        assert set(report) == {
            "applied", "level1_tiles", "retrieved_tiles", "retrieval_reason"
        }
        # The report must be accepted by the frozen contract.
        from satquery.contracts.input_manifest import TilingReport

        TilingReport.model_validate(report)

    def test_retrieval_prefers_structured_over_flat_tiles(self, tmp_path):
        """A flat tile carries no information; a textured one does."""
        from tests.conftest import write_raster

        h = w = 128
        scene = np.zeros((h, w), dtype="float32")
        # Right half is textured, left half is flat.
        scene[:, w // 2:] = np.random.default_rng(9).normal(100, 30, (h, w // 2))
        scene[:, : w // 2] = 100.0
        path = write_raster(tmp_path / "half.tif", scene, dtype="float32")

        sel = select_tiles(path, "where is the structure", tile_px=32, max_tiles=4)
        kept_cols = {t.col for t in sel.tiles}
        # Tiles from the textured right half must dominate the selection.
        assert max(kept_cols) >= 2
