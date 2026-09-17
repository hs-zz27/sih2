"""Tile pyramid and coarse-to-fine retrieval (plan task 2.10).

Phase 1 processed whole scenes in memory. Real Cartosat products are
7687x7640 (59 megapixels) and the plan expects 8000x8000; at that size a
single float64 band is ~450 MiB and the index engine exhausted memory on a
16 GB machine. Working in float32 bought a factor of two, which was enough to
survive one scene but is not a structural answer.

This module is the structural answer. Two capabilities:

* **Windowed iteration** - stream a scene tile by tile so peak memory depends
  on the tile size, not the scene size. A statistic accumulated across tiles
  (mean, histogram, threshold) is identical to the whole-scene result.
* **Coarse-to-fine retrieval** - most questions concern part of a scene. Score
  every tile cheaply on a decimated overview, keep the most relevant ones, and
  run the expensive work only there. The plan requires the retrieval decision
  to be logged, so `TileSelection` records why each tile was kept.

Retrieval never silently drops data: when a query has no spatial focus, the
selection returns every tile and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

# Default tile edge. 1024 keeps a float32 tile at ~4 MiB per band, so even a
# dozen intermediates stay comfortably inside a laptop's memory.
DEFAULT_TILE_PX = 1024

# Scenes below this longest edge are processed whole - tiling has overhead and
# buys nothing at small sizes.
TILING_TRIGGER_PX = 4096

# Longest edge of the overview used for cheap tile scoring.
OVERVIEW_MAX_PX = 1024


@dataclass(frozen=True)
class Tile:
    """One tile's position within the scene."""

    index: int
    row: int
    col: int
    window: Window

    @property
    def key(self) -> str:
        return f"r{self.row}c{self.col}"


@dataclass
class TileSelection:
    """Which tiles were kept, and the stated reason."""

    tiles: list[Tile]
    total_tiles: int
    reason: str
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def retrieved(self) -> int:
        return len(self.tiles)

    def to_report(self) -> dict:
        return {
            "applied": self.total_tiles > 1,
            "level1_tiles": self.total_tiles,
            "retrieved_tiles": self.retrieved,
            "retrieval_reason": self.reason,
        }


def needs_tiling(width: int, height: int, trigger: int = TILING_TRIGGER_PX) -> bool:
    return max(width, height) >= trigger


def plan_tiles(
    width: int, height: int, tile_px: int = DEFAULT_TILE_PX
) -> list[Tile]:
    """Cover a raster with non-overlapping tiles, row-major.

    Edge tiles are clipped rather than padded, so every pixel belongs to
    exactly one tile and accumulated statistics stay exact.
    """
    tiles: list[Tile] = []
    index = 0
    for row, y in enumerate(range(0, height, tile_px)):
        for col, x in enumerate(range(0, width, tile_px)):
            h = min(tile_px, height - y)
            w = min(tile_px, width - x)
            tiles.append(Tile(index, row, col, Window(x, y, w, h)))
            index += 1
    return tiles


def read_overview(path: str | Path, band: int = 1, max_edge: int = OVERVIEW_MAX_PX):
    """Read a decimated view of one band, for cheap whole-scene scoring."""
    with rasterio.open(path) as src:
        scale = max(src.width, src.height) / max_edge
        out_h = max(1, int(src.height / scale)) if scale > 1 else src.height
        out_w = max(1, int(src.width / scale)) if scale > 1 else src.width
        arr = src.read(
            band, out_shape=(out_h, out_w),
            resampling=Resampling.average, masked=True,
        )
    return np.ma.filled(arr.astype("float32"), np.nan)


def iter_tiles(
    path: str | Path, band: int = 1, tile_px: int = DEFAULT_TILE_PX,
    tiles: list[Tile] | None = None,
) -> Iterator[tuple[Tile, np.ndarray]]:
    """Yield (tile, array) pairs, reading one tile at a time.

    Peak memory is one tile, not one scene - the whole point of this module.
    """
    with rasterio.open(path) as src:
        plan = tiles if tiles is not None else plan_tiles(src.width, src.height, tile_px)
        for tile in plan:
            arr = src.read(band, window=tile.window, masked=True)
            yield tile, np.ma.filled(arr.astype("float32"), np.nan)


class StreamingStats:
    """Exact mean/variance/min/max accumulated across tiles.

    Uses Welford's algorithm rather than accumulating sum and sum-of-squares:
    the naive form loses precision on large pixel counts, and a 59-megapixel
    scene is exactly where that starts to matter.
    """

    def __init__(self) -> None:
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, arr: np.ndarray) -> None:
        values = arr[np.isfinite(arr)]
        if values.size == 0:
            return

        batch_n = int(values.size)
        batch_mean = float(values.mean())
        batch_m2 = float(((values - batch_mean) ** 2).sum())

        # Chan et al. parallel variance combination.
        delta = batch_mean - self._mean
        total = self.n + batch_n
        self._m2 += batch_m2 + delta * delta * self.n * batch_n / total
        self._mean += delta * batch_n / total
        self.n = total

        self.min = min(self.min, float(values.min()))
        self.max = max(self.max, float(values.max()))

    @property
    def mean(self) -> float:
        return self._mean if self.n else float("nan")

    @property
    def variance(self) -> float:
        return self._m2 / self.n if self.n else float("nan")

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance)) if self.n else float("nan")

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean": self.mean,
            "std": self.std,
            "min": self.min if self.n else float("nan"),
            "max": self.max if self.n else float("nan"),
        }


def scene_stats(path: str | Path, band: int = 1, tile_px: int = DEFAULT_TILE_PX) -> dict:
    """Whole-scene statistics computed tile by tile, in bounded memory."""
    stats = StreamingStats()
    for _, arr in iter_tiles(path, band=band, tile_px=tile_px):
        stats.update(arr)
    return stats.to_dict()


# Query words that indicate the answer concerns a specific part of the scene.
# Without one of these, retrieval keeps every tile.
_SPATIAL_HINTS = (
    "where", "locate", "find", "show me", "nearest", "closest", "corner",
    "region", "area near", "around", "part of",
)


def select_tiles(
    path: str | Path,
    query: str,
    *,
    tile_px: int = DEFAULT_TILE_PX,
    max_tiles: int | None = None,
    band: int = 1,
) -> TileSelection:
    """Choose which tiles to process for `query`.

    Scoring uses tile variance measured on a decimated overview: featureless
    tiles (open water, uniform cloud, nodata fill) carry little information,
    while structured tiles do. This is a cheap, modality-agnostic proxy - it
    needs no model and works identically on optical and SAR.

    It is deliberately conservative. Retrieval only narrows when the query
    actually asks about a location AND a cap is requested; otherwise every
    tile is returned, because silently discarding scene content would make
    "how much X is in this image" quietly wrong.
    """
    with rasterio.open(path) as src:
        width, height = src.width, src.height

    tiles = plan_tiles(width, height, tile_px)

    if len(tiles) <= 1:
        return TileSelection(tiles, len(tiles), "scene fits in a single tile")

    spatial = any(hint in query.lower() for hint in _SPATIAL_HINTS)
    if max_tiles is None or max_tiles >= len(tiles) or not spatial:
        reason = (
            "query is not spatially focused; all tiles retained so scene-wide "
            "statistics remain correct"
            if not spatial
            else "no tile budget set; all tiles retained"
        )
        return TileSelection(tiles, len(tiles), reason)

    # Score on the overview: one decimated read instead of N tile reads.
    overview = read_overview(path, band=band)
    oh, ow = overview.shape
    y_scale, x_scale = oh / height, ow / width

    scores: dict[str, float] = {}
    for tile in tiles:
        w = tile.window
        y0 = int(w.row_off * y_scale)
        y1 = max(y0 + 1, int((w.row_off + w.height) * y_scale))
        x0 = int(w.col_off * x_scale)
        x1 = max(x0 + 1, int((w.col_off + w.width) * x_scale))
        patch = overview[y0:y1, x0:x1]
        finite = patch[np.isfinite(patch)]
        scores[tile.key] = float(finite.std()) if finite.size > 1 else 0.0

    ranked = sorted(tiles, key=lambda t: scores[t.key], reverse=True)[:max_tiles]
    ranked.sort(key=lambda t: t.index)  # restore reading order

    return TileSelection(
        ranked,
        len(tiles),
        f"spatially focused query; kept the {len(ranked)} of {len(tiles)} tiles "
        "with the highest overview variance (most structured content)",
        scores,
    )
