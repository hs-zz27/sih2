"""Place and climate lookup from a bundled categorical raster.

The footprint on `ImageMeta` says where a scene is in degrees. This turns
those degrees into words - a country, a Koppen climate zone - so a
description can say "in Lithuania, in the warm-summer continental zone"
rather than only "54.9012 N, 21.0834 E".

## Why a raster, and why no new dependency

Country boundaries are naturally polygons, and a point-in-polygon test wants
`shapely`. `shapely` is importable in the current environment but is NOT in
`requirements.txt`; it arrives transitively, and building on an undeclared
transitive import is the exact mistake the pillow pin in that file was added
to correct. Categorical rasters are read by `rasterio`, which IS declared, so
both layers are rasters and this module adds no dependency at all.

## Activation

Opt-in via `SATQUERY_GAZETTEER`, pointing at a directory, matching how
`caption_v1`, `rs_vqa_v1` and the NLI backend gate their assets. Without it
every lookup returns an empty `Place` and the answer simply says nothing
about the region - which is the correct behaviour, not a degraded one. The
data is not in the repository: it is third-party, it carries its own licence
and attribution, and `scripts/fetch_geo.py` fetches it under the same
trust-on-first-use digest rule as the model and dataset fetchers.

## Borders, and why a single sample is not enough

A categorical raster at any resolution puts a hard edge where the world has a
soft one. A point 3 km inside one country and a point 3 km inside its
neighbour can land in the same coarse cell, and reporting that cell's label
as fact would be asserting a country the data does not actually establish.

So a lookup reads a 3x3 window, not a pixel. A label is asserted only when
the whole window agrees. When it does not, the label is still returned but
flagged in `ambiguous`, and the caller says "near the border of" or says
nothing - it never picks the centre cell and presents it as settled. This is
the same principle as `landcover_v1` abstaining per class between its
thresholds: where the evidence does not decide, the answer says so.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

ENV_GAZETTEER = "SATQUERY_GAZETTEER"

# Layers this module knows how to read. A directory supplying only one of
# them is fine and expected - the climate raster is small and freely
# licensed, national boundaries less uniformly so.
LAYERS = ("country", "climate")

# Half-width of the agreement window, in pixels. 1 gives the 3x3 described
# above: the sampled cell plus every cell touching it.
_WINDOW = 1


@dataclass(frozen=True)
class Place:
    """What the rasters say about one coordinate.

    Every field is optional because every layer is optional, and a field is
    `None` both when its layer is absent and when the coordinate fell on
    nodata. The caller cannot tell those apart and does not need to: in both
    cases the data does not support naming it.
    """

    country: str | None = None
    climate: str | None = None
    # Description for the climate code, e.g. "cold, no dry season, warm
    # summer" for Dfb, when the legend supplies one.
    climate_description: str | None = None
    # Layers whose window disagreed. Carried rather than used to drop the
    # label, so the caller can hedge instead of going silent.
    ambiguous: frozenset[str] = field(default_factory=frozenset)
    # Attribution strings from each legend, for the trace. Third-party data
    # under a CC BY licence has to be credited wherever it is used.
    sources: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.country or self.climate)


def is_available() -> tuple[bool, str]:
    """Whether a usable gazetteer directory is configured.

    Mirrors `tools.caption.is_available`: contents first, so the message
    names the thing the operator can actually fix.
    """
    path = os.getenv(ENV_GAZETTEER)
    if not path:
        return False, f"{ENV_GAZETTEER} is not set"
    directory = Path(path)
    if not directory.is_dir():
        return False, f"gazetteer directory not found: {path}"
    present = [name for name in LAYERS if (directory / f"{name}.tif").exists()]
    if not present:
        return False, f"no layer rasters in {path} (looked for {', '.join(LAYERS)})"
    try:
        import rasterio  # noqa: F401
    except ImportError:  # pragma: no cover - rasterio is a declared dependency
        return False, "rasterio is not installed"
    return True, f"ready ({', '.join(present)})"


class _Layer:
    """One categorical raster plus its legend, opened once."""

    def __init__(self, raster: Path, legend: Path | None):
        import rasterio

        self.dataset = rasterio.open(raster)
        self.nodata = self.dataset.nodata
        self.labels: dict[int, str] = {}
        self.descriptions: dict[int, str] = {}
        self.attribution: str | None = None

        if legend is not None and legend.exists():
            try:
                payload = json.loads(legend.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                # A corrupt legend degrades this layer to "no names" rather
                # than failing the query. Same reasoning as the vocab.json
                # check in tools/caption.py, one step softer: a missing label
                # is survivable, a missing vocabulary is not.
                payload = {}
            self.labels = {
                int(k): str(v) for k, v in (payload.get("labels") or {}).items()
            }
            self.descriptions = {
                int(k): str(v)
                for k, v in (payload.get("descriptions") or {}).items()
            }
            attribution = payload.get("attribution")
            self.attribution = str(attribution) if attribution else None

    def _blank(self) -> int:
        """The value meaning "nothing here" for this raster."""
        return int(self.nodata) if self.nodata is not None else 0

    def sample(self, latitude: float, longitude: float) -> tuple[int | None, bool]:
        """(code, ambiguous) at a coordinate, from a 3x3 agreement window.

        `code` is None when the point is outside the raster or on nodata.
        `ambiguous` is True when the window held more than one valid code.
        """
        import numpy as np

        blank = self._blank()
        row, col = self.dataset.index(longitude, latitude)
        # `boundless` so a coordinate near the antimeridian or a pole reads
        # the fill value instead of raising on a window off the edge.
        window = self.dataset.read(
            1,
            window=(
                (row - _WINDOW, row + _WINDOW + 1),
                (col - _WINDOW, col + _WINDOW + 1),
            ),
            boundless=True,
            fill_value=blank,
        )
        values = np.asarray(window).ravel()
        values = values[values != blank]
        if values.size == 0:
            return None, False

        # The centre is what the coordinate actually falls in; the rest of
        # the window only decides whether the centre is trustworthy.
        code = int(np.asarray(window).reshape(2 * _WINDOW + 1, -1)[_WINDOW, _WINDOW])
        if code == blank:
            return None, False
        return code, bool(np.unique(values).size > 1)

    def close(self) -> None:
        self.dataset.close()


class _Handle:
    """Open layers, shared process-wide."""

    _instance: "_Handle | None" = None
    _directory: Path | None = None
    _lock = threading.Lock()

    def __init__(self, directory: Path):
        self.layers: dict[str, _Layer] = {}
        for name in LAYERS:
            raster = directory / f"{name}.tif"
            if raster.exists():
                self.layers[name] = _Layer(raster, directory / f"{name}.json")

    @classmethod
    def get(cls, directory: Path) -> "_Handle":
        # Keyed on the directory, not merely cached: a process that is
        # pointed at a different gazetteer must not keep serving the old one.
        if cls._instance is None or cls._directory != directory:
            with cls._lock:
                if cls._instance is None or cls._directory != directory:
                    cls.reset()
                    cls._instance = cls(directory)
                    cls._directory = directory
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Close and drop the cached handle."""
        if cls._instance is not None:
            for layer in cls._instance.layers.values():
                layer.close()
        cls._instance = None
        cls._directory = None


def lookup(latitude: float | None, longitude: float | None) -> Place:
    """Name the region containing a coordinate, as far as the data allows.

    Never raises. A missing gazetteer, an unreadable raster, a coordinate in
    the ocean and a coordinate on a border all produce a `Place` carrying
    less, because in every one of those cases the honest answer is that the
    data does not establish the name.
    """
    if latitude is None or longitude is None:
        return Place()
    ok, _ = is_available()
    if not ok:
        return Place()

    try:
        handle = _Handle.get(Path(os.environ[ENV_GAZETTEER]))
    except Exception:  # noqa: BLE001 - a bad raster must not fail the query
        return Place()

    found: dict[str, str] = {}
    description: str | None = None
    ambiguous: set[str] = set()
    sources: list[str] = []

    for name, layer in handle.layers.items():
        try:
            code, uncertain = layer.sample(latitude, longitude)
        except Exception:  # noqa: BLE001 - as above, per layer
            continue
        if code is None:
            continue
        label = layer.labels.get(code)
        if label is None:
            # A code with no legend entry is not a name. Reporting the raw
            # integer would be worse than silence.
            continue
        found[name] = label
        if uncertain:
            ambiguous.add(name)
        if name == "climate":
            description = layer.descriptions.get(code)
        if layer.attribution:
            sources.append(layer.attribution)

    return Place(
        country=found.get("country"),
        climate=found.get("climate"),
        climate_description=description,
        ambiguous=frozenset(ambiguous),
        # dict.fromkeys rather than set(): two layers from the same publisher
        # credit it once, in the order the layers were read.
        sources=tuple(dict.fromkeys(sources)),
    )
