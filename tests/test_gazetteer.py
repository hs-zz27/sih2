"""Gazetteer tests (offline place and climate lookup).

The rasters themselves are third-party and are not in the repository, so
every test here builds a synthetic one. That is not a compromise: what needs
pinning is the decision logic - when a label is asserted, when it is hedged,
and when the module stays silent - and a synthetic raster exercises all three
without a 200 MB download or a licence.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery.geo import gazetteer
from satquery.geo.gazetteer import ENV_GAZETTEER, is_available, lookup

# A 1-degree global grid: west of the split is code 1, east is code 2.
_SPLIT_COL = 240
# The longitude that column falls on, given the transform below. Derived
# rather than restated, so the border tests cannot drift from the fixture.
_SPLIT_LON = _SPLIT_COL - 180
_WIDTH, _HEIGHT = 360, 180
_NODATA = 255


def _write(path, array, nodata=_NODATA):
    with rasterio.open(
        path, "w", driver="GTiff", height=_HEIGHT, width=_WIDTH, count=1,
        dtype="uint8", crs="EPSG:4326",
        transform=from_origin(-180, 90, 1, 1), nodata=nodata,
    ) as dst:
        dst.write(array, 1)


@pytest.fixture
def gazetteer_dir(tmp_path, monkeypatch):
    """A two-layer gazetteer: a country split at 60E, uniform climate."""
    countries = np.where(
        np.arange(_WIDTH)[None, :] >= _SPLIT_COL, 2, 1
    ).repeat(_HEIGHT, 0).astype("uint8")
    # A polar band of nodata, to stand for ocean or unmapped area.
    countries[:20, :] = _NODATA
    _write(tmp_path / "country.tif", countries)
    (tmp_path / "country.json").write_text(json.dumps({
        "labels": {"1": "Westland", "2": "Eastland"},
        "attribution": "Synthetic boundaries, public domain",
    }), encoding="utf-8")

    _write(tmp_path / "climate.tif", np.full((_HEIGHT, _WIDTH), 3, dtype="uint8"))
    (tmp_path / "climate.json").write_text(json.dumps({
        "labels": {"3": "Aw"},
        "descriptions": {"3": "tropical, savannah"},
        "attribution": "Synthetic climate, CC BY 4.0",
    }), encoding="utf-8")

    monkeypatch.setenv(ENV_GAZETTEER, str(tmp_path))
    gazetteer._Handle.reset()
    yield tmp_path
    gazetteer._Handle.reset()


class TestAvailability:
    def test_unset_is_reported_by_name(self, monkeypatch):
        monkeypatch.delenv(ENV_GAZETTEER, raising=False)
        ok, reason = is_available()
        assert ok is False
        assert ENV_GAZETTEER in reason

    def test_a_missing_directory_names_the_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_GAZETTEER, str(tmp_path / "absent"))
        ok, reason = is_available()
        assert ok is False
        assert "absent" in reason

    def test_an_empty_directory_names_the_layers_it_wanted(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv(ENV_GAZETTEER, str(tmp_path))
        ok, reason = is_available()
        assert ok is False
        assert "country" in reason and "climate" in reason

    def test_a_populated_directory_is_ready(self, gazetteer_dir):
        ok, reason = is_available()
        assert ok is True
        assert "country" in reason


class TestLookup:
    def test_a_point_well_inside_a_region_is_named(self, gazetteer_dir):
        place = lookup(0.0, 90.0)
        assert place.country == "Eastland"
        assert place.ambiguous == frozenset()

    def test_the_other_side_of_the_split_is_the_other_region(self, gazetteer_dir):
        assert lookup(0.0, -90.0).country == "Westland"

    def test_climate_carries_its_description(self, gazetteer_dir):
        place = lookup(0.0, 90.0)
        assert place.climate == "Aw"
        assert place.climate_description == "tropical, savannah"

    def test_nodata_yields_no_name(self, gazetteer_dir):
        """An unmapped cell is not a place. The climate layer still answers,
        which is why this asserts on `country` and not on the whole Place."""
        assert lookup(85.0, -90.0).country is None

    def test_attribution_travels_with_the_answer(self, gazetteer_dir):
        sources = lookup(0.0, 90.0).sources
        assert any("public domain" in s for s in sources)
        assert any("CC BY" in s for s in sources)


class TestBorders:
    """The 3x3 agreement window. A hard edge in the data is not a hard edge
    in the world, and a scene sitting on one must not get a confident name."""

    def test_a_point_on_the_split_is_flagged_ambiguous(self, gazetteer_dir):
        place = lookup(0.0, float(_SPLIT_LON))
        assert place.country is not None
        assert "country" in place.ambiguous

    def test_a_point_well_away_from_the_split_is_not(self, gazetteer_dir):
        assert "country" not in lookup(0.0, 90.0).ambiguous

    def test_a_uniform_layer_is_never_ambiguous(self, gazetteer_dir):
        """Same coordinate, same window - only the country layer has an edge
        there, so only the country label may be hedged."""
        assert "climate" not in lookup(0.0, float(_SPLIT_LON)).ambiguous


class TestDegradation:
    """Every failure mode returns less, never raises. A query that reaches
    this module must not be able to fail because of it."""

    def test_no_gazetteer_returns_an_empty_place(self, monkeypatch):
        monkeypatch.delenv(ENV_GAZETTEER, raising=False)
        place = lookup(0.0, 90.0)
        assert not place
        assert place.country is None

    def test_missing_coordinates_return_an_empty_place(self, gazetteer_dir):
        assert not lookup(None, None)

    def test_a_code_with_no_legend_entry_is_not_reported(
        self, monkeypatch, tmp_path
    ):
        """Reporting the raw integer would be worse than silence."""
        _write(tmp_path / "country.tif", np.full((_HEIGHT, _WIDTH), 7, dtype="uint8"))
        (tmp_path / "country.json").write_text(
            json.dumps({"labels": {"1": "Westland"}}), encoding="utf-8"
        )
        monkeypatch.setenv(ENV_GAZETTEER, str(tmp_path))
        gazetteer._Handle.reset()
        assert lookup(0.0, 90.0).country is None
        gazetteer._Handle.reset()

    def test_a_corrupt_legend_degrades_to_no_names(self, monkeypatch, tmp_path):
        _write(tmp_path / "country.tif", np.ones((_HEIGHT, _WIDTH), dtype="uint8"))
        (tmp_path / "country.json").write_text("{not json", encoding="utf-8")
        monkeypatch.setenv(ENV_GAZETTEER, str(tmp_path))
        gazetteer._Handle.reset()
        assert lookup(0.0, 90.0).country is None
        gazetteer._Handle.reset()

    def test_a_layer_without_a_legend_at_all_is_survivable(
        self, monkeypatch, tmp_path
    ):
        _write(tmp_path / "country.tif", np.ones((_HEIGHT, _WIDTH), dtype="uint8"))
        monkeypatch.setenv(ENV_GAZETTEER, str(tmp_path))
        gazetteer._Handle.reset()
        assert lookup(0.0, 90.0).country is None
        gazetteer._Handle.reset()

    def test_repointing_the_env_var_reloads(self, gazetteer_dir, tmp_path,
                                            monkeypatch):
        """The handle is cached process-wide. Caching it by nothing would
        make a second gazetteer in the same process unreachable."""
        assert lookup(0.0, 90.0).country == "Eastland"
        other = tmp_path / "other"
        other.mkdir()
        _write(other / "country.tif", np.full((_HEIGHT, _WIDTH), 1, dtype="uint8"))
        (other / "country.json").write_text(
            json.dumps({"labels": {"1": "Elsewhere"}}), encoding="utf-8"
        )
        monkeypatch.setenv(ENV_GAZETTEER, str(other))
        assert lookup(0.0, 90.0).country == "Elsewhere"
