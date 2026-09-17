"""Evidence pack export tests (plan task 2.11).

The acceptance criterion is that the output opens correctly georeferenced in
QGIS, so the tests focus on the two things that silently break that: GeoJSON
CRS, and raster integrity.
"""

from __future__ import annotations

import hashlib
import json
import zipfile

import pytest
import rasterio

from satquery.controller.pipeline import Controller
from satquery.report.evidence_pack import (
    GEOJSON_CRS,
    boxes_to_geojson,
    export,
    raster_footprint,
)


@pytest.fixture(scope="module")
def trace(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    import tests.conftest as cf
    import numpy as np

    scene = cf._structured_scene(128, 128, seed=2)
    bands = np.stack([
        scene * 800 + 200, scene * 900 + 250, scene * 700 + 180,
        scene * 2200 + 400, scene * 1100 + 300, scene * 800 + 220,
    ]).astype("uint16")
    path = cf.write_raster(
        out / "scene.tif", bands,
        band_names=["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
    )
    return Controller().run([path], "Classify the land cover."), out


class TestFootprint:
    def test_footprint_is_wgs84(self, trace):
        _, out = trace
        feature = raster_footprint(out / "scene.tif")
        assert feature is not None
        lon, lat = feature["geometry"]["coordinates"][0][0]
        # A UTM easting would be ~500000, far outside valid longitude.
        assert -180 <= lon <= 180
        assert -90 <= lat <= 90

    def test_source_crs_preserved(self, trace):
        _, out = trace
        feature = raster_footprint(out / "scene.tif")
        assert "32643" in feature["properties"]["source_crs"]

    def test_missing_crs_yields_no_footprint(self, no_crs_raster):
        assert raster_footprint(no_crs_raster) is None

    def test_unreadable_file_does_not_raise(self, tmp_path):
        bad = tmp_path / "bad.tif"
        bad.write_bytes(b"not a raster")
        assert raster_footprint(bad) is None


class TestExport:
    def test_creates_zip(self, trace, tmp_path):
        t, out = trace
        archive = export(t, tmp_path, artifact_dir="artifacts")
        assert archive.exists() and archive.suffix == ".zip"

    def test_contains_expected_members(self, trace, tmp_path):
        t, out = trace
        archive = export(t, tmp_path, artifact_dir="artifacts")
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
        assert "trace.json" in names
        assert "evidence.json" in names
        assert "answer.txt" in names
        assert "footprint.geojson" in names

    def test_evidence_manifest_is_valid_json(self, trace, tmp_path):
        t, out = trace
        archive = export(t, tmp_path, artifact_dir="artifacts")
        with zipfile.ZipFile(archive) as zf:
            evidence = json.loads(zf.read("evidence.json"))
        assert evidence["run_id"] == t.run_id
        assert evidence["geojson_crs"] == GEOJSON_CRS
        assert evidence["files"]

    def test_checksums_match_contents(self, trace, tmp_path):
        """The integrity claim must actually hold."""
        t, out = trace
        archive = export(t, tmp_path, artifact_dir="artifacts")
        with zipfile.ZipFile(archive) as zf:
            evidence = json.loads(zf.read("evidence.json"))
            for entry in evidence["files"]:
                digest = hashlib.sha256(zf.read(entry["file"])).hexdigest()
                assert digest == entry["sha256"], entry["file"]

    def test_geojson_parses_and_is_a_feature_collection(self, trace, tmp_path):
        t, out = trace
        archive = export(t, tmp_path, artifact_dir="artifacts")
        with zipfile.ZipFile(archive) as zf:
            fc = json.loads(zf.read("footprint.geojson"))
        assert fc["type"] == "FeatureCollection"
        assert fc["features"]

    def test_directory_mode_leaves_files_on_disk(self, trace, tmp_path):
        t, out = trace
        pack = export(t, tmp_path, artifact_dir="artifacts", zip_output=False)
        assert pack.is_dir()
        assert (pack / "evidence.json").exists()

    def test_answer_text_records_confidence(self, trace, tmp_path):
        t, out = trace
        pack = export(t, tmp_path, zip_output=False)
        text = (pack / "answer.txt").read_text(encoding="utf-8")
        assert t.answer[:20] in text
        assert "Confidence" in text

    def test_verification_carried_into_manifest(self, trace, tmp_path):
        t, out = trace
        pack = export(t, tmp_path, zip_output=False)
        evidence = json.loads((pack / "evidence.json").read_text(encoding="utf-8"))
        assert "physics_agreement" in evidence["verification"]
        assert "built_up_path" in evidence["verification"]

    def test_rasters_copied_bit_for_bit(self, trace, tmp_path):
        """Re-encoding could change values in a bundle meant for verification."""
        t, out = trace
        pack = export(t, tmp_path, artifact_dir="artifacts", zip_output=False)
        copied = sorted((pack / "rasters").glob("*.tif"))
        if not copied:
            pytest.skip("no index rasters were written for this run")
        for path in copied:
            source = pytest.importorskip("pathlib").Path("artifacts") / t.run_id / path.name
            if source.exists():
                assert path.read_bytes() == source.read_bytes()

    def test_copied_rasters_stay_georeferenced(self, trace, tmp_path):
        t, out = trace
        pack = export(t, tmp_path, artifact_dir="artifacts", zip_output=False)
        for path in sorted((pack / "rasters").glob("*.tif")):
            with rasterio.open(path) as src:
                assert src.crs is not None
                assert src.transform.a != 0


class TestDetections:
    def test_no_boxes_yields_none(self, trace):
        t, _ = trace
        # A land-cover run produces no bounding boxes.
        assert boxes_to_geojson(t) is None

    def test_boxes_export_when_present(self, msi_6band, tmp_path):
        t = Controller().run([msi_6band], "Show me where the roads are.")
        result = boxes_to_geojson(t)
        if result is None:
            pytest.skip("grounding stub produced no usable boxes")
        assert result["type"] == "FeatureCollection"
        lon, lat = result["features"][0]["geometry"]["coordinates"][0][0]
        assert -180 <= lon <= 180 and -90 <= lat <= 90
