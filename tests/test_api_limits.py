"""API resource limits (security hardening).

Two gaps found while auditing Phase 3, both unbounded and both reachable by
an unauthenticated caller:

* `shutil.copyfileobj` wrote whatever the client sent, so one request could
  fill the disk. Neither Starlette nor uvicorn imposes a default limit.
* Nothing ever deleted a run's temp directory, so uploaded imagery and run
  artifacts accumulated for the life of the process.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from satquery.api import main as api


@pytest.fixture
def client():
    return TestClient(api.app)


class TestUploadSizeLimit:
    def test_an_oversized_upload_is_refused_with_413(self, client, monkeypatch):
        monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 1024)
        payload = b"\x00" * (4 * 1024)
        response = client.post(
            "/runs",
            data={"query": "Describe this image."},
            files={"images": ("big.tif", io.BytesIO(payload), "image/tiff")},
        )
        assert response.status_code == 413
        assert "upload limit" in response.text

    def test_the_partial_file_is_not_left_on_disk(self, client, monkeypatch):
        """Refusing after writing 256 MB of a 4 GB upload would be no defence."""
        monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 1024)
        before = set(Path(tempfile.gettempdir()).glob("satquery_run_*"))
        client.post(
            "/runs",
            data={"query": "Describe this image."},
            files={"images": ("big.tif", io.BytesIO(b"\x00" * 8192), "image/tiff")},
        )
        after = set(Path(tempfile.gettempdir()).glob("satquery_run_*"))
        assert after == before

    def test_a_normal_upload_still_succeeds(self, client, msi_6band):
        with msi_6band.open("rb") as fh:
            response = client.post(
                "/runs",
                data={"query": "Classify the land cover."},
                files={"images": ("msi.tif", fh, "image/tiff")},
            )
        assert response.status_code == 200
        assert response.json()["answer"]

    def test_the_limit_is_configurable(self, monkeypatch):
        """So a deployment handling full Cartosat scenes can raise it."""
        assert api.MAX_UPLOAD_BYTES >= 64 * 1024 * 1024


class TestRunDirectoryRetention:
    def test_pruning_keeps_only_the_newest(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        made = []
        for i in range(8):
            d = tmp_path / f"satquery_run_{i:02d}"
            d.mkdir()
            (d / "upload.tif").write_bytes(b"x")
            import os
            os.utime(d, (i * 100, i * 100))
            made.append(d)

        api._prune_run_dirs(keep=3)
        surviving = sorted(p.name for p in tmp_path.glob("satquery_run_*"))
        assert len(surviving) == 3
        # Newest kept, oldest gone.
        assert "satquery_run_07" in surviving
        assert "satquery_run_00" not in surviving

    def test_report_directories_are_pruned_too(self, tmp_path, monkeypatch):
        """These leaked as well; the first prune only matched satquery_run_*."""
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        for i in range(5):
            (tmp_path / f"satquery_report_{i}").mkdir()
        api._prune_run_dirs(keep=1)
        assert len(list(tmp_path.glob("satquery_report_*"))) <= 1

    def test_unrelated_directories_are_never_touched(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        keep_me = tmp_path / "someone_elses_data"
        keep_me.mkdir()
        (tmp_path / "satquery_run_00").mkdir()
        api._prune_run_dirs(keep=0)
        assert keep_me.exists()

    def test_pruning_never_raises(self, tmp_path, monkeypatch):
        """Failing to reclaim disk must not fail the run that just succeeded."""
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "gone"))
        api._prune_run_dirs(keep=1)


class TestStreamedAndSyncPathsAgree:
    """The SSE path used to reach past the controller into the executor.

    That duplicated the pipeline, and the copies drifted: the streamed path
    never passed `config_excluded`, so the task 3.8 notice was missing from
    the streamed answer - the path the frontend actually uses. These tests
    pin the two paths to the same behaviour.
    """

    def _stream_trace(self, client, image, query):
        import json as _json

        with image.open("rb") as fh:
            response = client.post(
                "/runs/stream",
                data={"query": query},
                files={"images": ("msi.tif", fh, "image/tiff")},
            )
        assert response.status_code == 200
        for block in response.text.split("\n\n"):
            if block.startswith("event: complete"):
                return _json.loads(block.split("data: ", 1)[1])
        raise AssertionError(f"no complete event: {response.text[:300]}")

    def test_streamed_answer_carries_the_config_exclusion_notice(
        self, client, msi_6band
    ):
        """A change query on one image. This is the case that regressed."""
        trace = self._stream_trace(
            client, msi_6band, "Produce a change mask for these images."
        )
        assert trace["routing"]["config_excluded_task"] == "TEMPORAL_CHANGE_MAP"
        assert "TEMPORAL_CHANGE_MAP" in trace["answer"]

    def test_streamed_and_sync_answers_match(self, client, msi_6band):
        streamed = self._stream_trace(
            client, msi_6band, "Produce a change mask for these images."
        )
        with msi_6band.open("rb") as fh:
            sync = client.post(
                "/runs",
                data={"query": "Produce a change mask for these images."},
                files={"images": ("msi.tif", fh, "image/tiff")},
            ).json()
        assert streamed["answer"] == sync["answer"]
        assert streamed["routing"] == sync["routing"]
        assert streamed["abstained"] == sync["abstained"]

    def test_streamed_abstention_carries_its_resolving_input(
        self, client, no_crs_raster
    ):
        trace = self._stream_trace(client, no_crs_raster, "Describe this image.")
        assert trace["abstained"]
        assert trace["abstain_resolving_input"]


class TestMapOverlays:
    """Georeferenced overlays for the map viewer (plan task 1.6).

    The server reprojects to EPSG:3857 and returns the extent in a header, so
    the client places the image without carrying proj4 and a registry entry
    for every UTM zone an Indian scene can land in.
    """

    def _run(self, client, image, query="Classify the land cover."):
        with image.open("rb") as fh:
            response = client.post(
                "/runs",
                data={"query": query},
                files={"images": ("m.tif", fh, "image/tiff")},
            )
        assert response.status_code == 200
        return response.json()["run_id"]

    def test_overlays_are_listed_with_availability(self, client, msi_6band):
        run_id = self._run(client, msi_6band)
        body = client.get(f"/runs/{run_id}/overlays").json()
        assert body["overlays"]
        assert all("key" in o and "available" in o for o in body["overlays"])

    def test_an_overlay_is_png_with_a_projected_extent(self, client, msi_6band):
        run_id = self._run(client, msi_6band)
        overlays = client.get(f"/runs/{run_id}/overlays").json()["overlays"]
        key = next(o["key"] for o in overlays if o["available"])

        response = client.get(f"/runs/{run_id}/overlay/{key}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.headers["X-Projection"] == "EPSG:3857"

        extent = [float(v) for v in response.headers["X-Extent"].split(",")]
        assert len(extent) == 4
        minx, miny, maxx, maxy = extent
        assert minx < maxx and miny < maxy
        # Web Mercator is bounded; a wild extent means the reprojection failed
        # silently and the layer would land in the ocean.
        assert all(abs(v) < 2.01e7 for v in extent)

    def test_the_overlay_carries_an_alpha_channel(self, client, msi_6band):
        """Nodata must be transparent, or a mask covers the basemap with a
        black rectangle instead of overlaying it."""
        import io

        from PIL import Image

        run_id = self._run(client, msi_6band)
        overlays = client.get(f"/runs/{run_id}/overlays").json()["overlays"]
        key = next(o["key"] for o in overlays if o["available"])
        response = client.get(f"/runs/{run_id}/overlay/{key}")
        assert Image.open(io.BytesIO(response.content)).mode == "RGBA"

    def test_the_extent_header_is_exposed_to_the_browser(self, client, msi_6band):
        """A cross-origin fetch cannot read it without this, and the frontend
        is served from a different port."""
        run_id = self._run(client, msi_6band)
        overlays = client.get(f"/runs/{run_id}/overlays").json()["overlays"]
        key = next(o["key"] for o in overlays if o["available"])
        exposed = client.get(
            f"/runs/{run_id}/overlay/{key}"
        ).headers["Access-Control-Expose-Headers"]
        assert "X-Extent" in exposed

    def test_an_unknown_key_is_404_and_lists_what_exists(self, client, msi_6band):
        run_id = self._run(client, msi_6band)
        response = client.get(f"/runs/{run_id}/overlay/not_a_layer")
        assert response.status_code == 404
        assert "available" in response.text

    def test_an_unknown_run_is_404(self, client):
        assert client.get("/runs/run_nope/overlays").status_code == 404

    def test_a_stub_artifact_path_is_410_not_a_crash(self, client, msi_6band):
        """The stubs report fake paths; asking for one must degrade, and 410
        (gone) is the honest status rather than 404 or 500."""
        run_id = self._run(client, msi_6band)
        overlays = client.get(f"/runs/{run_id}/overlays").json()["overlays"]
        missing = [o["key"] for o in overlays if not o["available"]]
        if not missing:
            pytest.skip("no unavailable artifact in this run")
        assert client.get(
            f"/runs/{run_id}/overlay/{missing[0]}"
        ).status_code == 410
