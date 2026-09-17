"""Offline hardening (plan task 3.9).

`make offline-test` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`,
which is necessary and not sufficient: those variables only stop the
huggingface libraries from reaching out. They say nothing about a stray
`requests.get`, a `urllib.urlopen`, or a library that quietly checks for
updates.

These tests block the socket layer itself. Any attempt to open a network
connection raises `OfflineViolation`, which names the address, so a failure
points at what tried to connect rather than at a timeout twenty seconds
later.

The scope is the runtime path: ingest, routing, the index engine, the
verifier, the entailment gate's deterministic backend, confidence,
calibration, abstention, and the API. Model *downloads* are explicitly out of
scope - `scripts/fetch_models.py` is meant to reach the network, and is not
part of a cold boot.
"""

from __future__ import annotations

import socket

import pytest

from satquery.controller.pipeline import Controller


class OfflineViolation(RuntimeError):
    """Raised instead of opening a socket, naming the attempted address."""


@pytest.fixture
def no_network(monkeypatch):
    """Make every outbound connection raise, loopback included.

    Loopback is blocked too. A test that allows 127.0.0.1 would pass against a
    service the demo laptop happens to be running, which is exactly the
    failure mode a venue with no network produces.
    """

    def blocked(*args, **kwargs):
        raise OfflineViolation(f"network access attempted: {args!r}")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket, "gethostbyname", blocked)
    return blocked


LOOPBACK = {"127.0.0.1", "::1", "localhost", ""}


@pytest.fixture
def no_external_network(monkeypatch):
    """Block everything except loopback.

    Needed only for the API test. On Windows asyncio's `ProactorEventLoop`
    builds its self-pipe out of a **loopback TCP socket**, so the strict
    fixture above kills the event loop before any application code runs -
    which tests the harness, not the product.

    Loopback stays open here and nothing else does. That is a real weakening,
    so it is confined to the one test that needs it, and
    `test_pipeline_offline` above still runs under the strict fixture: the
    controller, not the web framework, is where a hidden download would live.
    """

    def host_of(address) -> str:
        if isinstance(address, (tuple, list)) and address:
            return str(address[0])
        return str(address)

    def guard(original):
        def wrapper(*args, **kwargs):
            address = args[1] if len(args) > 1 else kwargs.get("address")
            if address is not None and host_of(address) not in LOOPBACK:
                raise OfflineViolation(f"external network attempted: {address!r}")
            return original(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(socket.socket, "connect", guard(socket.socket.connect))
    monkeypatch.setattr(
        socket.socket, "connect_ex", guard(socket.socket.connect_ex)
    )

    real_create = socket.create_connection

    def create_connection(address, *a, **kw):
        if host_of(address) not in LOOPBACK:
            raise OfflineViolation(f"external network attempted: {address!r}")
        return real_create(address, *a, **kw)

    monkeypatch.setattr(socket, "create_connection", create_connection)
    return guard


class TestTheGuardItself:
    def test_the_fixture_actually_blocks_a_connection(self, no_network):
        """A guard that does not guard would make every test below vacuous."""
        with pytest.raises(OfflineViolation):
            socket.create_connection(("example.com", 80))

    def test_loopback_is_blocked_too(self, no_network):
        with pytest.raises(OfflineViolation):
            socket.create_connection(("127.0.0.1", 8000))


class TestPipelineOffline:
    def test_full_single_image_run_needs_no_network(self, no_network, msi_6band):
        controller = Controller()
        trace = controller.run([msi_6band], "Classify the land cover.")
        assert trace.answer
        assert not trace.abstained

    def test_bitemporal_run_needs_no_network(
        self, no_network, msi_6band, msi_6band_t2
    ):
        controller = Controller()
        trace = controller.run(
            [msi_6band, msi_6band_t2], "Describe what changed between the images."
        )
        assert trace.answer

    def test_crossmodal_run_needs_no_network(
        self, no_network, msi_6band, sar_dualpol
    ):
        controller = Controller()
        trace = controller.run(
            [msi_6band, sar_dualpol],
            "Combine the optical and radar images to find buildings.",
        )
        assert trace.answer

    def test_abstention_path_needs_no_network(self, no_network, no_crs_raster):
        controller = Controller()
        trace = controller.run([no_crs_raster], "Describe this image.")
        assert trace.abstained
        assert trace.abstain_resolving_input

    def test_controller_construction_needs_no_network(self, no_network):
        """The intent classifier fits on construction; it must fit offline."""
        controller = Controller()
        assert controller.router.classifier.classes_


class TestGazetteerOffline:
    """The gazetteer reads operator-installed rasters from disk. It must do
    so without a socket - it is on the query path, and task 3.9's guarantee
    covers the query path whether or not optional data is installed."""

    @pytest.fixture
    def installed(self, tmp_path, monkeypatch):
        import json

        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        from satquery.geo import gazetteer

        with rasterio.open(
            tmp_path / "country.tif", "w", driver="GTiff", height=180,
            width=360, count=1, dtype="uint8", crs="EPSG:4326",
            transform=from_origin(-180, 90, 1, 1), nodata=255,
        ) as dst:
            dst.write(np.ones((180, 360), dtype="uint8"), 1)
        (tmp_path / "country.json").write_text(
            json.dumps({"labels": {"1": "Testland"}, "attribution": "synthetic"}),
            encoding="utf-8",
        )
        monkeypatch.setenv("SATQUERY_GAZETTEER", str(tmp_path))
        gazetteer._Handle.reset()
        yield tmp_path
        gazetteer._Handle.reset()

    def test_a_lookup_needs_no_network(self, no_network, installed):
        from satquery.geo import lookup

        assert lookup(0.0, 90.0).country == "Testland"

    def test_a_full_run_with_a_gazetteer_needs_no_network(
        self, no_network, installed, msi_6band
    ):
        """The layer that turns coordinates into names is the one most
        likely to be tempted into a lookup service. It never is."""
        trace = Controller().run([msi_6band], "Describe this image.", run_id="r")
        assert "Testland" in trace.answer
        assert trace.data_sources == ["synthetic"]


class TestComponentsOffline:
    def test_calibration_registry_loads_offline(self, no_network):
        from satquery.controller.calibration import load_registry, reset_cache

        reset_cache()
        try:
            assert load_registry().status == "loaded"
        finally:
            reset_cache()

    def test_abstention_thresholds_load_offline(self, no_network):
        from satquery.controller.abstention import AbstentionPolicy

        assert AbstentionPolicy.load().min_final_confidence >= 0.0

    def test_entailment_gate_deterministic_backend_is_offline(self, no_network):
        """The always-available backend must never need a model.

        The NLI backend is opt-in via SATQUERY_NLI and reads a LOCAL
        checkpoint with `local_files_only=True`, so it is offline-safe too -
        but only the deterministic one is exercised here, because CI has no
        checkpoint and a skip would leave this untested where it matters.
        """
        from satquery.verify.entailment import run_gate

        payload = {"indices": {"ndvi": {"fraction_above_threshold": 0.62}}}
        result = run_gate("Vegetation covers 62% of the scene.", payload)
        assert result.retained == 1
        assert result.backend == "deterministic"

    def test_capability_matrix_loads_offline(self, no_network):
        from pathlib import Path

        from satquery.controller.matrix_loader import load_matrix

        assert load_matrix(Path("configs/capability_matrix.yaml")).tasks


class TestApiOffline:
    def test_api_serves_a_run_offline(self, no_external_network, msi_6band):
        """The API is the demo surface; it must boot and answer with no network.

        Runs under the loopback-permitting fixture - see its docstring - so
        this asserts no EXTERNAL connection is made. The strict fixture covers
        the controller path, which is where a hidden download would live.
        """
        from fastapi.testclient import TestClient

        from satquery.api.main import app

        client = TestClient(app)
        with msi_6band.open("rb") as fh:
            response = client.post(
                "/runs",
                data={"query": "Classify the land cover."},
                files={"images": ("msi.tif", fh, "image/tiff")},
            )
        assert response.status_code == 200
        assert response.json()["answer"]


class TestNoNetworkImports:
    def test_runtime_package_does_not_import_a_network_client(self):
        """A network client on the import path is a latent cold-boot failure.

        Importing `requests` does not connect, but a module that imports it at
        runtime is one line away from doing so, and the failure would appear
        on the venue laptop rather than here.
        """
        import subprocess
        import sys

        code = (
            "import sys; import satquery.controller.pipeline; "
            "import satquery.api.main; "
            "bad = [m for m in ('requests', 'httpx', 'urllib3', 'aiohttp') "
            "      if m in sys.modules]; "
            "print(','.join(bad))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        loaded = [m for m in out.stdout.strip().split(",") if m]
        # urllib3 arrives transitively through the FastAPI test stack in some
        # environments; the assertion names what is present so a NEW one is
        # visible rather than silently accepted.
        assert set(loaded) <= {"urllib3"}, f"network clients imported: {loaded}"
