"""The overlay lifecycle: a run exists before it can be queried.

`POST /runs/stream` emits `run_started` with the run id as soon as the run is
created, but the trace is only written when the run finishes. Until then
`GET /runs/{id}/overlays` answers 404 - `list_overlays` requires
`record["trace"]`, and that is correct: there are no artifacts yet to list.

The frontend mounted `MapView` on `run_started` and fetched `/overlays`
immediately, so a healthy run painted "Error: HTTP 404" over itself and then
completed normally.

These tests fix both halves of the invariant:

    run incomplete -> no overlays fetch
    run complete   -> overlays fetch allowed

The backend half is exercised against the real API. The frontend half is
asserted against the source, in the manner `tests/test_packaging.py` already
uses for the Dockerfiles - this repository has no JavaScript test runner, and
adding one for two assertions would be a heavier change than the fix.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from satquery.api import main as api_main

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "frontend" / "app" / "page.tsx"
MAPVIEW = ROOT / "frontend" / "app" / "MapView.tsx"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "_STORE", None, raising=False)
    monkeypatch.setenv("SATQUERY_RUN_DB", str(tmp_path / "runs.db"))
    with TestClient(api_main.app) as c:
        yield c


class TestOverlaysBeforeCompletion:
    """A created-but-unfinished run has no overlays to list."""

    def test_a_run_that_does_not_exist_is_404(self, client):
        response = client.get("/runs/does-not-exist/overlays")
        assert response.status_code == 404

    def test_a_created_run_without_a_trace_is_404(self, client):
        # Exactly the window the frontend was racing: the store row exists,
        # the trace does not.
        api_main.get_store().create("run_in_flight", "Describe this image.")
        response = client.get("/runs/run_in_flight/overlays")
        assert response.status_code == 404

    def test_the_404_is_the_backend_behaving_correctly(self, client):
        """Pinned deliberately: the fix must not be to make this a 200.

        Returning an empty overlay list for a run still in flight would be
        indistinguishable from a completed run that produced no georeferenced
        artifacts, and the map would silently show nothing instead of waiting.
        """
        api_main.get_store().create("run_in_flight", "Describe this image.")
        assert client.get("/runs/run_in_flight/overlays").status_code == 404


class TestOverlaysAfterCompletion:
    """A completed run answers, and the frontend is allowed to ask."""

    def test_completed_run_lists_overlays(self, client, msi_6band):
        with msi_6band.open("rb") as fh:
            created = client.post(
                "/runs",
                data={"query": "Classify the land cover."},
                files={"images": ("msi.tif", fh, "image/tiff")},
            )
        assert created.status_code == 200
        run_id = created.json()["run_id"]

        response = client.get(f"/runs/{run_id}/overlays")
        assert response.status_code == 200
        body = response.json()
        assert body["run_id"] == run_id
        assert isinstance(body["overlays"], list)

    def test_every_listed_overlay_is_fetchable(self, client, msi_6band):
        with msi_6band.open("rb") as fh:
            run_id = client.post(
                "/runs",
                data={"query": "Classify the land cover."},
                files={"images": ("msi.tif", fh, "image/tiff")},
            ).json()["run_id"]

        overlays = client.get(f"/runs/{run_id}/overlays").json()["overlays"]
        available = [o for o in overlays if o["available"]]
        assert available, "a land-cover run should produce index rasters"
        for overlay in available:
            drawn = client.get(f"/runs/{run_id}/overlay/{overlay['key']}")
            assert drawn.status_code == 200, overlay["key"]
            assert drawn.headers.get("X-Extent"), "the client needs the extent"


class TestStreamCompletesBeforeItAnnounces:
    """The ordering the frontend fix depends on.

    `store.complete()` is called before the `complete` event is emitted, so a
    client that waits for that event can query the run immediately. If this
    ever inverted, gating on `complete` would not be enough.
    """

    def test_store_complete_precedes_the_complete_event(self):
        source = (ROOT / "satquery" / "api" / "main.py").read_text(encoding="utf-8")
        persisted = source.index("store.complete(run_id, trace)")
        announced = source.index('events.put(("complete"')
        assert persisted < announced


class TestFrontendDoesNotAskEarly:
    """The frontend half of the invariant, asserted against the source."""

    def test_mapview_accepts_a_readiness_gate(self):
        assert "ready" in MAPVIEW.read_text(encoding="utf-8")

    def test_the_overlays_fetch_is_gated(self):
        text = MAPVIEW.read_text(encoding="utf-8")
        fetch_at = text.index("/overlays`)")
        guard_at = text.index("if (!ready) return;")
        effect_at = text.rindex("useEffect", 0, guard_at)
        assert effect_at < guard_at < fetch_at, (
            "the readiness guard must precede the overlays fetch in the same effect"
        )

    def test_the_gate_is_in_the_effect_dependencies(self):
        text = MAPVIEW.read_text(encoding="utf-8")
        assert "}, [runId, ready]);" in text, (
            "the effect must re-run when the run completes"
        )

    def test_page_only_marks_ready_on_the_complete_event(self):
        text = PAGE.read_text(encoding="utf-8")
        # Matched on the element rather than on one exact line: this asserted a
        # single-line `<MapView runId={runId} ready={runComplete} />` and broke
        # the moment the map gained a `footprint` prop and wrapped across
        # lines. The invariant is that the map is gated on `runComplete`, not
        # how many props happen to sit beside it.
        mounts = [
            text[at : text.index("/>", at)]
            for at in range(len(text))
            if text.startswith("<MapView", at)
        ]
        assert mounts, "the query page must still mount MapView"
        assert any("ready={runComplete}" in mount for mount in mounts), (
            "MapView must be gated on runComplete, so overlays are not "
            "requested before the trace is persisted"
        )
        # setRunComplete(true) must sit inside the `complete` branch, never in
        # the run_started branch.
        complete_branch = text.index("event.name === 'complete'")
        started_branch = text.index("event.name === 'run_started'")
        set_true = text.index("setRunComplete(true)")
        assert complete_branch < set_true
        assert not (started_branch < set_true < complete_branch)

    def test_a_new_run_clears_the_previous_readiness(self):
        assert "setRunComplete(false);" in PAGE.read_text(encoding="utf-8")

    def test_genuine_errors_are_still_surfaced(self):
        # The fix must not swallow failures: the catch that sets the error
        # state has to survive.
        text = MAPVIEW.read_text(encoding="utf-8")
        assert "setError(String(e))" in text
        assert "HTTP ${r.status}" in text


class TestConfidenceIsShownAsAScore:
    """`calibration.py` reports "score is not a calibratable probability"."""

    def test_the_live_page_does_not_render_a_percentage(self):
        text = PAGE.read_text(encoding="utf-8")
        assert not re.search(r"confidence\.final\s*\*\s*100", text), (
            "rendering the score as a percentage asserts a calibrated probability"
        )

    def test_the_live_page_renders_the_score(self):
        assert "confidence.final.toFixed(2)" in PAGE.read_text(encoding="utf-8")

    def test_the_permalink_page_already_agreed(self):
        # It always showed a score; this test pins the two pages together so
        # they cannot drift apart again.
        permalink = ROOT / "frontend" / "app" / "runs" / "[runId]" / "page.tsx"
        text = permalink.read_text(encoding="utf-8")
        assert not re.search(r"confidence\.final\s*\*\s*100", text)

    def test_the_backend_still_calls_it_uncalibrated(self):
        """The UI change is only honest while the backend says this."""
        text = (ROOT / "satquery" / "controller" / "calibration.py").read_text(
            encoding="utf-8"
        )
        assert "uncalibrated (score is not a calibratable probability)" in text
