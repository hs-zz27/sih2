"""PDF report, model registry and benchmark page (plan task 3.12).

The PDF tests assert what the document SAYS, not that a file was produced.
`export_pdf(compress=False)` writes uncompressed content streams so the text
can be read out of the bytes without adding a PDF parser as a test
dependency - a test that only checked the file was non-empty would pass for a
blank page.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from satquery.controller.pipeline import Controller
from satquery.ingest import ingest
from satquery.jsonsafe import json_safe
from satquery.report.pdf_report import export_pdf, raster_preview
from satquery.report.registry import benchmarks, model_registry

# reportlab is documented as optional, so the tests must behave as if it is:
# skipping here is what makes that claim true rather than aspirational. It is
# in the `dev` extra, so CI does run these.
reportlab = pytest.importorskip("reportlab", reason="reportlab is optional")


def pdf_text(path: Path) -> str:
    raw = path.read_bytes().decode("latin-1", "replace")
    return " ".join(re.findall(r"\((.*?)\)\s*Tj", raw))


@pytest.fixture(scope="module")
def trace(tmp_path_factory):
    from evaluation.scenes import build_configurations

    scenes = build_configurations(tmp_path_factory.mktemp("report"))
    return Controller().run_on_manifest(
        ingest(scenes["SINGLE"]), "Classify the land cover."
    )


@pytest.fixture(scope="module")
def rendered(trace, tmp_path_factory):
    out = tmp_path_factory.mktemp("pdf") / "report.pdf"
    export_pdf(trace, out, compress=False)
    return out, pdf_text(out)


class TestPdfContent:
    def test_the_file_is_a_pdf_with_pages(self, rendered):
        path, _ = rendered
        raw = path.read_bytes()
        assert raw.startswith(b"%PDF-")
        assert b"/Type /Page" in raw or b"/Type/Page" in raw

    def test_the_answer_appears(self, rendered, trace):
        _, text = rendered
        assert trace.answer.split(".")[0][:40] in text

    def test_the_confidence_breakdown_appears(self, rendered):
        _, text = rendered
        for component in ("model", "agreement", "input_quality"):
            assert component in text
        assert "CONFIDENCE" in text

    def test_the_routing_decision_appears(self, rendered, trace):
        _, text = rendered
        assert trace.routing.selected_task in text
        assert "ROUTING" in text

    def test_the_entailment_gate_statistics_appear(self, rendered):
        _, text = rendered
        assert "Entailment gate" in text
        assert "Unverifiable" in text

    def test_index_rasters_are_embedded_as_images(self, rendered):
        """"Renders with maps and indices" is the acceptance criterion."""
        path, _ = rendered
        assert path.read_bytes().count(b"/Image") >= 1

    def test_an_unmeasured_ece_is_printed_as_unmeasured(self, rendered):
        """The sentinel exists so -1.0 is never shown as a measurement."""
        _, text = rendered
        assert "-1.0" not in text
        assert "not measured" in text or "ECE=" in text

    def test_an_abstention_prints_its_resolving_input(self, tmp_path, no_crs_raster):
        trace = Controller().run([no_crs_raster], "Describe this image.")
        out = tmp_path / "abstain.pdf"
        export_pdf(trace, out, compress=False)
        text = pdf_text(out)
        assert "ABSTAINED" in text
        assert "What would resolve it" in text

    def test_a_missing_raster_does_not_lose_the_report(self, tmp_path):
        assert raster_preview(tmp_path / "absent.tif", tmp_path) is None

    def test_artifact_paths_reach_the_trace(self, trace):
        """They were written to disk but never recorded, so nothing could
        find them - which is why the first PDF had no images in it."""
        assert trace.artifact_paths
        assert set(trace.artifact_paths) <= set(trace.artifacts)


class TestModelRegistry:
    @pytest.fixture(scope="class")
    def registry(self):
        return model_registry()

    def test_checkpoints_are_listed(self, registry):
        """Each trained checkpoint directory becomes one registry entry.

        Skips when the tree holds none. This asserted a **precondition of the
        machine** - that somebody's trained weights are on disk - rather than
        a behaviour of the code, and on 2026-08-30 that precondition stopped
        holding: `checkpoints/` was deleted and could not be recovered
        (`docs/00` §3.6 L26). Fabricating a checkpoint to keep the assertion
        green would make the registry page look populated when it is empty,
        which is the opposite of what this page is for. The behaviour that
        matters when there are none is asserted below instead.
        """
        from satquery.report.registry import CHECKPOINT_DIR

        on_disk = [p for p in CHECKPOINT_DIR.glob("*") if p.is_dir()] \
            if CHECKPOINT_DIR.exists() else []
        if not on_disk:
            pytest.skip(
                f"no checkpoint directories under {CHECKPOINT_DIR} - the "
                "registry has nothing to list (docs/00 §3.6 L26)"
            )

        assert registry["checkpoints"]
        for entry in registry["checkpoints"]:
            assert entry["name"] and "metrics" in entry and "training" in entry

    def test_an_empty_checkpoint_tree_yields_an_empty_list_not_an_error(
        self, tmp_path
    ):
        """The state the system is actually in, and it must not be a failure.

        `/models` renders empty rather than 500-ing, and says nothing it
        cannot read off disk.
        """
        from satquery.report.registry import checkpoint_entries

        assert checkpoint_entries(tmp_path) == []
        assert checkpoint_entries(tmp_path / "absent") == []

    def test_metrics_are_json_serialisable(self, registry):
        """Training metrics contain NaN for classes with no positives."""
        import json

        json.dumps(json_safe(registry))

    def test_caveats_travel_with_the_numbers(self, registry):
        """The whole point of the page.

        A registry showing mAP 0.2854 without "official test shard, not
        comparable to the v0 number" is how a reader forms a wrong
        impression of a real measurement.
        """
        named = {c["name"]: c for c in registry["checkpoints"]}
        if "track_a_full_base" in named:
            caveat = named["track_a_full_base"]["caveat"]
            assert caveat and "not comparable" in caveat.lower()

    def test_rejected_calibrations_are_shown_not_hidden(self, registry):
        assert "rejected" in registry["calibration"]


class TestBenchmarkPage:
    @pytest.fixture(scope="class")
    def data(self):
        return benchmarks()

    def test_reports_are_aggregated(self, data):
        assert data["available"]

    def test_missing_reports_are_named_not_dropped(self, data):
        """A page that silently omits an ungenerated report looks complete."""
        assert isinstance(data["missing"], list)
        for entry in data["missing"]:
            assert entry["name"] and entry["expected_at"]

    def test_each_entry_records_the_file_it_came_from(self, data):
        for name, entry in data["available"].items():
            assert entry["source"], name
            assert entry["data"] is not None, name

    def test_it_says_how_to_regenerate(self, data):
        assert data["regenerate_with"]
