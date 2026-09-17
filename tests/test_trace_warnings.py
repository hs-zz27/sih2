"""Executor warnings reach the trace.

The executor collected warnings from its first version - a tool that failed
and was skipped, a stub standing in for a model, a selective head that
asserted nothing - and then returned a Trace without them. Every append was
write-only, so a degraded run and a healthy one produced traces that differed
only in details a reader would have to already suspect. The auditable
execution summary (PS M8) is exactly where that difference belongs.
"""

from __future__ import annotations

import pytest

from satquery.contracts.trace import Trace
from satquery.controller.pipeline import Controller
from satquery.tools.stubs import REGISTRY, STUB_WARNING


@pytest.fixture
def controller():
    return Controller()


def _explode(monkeypatch, name: str) -> None:
    tool = REGISTRY[name]

    class Exploding:
        def __init__(self):
            self.name = getattr(tool, "name", name)
            self.version = getattr(tool, "version", "0.0.0")

        def run(self, manifest, params):
            raise RuntimeError("simulated CUDA out of memory")

        def run_batch(self, manifests, params):
            raise RuntimeError("simulated CUDA out of memory")

    monkeypatch.setitem(REGISTRY, name, Exploding())


def test_a_skipped_tool_is_recorded(controller, monkeypatch, msi_6band):
    """landcover_v1 declares a fallback for captioning, so its failure degrades."""
    _explode(monkeypatch, "landcover_v1")
    trace = controller.run([msi_6band], "Describe this image.")
    assert str(trace.routing.selected_task) == "SINGLE_CAPTION"
    assert any(
        "landcover_v1" in w and "continuing degraded" in w for w in trace.warnings
    ), trace.warnings


def test_an_aborting_tool_is_recorded(controller, monkeypatch, msi_6band):
    _explode(monkeypatch, "index_engine_v1")
    trace = controller.run([msi_6band], "Classify the land cover.")
    assert trace.abstained
    assert any("index_engine_v1" in w for w in trace.warnings), trace.warnings


def test_a_clean_input_with_a_healthy_plan_has_no_failure_warning(controller, msi_6band):
    trace = controller.run([msi_6band], "Describe this image.")
    assert not any("failed" in w for w in trace.warnings), trace.warnings


def test_shared_notices_are_not_repeated(controller, msi_6band):
    """Each stub appends STUB_WARNING; the trace should say it once."""
    trace = controller.run([msi_6band], "Describe this image.")
    assert len(trace.warnings) == len(set(trace.warnings))
    if any(s.confidence_method == "stub" for s in trace.execution):
        assert trace.warnings.count(STUB_WARNING) == 1


def test_traces_stored_before_the_field_still_load(controller, msi_6band):
    stored = controller.run([msi_6band], "Describe this image.").model_dump()
    stored.pop("warnings")
    assert Trace.model_validate(stored).warnings == []
