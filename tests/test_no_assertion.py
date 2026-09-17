"""A selective head that asserts nothing must not veto the answer.

The defect, measured 2026-09-12 on the demo bundle with the learned tools
enabled: `landcover_v1` asserted no class on the Cartosat scenes - correctly,
they are 4-band 1.6 m imagery far outside its BigEarthNet training - and
reported confidence 0.0. The executor takes the minimum over learned tools,
so 0.0 became the model confidence for the whole SINGLE_CAPTION plan, and
four of nine beats abstained with "the model itself was unconfident" while
`caption_v1` had answered at 0.56 and `index_engine_v1` - the matrix's declared
fallback for landcover - had already run. The plan was answerable and said not.

v1 passed those beats only because it asserted classes at 0.98 on imagery it
had never seen. Overconfidence was masking the defect, not the absence of it.

The fix gives "I made no claim" its own `confidence_method`, `no_assertion`,
which the executor excludes from the minimum. The guard that keeps that
honest: a plan in which NO learned tool scored is capped exactly as a stubbed
one is, because the combined number would otherwise describe input quality
alone while reading as a model result. (The executor also appends a warning
naming the silent tool, but its `warnings` list has never been written into
the trace - every append in executor.py is write-only - so that is not
asserted here.)
"""

from __future__ import annotations

import pytest

from satquery.contracts.tool_result import ToolResult
from satquery.tools.stubs import REGISTRY, StubPayload


class _Silent:
    """A landcover head whose every class fell in its abstention band."""

    name = "landcover"
    version = "test-silent"

    def run(self, manifest, params):
        return ToolResult(
            tool="landcover", version=self.version,
            payload=StubPayload(data={"asserted": [], "abstained": [], "denied": []}),
            artifacts=[], confidence=0.0, confidence_method="no_assertion",
            model_card="test", runtime_ms=1,
            warnings=["asserted nothing at the measured threshold"],
        )

    def run_batch(self, manifests, params):
        return [self.run(m, params) for m in manifests]


class _Confident:
    """A captioner that answered with a real, calibratable score."""

    name = "caption"
    version = "test-confident"

    def __init__(self, confidence: float = 0.56):
        self.confidence = confidence

    def run(self, manifest, params):
        return ToolResult(
            tool="caption", version=self.version,
            payload=StubPayload(data={"caption": "an airport beside farmland"}),
            artifacts=[], confidence=self.confidence, confidence_method="logprob",
            model_card="test", runtime_ms=1, warnings=[],
        )

    def run_batch(self, manifests, params):
        return [self.run(m, params) for m in manifests]


def _run(monkeypatch, msi_6band, landcover, caption):
    from satquery.controller.pipeline import Controller

    monkeypatch.setitem(REGISTRY, "landcover_v1", landcover)
    monkeypatch.setitem(REGISTRY, "caption_v1", caption)
    return Controller().run([msi_6band], "Describe this scene.")


def test_a_silent_head_does_not_veto_a_confident_answer(monkeypatch, msi_6band):
    trace = _run(monkeypatch, msi_6band, _Silent(), _Confident(0.56))
    assert trace.routing.selected_task == "SINGLE_CAPTION"
    assert trace.abstained is False, trace.answer
    assert trace.confidence.components.model == pytest.approx(0.56, abs=1e-6), \
        "the model component must come from the tool that made a claim"


def test_the_silent_step_is_recorded_as_such(monkeypatch, msi_6band):
    trace = _run(monkeypatch, msi_6band, _Silent(), _Confident())
    step = next(s for s in trace.execution if s.tool == "landcover_v1")
    assert step.confidence_method == "no_assertion"
    assert step.confidence == 0.0


def test_a_plan_where_no_learned_tool_claimed_anything_is_capped(monkeypatch, msi_6band):
    """Silence everywhere is not a HIGH-confidence model result.

    With every learned tool silent the minimum is never set and stays at
    the 1.0 it starts from. Left alone, the combined score would describe
    input quality and physics agreement while reading as a model result -
    the exact situation the stub cap exists for, so the same cap applies.
    """
    from satquery.controller.confidence import STUB_CONFIDENCE_CAP

    class _SilentCaption(_Silent):
        name = "caption"

        def run(self, manifest, params):
            return super().run(manifest, params).model_copy(update={"tool": "caption"})

    trace = _run(monkeypatch, msi_6band, _Silent(), _SilentCaption())
    assert trace.confidence.final <= STUB_CONFIDENCE_CAP
    assert trace.confidence.band == "LOW"


def test_a_genuine_zero_from_a_claiming_tool_still_declines(monkeypatch, msi_6band):
    """`no_assertion` must not become a loophole for a real 0.0.

    A tool that DID make a claim and scored it 0.0 - change_vqa's documented
    "cannot measure" - still sets the minimum and still abstains.
    """
    trace = _run(monkeypatch, msi_6band, _Silent(), _Confident(0.0))
    assert trace.abstained is True


def test_no_assertion_is_not_calibratable():
    from satquery.controller.calibration import CALIBRATABLE_CONFIDENCE_METHODS

    assert "no_assertion" not in CALIBRATABLE_CONFIDENCE_METHODS
