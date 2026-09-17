from pathlib import Path
from typing import Any

from satquery.contracts.tool_result import ToolResult, ToolPayload, Artifact
from satquery.contracts.input_manifest import InputManifest
from satquery.tools.base import ToolProtocol


# Every stub answer carries this, and every stub emits the warning below.
#
# The registry falls back to a stub whenever a learned tool is unavailable -
# no checkpoint, no environment variable, or a checkpoint that cannot be read.
# That is the design, and it is what keeps CI green without torch. What was
# wrong is that the stubs' *text* was indistinguishable from a real answer:
# `RSVQAStub` returned "fake answer string" at 0.85 confidence, and
# `ChangeCaptionStub` returned "A new building was constructed in the center."
# - a plausible, specific, entirely fabricated finding, recorded as the golden
# answer to the PS's own "what changed and where" query.
#
# Found on 2026-09-01 while testing a real image: with the Track B adapter
# destroyed (docs/00 section 3.6 L31b), a VQA query answered "fake answer
# string" with confidence 0.8973 HIGH and nothing in the answer said why.
#
# The marker is a prefix rather than a replacement so the payload shape and
# every downstream consumer stay unchanged; the warning is what a trace reader
# and the report page pick up.
# A stub measures nothing, so it reports nothing to measure.
#
# The eight learned-tool stubs used to report confidences of 0.80-0.95 with
# `confidence_method="threshold_rule"` - indistinguishable, to the combiner,
# from a real head's score. `Executor` takes the MINIMUM over the learned
# tools that ran, so a stubbed VQA answer reached the user as
# "0.9473 HIGH" beside the text "[STUB - no model loaded]".
#
# The fix is not a special case in the combiner: it is to stop the stub
# feeding it a fabricated number. `geometric_mean` already collapses to zero
# when any component is zero - deliberately, so "a confident model on a
# corrupt input must not score 0.66" - so a stub reporting 0.0 yields a final
# score of 0.0 and a LOW band through the ordinary arithmetic, with no
# override, no new band value, and no change to calibration.
#
# `confidence_method="stub"` is what makes it auditable: it is not in
# CALIBRATABLE_CONFIDENCE_METHODS, so the calibration path is untouched, and
# the trace names the reason the component is zero.
STUB_CONFIDENCE = 0.0
STUB_CONFIDENCE_METHOD = "stub"

STUB_NOTICE = "[STUB - no model loaded]"
STUB_WARNING = (
    "this answer came from a placeholder, not a trained model: the learned "
    "tool was unavailable and the registry fell back to its stub. The text is "
    "fixed and carries no information about these inputs"
)


def stub_text(text: str) -> str:
    """Prefix a stub's user-visible text so it cannot be read as a result."""
    return f"{STUB_NOTICE} {text}"


class StubPayload(ToolPayload):
    data: dict[str, Any]


class RSVQAStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={
            "answer": stub_text("no answer - the VQA model is not loaded"),
        })
        return ToolResult(
            tool="rs_vqa",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[],
            confidence=STUB_CONFIDENCE,
            confidence_method=STUB_CONFIDENCE_METHOD,
            model_card="stub_vqa_model",
            runtime_ms=120,
            warnings=[STUB_WARNING]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class CaptionStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={
            "caption": stub_text("no caption - the captioning model is not loaded"),
        })
        return ToolResult(
            tool="caption",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[],
            confidence=STUB_CONFIDENCE,
            confidence_method=STUB_CONFIDENCE_METHOD,
            model_card="stub_caption_model",
            runtime_ms=150,
            warnings=[STUB_WARNING]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class GroundingStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={
            "bounding_boxes": [{"xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}],
        })
        artifact = Artifact(
            key="grounding_bboxes",
            kind="geojson",
            path=Path("/fake/grounding.geojson"),
            description="Fake bounding boxes"
        )
        return ToolResult(
            tool="grounding",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[artifact],
            confidence=STUB_CONFIDENCE,
            confidence_method=STUB_CONFIDENCE_METHOD,
            model_card="stub_grounding_model",
            runtime_ms=100,
            warnings=[STUB_WARNING]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class LandcoverStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={"classes": ["forest", "water"]})
        artifact = Artifact(
            key="landcover_map",
            kind="geotiff",
            path=Path("/fake/landcover.tif"),
            description="Fake landcover map"
        )
        return ToolResult(
            tool="landcover",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[artifact],
            confidence=STUB_CONFIDENCE,
            confidence_method=STUB_CONFIDENCE_METHOD,
            model_card="stub_landcover_model",
            runtime_ms=200,
            warnings=[STUB_WARNING]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class OptSARFusionStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={"fusion_status": "success"})
        artifact = Artifact(
            key="fused_image",
            kind="geotiff",
            path=Path("/fake/optsar_fusion.tif"),
            description="Fake optical-SAR fusion"
        )
        return ToolResult(
            tool="optsar_fusion",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[artifact],
            confidence=STUB_CONFIDENCE,
            confidence_method=STUB_CONFIDENCE_METHOD,
            model_card="stub_fusion_model",
            runtime_ms=500,
            warnings=[STUB_WARNING]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class ChangeMaskStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={"changed_pixels": 420})
        artifact = Artifact(
            key="change_mask",
            kind="png",
            path=Path("/fake/change_mask.png"),
            description="Fake change mask"
        )
        return ToolResult(
            tool="change_mask",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[artifact],
            confidence=STUB_CONFIDENCE,
            confidence_method=STUB_CONFIDENCE_METHOD,
            model_card="stub_change_mask_model",
            runtime_ms=300,
            warnings=[STUB_WARNING]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class ChangeCaptionStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={
            "caption": stub_text("no change description - the model is not loaded"),
        })
        return ToolResult(
            tool="change_caption",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[],
            confidence=STUB_CONFIDENCE,
            confidence_method=STUB_CONFIDENCE_METHOD,
            model_card="stub_change_caption_model",
            runtime_ms=180,
            warnings=[STUB_WARNING]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class ChangeVQAStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={
            "answer": stub_text("no answer - the change-VQA model is not loaded"),
        })
        return ToolResult(
            tool="change_vqa",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[],
            confidence=STUB_CONFIDENCE,
            confidence_method=STUB_CONFIDENCE_METHOD,
            model_card="stub_change_vqa_model",
            runtime_ms=170,
            warnings=[STUB_WARNING]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


class IndexEngineStub(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        payload = StubPayload(data={
            "NDVI": 0.65,
            "NDWI": -0.2,
        })
        return ToolResult(
            tool="index_engine",
            version="0.1.0-stub",
            payload=payload,
            artifacts=[],
            confidence=1.0,
            confidence_method="deterministic",
            model_card="math",
            runtime_ms=10,
            warnings=[STUB_WARNING]
        )

    def run_batch(self, manifests: list[InputManifest], params: dict[str, Any]) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]


# index_engine_v1 is the first tool promoted from stub to real (plan task 1.2).
# Imported here rather than at module top to keep the stub definitions above
# free of any dependency on the real implementation.
from satquery.tools.index_engine import IndexEngine  # noqa: E402
from satquery.tools.change_vqa import ChangeVQATemplate  # noqa: E402


def _fusion_tool():
    """Real triad when a checkpoint is configured, else the stub."""
    from satquery.tools.optsar_fusion import OptSARFusionTool, is_available

    return OptSARFusionTool() if is_available()[0] else OptSARFusionStub()


def _change_mask_tool():
    """Real detector when a checkpoint is configured, else the stub."""
    from satquery.tools.change_mask import ChangeMaskTool, is_available

    return ChangeMaskTool() if is_available()[0] else ChangeMaskStub()

# rs_vqa_v1 uses the real QLoRA adapter only when SATQUERY_VQA_BASE and
# SATQUERY_VQA_ADAPTER are both set and the GPU stack is importable.
# Otherwise the stub stays, so CI and GPU-less machines keep a green suite
# rather than half-loading a model and answering badly.
def _vqa_tool():
    from satquery.tools.rs_vqa import RSVQATool, is_available

    available, _reason = is_available()
    return RSVQATool() if available else RSVQAStub()


def _landcover_tool():
    """Real Track A head when SATQUERY_LANDCOVER is set, else the stub.

    Wired with selective prediction rather than a 0.5 threshold - task 3.6
    measured that this head is worse than trivial at 0.5.
    """
    from satquery.tools.landcover import LandcoverTool, is_available

    return LandcoverTool() if is_available()[0] else LandcoverStub()


def _caption_tool():
    """Real captioner when SATQUERY_CAPTION is set, else the stub (task 2.8)."""
    from satquery.tools.caption import CaptionTool, is_available

    return CaptionTool() if is_available()[0] else CaptionStub()


def _grounding_tool():
    """Real grounder when SATQUERY_GROUNDING is set, else the stub (2.7)."""
    from satquery.tools.grounding import GroundingTool, is_available

    return GroundingTool() if is_available()[0] else GroundingStub()


def _change_vqa_tool():
    """Semantic change path when SATQUERY_CHANGE_VQA is set (task 2.6).

    The template path is not a stub - it is a real deterministic answerer - so
    the fallback here is a working tool rather than a placeholder. Setting the
    variable adds the semantic path behind it, which is what makes CDVQA's
    per-class questions answerable at all.
    """
    from satquery.tools.change_vqa import ChangeVQASemantic, semantic_available

    return ChangeVQASemantic() if semantic_available()[0] else ChangeVQATemplate()


def _change_caption_tool():
    """Real change captioner when SATQUERY_CHANGE_CAPTION is set (task 2.5)."""
    from satquery.tools.change_caption import ChangeCaptionTool, is_available

    return ChangeCaptionTool() if is_available()[0] else ChangeCaptionStub()


REGISTRY = {
    "rs_vqa_v1": _vqa_tool(),
    "caption_v1": _caption_tool(),
    "grounding_v1": _grounding_tool(),
    "landcover_v1": _landcover_tool(),
    "optsar_fusion_v1": _fusion_tool(),
    "change_mask_v1": _change_mask_tool(),
    "change_caption_v1": _change_caption_tool(),
    "change_vqa_v1": _change_vqa_tool(),
    "index_engine_v1": IndexEngine(),
}
