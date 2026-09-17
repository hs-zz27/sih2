"""Readiness reports what the registry serves, so a demo is not recorded on stubs."""

from __future__ import annotations

from fastapi.testclient import TestClient

from satquery.api.main import app
from satquery.readiness import tool_readiness


class _FakeTool:
    pass


def _registry(**classes):
    """A registry whose entries are instances of classes with the given names."""
    return {tool: type(cls_name, (_FakeTool,), {})() for tool, cls_name in classes.items()}


LEARNED = dict(
    rs_vqa_v1="RSVQATool", caption_v1="CaptionTool", grounding_v1="GroundingTool",
    landcover_v1="LandcoverTool", optsar_fusion_v1="OptSARFusionTool",
    change_mask_v1="ChangeMaskTool", change_caption_v1="ChangeCaptionTool",
    change_vqa_v1="ChangeVQASemantic", index_engine_v1="IndexEngine",
)


def test_all_models_live_is_demo_ready():
    r = tool_readiness(_registry(**LEARNED))
    assert r["demo_ready"] is True
    assert r["stubs"] == []
    assert (r["learned_live"], r["learned_total"]) == (8, 8)


def test_one_stub_blocks_the_demo_and_names_itself():
    r = tool_readiness(_registry(**{**LEARNED, "caption_v1": "CaptionStub"}))
    assert r["demo_ready"] is False
    assert r["stubs"] == ["caption_v1"]
    caption = next(t for t in r["tools"] if t["tool"] == "caption_v1")
    assert caption["status"] == "stub"
    assert caption["reason"]  # the availability check's own explanation


def test_change_vqa_template_is_a_real_fallback_not_a_stub():
    r = tool_readiness(_registry(**{**LEARNED, "change_vqa_v1": "ChangeVQATemplate"}))
    assert r["demo_ready"] is True
    assert r["learned_live"] == 7
    status = {t["tool"]: t["status"] for t in r["tools"]}
    assert status["change_vqa_v1"] == "deterministic"
    assert status["index_engine_v1"] == "deterministic"


def test_default_registry_without_checkpoints_is_not_ready():
    """The CI and laptop condition: no checkpoints configured."""
    r = tool_readiness()
    if r["stubs"]:
        assert r["demo_ready"] is False
        for t in r["tools"]:
            if t["status"] == "stub":
                assert t["reason"] and t["reason"] != "ready"


def test_endpoint_matches_the_module():
    body = TestClient(app).get("/readiness").json()
    assert set(body) >= {"learned_live", "learned_total", "stubs", "demo_ready", "gpu", "tools"}
    assert len(body["tools"]) == 9
