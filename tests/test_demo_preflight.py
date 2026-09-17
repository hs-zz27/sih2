"""The pre-flight verdict: stubs, a dead API or a dead UI must be NO-GO."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from demo_preflight import evaluate  # noqa: E402


def ready(demo_ready=True, cuda=True):
    tools = [{"tool": "caption_v1", "status": "learned" if demo_ready else "stub",
              "reason": "ready" if demo_ready else "SATQUERY_CAPTION is not set"}]
    return {"learned_live": 8 if demo_ready else 7, "learned_total": 8,
            "stubs": [] if demo_ready else ["caption_v1"], "demo_ready": demo_ready,
            "gpu": {"cuda": cuda, "name": "RTX 4060" if cuda else None, "note": ""}, "tools": tools}


BASE = dict(health={"version": "0.1"}, web_up=True, free_disk_gb=100.0, ram_gb=32.0, bundle_built=True)


def levels(lines):
    return [level for level, _ in lines]


def test_everything_live_is_go():
    go, lines = evaluate(readiness=ready(), **BASE)
    assert go and "FAIL" not in levels(lines)


def test_a_stub_is_no_go_and_says_why():
    go, lines = evaluate(readiness=ready(demo_ready=False), **BASE)
    assert not go
    assert any("SATQUERY_CAPTION is not set" in m for _, m in lines)


def test_allow_stubs_downgrades_to_warning():
    go, lines = evaluate(readiness=ready(demo_ready=False), allow_stubs=True, **BASE)
    assert go and "WARN" in levels(lines)


def test_api_down_is_no_go():
    go, lines = evaluate(**{**BASE, "health": None}, readiness=None)
    assert not go
    assert any("API not reachable" in m for _, m in lines)


def test_ui_down_is_no_go():
    go, _ = evaluate(readiness=ready(), **{**BASE, "web_up": False})
    assert not go


def test_missing_gpu_low_disk_and_no_bundle_warn_but_do_not_block():
    go, lines = evaluate(readiness=ready(cuda=False),
                         **{**BASE, "free_disk_gb": 5.0, "ram_gb": 8.0, "bundle_built": False})
    assert go
    assert levels(lines).count("WARN") == 4
