"""Which tools are serving real models right now, and is that enough to demo.

Every learned tool falls back to a stub when its checkpoint is absent. That is
the right behaviour for CI and for a GPU-less laptop, and it is invisible from
the outside: the API answers, the UI renders, and every answer reads
"[STUB - no model loaded]". A demo video recorded in that state looks broken,
and nothing warned the person recording it. The only way to find out was to
run a query and read the answer.

This reports, per tool, what the registry is *actually serving in this
process* - the class it holds, not what the environment suggests - with the
availability check's reason alongside, so "stub" always comes with "because
SATQUERY_CAPTION is not set" or "because torch is not installed".
"""

from __future__ import annotations

from importlib import import_module

# tool -> (module holding its availability check, name of that function)
_CHECKS: dict[str, tuple[str, str]] = {
    "rs_vqa_v1": ("satquery.tools.rs_vqa", "is_available"),
    "caption_v1": ("satquery.tools.caption", "is_available"),
    "grounding_v1": ("satquery.tools.grounding", "is_available"),
    "landcover_v1": ("satquery.tools.landcover", "is_available"),
    "optsar_fusion_v1": ("satquery.tools.optsar_fusion", "is_available"),
    "change_mask_v1": ("satquery.tools.change_mask", "is_available"),
    "change_caption_v1": ("satquery.tools.change_caption", "is_available"),
    "change_vqa_v1": ("satquery.tools.change_vqa", "semantic_available"),
}

# Tools whose fallback is a working deterministic answerer, not a placeholder.
# Their absence degrades a capability; it does not produce stub text.
_DETERMINISTIC_FALLBACK = {"change_vqa_v1"}


def _reason(tool: str) -> str:
    module, fn = _CHECKS[tool]
    try:
        return getattr(import_module(module), fn)()[1]
    except Exception as exc:  # noqa: BLE001 - a broken check is itself a reason
        return f"availability check failed: {type(exc).__name__}: {exc}"


def _gpu() -> dict:
    try:
        import torch
    except ImportError:
        return {"cuda": False, "name": None, "note": "torch is not installed"}
    if not torch.cuda.is_available():
        return {"cuda": False, "name": None, "note": "no CUDA device"}
    return {"cuda": True, "name": torch.cuda.get_device_name(0), "note": ""}


def tool_readiness(registry: dict | None = None) -> dict:
    """Per-tool status plus an overall demo verdict.

    status is one of:
      * ``learned``        - a trained model is being served
      * ``stub``           - a placeholder; answers carry "[STUB ...]"
      * ``deterministic``  - no model involved by design (index engine), or a
                             real rule-based fallback (change-VQA template)
    """
    if registry is None:
        from satquery.tools.stubs import REGISTRY as registry

    tools = []
    for name, instance in registry.items():
        cls = type(instance).__name__
        if name == "index_engine_v1":
            status, reason = "deterministic", "no model by design"
        elif cls.endswith("Stub"):
            status, reason = "stub", _reason(name)
        elif name in _DETERMINISTIC_FALLBACK and "Template" in cls:
            status, reason = "deterministic", _reason(name)
        else:
            status, reason = "learned", "ready"
        tools.append({"tool": name, "status": status, "served_by": cls, "reason": reason})

    learnable = [t for t in tools if t["tool"] in _CHECKS]
    live = [t for t in learnable if t["status"] == "learned"]
    stubs = [t["tool"] for t in tools if t["status"] == "stub"]
    return {
        "learned_live": len(live),
        "learned_total": len(learnable),
        "stubs": stubs,
        # Ready means no answer can come out as placeholder text. The
        # change-VQA template is allowed: it answers, it just answers fewer
        # question types.
        "demo_ready": not stubs,
        "gpu": _gpu(),
        "tools": tools,
    }
