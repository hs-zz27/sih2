"""`grounding_v1` backed by the trained referring grounder (plan task 2.7).

Task 2.7 trained the grounder and reported mIoU 0.1405 / Acc@0.5 0.0762 on
DIOR-RSVG, but no tool wired it into the pipeline - the registry kept the
stub, so the model was unreachable from a query. This is that wiring.

Opt-in via `SATQUERY_GROUNDING`, the same pattern the other learned tools use.

## Read the metric before trusting a box

Published DIOR-RSVG results reach roughly 70-80% Acc@0.5. **This model reaches
7.6%**, and the cause is architectural rather than mysterious: it
global-average-pools the visual feature map before regressing the box, which
discards exactly the spatial information localisation depends on, so it can
only learn an "average" box. Task 2.7 recorded that, and wiring the model in
does not change it.

The tool therefore reports the box **with the model's own weak confidence**,
and the three-component combiner and the abstention policy are what stop a
near-random box being presented as a finding. It is wired because "the
pipeline cannot reach a model we trained" is a worse state than "the pipeline
reaches a model whose limitations are measured and recorded" - not because
the box is good.

Boxes are emitted in pixel coordinates. `satquery/report/evidence_pack.py`
projects them to GeoJSON using the image transform, which is where task 2.7's
"boxes exported as GeoJSON" is satisfied.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from satquery.contracts.input_manifest import InputManifest
from satquery.contracts.tool_result import ToolPayload, ToolResult
from satquery.tools.base import ToolProtocol
from satquery.tools.provenance import record
from satquery.tools.sidecars import readable_json
from satquery.tools.imaging import to_rgb_preview

TOOL_NAME = "grounding"
TOOL_VERSION = "1.0.0"
ENV_CHECKPOINT = "SATQUERY_GROUNDING"


class GroundingPayload(ToolPayload):
    data: dict[str, Any]


def is_available() -> tuple[bool, str]:
    path = os.getenv(ENV_CHECKPOINT)
    if not path:
        return False, f"{ENV_CHECKPOINT} is not set"
    if not Path(path).exists():
        return False, f"checkpoint not found: {path}"
    # Checkpoint contents before environment - see caption.py.
    # Readable, not merely present - see caption.py. Restoring this
    # checkpoint from a shadow copy returned vocab.json as 1,106 bytes of NUL,
    # and this check answered "ready" until it was taught to parse the file.
    ok, reason = readable_json(Path(path) / "vocab.json", expect=dict)
    if not ok:
        return False, reason
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch is not installed"
    return True, "ready"


class _Handle:
    _instance: "_Handle | None" = None
    _lock = threading.Lock()

    def __init__(self, checkpoint: Path):
        import torch

        from training.common.checkpointing import (
            find_latest_checkpoint,
            load_checkpoint,
            safe_torch_load,
        )
        from training.train_grounding import build_model

        self.vocab: dict[str, int] = json.loads(
            (checkpoint / "vocab.json").read_text(encoding="utf-8")
        )
        latest = find_latest_checkpoint(checkpoint) or checkpoint
        payload = safe_torch_load(latest)
        extra = payload.get("extra") or {}
        dim = extra.get("dim", 128)
        # Which architecture wrote these weights. A checkpoint from before
        # Phase 5 has no `arch` field, so the default is v1 and every
        # existing checkpoint rebuilds exactly as it always did. Guessing
        # instead would load v1 weights into a v2 graph and fail on a key
        # mismatch that says nothing about the cause.
        # `pretrained` selects a different backbone, so it must be known
        # before the graph is built - exactly like `arch`. Two checkpoints
        # were written before it was recorded; for those it is inferred from
        # the weights: only the pretrained variant has a `proj.` projection
        # after its 2048-wide ResNet-50 (the from-scratch trunk is Identity
        # there). The same pattern `landcover` uses to infer `has_gsd`.
        state = payload.get("model_state_dict", {})
        pretrained = extra.get("pretrained")
        if pretrained is None:
            pretrained = any(k.startswith("proj.") for k in state)
        model = build_model(vocab_size=len(self.vocab), dim=dim,
                            arch=extra.get("arch", "v1"), pretrained=pretrained)
        load_checkpoint(latest, model, map_location="cpu")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.torch = torch
        self.path = str(latest)
        # The bytes that are now in memory, hashed once per process, so
        # `Trace.weights_hashes` names the weights that produced the answer
        # rather than being empty. See satquery/tools/provenance.py.
        record("grounding_v1", latest)

    @classmethod
    def get(cls, checkpoint: Path) -> "_Handle":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(checkpoint)
        return cls._instance


class GroundingTool(ToolProtocol):
    name = TOOL_NAME
    version = TOOL_VERSION

    def run(self, manifest: InputManifest, params: dict) -> ToolResult:
        started = time.perf_counter()
        handle = _Handle.get(Path(os.environ[ENV_CHECKPOINT]))
        torch = handle.torch

        from training.train_grounding import IMAGE_SIZE, encode_text

        phrase = str(params.get("_query") or "").strip()
        warnings: list[str] = []
        if not phrase:
            phrase = "the main object"
            warnings.append("no referring expression supplied; used a generic one")

        meta = manifest.images[0]
        image, _ = to_rgb_preview(meta, max_edge=IMAGE_SIZE)
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        array = np.asarray(image, dtype="float32").transpose(2, 0, 1) / 255.0

        tokens = encode_text(phrase, handle.vocab)
        with torch.no_grad():
            box = handle.model(
                torch.from_numpy(array).unsqueeze(0).to(handle.device),
                torch.from_numpy(np.asarray(tokens)).unsqueeze(0).to(handle.device),
            )[0].tolist()

        # The head emits normalised (cx, cy, w, h) through a sigmoid, so the
        # box is inside the frame by construction. Converted to pixel corners
        # against the ACTUAL image size, not the 224px model input, or every
        # exported box would be wrong on any scene that is not square.
        cx, cy, w, h = box
        width, height = meta.width, meta.height
        x0 = max(0.0, (cx - w / 2) * width)
        y0 = max(0.0, (cy - h / 2) * height)
        x1 = min(float(width), (cx + w / 2) * width)
        y1 = min(float(height), (cy + h / 2) * height)

        # There is no objectness head, so there is no learned score to report.
        # Fabricating one would be worse than reporting the measured ceiling:
        # this is the model's Acc@0.5 on its own test split, which is what a
        # box from it is actually worth.
        confidence = 0.0762

        return ToolResult(
            tool=TOOL_NAME,
            version=TOOL_VERSION,
            payload=GroundingPayload(
                data={
                    "phrase": phrase,
                    "bounding_boxes": [
                        {
                            "x0": round(x0, 2), "y0": round(y0, 2),
                            "x1": round(x1, 2), "y1": round(y1, 2),
                            "label": phrase,
                        }
                    ],
                    "normalised_cxcywh": [round(v, 6) for v in box],
                    "image_size": [width, height],
                }
            ),
            artifacts=[],
            confidence=confidence,
            # Not a per-prediction score: a fixed dataset-level accuracy. Named
            # so nobody mistakes it for the model's own certainty.
            confidence_method="threshold_rule",
            model_card=f"referring grounder ({Path(handle.path).name})",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=warnings + [
                "grounding Acc@0.5 is 0.0762 on DIOR-RSVG against ~70-80% "
                "published; the head pools away spatial detail before "
                "regressing the box, so this localisation is weak by design"
            ],
        )

    def run_batch(self, manifests, params):
        return [self.run(m, params) for m in manifests]
