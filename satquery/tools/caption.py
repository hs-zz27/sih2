"""`caption_v1` backed by the trained scene captioner (plan task 2.8).

Task 2.8 trained the captioner and reported BLEU-4 0.2446 on RSICD, but no
tool ever wired it into the pipeline - the registry kept the stub, so the
model was unreachable from a query. This is that wiring.

Activation is explicit and opt-in via `SATQUERY_CAPTION`, matching
`rs_vqa_v1` and `change_mask_v1`: without it the stub stays, so CI and
GPU-less machines keep a green suite rather than half-loading a model.

## The confidence this reports, and why it is not a probability

Mean per-token probability over a greedy decode, reported as
`confidence_method="logprob"`. That is a *fluency* signal, not P(correct) -
the model can be certain of every token in a caption describing the wrong
scene, which is exactly what task 2.8 measured: fluent, plausible
remote-sensing prose with only 13.4% unique captions. It is therefore
excluded from `CALIBRATABLE_CONFIDENCE_METHODS` along with everything else,
and it feeds the confidence combiner as one weak signal.

**The caption is the half of the answer the entailment gate exists for.**
Anything the physics can measure is described deterministically by
`synth/narrative.py` and checked against the indices; this model handles only
genuinely open-ended description, and task 3.5's gate removes any sentence it
produces that contradicts a measured index.
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

TOOL_NAME = "caption"
TOOL_VERSION = "1.0.0"
ENV_CHECKPOINT = "SATQUERY_CAPTION"

def image_size() -> int:
    """The size the weights were fitted at, read from the training module.

    Looked up lazily, NOT imported at module scope. `training.train_caption`
    imports torch at import time, and `stubs.py` builds the registry when it
    is imported - so a module-level import here made the entire package
    unimportable on any machine without torch, which is every CI runner.
    Restating the number instead would let the tool drift from the weights,
    and a size mismatch degrades quality silently: the model still runs, it
    just sees the wrong scale.
    """
    from training.train_caption import IMAGE_SIZE

    return IMAGE_SIZE


class CaptionPayload(ToolPayload):
    data: dict[str, Any]


def is_available() -> tuple[bool, str]:
    path = os.getenv(ENV_CHECKPOINT)
    if not path:
        return False, f"{ENV_CHECKPOINT} is not set"
    if not Path(path).exists():
        return False, f"checkpoint not found: {path}"
    # Checkpoint contents first, environment second: a missing vocab.json is a
    # mistake in the path the operator just supplied, and naming it is more
    # useful than "torch is not installed" when both are true.
    # Readable, not merely present. A vocab.json that exists but cannot be
    # parsed made this function answer "ready" and the loader raise
    # JSONDecodeError - the tool reported available and then failed, which is
    # exactly what the registry's stub fallback exists to prevent. Measured
    # 2026-08-31, when a shadow-copy restore returned this file as 28,130
    # bytes of NUL. The vocabulary is built from the training captions and
    # saved beside the weights; without it the token ids decode to nothing.
    ok, reason = readable_json(Path(path) / "vocab.json", expect=dict)
    if not ok:
        return False, reason
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch is not installed"
    return True, "ready"


class _Handle:
    """Lazily loaded captioner, shared process-wide."""

    _instance: "_Handle | None" = None
    _lock = threading.Lock()

    def __init__(self, checkpoint: Path):
        import torch

        from training.common.checkpointing import (
            find_latest_checkpoint,
            load_checkpoint,
            safe_torch_load,
        )
        from training.train_caption import build_model

        self.vocab: dict[str, int] = json.loads(
            (checkpoint / "vocab.json").read_text(encoding="utf-8")
        )
        self.inverse = {i: w for w, i in self.vocab.items()}

        latest = find_latest_checkpoint(checkpoint) or checkpoint
        payload = safe_torch_load(latest)
        extra = payload.get("extra") or {}
        dim = extra.get("dim", 192)
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
        record("caption_v1", latest)

    @classmethod
    def get(cls, checkpoint: Path) -> "_Handle":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(checkpoint)
        return cls._instance


def _image_array(meta, size: int) -> np.ndarray:
    """RGB array in the layout the captioner trained on.

    Routed through `to_rgb_preview`, the same function the VQA tool and the
    browser preview endpoint use, so what the model sees is what a user is
    shown - band selection and stretch included.
    """
    image, _ = to_rgb_preview(meta, max_edge=size)
    image = image.resize((size, size))
    return np.asarray(image, dtype="float32").transpose(2, 0, 1) / 255.0


class CaptionTool(ToolProtocol):
    name = TOOL_NAME
    version = TOOL_VERSION

    def run(self, manifest: InputManifest, params: dict) -> ToolResult:
        started = time.perf_counter()
        checkpoint = Path(os.environ[ENV_CHECKPOINT])
        handle = _Handle.get(checkpoint)
        torch = handle.torch

        from training.train_change_caption import EOS, PAD

        array = _image_array(manifest.images[0], image_size())
        batch = torch.from_numpy(array).unsqueeze(0).to(handle.device)

        with torch.no_grad():
            tokens = handle.model.generate(batch)[0].tolist()

        words: list[str] = []
        for token in tokens:
            if token in (EOS, PAD):
                break
            word = handle.inverse.get(int(token))
            if word and not word.startswith("<"):
                words.append(word)
        caption = " ".join(words).strip()

        warnings: list[str] = []
        if not caption:
            # Never return an empty string: the executor turns an empty answer
            # into a named abstention, and "the captioner emitted nothing" is
            # more useful to a reader than silence.
            caption = "No caption could be generated for this image."
            warnings.append("captioner produced no tokens")

        confidence = _mean_token_probability(torch, handle, batch, tokens)

        return ToolResult(
            tool=TOOL_NAME,
            version=TOOL_VERSION,
            payload=CaptionPayload(
                data={
                    "caption": caption,
                    "n_tokens": len(words),
                    "vocab_size": len(handle.vocab),
                }
            ),
            artifacts=[],
            confidence=confidence,
            # Fluency, not correctness - see the module docstring.
            confidence_method="logprob",
            model_card=f"scene captioner ({Path(handle.path).name})",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=warnings,
        )

    def run_batch(self, manifests, params):
        return [self.run(m, params) for m in manifests]


def _mean_token_probability(torch, handle, batch, tokens) -> float:
    """Mean probability of the tokens the greedy decode chose.

    Re-runs the model teacher-forced on its own output through `forward`,
    the one entry point every captioner - v1 GRU or v2 transformer - shares:
    `model(image, tokens) -> (B, T, V)` logits, where position i predicts
    token i+1. Feeding `[BOS] + tokens[:-1]` therefore yields, at position i,
    the distribution from which `tokens[i]` was chosen.

    The first version stepped v1's `.vision`, `.gru` and `.embed` by hand.
    That is the same computation, and it broke on every v2 checkpoint at
    inference - after the loader had accepted them - because v2 has none of
    those attributes. Found by the demo bundle on 2026-09-12: four of nine
    beats reported "caption_v1 failed". A tool that depends on a model's
    internals beyond its forward contract is a tool that only works on the
    model it was written against.
    """
    if not tokens:
        return 0.0
    from training.train_change_caption import BOS

    with torch.no_grad():
        prefix = torch.tensor([[BOS] + [int(x) for x in tokens[:-1]]],
                              dtype=torch.long, device=handle.device)
        logits = handle.model(batch, prefix).float()          # (1, T, V)
        step = torch.softmax(logits[0], dim=-1)                # (T, V)
        idx = torch.tensor([int(x) for x in tokens], device=handle.device)
        probs = step[torch.arange(len(tokens), device=handle.device), idx]
    return round(float(probs.mean()), 6)
