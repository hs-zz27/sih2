"""`change_caption_v1`, mask-conditioned (plan task 2.5).

Task 2.5 trained the captioner and reported BLEU-4 0.3063 on changed pairs,
but no tool wired it into the pipeline - the registry kept the stub, so the
model was unreachable from a query. This is that wiring.

Opt-in via `SATQUERY_CHANGE_CAPTION`, matching the other learned tools.

## Mask conditioning is the point, and it constrains this tool

The captioner takes the change **mask** as a third input, not just the two
dates. That is what task 2.5 was for: the captioner starts from *where* the
change is and spends capacity describing it rather than locating it, and the
prose cannot disagree with the mask the system already exported.

So this tool needs a mask. It gets one from `change_mask_v1` when that tool is
configured, and **falls back to a zero mask otherwise** - which is reported in
the warnings, because an unconditioned caption from a mask-conditioned model
is running it outside its training distribution. Silently substituting zeros
would produce fluent text with no stated reason to distrust it.

## Read the metric before trusting the caption

BLEU-4 0.5686 aggregate is the mean of a trivial half and the real task:
LEVIR-CC is ~50/50 changed against unchanged, and the unchanged half is
answered correctly by the single string "there is no difference". Only the
**changed-pair figure of 0.3063** is meaningful, and 51% of the model's output
is the majority string. Task 2.5 recorded that; wiring it in does not change
it.
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

TOOL_NAME = "change_caption"
TOOL_VERSION = "1.0.0"
ENV_CHECKPOINT = "SATQUERY_CHANGE_CAPTION"


class ChangeCaptionPayload(ToolPayload):
    data: dict[str, Any]


def is_available() -> tuple[bool, str]:
    path = os.getenv(ENV_CHECKPOINT)
    if not path:
        return False, f"{ENV_CHECKPOINT} is not set"
    if not Path(path).exists():
        return False, f"checkpoint not found: {path}"
    # Checkpoint contents before environment - see caption.py.
    # Readable, not merely present - see caption.py.
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
        from training.train_change_caption import build_model

        self.vocab: dict[str, int] = json.loads(
            (checkpoint / "vocab.json").read_text(encoding="utf-8")
        )
        self.inverse = {i: w for w, i in self.vocab.items()}
        latest = find_latest_checkpoint(checkpoint) or checkpoint
        payload = safe_torch_load(latest)
        extra = payload.get("extra") or {}
        dim = extra.get("dim", 128)
        # Which architecture wrote these weights. A checkpoint from before
        # Phase 5 has no `arch` field, so the default is v1 and every
        # existing checkpoint rebuilds exactly as it always did. Guessing
        # instead would load v1 weights into a v2 graph and fail on a key
        # mismatch that says nothing about the cause.
        model = build_model(vocab_size=len(self.vocab), dim=dim,
                            arch=extra.get("arch", "v1"))
        load_checkpoint(latest, model, map_location="cpu")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.torch = torch
        self.path = str(latest)
        # The bytes that are now in memory, hashed once per process, so
        # `Trace.weights_hashes` names the weights that produced the answer
        # rather than being empty. See satquery/tools/provenance.py.
        record("change_caption_v1", latest)

    @classmethod
    def get(cls, checkpoint: Path) -> "_Handle":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(checkpoint)
        return cls._instance


def _rgb(meta, size: int) -> np.ndarray:
    image, _ = to_rgb_preview(meta, max_edge=size)
    image = image.resize((size, size))
    return np.asarray(image, dtype="float32").transpose(2, 0, 1) / 255.0


def _change_mask(manifest: InputManifest, size: int) -> tuple[np.ndarray, list[str]]:
    """The mask this captioner is conditioned on, or zeros with a warning."""
    from satquery.tools import change_mask as cm

    available, reason = cm.is_available()
    if not available:
        return np.zeros((1, size, size), dtype="float32"), [
            f"no change mask available ({reason}); captioning with a zero "
            "mask, which is outside the mask-conditioned model's training "
            "distribution - treat the wording as unanchored"
        ]

    import rasterio

    result = cm.ChangeMaskTool().run(manifest, {})
    mask_artifact = next(
        (a for a in result.artifacts if "change" in a.key.lower()), None
    )
    if mask_artifact is None:
        return np.zeros((1, size, size), dtype="float32"), [
            "change_mask_v1 produced no raster; captioning with a zero mask"
        ]
    with rasterio.open(mask_artifact.path) as src:
        data = src.read(1, out_shape=(size, size)).astype("float32")
    return (data > 0).astype("float32")[None, ...], []


class ChangeCaptionTool(ToolProtocol):
    name = TOOL_NAME
    version = TOOL_VERSION

    def run(self, manifest: InputManifest, params: dict) -> ToolResult:
        started = time.perf_counter()
        handle = _Handle.get(Path(os.environ[ENV_CHECKPOINT]))
        torch = handle.torch

        from training.train_change_caption import BOS, EOS, PAD, PATCH

        if len(manifest.images) < 2:
            # The router's config gating makes this unreachable, but a tool
            # that assumes its inputs is one refactor away from a traceback.
            raise ValueError(
                "change_caption_v1 needs a bi-temporal pair; got "
                f"{len(manifest.images)} image(s)"
            )

        a = _rgb(manifest.images[0], PATCH)
        b = _rgb(manifest.images[1], PATCH)
        mask, warnings = _change_mask(manifest, PATCH)

        with torch.no_grad():
            tokens = handle.model.generate(
                torch.from_numpy(a).unsqueeze(0).to(handle.device),
                torch.from_numpy(b).unsqueeze(0).to(handle.device),
                torch.from_numpy(mask).unsqueeze(0).to(handle.device),
            )[0].tolist()

        words: list[str] = []
        for token in tokens:
            if token in (EOS, PAD):
                break
            word = handle.inverse.get(int(token))
            if word and not word.startswith("<"):
                words.append(word)
        caption = " ".join(words).strip()
        if not caption:
            caption = "No change description could be generated for this pair."
            warnings.append("change captioner produced no tokens")

        changed_fraction = float(mask.mean())

        return ToolResult(
            tool=TOOL_NAME,
            version=TOOL_VERSION,
            payload=ChangeCaptionPayload(
                data={
                    "caption": caption,
                    "mask_conditioned": not any("zero mask" in w for w in warnings),
                    "changed_fraction": round(changed_fraction, 6),
                    "n_tokens": len(words),
                }
            ),
            artifacts=[],
            # Fraction of the mask that is changed is NOT a confidence, so the
            # model's own token probabilities are used instead - see caption.py
            # on why that is fluency rather than correctness.
            confidence=_mean_token_probability(torch, handle, a, b, mask, tokens),
            confidence_method="logprob",
            model_card=f"mask-conditioned change captioner ({Path(handle.path).name})",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=warnings,
        )

    def run_batch(self, manifests, params):
        return [self.run(m, params) for m in manifests]


def _mean_token_probability(torch, handle, a, b, mask, tokens) -> float:
    """Mean probability of the chosen tokens, via `forward`. See caption.py.

    `model(a, b, mask, tokens) -> (B, T, V)`, position i predicting token
    i+1, is the contract both the v1 GRU and the v2 transformer honour.
    Stepping the GRU by hand, as this did before, is the same arithmetic on
    v1 and an AttributeError on v2.
    """
    if not tokens:
        return 0.0
    from training.train_change_caption import BOS

    with torch.no_grad():
        ab = torch.from_numpy(a).unsqueeze(0).to(handle.device)
        bb = torch.from_numpy(b).unsqueeze(0).to(handle.device)
        mb = torch.from_numpy(mask).unsqueeze(0).to(handle.device)
        prefix = torch.tensor([[BOS] + [int(x) for x in tokens[:-1]]],
                              dtype=torch.long, device=handle.device)
        logits = handle.model(ab, bb, mb, prefix).float()     # (1, T, V)
        step = torch.softmax(logits[0], dim=-1)
        idx = torch.tensor([int(x) for x in tokens], device=handle.device)
        probs = step[torch.arange(len(tokens), device=handle.device), idx]
    return round(float(probs.mean()), 6)
