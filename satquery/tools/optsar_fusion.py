"""`optsar_fusion_v1` in triad mode (plan task 2.3).

The PS requires extracting *complementary* information from a co-registered
optical + SAR pair. Per docs/01 section 5.2, a fused number alone does not
demonstrate complementarity, so every query runs three passes - optical-only,
SAR-only, fused - and reports a per-query complementarity score:

    gain      = fused - max(optical, sar)
    agreement = how often the two modalities pick the same classes
    attribution = which modality drove each class

MEASURED CAVEAT, reported rather than buried: on WHU-OPT-SAR scene-level
multi-label classification the gain is about -0.006 - fusion does not help.
Both modalities can independently answer "is there water somewhere in this
tile", leaving nothing for fusion to add. Complementarity is inherently
spatial (SAR seeing through cloud, separating water from shadow), so
demonstrating it needs a per-pixel segmentation head, which is task 2.9/3.x
work. The triad machinery here is correct and reports the honest number;
the number is currently ~zero.

Confidence is the mean fused probability over the classes the tool actually
asserted - those at or above `PRESENCE_THRESHOLD` - reported as
`confidence_method="mean_asserted_probability"`. Unlike the change mask's
sharpness this genuinely is a probability: it is the head's own estimate of
its precision over the positive predictions it made. It is still not a
calibratable P(correct) for this answer, for two reasons worth keeping
straight:

* it is conditioned on `p >= PRESENCE_THRESHOLD`, so it describes only the
  asserted subset and says nothing about classes the tool stayed silent on;
* it is an aggregate. A fitted calibration is a *nonlinear* map, so applying
  it to a mean of probabilities is not the same as calibrating each class and
  then averaging. Calibrating this head means transforming `p_fused`
  per class before this line, not transforming the number it produces.

Until Phase 3 this was labelled `softmax_temp_scaled`; nothing was ever
temperature-scaled.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling

from satquery.contracts.input_manifest import InputManifest
from satquery.contracts.tool_result import ToolPayload, ToolResult
from satquery.tools.base import ToolProtocol
from satquery.tools.provenance import record

TOOL_NAME = "optsar_fusion"
TOOL_VERSION = "1.0.0-triad"
ENV_CHECKPOINT = "SATQUERY_FUSION"
PATCH = 120
PRESENCE_THRESHOLD = 0.5

OPTICAL_MODALITIES = {"OPTICAL", "MSI", "PAN"}


class FusionPayload(ToolPayload):
    data: dict[str, Any]


def is_available() -> tuple[bool, str]:
    path = os.getenv(ENV_CHECKPOINT)
    if not path:
        return False, f"{ENV_CHECKPOINT} is not set"
    if not Path(path).exists():
        return False, f"checkpoint not found: {path}"
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch is not installed"
    return True, "ready"


class _Handle:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, checkpoint: Path):
        import torch

        from training.common.checkpointing import find_latest_checkpoint, load_checkpoint
        from training.train_optsar_fusion import build_model

        latest = find_latest_checkpoint(checkpoint) or checkpoint
        from training.common.checkpointing import safe_torch_load

        extra = (safe_torch_load(latest).get("extra") or {})
        # Which architecture wrote these weights. A checkpoint from before
        # Phase 5 has no `arch` field, so the default is v1 and every
        # existing checkpoint rebuilds exactly as it always did. Guessing
        # instead would load v1 weights into a v2 graph and fail on a key
        # mismatch that says nothing about the cause.
        model = build_model(extra.get("dim", 32),
                            arch=extra.get("arch", "v1"))
        load_checkpoint(latest, model, map_location="cpu")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.torch = torch
        self.path = str(latest)
        # The bytes that are now in memory, hashed once per process, so
        # `Trace.weights_hashes` names the weights that produced the answer
        # rather than being empty. See satquery/tools/provenance.py.
        record("optsar_fusion_v1", latest)

    @classmethod
    def get(cls, checkpoint: Path) -> "_Handle":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(checkpoint)
        return cls._instance


def _read(meta, n_bands: int) -> np.ndarray:
    """Read and standardise `n_bands` channels at the trained patch size."""
    with rasterio.open(meta.path) as src:
        count = min(src.count, n_bands)
        arr = src.read(
            list(range(1, count + 1)), out_shape=(count, PATCH, PATCH),
            resampling=Resampling.bilinear, masked=True,
        ).astype("float32")
    arr = np.ma.filled(arr, np.nan)
    out = []
    for band in arr:
        finite = band[np.isfinite(band)]
        if finite.size:
            mean, std = float(finite.mean()), float(finite.std()) or 1.0
            band = (band - mean) / std
        out.append(np.nan_to_num(band))
    stacked = np.stack(out)
    while stacked.shape[0] < n_bands:
        stacked = np.concatenate([stacked, stacked[-1:]])
    return stacked.astype("float32")


class OptSARFusionTool(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()

        optical = next(
            (i for i in manifest.images if i.modality in OPTICAL_MODALITIES), None
        )
        sar = next((i for i in manifest.images if i.modality == "SAR"), None)
        if optical is None or sar is None:
            raise ValueError(
                "cross-modal fusion needs one optical and one SAR image; got "
                f"{[i.modality for i in manifest.images]}"
            )

        handle = _Handle.get(Path(os.environ[ENV_CHECKPOINT]))
        torch = handle.torch

        o = _read(optical, 4)[None]
        s = _read(sar, 1)[None]
        with torch.no_grad():
            lo, ls, lf = handle.model(
                torch.from_numpy(o).to(handle.device),
                torch.from_numpy(s).to(handle.device),
            )
            p_opt = torch.sigmoid(lo)[0].cpu().numpy()
            p_sar = torch.sigmoid(ls)[0].cpu().numpy()
            p_fused = torch.sigmoid(lf)[0].cpu().numpy()

        from training.prepare.whu_opt_sar import CLASSES

        def present(p):
            return {CLASSES[i] for i, v in enumerate(p) if v >= PRESENCE_THRESHOLD}

        set_o, set_s, set_f = present(p_opt), present(p_sar), present(p_fused)
        union = set_o | set_s
        # Low agreement means the modalities disagree, which is exactly when
        # fusion had something to resolve.
        agreement = len(set_o & set_s) / len(union) if union else 1.0

        attribution = {}
        for i, name in enumerate(CLASSES):
            if p_fused[i] < PRESENCE_THRESHOLD:
                continue
            attribution[name] = (
                "optical" if p_opt[i] > p_sar[i] else
                "sar" if p_sar[i] > p_opt[i] else "equal"
            )

        # Per-query proxy for the offline gain: how much more confident the
        # fused head is than the better single modality, averaged over the
        # classes it actually asserts.
        asserted = [i for i in range(len(CLASSES)) if p_fused[i] >= PRESENCE_THRESHOLD]
        gain = (
            float(np.mean([p_fused[i] - max(p_opt[i], p_sar[i]) for i in asserted]))
            if asserted else 0.0
        )

        classes_text = ", ".join(sorted(set_f)) or "no class above threshold"
        answer = (
            f"Fused optical+SAR analysis identifies: {classes_text}. "
            f"The two modalities agreed on {agreement:.0%} of detected classes."
        )

        payload = FusionPayload(data={
            "answer": answer,
            "mode": "triad",
            "optical_only": {CLASSES[i]: round(float(p_opt[i]), 4) for i in range(len(CLASSES))},
            "sar_only": {CLASSES[i]: round(float(p_sar[i]), 4) for i in range(len(CLASSES))},
            "fused": {CLASSES[i]: round(float(p_fused[i]), 4) for i in range(len(CLASSES))},
            "complementarity": {
                "gain": round(gain, 6),
                "modality_agreement": round(agreement, 4),
                "attribution": attribution,
                "note": (
                    "Offline gain on WHU-OPT-SAR scene-level classification is "
                    "about -0.006: fusion does not currently beat optical alone. "
                    "Scene-level presence is too coarse a task for "
                    "complementarity to appear; a per-pixel head is required."
                ),
            },
        })

        return ToolResult(
            tool=TOOL_NAME, version=TOOL_VERSION, payload=payload, artifacts=[],
            confidence=float(np.mean([p_fused[i] for i in asserted])) if asserted else 0.0,
            confidence_method=(
                "mean_asserted_probability" if asserted else "no_assertion"
            ),
            model_card=f"optical-SAR triad ({Path(handle.path).name})",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=(
                [] if asserted else ["no class exceeded the presence threshold"]
            ),
        )

    def run_batch(self, manifests, params):
        return [self.run(m, params) for m in manifests]
