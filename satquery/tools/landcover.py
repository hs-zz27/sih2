"""`landcover_v1` backed by Track A, with selective prediction (2.1, 3.3, 3.6).

Track A was trained and measured in task 2.1 (mAP 0.2854 on the official
BigEarthNet test shard) and no tool ever loaded it. Land-cover answers came
from the narrative synthesiser over measured indices, which is honest but
means the trained encoder was unreachable from a query.

Wiring it naively would have been worse than leaving it out, and task 3.6
measured why:

| decision rule | per (patch, class) error |
|---|---|
| always predict negative | **0.1834** |
| the head at threshold 0.5 | **0.2064** |

**At 0.5 this head is worse than always saying "no".** A tool that thresholded
at 0.5 and printed the surviving class names would be presenting something
worse than a constant, dressed as a model output.

What the head does have is *ranking* - mAP is threshold-free and 0.285 carries
real signal - and ranking is exactly what selective prediction consumes. So
this tool does three things the naive version does not:

1. **Calibrates per class**, using the affine transform task 3.3 fitted for
   `SINGLE_LANDCOVER` (ECE 0.0638 -> 0.0470). Per class is the *correct place*
   for it: the transform is nonlinear, so calibrating each class and then
   aggregating is not the same as calibrating an aggregate, which is why the
   runtime combiner refuses to touch aggregate scores.
2. **Uses a measured decision threshold**, not 0.5 - and from the *right*
   curve. `configs/thresholds.yaml` carries 0.70, where this head's positive
   assertions are ~91% precise. An earlier version used 0.8440 off the
   symmetric per-decision risk curve, which counts confident NEGATIVES (82% of
   the data) as coverage and is the wrong question for a tool that asserts
   classes.
3. **Abstains per class.** A class is asserted above the threshold, denied
   below its mirror, and *abstained on* in between - and the count of
   abstentions is reported, because "we could not call 12 of 19 classes" is
   the honest shape of this model's competence.

**Recall is 0.25%.** At 91% precision this head asserts almost nothing, which
is what mAP 0.285 buys. That is stated rather than buried, and it is survivable
because `synth/narrative.py` answers every land-cover query from measured
indices regardless - this tool adds high-precision class names when it can and
stays silent otherwise.

The confidence reported is the mean calibrated probability over the classes
actually asserted. That is a probability, but it is an **aggregate over a
threshold-selected subset**, so it is not P(correct) for the answer and is
excluded from `CALIBRATABLE_CONFIDENCE_METHODS` - the same reasoning as
`optsar_fusion`.
"""

from __future__ import annotations

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

TOOL_NAME = "landcover"
TOOL_VERSION = "1.0.0"
ENV_CHECKPOINT = "SATQUERY_LANDCOVER"

# Fallback if configs/thresholds.yaml is unreadable. Deliberately the measured
# value rather than 0.5, so a missing config degrades to the right behaviour.
DEFAULT_DECISION_CONFIDENCE = 0.70

# BigEarthNet-19 class names, in the order the head emits them.
#
# The authoritative list is `training/prepare/bigearthnet.CLASSES` - it is what
# built the label vectors the head was fitted against, and it is ALPHABETICAL.
# An earlier version of this file restated it from memory in a plausible but
# different order, so every assertion printed the wrong class name: index 0 was
# labelled "urban fabric" when it is "Agro-forestry areas". Restating a label
# ordering is exactly the kind of thing that looks right and silently is not.
#
# Read from the training module at first use so the tool cannot drift, with
# this copy as the fallback for an environment where that module will not
# import.
_FALLBACK_CLASS_NAMES = [
    "Agro-forestry areas",
    "Arable land",
    "Beaches, dunes, sands",
    "Broad-leaved forest",
    "Coastal wetlands",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Industrial or commercial units",
    "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of "
    "natural vegetation",
    "Marine waters",
    "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas",
    "Pastures",
    "Permanent crops",
    "Transitional woodland, shrub",
    "Urban fabric",
]


def class_names() -> list[str]:
    """The label ordering the head was trained against."""
    try:
        from training.prepare.bigearthnet import CLASSES

        if len(CLASSES) == len(_FALLBACK_CLASS_NAMES):
            return list(CLASSES)
    except Exception:  # noqa: BLE001 - fall back rather than fail a query
        pass
    return list(_FALLBACK_CLASS_NAMES)


class LandcoverPayload(ToolPayload):
    data: dict[str, Any]


def is_available() -> tuple[bool, str]:
    path = os.getenv(ENV_CHECKPOINT)
    if not path:
        return False, f"{ENV_CHECKPOINT} is not set"
    if not Path(path).exists():
        return False, f"checkpoint not found: {path}"
    # Band statistics before torch: without them the tool CANNOT run
    # correctly, and running it anyway is the failure this check exists to
    # prevent. Per-image standardisation instead of the training statistics
    # produced a head that asserted class 0 on every patch at 0.9 confidence
    # and was wrong every time - confidently, silently wrong.
    # Readable, not merely present. `track_a_full_multires/band_stats.json`
    # came back from a shadow-copy restore as 1,156 bytes of NUL on
    # 2026-08-31, and an existence check would have declared that head ready.
    ok, reason = readable_json(Path(path) / "band_stats.json", expect=dict)
    if not ok:
        return False, (
            f"{reason}; regenerate it with training.track_a_full.compute_stats "
            "over the training shards. The head cannot be normalised without "
            "it and will emit confident nonsense."
        )
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch is not installed"
    return True, "ready"


def decision_confidence(path=None) -> float:
    """The measured selective-prediction threshold, or the measured default."""
    from satquery.controller.abstention import (
        DEFAULT_THRESHOLDS_PATH,
        ENV_THRESHOLDS,
    )

    target = Path(
        path or os.environ.get(ENV_THRESHOLDS) or DEFAULT_THRESHOLDS_PATH
    )
    if not target.exists():
        return DEFAULT_DECISION_CONFIDENCE
    try:
        import yaml

        blob = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        value = float((blob.get("landcover") or {}).get(
            "decision_confidence", DEFAULT_DECISION_CONFIDENCE
        ))
        # A threshold at or below 0.5 would assert every class and reproduce
        # the worse-than-trivial behaviour this tool exists to avoid.
        return value if 0.5 < value < 1.0 else DEFAULT_DECISION_CONFIDENCE
    except Exception:  # noqa: BLE001 - degradation, not a crash
        return DEFAULT_DECISION_CONFIDENCE


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
        from training.track_a_full import build_model

        latest = find_latest_checkpoint(checkpoint) or checkpoint
        payload = safe_torch_load(latest)
        state = payload.get("model_state_dict", {})
        # GSD conditioning must match the checkpoint or load_state_dict leaves
        # the FiLM layers randomly initialised - a silent partial load, which
        # is how people end up reporting a random encoder as fine-tuned.
        has_gsd = any(k.startswith("gsd_mlp.") for k in state)
        extra = payload.get("extra") or {}
        dim = extra.get("dim", 64)
        # Which architecture wrote these weights. A checkpoint from before
        # Phase 5 has no `arch` field, so the default is v1 and every
        # existing checkpoint rebuilds exactly as it always did. Guessing
        # instead would load v1 weights into a v2 graph and fail on a key
        # mismatch that says nothing about the cause.
        arch = extra.get("arch", "v1")
        model = build_model(dim=dim, gsd_conditioning=has_gsd, arch=arch)
        load_checkpoint(latest, model, map_location="cpu")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.has_gsd = has_gsd
        self.checkpoint = checkpoint
        self.torch = torch
        self.path = str(latest)
        # The bytes that are now in memory, hashed once per process, so
        # `Trace.weights_hashes` names the weights that produced the answer
        # rather than being empty. See satquery/tools/provenance.py.
        record("landcover_v1", latest)

    @classmethod
    def get(cls, checkpoint: Path) -> "_Handle":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(checkpoint)
        return cls._instance


def load_band_stats(checkpoint: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Dataset-level band statistics saved beside the weights."""
    import json

    blob = json.loads((checkpoint / "band_stats.json").read_text(encoding="utf-8"))
    return (
        np.asarray(blob["mean"], dtype="float32"),
        np.asarray(blob["std"], dtype="float32"),
        float(blob.get("reflectance_scale", 10000.0)),
    )


def _band_cube(meta, stats, size: int = 120):
    """Read the image into the encoder's canonical 12-band slots.

    Returns (cube, presence). The presence mask is what makes a 4-band VNIR
    sensor legal input: the encoder was trained with band dropout, so it
    averages over whichever bands are actually there rather than assuming
    twelve.
    """
    import rasterio
    from training.track_a_full import BAND_NAMES_12

    # Canonical Sentinel-2 names to the encoder's slot order.
    alias = {
        "BLUE": "B02", "GREEN": "B03", "RED": "B04", "NIR": "B08",
        "SWIR1": "B11", "SWIR2": "B12",
    }
    mean, std, scale = stats
    cube = np.zeros((len(BAND_NAMES_12), size, size), dtype="float32")
    presence = np.zeros(len(BAND_NAMES_12), dtype="float32")

    with rasterio.open(meta.path) as src:
        for index, name in enumerate(meta.bands or []):
            slot_name = alias.get(str(name).upper())
            if slot_name is None or slot_name not in BAND_NAMES_12:
                continue
            slot = BAND_NAMES_12.index(slot_name)
            band = src.read(
                index + 1, out_shape=(size, size), masked=True
            ).astype("float32")
            band = np.ma.filled(band, np.nan)
            # EXACTLY the training transform: divide by the reflectance scale,
            # then standardise with the DATASET-level statistics. Per-image
            # standardisation is a different distribution and the head has
            # never seen it.
            band = band / scale
            band = (band - mean[slot]) / (std[slot] or 1.0)
            cube[slot] = np.nan_to_num(band)
            presence[slot] = 1.0
    return cube, presence


class LandcoverTool(ToolProtocol):
    name = TOOL_NAME
    version = TOOL_VERSION

    def run(self, manifest: InputManifest, params: dict) -> ToolResult:
        started = time.perf_counter()
        handle = _Handle.get(Path(os.environ[ENV_CHECKPOINT]))
        torch = handle.torch

        from satquery.controller.calibration import load_registry
        from training.track_a_full import BEN_GSD_M

        meta = manifest.images[0]
        cube, presence = _band_cube(meta, load_band_stats(handle.checkpoint))
        warnings: list[str] = []
        if presence.sum() == 0:
            raise ValueError(
                "no recognised spectral bands; landcover_v1 needs at least one "
                "of BLUE/GREEN/RED/NIR/SWIR1/SWIR2"
            )

        gsd = float(meta.gsd_m or BEN_GSD_M)
        with torch.no_grad():
            logits = handle.model(
                torch.from_numpy(cube).unsqueeze(0).to(handle.device),
                torch.from_numpy(presence).unsqueeze(0).to(handle.device),
                torch.full((1,), gsd, device=handle.device),
            )[0].cpu().numpy()

        # Calibrate PER CLASS, which is the correct place for it: the fitted
        # transform is nonlinear, so calibrating each class then aggregating
        # is not the same as transforming an aggregate.
        entry = load_registry().lookup("SINGLE_LANDCOVER")
        if entry is not None:
            # From the LOGIT, not from a probability made from it. A float32
            # sigmoid saturates at z ~ 17 and `apply` clamps at 13.8; this
            # head reaches +71, and the round trip flattened its top 33,620
            # decisions onto one value. `apply_logit` is the same transform
            # without the two lossy steps in front of it.
            probs = np.array([entry.apply_logit(float(z)) for z in logits])
            calibration = f"{entry.method}:SINGLE_LANDCOVER"
        else:
            probs = 1.0 / (1.0 + np.exp(-logits))
            calibration = "uncalibrated (no accepted fit for this head)"
            warnings.append(
                "no calibration entry for SINGLE_LANDCOVER; probabilities are "
                "the head's raw sigmoid"
            )

        threshold = decision_confidence()
        names = class_names()
        asserted, denied, abstained = [], [], []
        for i, p in enumerate(probs):
            name = names[i] if i < len(names) else f"class_{i}"
            if p >= threshold:
                asserted.append({"class": name, "probability": round(float(p), 4)})
            elif p <= 1.0 - threshold:
                denied.append(name)
            else:
                # The head cannot call this class at the measured risk budget.
                abstained.append({"class": name, "probability": round(float(p), 4)})

        if asserted:
            listed = ", ".join(a["class"] for a in asserted)
            answer = f"Detected land-cover classes: {listed}."
        else:
            answer = (
                "No land-cover class could be asserted at the measured "
                f"confidence threshold ({threshold:.3f}); the head was "
                f"undecided on {len(abstained)} of {len(probs)} classes."
            )

        if abstained:
            warnings.append(
                f"abstained on {len(abstained)} of {len(probs)} classes: below "
                f"the {threshold:.2f} confidence at which this head's "
                f"assertions are ~91% precise. At 0.5 it is worse than always "
                f"predicting negative (task 3.6), so a full 19-class map from "
                f"it would be misleading."
            )

        confidence = (
            float(np.mean([a["probability"] for a in asserted])) if asserted else 0.0
        )

        return ToolResult(
            tool=TOOL_NAME,
            version=TOOL_VERSION,
            payload=LandcoverPayload(
                data={
                    "answer": answer,
                    "labels": [a["class"] for a in asserted],
                    "asserted": asserted,
                    "abstained": abstained,
                    "n_denied": len(denied),
                    "decision_confidence": threshold,
                    "calibration": calibration,
                    "bands_present": int(presence.sum()),
                    "gsd_m": gsd,
                }
            ),
            artifacts=[],
            confidence=round(confidence, 6),
            # A probability, but an aggregate over a threshold-selected
            # subset - not P(correct) for the answer. Same reasoning as
            # optsar_fusion; see CALIBRATABLE_CONFIDENCE_METHODS.
            confidence_method=(
                "mean_asserted_probability" if asserted else "no_assertion"
            ),
            model_card=f"Track A land-cover head ({Path(handle.path).name})",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=warnings,
        )

    def run_batch(self, manifests, params):
        return [self.run(m, params) for m in manifests]
