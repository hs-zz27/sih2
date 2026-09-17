"""`change_vqa_v1` deterministic template path (task 2.6).

The plan is explicit that the template path comes *first*, before the CDVQA
head. The reason is sound: "how much did the water area change" has an exact
arithmetic answer from two index rasters, and asking a neural model to
estimate a number the physics already knows exactly is strictly worse - it
adds error and removes auditability.

So this answers area-delta questions by measuring, and defers anything it
cannot measure rather than guessing. `confidence_method` is "deterministic"
because the arithmetic has no uncertainty; the uncertainty lives in the
thresholds, which are reported separately.
"""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np

from satquery.contracts.input_manifest import InputManifest
from satquery.contracts.tool_result import ToolPayload, ToolResult
from satquery.ingest.reader import read_canonical_band
from satquery.tools.base import ToolProtocol
from satquery.tools.provenance import record
from satquery.verify.indices import mndwi, ndbi, ndvi, ndwi, swir_free_builtup_proxy
from satquery.verify.thresholding import adaptive_threshold, apply_threshold
from satquery.verify.verifier import SUBJECT_TERMS

TOOL_NAME = "change_vqa"
TOOL_VERSION = "1.0.0-template"

# Priors used when the histogram has no bimodal split to threshold on. The
# built-up prior is the proxy's own midpoint: unlike an index in [-1, 1] the
# proxy is already a likelihood in [0, 1], so 0.5 is the neutral cut.
FIXED_PRIORS = {"vegetation": 0.3, "water": 0.0, "built_up": 0.5}

# Question shapes this path can answer exactly.
_QUANTITY_RE = re.compile(
    r"\b(how much|how many|by what|what (?:is|was) the (?:net )?change|"
    r"quantify|percentage|percent|area)\b", re.IGNORECASE
)
_DIRECTION_RE = re.compile(
    r"\b(increase|decrease|grow|shrink|expand|reduce|more|less|gain|loss|lost)\b",
    re.IGNORECASE,
)


class ChangeVQAPayload(ToolPayload):
    data: dict[str, Any]


def subject_of(question: str) -> str | None:
    lowered = question.lower()
    for subject, terms in SUBJECT_TERMS.items():
        if any(t in lowered for t in terms):
            return subject
    return None


def _index_for(subject: str, meta) -> tuple[np.ndarray, str] | None:
    """Compute the index that measures `subject` for one image."""
    bands = set(meta.bands)
    if subject == "vegetation" and {"RED", "NIR"} <= bands:
        return ndvi(read_canonical_band(meta, "RED"),
                    read_canonical_band(meta, "NIR")), "ndvi"
    if subject == "water":
        if {"GREEN", "SWIR1"} <= bands:
            return mndwi(read_canonical_band(meta, "GREEN"),
                         read_canonical_band(meta, "SWIR1")), "mndwi"
        if {"GREEN", "NIR"} <= bands:
            return ndwi(read_canonical_band(meta, "GREEN"),
                        read_canonical_band(meta, "NIR")), "ndwi"
    if subject == "built_up":
        # Added 2026-08-30. The PS's fifth representative query is "Has the
        # built-up area increased, decreased, or remained unchanged?" and this
        # function knew only vegetation and water, so that query abstained -
        # found by rehearsing the demo rather than by any test.
        #
        # The subject/index mapping already existed in the verifier
        # (SUBJECT_INDICES: built_up -> ndbi, builtup_proxy); only this path
        # had not implemented it. Same ordering, same SWIR-free fallback,
        # which is Axiom 2: Cartosat-2S has no SWIR, so the proxy is the
        # operative path on the target sensor rather than a contingency.
        if {"SWIR1", "NIR"} <= bands:
            return ndbi(read_canonical_band(meta, "SWIR1"),
                        read_canonical_band(meta, "NIR")), "ndbi"
        if {"RED", "NIR"} <= bands:
            return swir_free_builtup_proxy(
                read_canonical_band(meta, "RED"),
                read_canonical_band(meta, "NIR"),
            ), "builtup_proxy"
    return None


def measure_change(subject: str, t1, t2) -> dict | None:
    """Fraction of each scene above the index threshold, and the delta.

    The threshold is derived from t1 and applied to both dates. Re-deriving it
    per date would let a threshold shift masquerade as real change, which is
    the classic way naive change detection invents results.
    """
    a = _index_for(subject, t1)
    b = _index_for(subject, t2)
    if a is None or b is None:
        return None

    arr1, index_name = a
    arr2, _ = b

    threshold = adaptive_threshold(arr1, fixed_prior=FIXED_PRIORS.get(subject, 0.0))
    m1 = apply_threshold(arr1, threshold)
    m2 = apply_threshold(arr2, threshold)

    f1 = float(m1.sum()) / max(m1.size, 1)
    f2 = float(m2.sum()) / max(m2.size, 1)
    gsd = t1.gsd_m or 0.0
    area_km2 = (f2 - f1) * m1.size * gsd * gsd / 1e6

    return {
        "index": index_name,
        "threshold": round(threshold.value, 6),
        "threshold_method": threshold.method,
        "fraction_t1": round(f1, 6),
        "fraction_t2": round(f2, 6),
        "delta_fraction": round(f2 - f1, 6),
        "relative_change": round((f2 - f1) / f1, 6) if f1 > 0 else None,
        "delta_area_km2": round(area_km2, 4),
    }


# Subject keys are internal identifiers; a demo audience should not read
# "built_up" in a sentence.
DISPLAY_NAMES = {"built_up": "built-up"}


def phrase(subject: str, m: dict) -> str:
    subject = DISPLAY_NAMES.get(subject, subject)
    delta = m["delta_fraction"]
    direction = "increased" if delta > 0 else "decreased" if delta < 0 else "did not change"
    if delta == 0:
        return f"The {subject} extent did not change measurably between the two dates."
    rel = (
        f" ({abs(m['relative_change']):.0%} relative)"
        if m["relative_change"] is not None else ""
    )
    return (
        f"The {subject} extent {direction} from {m['fraction_t1']:.1%} to "
        f"{m['fraction_t2']:.1%} of the scene{rel}, a change of about "
        f"{abs(m['delta_area_km2']):.2f} km2, measured by {m['index'].upper()}."
    )


class ChangeVQATemplate(ToolProtocol):
    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        warnings: list[str] = []
        question = str(params.get("_query") or "")

        if len(manifest.images) != 2:
            return self._defer(
                started, "change questions need two images; only one was supplied"
            )

        t1, t2 = manifest.images
        subject = subject_of(question)
        answerable = bool(_QUANTITY_RE.search(question) or _DIRECTION_RE.search(question))

        if subject is None or not answerable:
            return self._defer(
                started,
                "this question is not an area-change measurement; the "
                "deterministic path defers to the learned change-VQA head "
                "(task 2.6, not yet trained)",
            )

        measured = measure_change(subject, t1, t2)
        if measured is None:
            return self._defer(
                started,
                f"cannot measure {subject} change: the required bands are not "
                f"present in both images ({t1.bands} / {t2.bands})",
            )

        if measured["threshold_method"] == "fixed_prior":
            warnings.append(
                f"{measured['index']} threshold fell back to a fixed prior; "
                "the change estimate is less reliable"
            )

        payload = ChangeVQAPayload(
            data={
                "answer": phrase(subject, measured),
                "question": question,
                "subject": subject,
                "measurement": measured,
                "path": "deterministic_template",
            }
        )
        return ToolResult(
            tool=TOOL_NAME, version=TOOL_VERSION, payload=payload, artifacts=[],
            confidence=1.0 if measured["threshold_method"] != "fixed_prior" else 0.6,
            confidence_method="deterministic",
            model_card="change_vqa_v1 template path (closed-form index delta)",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=warnings,
        )

    def _defer(self, started: float, reason: str) -> ToolResult:
        """Say what cannot be measured rather than guessing at it."""
        return ToolResult(
            tool=TOOL_NAME, version=TOOL_VERSION,
            payload=ChangeVQAPayload(data={"answer": "", "deferred": True,
                                           "reason": reason, "path": "deferred"}),
            artifacts=[], confidence=0.0, confidence_method="deterministic",
            model_card="change_vqa_v1 template path (deferred)",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=[reason],
        )

    def run_batch(self, manifests, params):
        return [self.run(m, params) for m in manifests]


# ---------------------------------------------------------------------------
# Semantic path (task 2.6's learned half)
# ---------------------------------------------------------------------------
#
# The deterministic path above measures vegetation with NDVI and water with
# MNDWI/NDWI, so it cannot fire on RGB-only imagery - which is every image in
# CDVQA, the PS's prescribed change-VQA benchmark. That is why the benchmark
# measured 0.0000 (docs/phase1-status.md).
#
# This path fills the gap without giving up the design: a semantic change
# segmenter predicts a class map per date, and satquery.verify.semantic_change
# derives the answer from those maps by counting pixels. The neural component
# decides *what is where*; the answer is arithmetic, which is the same division
# of labour the index path uses.
#
# Precedence is deterministic-first, then semantic, then defer. The two answer
# different question shapes - the index path returns a measured km2 delta, the
# semantic path a closed-vocabulary class or bin - and on multispectral
# imagery the closed-form measurement is the better answer. On RGB the index
# path defers immediately, so the semantic path is reachable rather than
# shadowed. That ordering is asserted in the tests, because a precedence rule
# that silently shadows the second path is a mistake this project has already
# made once (the task-3.5 entailment gate).

import os  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402

from satquery.verify import semantic_change  # noqa: E402

ENV_SEMANTIC = "SATQUERY_CHANGE_VQA"
SEMANTIC_VERSION = "1.1.0-semantic"


def semantic_available() -> tuple[bool, str]:
    path = os.getenv(ENV_SEMANTIC)
    if not path:
        return False, f"{ENV_SEMANTIC} is not set"
    if not Path(path).exists():
        return False, f"checkpoint not found: {path}"
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch is not installed"
    return True, "ready"


class _SemanticHandle:
    """Loaded once per process, like the other learned tools."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, checkpoint: Path):
        import torch

        from training.common.checkpointing import safe_torch_load
        from training.train_change_vqa import build_model, build_pretrained_model

        payload = safe_torch_load(checkpoint)
        state = payload.get("model", payload)
        # Hyperparameters may sit at the top level (the checkpoints written
        # before Phase 5) or under `extra` (everything written since). Read
        # both, top level first, so neither shape falls back to a default that
        # silently rebuilds the wrong-sized model.
        extra = payload.get("extra") or {}
        dim = payload.get("dim", extra.get("dim", 48))

        # The checkpoint records which encoder it was trained with. Guessing
        # would load ImageNet weights into a from-scratch graph, or fail on a
        # key mismatch that says nothing about the cause. `arch` is the same
        # question one level down: absent means v1, so every existing
        # checkpoint rebuilds exactly as it always did.
        pretrained = payload.get("pretrained", extra.get("pretrained"))
        if pretrained:
            model = build_pretrained_model(dim)
        else:
            model = build_model(dim, arch=extra.get("arch", "v1"))
        model.load_state_dict(state)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.torch = torch
        self.path = str(checkpoint)
        # See satquery/tools/provenance.py. Recorded only on the semantic
        # path: the template path loads no weights, and giving it a digest
        # would claim bytes that were never read.
        record("change_vqa_v1", checkpoint)

    @classmethod
    def get(cls, checkpoint: Path) -> "_SemanticHandle":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(checkpoint)
        return cls._instance


def read_rgb_as_trained(meta, size: int) -> np.ndarray:
    """Read RGB the way the segmenter was trained to see it.

    Training read 8-bit PNGs and divided by 255 - no stretch. Applying the
    percentile stretch that `change_mask_v1` uses would hand the model a
    different input distribution than it learned on, which degrades it
    silently rather than loudly. So 8-bit data is scaled by 255 and only
    higher bit depths get a stretch, which they need because their range is
    sensor-dependent.
    """
    import rasterio
    from rasterio.enums import Resampling

    order = [b for b in ("RED", "GREEN", "BLUE") if b in meta.bands]
    with rasterio.open(meta.path) as src:
        indices = (
            [meta.bands.index(b) + 1 for b in order]
            if len(order) == 3
            else list(range(1, min(3, src.count) + 1))
        )
        while len(indices) < 3:
            indices.append(indices[-1])

        eight_bit = src.dtypes[0] == "uint8"
        channels = []
        for idx in indices:
            arr = src.read(
                idx, out_shape=(size, size),
                resampling=Resampling.bilinear, masked=True,
            ).astype("float32")
            arr = np.ma.filled(arr, np.nan)
            if eight_bit:
                arr = arr / 255.0
            else:
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    lo, hi = np.percentile(finite, [2, 98])
                    arr = np.clip((arr - lo) / (hi - lo), 0, 1) if hi > lo else arr * 0
            channels.append(np.nan_to_num(arr))
    return np.stack(channels).astype("float32")


def predict_class_maps(t1_meta, t2_meta, size: int = 512):
    """Both dates' semantic class maps, as integer arrays."""
    handle = _SemanticHandle.get(Path(os.environ[ENV_SEMANTIC]))
    torch = handle.torch
    a = read_rgb_as_trained(t1_meta, size)[None]
    b = read_rgb_as_trained(t2_meta, size)[None]
    with torch.no_grad():
        p1, p2 = handle.model(
            torch.from_numpy(a).to(handle.device),
            torch.from_numpy(b).to(handle.device),
        )
        return (
            p1.argmax(1)[0].cpu().numpy(),
            p2.argmax(1)[0].cpu().numpy(),
            handle.path,
        )


class ChangeVQASemantic(ChangeVQATemplate):
    """Deterministic index path first, then the semantic change head."""

    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        deterministic = super().run(manifest, params)
        if not deterministic.payload.data.get("deferred"):
            return deterministic

        started = time.perf_counter()
        question = str(params.get("_query") or "")
        if len(manifest.images) != 2:
            return deterministic

        t1_meta, t2_meta = manifest.images
        t1, t2 = predict_class_maps(t1_meta, t2_meta)[:2]
        derived = semantic_change.answer(question, t1, t2)

        if derived is None:
            # The segmenter ran and the arithmetic still does not answer this
            # question shape. Say that, rather than returning the index path's
            # "no NIR band" reason, which would be the wrong explanation.
            return self._defer(
                started,
                "the semantic change head ran, but this question is not one "
                "the change-map arithmetic answers",
            )

        areas1 = semantic_change.class_areas(t1)
        areas2 = semantic_change.class_areas(t2)
        parsed = semantic_change.parse_question(question)
        changed_fraction = float((t1 != semantic_change.UNCHANGED).mean())

        payload = ChangeVQAPayload(data={
            "answer": derived,
            "question": question,
            "subject": parsed.subject,
            "path": "semantic_change_map",
            "measurement": {
                "question_kind": parsed.kind,
                "date_scope": parsed.scope,
                "changed_fraction": round(changed_fraction, 6),
                "class_areas_t1": areas1,
                "class_areas_t2": areas2,
            },
        })
        return ToolResult(
            tool=TOOL_NAME, version=SEMANTIC_VERSION, payload=payload, artifacts=[],
            # The arithmetic over the maps is exact; the uncertainty is all in
            # the segmentation, and reporting 1.0 here would hide that. The
            # fraction of pixels the segmenter left unchanged is not a
            # correctness probability, so this is deliberately conservative
            # and fixed, and the segmenter's measured mIoU is the number that
            # belongs in the model card.
            confidence=0.6,
            confidence_method="segmentation_derived",
            model_card="change_vqa_v1 semantic path (SECOND 7-class change maps)",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=[],
        )
