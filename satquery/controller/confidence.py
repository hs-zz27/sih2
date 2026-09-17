"""Three-component confidence (plan task 1.3, extended in 3.4).

Confidence is not the model's softmax. A model can be certain and wrong, so
the score combines three independent signals:

* **model**      - the tool's own reported confidence
* **agreement**  - do the deterministic physics indices corroborate the claim?
* **input_quality** - were the inputs clean enough to support any answer?

They are combined with a *geometric* mean rather than an arithmetic one: the
geometric mean collapses toward zero if any single component collapses, which
is the behaviour we want. A confident model on a corrupt input should not
score 0.66 because two of three components were fine.

Phase 1 used equal weights and no calibration. Task 3.3 supplies the
calibration: when the head that produced the score has an accepted fit in
`configs/calibration.json`, the **model** component is recalibrated before it
is combined, and the trace reports the real method and held-out ECE instead
of the sentinel. Fitting the three component weights is still task 3.4, so
the combination below remains an unweighted geometric mean.
"""

from __future__ import annotations

import math

from satquery.contracts.input_manifest import InputManifest
from satquery.contracts.trace import (
    ConfidenceCalibrationTrace,
    ConfidenceComponentsTrace,
    ConfidenceTrace,
)
from satquery.controller.calibration import load_registry, method_label

HIGH_BAND = 0.75
MEDIUM_BAND = 0.45

# The ceiling on a run whose answer came from a placeholder rather than a
# model. Chosen to sit just below MEDIUM_BAND, so `band()` returns LOW through
# the ordinary comparison, and comfortably above the 0.25 abstention
# threshold, so a stubbed run still ANSWERS - which is what CI and the lite
# profile depend on - it simply cannot claim to be confident.
#
# Why a cap and not a zero component: a stub reporting 0.0 into the geometric
# mean collapses the score to 0.0, which trips abstention, and every stubbed
# run in the suite then refuses instead of answering. Measured 2026-09-01:
# 36 tests failed that way. The cap keeps the abstention policy untouched.
STUB_CONFIDENCE_CAP = 0.44

# Per-check penalties applied to the input-quality component.
WARN_PENALTY = 0.15
FAIL_PENALTY = 0.5

# Sentinel for "expected calibration error has not been measured yet".
# A real ECE is in [0, 1], so a negative value is unambiguous, and unlike NaN
# it survives JSON serialisation.
UNMEASURED_ECE = -1.0


def input_quality(manifest: InputManifest) -> float:
    """Quality of the inputs themselves, in [0, 1], from the check results."""
    if not manifest.checks:
        return 1.0
    score = 1.0
    for check in manifest.checks:
        if check.status == "WARN":
            score -= WARN_PENALTY
        elif check.status == "FAIL":
            score -= FAIL_PENALTY
    return max(0.0, min(1.0, score))


def physics_agreement(agreements: dict[str, float]) -> float:
    """Mean agreement across whatever physics checks were possible.

    With no applicable physics check the component is neutral (1.0) rather
    than zero - absence of corroboration is not evidence of contradiction,
    and the trace records that no check ran.
    """
    if not agreements:
        return 1.0
    values = list(agreements.values())
    return max(0.0, min(1.0, sum(values) / len(values)))


def geometric_mean(*values: float, weights: tuple[float, ...] | None = None) -> float:
    """Weighted geometric mean, zero if any component is zero.

    Weights are supported (task 3.4) but ship EQUAL, and the reason is worth
    stating rather than leaving as an apparent oversight: fitting them needs a
    labelled set of (components -> was the answer actually correct) pairs, and
    no such set exists. Corrected 2026-08-30 - the earlier wording said "every
    learned tool is still a stub", which stopped being true in Phase 2. The
    gap is narrower and unchanged in effect: the learned tools are real, and
    not one of them reports a probability of *correctness*, which is what a
    weight would have to be fitted against. See
    CALIBRATABLE_CONFIDENCE_METHODS for what each confidence method actually
    measures. Equal weights are the honest default; the mechanism is here so
    the fit is a config change rather than a code change once the data
    exists.
    """
    if not values:
        return 0.0
    if weights is None:
        weights = (1.0,) * len(values)
    if len(weights) != len(values):
        raise ValueError("weights and values must have the same length")

    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")

    accumulated = 0.0
    for value, weight in zip(values, weights, strict=True):
        if value <= 0:
            # A zero in any component collapses the score, which is the whole
            # reason for a geometric rather than arithmetic mean: a confident
            # model on a corrupt input must not score 0.66 because two of
            # three components were fine.
            return 0.0
        accumulated += weight * math.log(value)
    return math.exp(accumulated / total)


# Component weights, in the order (model, agreement, input_quality). Read from
# configs/thresholds.yaml under `confidence.weights` when present.
DEFAULT_WEIGHTS = (1.0, 1.0, 1.0)


def load_weights(path=None) -> tuple[float, float, float]:
    """Load component weights, falling back to equal on any problem."""
    import os
    from pathlib import Path

    from satquery.controller.abstention import (
        DEFAULT_THRESHOLDS_PATH,
        ENV_THRESHOLDS,
    )

    target = Path(
        path or os.environ.get(ENV_THRESHOLDS) or DEFAULT_THRESHOLDS_PATH
    )
    if not target.exists():
        return DEFAULT_WEIGHTS
    try:
        import yaml

        blob = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        section = (blob.get("confidence") or {}).get("weights") or {}
        weights = (
            float(section.get("model", 1.0)),
            float(section.get("agreement", 1.0)),
            float(section.get("input_quality", 1.0)),
        )
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            return DEFAULT_WEIGHTS
        return weights
    except Exception:  # noqa: BLE001 - degradation, not a crash
        return DEFAULT_WEIGHTS


def band(score: float) -> str:
    if score >= HIGH_BAND:
        return "HIGH"
    if score >= MEDIUM_BAND:
        return "MEDIUM"
    return "LOW"


def compute_confidence(
    model_confidence: float,
    manifest: InputManifest,
    agreements: dict[str, float],
    head: str | None = None,
    stubbed: bool = False,
) -> ConfidenceTrace:
    """Combine the three components into the reported confidence.

    `head` names the head that produced `model_confidence`, and is the key
    into the calibration registry. Passing None - which is correct whenever
    no learned tool contributed a score - leaves the value untouched and
    keeps the uncalibrated sentinel in the trace, because recalibrating a
    number that did not come from the fitted head would be worse than not
    calibrating at all.

    `stubbed` says that at least one tool in the plan was a placeholder rather
    than a trained model. The score is then capped at `STUB_CONFIDENCE_CAP`,
    so it can never be reported as HIGH: a stub measures nothing, and the
    combined number would otherwise describe input quality and physics
    agreement alone while reading as a model result.
    """
    raw_model = max(0.0, min(1.0, float(model_confidence)))

    registry = load_registry()
    entry = registry.lookup(head)
    model = entry.apply(raw_model) if entry is not None else raw_model

    agreement = physics_agreement(agreements)
    quality = input_quality(manifest)
    final = geometric_mean(model, agreement, quality, weights=load_weights())
    if stubbed:
        final = min(final, STUB_CONFIDENCE_CAP)

    return ConfidenceTrace(
        final=round(final, 6),
        band=band(final),  # type: ignore[arg-type]
        components=ConfidenceComponentsTrace(
            model=round(model, 6),
            agreement=round(agreement, 6),
            input_quality=round(quality, 6),
        ),
        calibration=ConfidenceCalibrationTrace(
            method=method_label(entry, registry, head),
            # For an affine fit T is the equivalent temperature (1/a); the
            # intercept is what actually did the work and is recorded in the
            # registry alongside the split it was fitted on.
            T=1.0 if entry is None else round(entry.T, 6),
            # -1.0 remains the documented "not measured" sentinel for any
            # head without an accepted fit. A real ECE is in [0, 1], so the
            # two can never be confused; NaN would break JSON and the SSE
            # stream, which is why the sentinel is negative rather than NaN.
            ece_after=UNMEASURED_ECE if entry is None else round(entry.ece_after, 6),
        ),
    )
