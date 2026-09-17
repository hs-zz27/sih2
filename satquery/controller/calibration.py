"""Runtime side of calibration (plan task 3.3).

`evaluation/calibrate.py` fits the parameters offline and writes
`configs/calibration.json`. This reads that registry and applies it, and it
is deliberately the *only* place a fitted parameter enters the running
system, so there is exactly one answer to "where did this number come from".

Three properties are load-bearing:

* **A head with no accepted fit is left alone.** `lookup` returns None and
  the trace keeps `method="uncalibrated", T=1.0, ece_after=-1.0`. Rejected
  fits are recorded in the registry's `rejected` block for the report and
  are never applied.
* **The transform is monotone**, so it cannot reorder anything. A calibrated
  confidence changes what a number claims, never which answer is given.
* **A missing or malformed registry is not an error.** The system ran
  uncalibrated for two phases and must still boot that way - offline, in
  CI, and on a fresh clone. A broken registry degrades to uncalibrated with
  a recorded reason rather than taking the process down.

## What this does NOT establish

The registry entries were fitted on a specific head's own output
distribution: land-cover on 19 per-class sigmoids, change on per-pixel
sigmoids. The runtime `model_confidence` is a scalar minimum across whichever
learned tools ran. Those are the same quantity only when the answer came from
that head. `apply_to_confidence` therefore requires the caller to name the
head, and the executor passes one only when a learned tool actually produced
the score. Everything else stays uncalibrated on purpose.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

# Overridable so tests, the offline profile and a packaged install can point
# at their own registry without patching module internals.
ENV_VAR = "SATQUERY_CALIBRATION"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "calibration.json"

# Probabilities are clamped away from the open ends before the inverse
# sigmoid, which is infinite at 0 and 1.
_EPS = 1e-6


@dataclass(frozen=True)
class CalibrationEntry:
    head: str
    method: str
    T: float
    a: float | None
    b: float | None
    ece_before: float
    ece_after: float
    n_fit: int
    n_eval: int
    dataset: str
    split_note: str

    def apply(self, probability: float) -> float:
        """Recalibrate a probability by transforming it in logit space.

        Clamps to [_EPS, 1-_EPS] first, so the logit it can recover is capped
        at +/-13.8. For a caller that HAS the logit, that cap is pure loss -
        use `apply_logit`. This entry point is kept for callers that only have
        a probability (the confidence combiner), where the clamp is the
        correct guard against log(0).
        """
        p = min(max(float(probability), _EPS), 1.0 - _EPS)
        return self.apply_logit(math.log(p / (1.0 - p)))

    def apply_logit(self, z: float) -> float:
        """Recalibrate from the logit directly. Lossless.

        WHY THIS EXISTS

        The Phase 5 land-cover head emits logits in [-54, +71]. Routed through
        `apply`, that became: float32 sigmoid saturates to exactly 1.0 above
        z ~ 17, the clamp then caps the recovered logit at 13.8, and the
        affine transform - fitted on the true range - maps everything above to
        the same value. Every one of the 33,620 most confident decisions
        served as p <= 0.365, the tool could assert nothing at any threshold,
        and its raw ranking was 89% precise in the top 56 the whole time.
        Measured 2026-09-12.

        For |z| below the clamp this is identical to `apply(sigmoid(z))`, so
        every v1 head - none of which exceeds it - is unaffected.
        """
        z = float(z)
        if self.method == "affine":
            z = (self.a if self.a is not None else 1.0) * z + (self.b or 0.0)
        else:
            z = z / max(self.T, _EPS)
        return 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))


@dataclass(frozen=True)
class Registry:
    entries: dict[str, CalibrationEntry]
    source: str
    status: str  # "loaded" | "missing" | "invalid"

    def lookup(self, head: str | None) -> CalibrationEntry | None:
        return self.entries.get(head) if head else None


# Which `ToolResult.confidence_method` values denote a probability that this
# calibration is even defined on.
#
# Calibration maps a probability of correctness to a better probability of
# correctness. **No tool currently reports one**, so this set is empty:
#
# * `deterministic` - the index engine. Arithmetic, no probability to fit.
# * `threshold_rule` - stub tools returning a fixed constant. A constant has
#   no reliability curve; recalibrating it would produce a number that looks
#   measured and is not.
# * `sharpness` - `change_mask_v1`. `mean(|p - 0.5|) * 2` measures how
#   decisive the detector was, not whether it was right. A detector that is
#   uniformly saturated and uniformly wrong scores 1.0.
# * `mean_asserted_probability` - `optsar_fusion`. Genuinely a probability,
#   but an aggregate over a threshold-selected subset. A fitted calibration
#   is nonlinear, so transforming a mean of probabilities is not the same as
#   calibrating each class and averaging. Calibrating this head means
#   transforming `p_fused` per class inside the tool.
# * `logprob` - `rs_vqa_v1`. The mean probability of the tokens a greedy
#   decode chose. That is fluency, not correctness: a model can be certain of
#   every token in a confidently wrong answer. The tool's own docstring has
#   always said this value "feeds the confidence combiner rather than being
#   reported as a probability of correctness"; task 3.3 nonetheless listed it
#   here, which was wrong. Removing it changes no observable behaviour - there
#   is no accepted fit for the VQA head either - but the gate should mean what
#   it says.
# * `segmentation_derived` - `change_vqa_v1`'s semantic path. A fixed
#   constant standing in for "the arithmetic is exact, the segmentation is
#   not". There is no per-answer probability to calibrate; the number that
#   describes this head's reliability is the segmenter's mIoU, which lives in
#   its model card rather than in a reliability curve.
#
# An empty set is the correct state, not a gap. The fitted parameters and
# their ECE tables are the deliverable of task 3.3; this path activates by
# itself the moment a tool reports a real per-head P(correct). The
# alternative - calibrating a stub's hardcoded 0.8 with the land-cover head's
# transform - would put a fabricated "calibrated" number in front of a judge.
CALIBRATABLE_CONFIDENCE_METHODS: frozenset[str] = frozenset()

_EMPTY_REASONS = {
    "missing": "no calibration registry on disk",
    "invalid": "calibration registry could not be parsed",
}

_cache: Registry | None = None


def registry_path() -> Path:
    override = os.environ.get(ENV_VAR)
    return Path(override) if override else DEFAULT_PATH


def load_registry(path: str | Path | None = None, *, refresh: bool = False) -> Registry:
    """Load and cache the registry. Never raises."""
    global _cache
    if _cache is not None and path is None and not refresh:
        return _cache

    target = Path(path) if path is not None else registry_path()
    if not target.exists():
        registry = Registry({}, str(target), "missing")
    else:
        try:
            blob = json.loads(target.read_text(encoding="utf-8"))
            entries = {
                head: CalibrationEntry(
                    head=head,
                    method=str(row.get("method", "temperature")),
                    T=float(row.get("T", 1.0)),
                    a=None if row.get("a") is None else float(row["a"]),
                    b=None if row.get("b") is None else float(row["b"]),
                    ece_before=float(row.get("ece_before", -1.0)),
                    ece_after=float(row.get("ece_after", -1.0)),
                    n_fit=int(row.get("n_fit", 0)),
                    n_eval=int(row.get("n_eval", 0)),
                    dataset=str(row.get("dataset", "")),
                    split_note=str(row.get("split_note", "")),
                )
                for head, row in (blob.get("heads") or {}).items()
            }
            registry = Registry(entries, str(target), "loaded")
        except (ValueError, TypeError, KeyError, OSError):
            registry = Registry({}, str(target), "invalid")

    if path is None:
        _cache = registry
    return registry


def reset_cache() -> None:
    """Drop the cached registry (tests, and a reload after re-fitting)."""
    global _cache
    _cache = None


def method_label(
    entry: CalibrationEntry | None, registry: Registry, head: str | None = None
) -> str:
    """What the trace should say the calibration method was.

    A head that has no entry is not simply "uncalibrated". Whether the score
    was not a probability at all, the registry was absent or unreadable, or
    the registry held no accepted fit for this head are four different
    situations with four different fixes, and the trace should not blur them
    into one word.
    """
    if entry is not None:
        return f"{entry.method}:{entry.head}"
    if head is None:
        return "uncalibrated (score is not a calibratable probability)"
    if registry.status != "loaded":
        return f"uncalibrated ({_EMPTY_REASONS[registry.status]})"
    return "uncalibrated (no accepted fit for this head)"
