"""Calibration: temperature scaling, ECE, and reliability diagrams (task 3.3).

A confidence score is only useful if it means what it says. "0.8" should be
right about 80% of the time. Nothing in Phase 1 or 2 checked that, which is
why every trace has carried `method="uncalibrated", T=1.0, ece_after=-1.0`
since task 1.3 - a deliberate sentinel rather than a fabricated number.

This module supplies the measurement and the fix:

* **ECE** - expected calibration error, the mean gap between a bin's stated
  confidence and its observed frequency, weighted by bin population.
* **Temperature scaling** - one scalar T per head, fitted by minimising NLL.
  It is monotone, so it cannot change any ranking: mAP, accuracy and AP are
  all identical before and after. It changes only what the numbers *claim*.
* **Reliability diagrams** - emitted as dependency-free SVG (the project has
  no plotting dependency and this is not worth adding one for).

Everything here is numpy + scipy. Producing logits needs torch; scoring and
fitting them does not, so this half runs anywhere, including in CI.

## Two traps this module is built to avoid

**Fitting and evaluating on the same data.** A temperature fitted on a set
and then scored on that same set always looks like an improvement. Every
entry point here splits the data first and reports `ece_after` on points the
fit never saw, with both split sizes recorded.

**Multi-label ECE is dominated by the negative class.** With 19 land-cover
labels and roughly 2-3 positive per patch, ~88% of class-instances are
negative. A model that emitted 0.0 everywhere would post an excellent ECE
while being useless. `positive_rate` is therefore reported next to every
multi-label ECE, and `brier` alongside it, so a suspiciously good number can
be recognised for what it is.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.optimize import minimize_scalar

Mode = Literal["multilabel", "multiclass"]

DEFAULT_BINS = 15

# Bounds for the temperature search. T < 1 sharpens, T > 1 softens; a fitted
# value pinned at either bound means the search saturated and the result
# should not be trusted, which `t_at_bound` records.
T_MIN = 0.05
T_MAX = 20.0


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(np.clip(z, -60.0, 60.0))
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class Bin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    empirical_frequency: float

    @property
    def gap(self) -> float:
        return abs(self.mean_confidence - self.empirical_frequency)


@dataclass
class CalibrationCurve:
    """A reliability curve plus the scalar summaries computed from it."""

    ece: float
    mce: float
    brier: float
    n: int
    positive_rate: float
    n_bins: int
    bins: list[Bin] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bins"] = [asdict(b) for b in self.bins]
        return d


def _curve(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int
) -> CalibrationCurve:
    """Bin `confidence` in [0,1] against binary `correct` and summarise.

    Equal-width bins, which is the standard definition and keeps bin edges
    comparable across models. Empty bins contribute nothing and are still
    emitted with `count=0` so a curve built from a thin region of the
    confidence range is visibly thin rather than silently interpolated.
    """
    confidence = np.asarray(confidence, dtype="float64").ravel()
    correct = np.asarray(correct, dtype="float64").ravel()
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must have the same shape")
    n = confidence.size
    if n == 0:
        raise ValueError("cannot compute a calibration curve from zero points")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins so 1.0 lands in the last bin rather than falling out.
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    bins: list[Bin] = []
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        sel = idx == b
        count = int(sel.sum())
        if count == 0:
            bins.append(Bin(float(edges[b]), float(edges[b + 1]), 0, 0.0, 0.0))
            continue
        mean_conf = float(confidence[sel].mean())
        freq = float(correct[sel].mean())
        gap = abs(mean_conf - freq)
        ece += (count / n) * gap
        mce = max(mce, gap)
        bins.append(Bin(float(edges[b]), float(edges[b + 1]), count, mean_conf, freq))

    return CalibrationCurve(
        ece=float(ece),
        mce=float(mce),
        brier=float(np.mean((confidence - correct) ** 2)),
        n=n,
        positive_rate=float(correct.mean()),
        n_bins=n_bins,
        bins=bins,
    )


def multilabel_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = DEFAULT_BINS
) -> CalibrationCurve:
    """Reliability of independent per-class sigmoid probabilities.

    Every (sample, class) pair is one binary prediction, so an (N, C) score
    matrix contributes N*C points. See the module docstring on why
    `positive_rate` must be read alongside the ECE.
    """
    probs = np.asarray(probs, dtype="float64")
    labels = np.asarray(labels, dtype="float64")
    if probs.shape != labels.shape:
        raise ValueError(
            f"shape mismatch: probs {probs.shape} vs labels {labels.shape}"
        )
    return _curve(probs.ravel(), labels.ravel(), n_bins)


def multiclass_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = DEFAULT_BINS
) -> CalibrationCurve:
    """Reliability of the top-1 softmax probability against top-1 correctness.

    This is the standard multiclass ECE (Guo et al. 2017): it asks only
    whether the winning class's stated probability matches how often that
    winner is right, and says nothing about the rest of the distribution.
    """
    probs = np.asarray(probs, dtype="float64")
    labels = np.asarray(labels, dtype="int64").ravel()
    if probs.ndim != 2:
        raise ValueError("multiclass probs must be (n_samples, n_classes)")
    if probs.shape[0] != labels.size:
        raise ValueError("probs and labels disagree on the number of samples")
    predicted = probs.argmax(axis=1)
    return _curve(probs.max(axis=1), (predicted == labels).astype("float64"), n_bins)


def _nll(
    logits: np.ndarray, labels: np.ndarray, mode: Mode, temperature: float
) -> float:
    t = max(temperature, 1e-6)
    if mode == "multilabel":
        p = np.clip(sigmoid(logits / t), 1e-12, 1 - 1e-12)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
    p = np.clip(softmax(logits / t), 1e-12, 1.0)
    return float(-np.mean(np.log(p[np.arange(labels.size), labels])))


@dataclass
class TemperatureFit:
    T: float
    nll_before: float
    nll_after: float
    n_fit: int
    t_at_bound: bool

    def to_dict(self) -> dict:
        return asdict(self)


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, mode: Mode
) -> TemperatureFit:
    """Fit a single scalar temperature by minimising NLL on the given data.

    NLL rather than ECE directly: NLL is smooth and strictly proper, whereas
    ECE is a binned statistic with a flat, non-convex landscape that a scalar
    optimiser can happily walk to a degenerate temperature. Minimising NLL is
    the standard choice and does not let the fit game the bin edges it will
    later be judged on.
    """
    logits = np.asarray(logits, dtype="float64")
    labels = np.asarray(labels)
    labels = (
        labels.astype("float64") if mode == "multilabel" else labels.astype("int64")
    )

    result = minimize_scalar(
        lambda t: _nll(logits, labels, mode, t),
        bounds=(T_MIN, T_MAX),
        method="bounded",
        options={"xatol": 1e-4},
    )
    t = float(result.x)
    return TemperatureFit(
        T=t,
        nll_before=_nll(logits, labels, mode, 1.0),
        nll_after=_nll(logits, labels, mode, t),
        n_fit=int(logits.shape[0]),
        t_at_bound=bool(t <= T_MIN * 1.01 or t >= T_MAX * 0.99),
    )


def apply_temperature(
    logits: np.ndarray, temperature: float, mode: Mode
) -> np.ndarray:
    """Convert logits to probabilities at the given temperature."""
    z = np.asarray(logits, dtype="float64") / max(float(temperature), 1e-6)
    return sigmoid(z) if mode == "multilabel" else softmax(z)


@dataclass
class AffineFit:
    """Platt scaling: p = sigmoid(a*z + b). Binary / multi-label only."""

    a: float
    b: float
    nll_before: float
    nll_after: float
    n_fit: int
    t_at_bound: bool = False

    @property
    def T(self) -> float:
        """Equivalent temperature, for reporting against a temperature fit."""
        return 1.0 / self.a if self.a else float("inf")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["T_equivalent"] = self.T
        return d


def _affine_nll(logits: np.ndarray, labels: np.ndarray, a: float, b: float) -> float:
    p = np.clip(sigmoid(a * logits + b), 1e-12, 1 - 1e-12)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def fit_affine(logits: np.ndarray, labels: np.ndarray) -> AffineFit:
    """Fit a slope and an intercept on the logits, minimising NLL.

    Temperature scaling is a slope with the intercept pinned at zero, so it
    can rescale confidence but cannot shift it. That is exactly the wrong
    shape for a head trained with a `pos_weight`: upweighting the positive
    class adds an approximately constant offset to every logit, and no single
    temperature removes a constant. Where that is the cause, this fits it and
    the report names the reason rather than recording an unexplained failure.
    """
    from scipy.optimize import minimize

    logits = np.asarray(logits, dtype="float64")
    labels = np.asarray(labels, dtype="float64")
    result = minimize(
        lambda ab: _affine_nll(logits, labels, ab[0], ab[1]),
        x0=np.array([1.0, 0.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 500},
    )
    a, b = float(result.x[0]), float(result.x[1])
    return AffineFit(
        a=a,
        b=b,
        nll_before=_affine_nll(logits, labels, 1.0, 0.0),
        nll_after=_affine_nll(logits, labels, a, b),
        n_fit=int(logits.shape[0]),
    )


def apply_affine(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    return sigmoid(a * np.asarray(logits, dtype="float64") + b)


@dataclass
class CalibrationReport:
    """Everything needed to justify a shipped temperature, or to reject one."""

    head: str
    mode: str
    method: str
    dataset: str
    split_note: str
    n_fit: int
    n_eval: int
    fit: dict
    before: dict
    after: dict
    ece_improvement: float
    accepted: bool
    rejection_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        verdict = "ACCEPTED" if self.accepted else f"REJECTED ({self.rejection_reason})"
        if self.method == "affine":
            params = f"a={self.fit['a']:.4f} b={self.fit['b']:+.4f}"
        else:
            params = f"T={self.fit['T']:.4f}"
        return (
            f"{self.head} [{self.method}]: {params}  "
            f"ECE {self.before['ece']:.4f} -> {self.after['ece']:.4f} "
            f"({self.ece_improvement:+.4f})  "
            f"n_fit={self.n_fit} n_eval={self.n_eval}  {verdict}"
        )


# A temperature fitted on fewer points than this is reported but never
# shipped. With a handful of confidence bins, a few dozen points cannot
# distinguish real miscalibration from sampling noise, and a temperature
# fitted on noise is worse than no temperature at all.
MIN_FIT_SAMPLES = 500


def calibrate_head(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    head: str,
    mode: Mode,
    dataset: str,
    split_note: str,
    method: Literal["temperature", "affine"] = "temperature",
    n_bins: int = DEFAULT_BINS,
    fit_fraction: float = 0.5,
    seed: int = 0,
    min_fit_samples: int = MIN_FIT_SAMPLES,
) -> CalibrationReport:
    """Split, fit on one half, and report ECE before/after on the other half.

    `split_note` is not decoration. It records which evaluation split the
    logits came from and what that split can and cannot distinguish, because
    a calibration curve measured on a set that cannot exercise the failure
    mode looks identical to a well-calibrated model.
    """
    logits = np.asarray(logits, dtype="float64")
    labels = np.asarray(labels)
    n = logits.shape[0]
    if n != labels.shape[0]:
        raise ValueError("logits and labels disagree on the number of samples")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = int(round(n * fit_fraction))
    fit_idx, eval_idx = order[:cut], order[cut:]
    if fit_idx.size == 0 or eval_idx.size == 0:
        raise ValueError(f"fit_fraction={fit_fraction} leaves one side empty at n={n}")

    if method == "affine":
        if mode != "multilabel":
            raise ValueError("affine (Platt) scaling is defined for multilabel only")
        fit: TemperatureFit | AffineFit = fit_affine(logits[fit_idx], labels[fit_idx])
        calibrated = apply_affine(logits[eval_idx], fit.a, fit.b)
    else:
        fit = fit_temperature(logits[fit_idx], labels[fit_idx], mode)
        calibrated = apply_temperature(logits[eval_idx], fit.T, mode)

    curve = multilabel_curve if mode == "multilabel" else multiclass_curve
    before = curve(
        apply_temperature(logits[eval_idx], 1.0, mode), labels[eval_idx], n_bins
    )
    after = curve(calibrated, labels[eval_idx], n_bins)

    accepted, reason = True, None
    if method == "affine" and fit.a <= 0:
        # A negative slope inverts every ranking while still being free to
        # post a fine ECE. Nothing downstream would notice, so it is caught
        # here rather than trusted to be impossible.
        accepted, reason = False, (
            f"fitted slope a={fit.a:.4f} is not positive, which would invert "
            "the ranking the head produces"
        )
    elif fit.n_fit < min_fit_samples:
        accepted, reason = False, (
            f"only {fit.n_fit} fitting samples, below the {min_fit_samples} "
            "needed for a temperature that is not fitted to noise"
        )
    elif fit.t_at_bound:
        accepted, reason = False, (
            f"temperature saturated at the search bound (T={fit.T:.3f})"
        )
    elif after.ece >= before.ece:
        accepted, reason = False, (
            f"ECE did not improve on held-out data "
            f"({before.ece:.4f} -> {after.ece:.4f})"
        )
    elif after.brier > before.brier:
        # ECE alone is gameable: a transform that squashes every probability
        # onto the base rate drives ECE to zero while saying nothing about
        # any individual sample. Brier is strictly proper and rises under
        # exactly that move, so requiring both to improve rules it out.
        accepted, reason = False, (
            f"ECE improved but Brier did not ({before.brier:.5f} -> "
            f"{after.brier:.5f}), which is the signature of a transform "
            "collapsing probabilities toward the base rate"
        )

    return CalibrationReport(
        head=head,
        mode=mode,
        method=method,
        dataset=dataset,
        split_note=split_note,
        n_fit=int(fit_idx.size),
        n_eval=int(eval_idx.size),
        fit=fit.to_dict(),
        before=before.to_dict(),
        after=after.to_dict(),
        ece_improvement=float(before.ece - after.ece),
        accepted=accepted,
        rejection_reason=reason,
    )


# --- Reliability diagram (SVG, no plotting dependency) -----------------------

_SVG_W, _SVG_H, _PAD = 320, 320, 44


def reliability_svg(curve: CalibrationCurve, title: str) -> str:
    """A reliability diagram as a standalone SVG string.

    Bars are observed frequency per bin; the diagonal is perfect calibration.
    Bin population is drawn as a faint histogram along the bottom, because a
    bar sitting far from the diagonal on a bin holding four points is noise,
    not evidence, and a diagram without the counts hides that.
    """
    plot = _SVG_W - 2 * _PAD

    def px(v: float) -> float:
        return _PAD + v * plot

    def py(v: float) -> float:
        return _SVG_H - _PAD - v * plot

    max_count = max((b.count for b in curve.bins), default=0) or 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_SVG_W}" '
        f'height="{_SVG_H}" viewBox="0 0 {_SVG_W} {_SVG_H}" '
        f'font-family="sans-serif" font-size="10">',
        f'<rect width="{_SVG_W}" height="{_SVG_H}" fill="white"/>',
        f'<text x="{_SVG_W / 2}" y="18" text-anchor="middle" font-size="12">'
        f"{title}</text>",
    ]

    for b in curve.bins:
        if not b.count:
            continue
        x0, x1 = px(b.lower), px(b.upper)
        h = (b.count / max_count) * (plot * 0.18)
        parts.append(
            f'<rect x="{x0:.1f}" y="{py(0) - h:.1f}" width="{x1 - x0:.1f}" '
            f'height="{h:.1f}" fill="#c8d6e5"/>'
        )
    for b in curve.bins:
        if not b.count:
            continue
        x0, x1 = px(b.lower), px(b.upper)
        y = py(b.empirical_frequency)
        parts.append(
            f'<rect x="{x0:.1f}" y="{y:.1f}" width="{x1 - x0:.1f}" '
            f'height="{py(0) - y:.1f}" fill="#2e86de" fill-opacity="0.55" '
            f'stroke="#1b4f72" stroke-width="0.5"/>'
        )

    parts += [
        f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(1)}" '
        f'stroke="#c0392b" stroke-width="1" stroke-dasharray="4 3"/>',
        f'<rect x="{_PAD}" y="{_PAD}" width="{plot}" height="{plot}" '
        f'fill="none" stroke="#333" stroke-width="1"/>',
        f'<text x="{_SVG_W / 2}" y="{_SVG_H - 12}" text-anchor="middle">'
        f"confidence</text>",
        f'<text x="14" y="{_SVG_H / 2}" text-anchor="middle" '
        f'transform="rotate(-90 14 {_SVG_H / 2})">observed frequency</text>',
        f'<text x="{px(0) + 4}" y="{_PAD + 14}">ECE {curve.ece:.4f} | '
        f"MCE {curve.mce:.4f} | n={curve.n:,}</text>",
        "</svg>",
    ]
    return "\n".join(parts)


def write_report(reports: list[CalibrationReport], path: str | Path) -> None:
    """Write the machine-readable calibration report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8"
    )
