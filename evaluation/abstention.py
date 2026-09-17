"""Risk-coverage analysis and AURC (plan task 3.6).

Abstention is only useful if the thing it abstains on is the thing it gets
wrong. Accuracy cannot show that; a risk-coverage curve can. Sort predictions
by confidence, answer the most confident fraction (the **coverage**) and
abstain on the rest, and plot the error rate among those answered (the
**risk**). A useful confidence signal produces a curve that starts low and
rises; a useless one produces a flat line at the base error rate.

**AURC** is the area under that curve. **E-AURC** is what should actually be
compared, and the distinction matters enough to state plainly:

    AURC depends mostly on how accurate the model is. A model with 30% error
    has a high AURC even with perfect confidence ranking, simply because it
    is wrong a lot. Comparing raw AURC across models compares their accuracy,
    not their confidence.

    E-AURC = AURC - AURC_optimal, where AURC_optimal is the area a perfect
    ranking would achieve at the same accuracy (every correct prediction
    ranked above every incorrect one). It is zero for a perfect ranking
    regardless of accuracy, so it isolates the only thing selective
    prediction is about.

Both are reported. E-AURC is the one that answers "is this confidence signal
worth anything".

Everything here is numpy-only, so it runs in CI without torch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RiskCoveragePoint:
    coverage: float
    risk: float
    n_answered: int
    threshold: float


@dataclass
class SelectiveResult:
    name: str
    n: int
    base_error: float
    aurc: float
    aurc_optimal: float
    e_aurc: float
    coverage_at_risk: dict
    curve: list[RiskCoveragePoint] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["curve"] = [asdict(p) for p in self.curve]
        return d

    def summary(self) -> str:
        return (
            f"{self.name}: n={self.n:,} base_error={self.base_error:.4f} "
            f"AURC={self.aurc:.4f} optimal={self.aurc_optimal:.4f} "
            f"E-AURC={self.e_aurc:.4f}"
        )


def risk_coverage_curve(
    confidence: np.ndarray, correct: np.ndarray
) -> list[RiskCoveragePoint]:
    """Risk at every coverage level, most-confident first.

    Ties are not broken arbitrarily in the caller's favour: the sort is
    stable on the confidence value alone, so a model that emits one constant
    confidence produces the flat line it deserves rather than a curve shaped
    by the order its predictions happened to arrive in.
    """
    confidence = np.asarray(confidence, dtype="float64").ravel()
    correct = np.asarray(correct, dtype="float64").ravel()
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must have the same shape")
    n = confidence.size
    if n == 0:
        raise ValueError("cannot build a risk-coverage curve from zero points")

    order = np.argsort(-confidence, kind="stable")
    errors = 1.0 - correct[order]
    cumulative_errors = np.cumsum(errors)
    counts = np.arange(1, n + 1)

    return [
        RiskCoveragePoint(
            coverage=float(k / n),
            risk=float(cumulative_errors[k - 1] / k),
            n_answered=int(k),
            threshold=float(confidence[order][k - 1]),
        )
        for k in counts
    ]


def aurc(curve: list[RiskCoveragePoint]) -> float:
    """Area under the risk-coverage curve, as the mean risk over coverages."""
    if not curve:
        return 0.0
    return float(np.mean([p.risk for p in curve]))


def optimal_aurc(correct: np.ndarray) -> float:
    """AURC of a perfect confidence ranking at the same accuracy.

    Every correct prediction ordered above every incorrect one. This is the
    floor E-AURC measures the distance from.
    """
    correct = np.asarray(correct, dtype="float64").ravel()
    perfect = risk_coverage_curve(correct, correct)
    return aurc(perfect)


def coverage_at_risk(
    curve: list[RiskCoveragePoint], targets: tuple[float, ...] = (0.05, 0.10, 0.20)
) -> dict:
    """Highest coverage whose risk stays at or below each target.

    This is the operationally useful reading: "answering the most confident
    62% of queries keeps the error rate under 10%". Reported as None when no
    coverage achieves the target, rather than as 0.0 - "impossible" and "we
    can answer nothing" are different statements.
    """
    result: dict[str, float | None] = {}
    for target in targets:
        achievable = [p.coverage for p in curve if p.risk <= target]
        result[f"risk<={target:.2f}"] = max(achievable) if achievable else None
    return result


def evaluate_selective(
    confidence: np.ndarray,
    correct: np.ndarray,
    name: str,
    note: str = "",
    max_curve_points: int = 200,
) -> SelectiveResult:
    """Full selective-prediction report for one confidence signal."""
    curve = risk_coverage_curve(confidence, correct)
    correct = np.asarray(correct, dtype="float64").ravel()

    # The stored curve is thinned for the report; every scalar above is
    # computed on the full curve first so nothing depends on the thinning.
    area = aurc(curve)
    optimal = optimal_aurc(correct)
    coverages = coverage_at_risk(curve)

    step = max(1, len(curve) // max_curve_points)
    thinned = curve[::step]
    if thinned[-1] is not curve[-1]:
        thinned.append(curve[-1])

    return SelectiveResult(
        name=name,
        n=int(correct.size),
        base_error=float(1.0 - correct.mean()),
        aurc=area,
        aurc_optimal=optimal,
        e_aurc=area - optimal,
        coverage_at_risk=coverages,
        curve=thinned,
        note=note,
    )


# --- Risk-coverage plot (SVG, no plotting dependency) ------------------------

_W, _H, _PAD = 340, 300, 46


def risk_coverage_svg(result: SelectiveResult, title: str | None = None) -> str:
    """Risk against coverage, with the base error rate drawn as a reference.

    The base-error line is the point of the plot. A confidence signal that
    tracks it is worthless no matter how low the absolute risk looks, and
    without the line on the chart that is not visible.
    """
    plot_w = _W - 2 * _PAD
    plot_h = _H - 2 * _PAD
    risks = [p.risk for p in result.curve] or [0.0]
    y_max = max(max(risks), result.base_error) * 1.15 or 1.0

    def px(coverage: float) -> float:
        return _PAD + coverage * plot_w

    def py(risk: float) -> float:
        return _H - _PAD - (risk / y_max) * plot_h

    points = " ".join(f"{px(p.coverage):.1f},{py(p.risk):.1f}" for p in result.curve)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_W}" height="{_H}" '
        f'viewBox="0 0 {_W} {_H}" font-family="sans-serif" font-size="10">',
        f'<rect width="{_W}" height="{_H}" fill="white"/>',
        f'<text x="{_W / 2}" y="18" text-anchor="middle" font-size="12">'
        f"{title or result.name}</text>",
        f'<line x1="{px(0)}" y1="{py(result.base_error):.1f}" '
        f'x2="{px(1)}" y2="{py(result.base_error):.1f}" stroke="#c0392b" '
        f'stroke-width="1" stroke-dasharray="4 3"/>',
        f'<text x="{px(1) - 4}" y="{py(result.base_error) - 4:.1f}" '
        f'text-anchor="end" fill="#c0392b">base error '
        f"{result.base_error:.3f}</text>",
        f'<polyline points="{points}" fill="none" stroke="#2e86de" '
        f'stroke-width="1.6"/>',
        f'<rect x="{_PAD}" y="{_PAD}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="#333" stroke-width="1"/>',
        f'<text x="{_W / 2}" y="{_H - 12}" text-anchor="middle">coverage</text>',
        f'<text x="14" y="{_H / 2}" text-anchor="middle" '
        f'transform="rotate(-90 14 {_H / 2})">risk (error among answered)</text>',
        f'<text x="{_PAD + 4}" y="{_PAD + 14}">AURC {result.aurc:.4f} | '
        f"E-AURC {result.e_aurc:.4f} | n={result.n:,}</text>",
        "</svg>",
    ]
    return "\n".join(parts)


def write_results(results: list[SelectiveResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8"
    )
