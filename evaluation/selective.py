"""Risk-coverage and AURC over the real heads (plan task 3.6).

Reuses the logits `evaluation/calibrate.py` cached for task 3.3, so this
needs neither a GPU nor the datasets on disk once that has run once.

Writes `docs/assets/abstention/selective.json` and one risk-coverage SVG per
signal.

Usage:
    python evaluation/selective.py
    python evaluation/selective.py --cache-dir artifacts/calibration/logits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.abstention import (  # noqa: E402
    SelectiveResult,
    evaluate_selective,
    risk_coverage_svg,
    write_results,
)
from evaluation.calibration import sigmoid, softmax  # noqa: E402

REPORT_DIR = Path("docs/assets/abstention")
CACHE_DIR = Path("artifacts/calibration/logits")


def landcover_signal(logits: np.ndarray, labels: np.ndarray):
    """Per-decision confidence for a multi-label head.

    Each (patch, class) pair is one binary decision. Its confidence is
    `max(p, 1-p)` - how far from undecided the head was - and it is correct
    when the thresholded decision matches the label. Treating the whole
    19-class vector as one prediction would make almost everything wrong and
    say nothing about which individual calls to trust.
    """
    probs = sigmoid(logits).ravel()
    truth = np.asarray(labels, dtype="float64").ravel()
    confidence = np.maximum(probs, 1.0 - probs)
    correct = ((probs >= 0.5).astype("float64") == truth).astype("float64")
    return confidence, correct


def intent_signal(logits: np.ndarray, labels: np.ndarray):
    """Top-1 softmax probability against top-1 correctness."""
    probs = softmax(logits)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(labels).ravel()).astype("float64")
    return confidence, correct


SIGNALS = {
    "landcover": (
        landcover_signal,
        "Track A land-cover head, per (patch, class) decision on the official "
        "BigEarthNet test shard. Measures whether the head's own sigmoid "
        "confidence ranks its correct calls above its incorrect ones. It says "
        "nothing about the SYSTEM's abstention, because no tool currently "
        "feeds this head's probability into the confidence combiner - see the "
        "runtime calibration note for task 3.3.",
    ),
    "intent": (
        intent_signal,
        "Tier-1 router on CLEAN_HOLDOUT, n=29. Far too small for a stable "
        "AURC - the curve is visibly stepped and a single item moves it - but "
        "this is the signal the router's LOW_CONFIDENCE_TOP1 gate actually "
        "uses, so its shape is worth seeing even at this n.",
    ),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    p.add_argument("--signals", nargs="+", default=sorted(SIGNALS),
                   choices=sorted(SIGNALS))
    args = p.parse_args()

    results: list[SelectiveResult] = []
    for name in args.signals:
        cached = args.cache_dir / f"{name}.npz"
        if not cached.exists():
            print(
                f"skipping {name}: no cached logits at {cached}. "
                f"Run evaluation/calibrate.py --heads {name} first.",
                file=sys.stderr,
            )
            continue

        blob = np.load(cached, allow_pickle=False)
        signal_fn, note = SIGNALS[name]
        confidence, correct = signal_fn(blob["logits"], blob["labels"])
        result = evaluate_selective(confidence, correct, name, note)
        results.append(result)
        print(result.summary())
        for target, coverage in result.coverage_at_risk.items():
            shown = "unreachable" if coverage is None else f"{coverage:.1%}"
            print(f"    coverage at {target}: {shown}")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / f"{name}_risk_coverage.svg"
        path.write_text(risk_coverage_svg(result), encoding="utf-8")
        print(f"    wrote {path}")

    if not results:
        print("no signals scored", file=sys.stderr)
        return 1

    out = args.out_dir / "selective.json"
    write_results(results, out)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
