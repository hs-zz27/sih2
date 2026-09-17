"""Derive the land-cover assertion threshold from cached logits, reproducibly.

`configs/thresholds.yaml` carries `landcover.decision_confidence: 0.70` with
`measured_precision: 0.9107` and a provenance note saying it came from
"positive-side precision/recall over 111,473 (patch, class) decisions ...
after the task 3.3 affine calibration". There was no script behind that
sentence - the number was derived by hand and recorded. This is the script.

Two reasons it has to exist now:

* **The threshold must be refitted for v2.** It was chosen on v1's calibrated
  score distribution. A v2 head with mAP 0.315 rather than 0.285 produces
  different scores, and the point at which its positive assertions are ~91%
  precise is not the same point. Reusing 0.70 would be asserting classes at an
  unmeasured precision, which is the exact thing the threshold exists to
  prevent.
* **A hand-derived number cannot be checked.** This script is validated by
  reproducing v1's own recorded figure from v1's own cache before it is
  trusted on v2 - see `--check` below and `tests/test_assertion_threshold.py`.

WHAT IT COMPUTES

For each candidate threshold t, over every (patch, class) decision on the
official test shard, after applying the head's calibration transform:

    precision(t) = P(class present | calibrated p >= t)
    recall(t)    = P(calibrated p >= t | class present)

and reports the smallest t at which precision reaches `--target-precision`.
That is the positive-side question - "when this tool asserts a class, how
often is it right?" - which the thresholds file says is the right one, as
opposed to the symmetric risk-coverage curve that counts confident negatives
as coverage.

Read-only: writes to `--out` only when asked.

Usage:
    # reproduce v1 from its own cache: must print precision ~0.9107 at 0.70
    python evaluation/assertion_threshold.py --check

    # derive for v2
    python evaluation/assertion_threshold.py \\
        --logits artifacts/calibration_v2/logits/landcover.npz \\
        --calibration configs/calibration.v2.json \\
        --out docs/assets/calibration_v2/assertion_threshold.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

V1_LOGITS = Path("artifacts/calibration/logits/landcover.npz")
V1_CALIBRATION = Path("configs/calibration.json")
HEAD = "SINGLE_LANDCOVER"
_EPS = 1e-6


def calibrated_probabilities(logits: np.ndarray, entry: dict | None) -> np.ndarray:
    """Apply the registry's transform to the LOGIT, as `apply_logit` does.

    Not via a probability: the tool now calls `CalibrationEntry.apply_logit`
    with the raw logit, so this must match that path. Routing through
    sigmoid-then-clamp - which is what the first version of this script did,
    mirroring the runtime as it then was - capped every logit at 13.8 and
    reported that no threshold could reach 0.90 precision, when the raw
    ranking was 89% precise in its top 56.
    """
    z = np.asarray(logits, dtype="float64")
    if entry is None:
        return 1.0 / (1.0 + np.exp(-z))
    if entry.get("method") == "affine":
        z = (entry.get("a") if entry.get("a") is not None else 1.0) * z + (entry.get("b") or 0.0)
    else:
        z = z / max(float(entry.get("T", 1.0)), _EPS)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def curve(probabilities: np.ndarray, labels: np.ndarray,
          thresholds: np.ndarray) -> list[dict]:
    """Positive-side precision and recall at each threshold."""
    p = probabilities.ravel()
    y = labels.ravel().astype(bool)
    positives = int(y.sum())
    rows = []
    for t in thresholds:
        asserted = p >= t
        n_asserted = int(asserted.sum())
        correct = int((asserted & y).sum())
        rows.append({
            "threshold": round(float(t), 4),
            "asserted": n_asserted,
            "precision": (correct / n_asserted) if n_asserted else None,
            "recall": (correct / positives) if positives else None,
        })
    return rows


def choose(rows: list[dict], target_precision: float, min_asserted: int) -> dict | None:
    """The smallest threshold whose precision reaches the target.

    Smallest, because a higher threshold than needed throws away recall for no
    gain in the property being bought. `min_asserted` stops a threshold that
    asserts three things at 100% from winning; three assertions is not a
    precision estimate.
    """
    for row in rows:
        if (row["precision"] is not None and row["precision"] >= target_precision
                and row["asserted"] >= min_asserted):
            return row
    return None


def run(logits_path: Path, calibration_path: Path | None, target: float,
        min_asserted: int, step: float = 0.01) -> dict:
    cache = np.load(logits_path)
    logits, labels = cache["logits"], cache["labels"]

    entry = None
    if calibration_path is not None and calibration_path.is_file():
        registry = json.loads(calibration_path.read_text(encoding="utf-8"))
        entry = registry.get("heads", {}).get(HEAD)

    probabilities = calibrated_probabilities(logits, entry)
    thresholds = np.round(np.arange(0.05, 1.0, step), 4)
    rows = curve(probabilities, labels, thresholds)
    chosen = choose(rows, target, min_asserted)

    return {
        "head": HEAD,
        "logits": str(logits_path),
        "calibration": str(calibration_path) if entry is not None else None,
        "calibration_method": entry.get("method") if entry else "none",
        "n_decisions": int(labels.size),
        "n_patches": int(labels.shape[0]),
        "n_classes": int(labels.shape[1]),
        "positive_rate": float(labels.mean()),
        "target_precision": target,
        "min_asserted": min_asserted,
        "chosen": chosen,
        "curve": rows,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Derive the land-cover assertion threshold.")
    p.add_argument("--logits", type=Path, default=V1_LOGITS)
    p.add_argument("--calibration", type=Path, default=V1_CALIBRATION)
    p.add_argument("--target-precision", type=float, default=0.90)
    p.add_argument("--min-asserted", type=int, default=50)
    p.add_argument("--out", type=Path)
    p.add_argument("--check", action="store_true",
                   help="reproduce v1's recorded 0.9107 @ 0.70 from v1's cache")
    args = p.parse_args()

    result = run(args.logits, args.calibration, args.target_precision, args.min_asserted)

    if args.check:
        at_070 = next(r for r in result["curve"] if abs(r["threshold"] - 0.70) < 1e-9)
        print(f"v1 cache at 0.70: precision {at_070['precision']:.4f}  "
              f"recall {at_070['recall']:.4f}  asserted {at_070['asserted']}")
        print(f"recorded in configs/thresholds.yaml: precision 0.9107  recall 0.0025")
        ok = abs(at_070["precision"] - 0.9107) < 0.002
        print("REPRODUCED" if ok else "DOES NOT REPRODUCE - do not trust this script on v2")
        return 0 if ok else 1

    print(f"{result['n_decisions']} decisions over {result['n_patches']} patches, "
          f"calibration: {result['calibration_method']}")
    print(f"{'t':>6} {'asserted':>9} {'precision':>10} {'recall':>8}")
    for row in result["curve"]:
        if row["threshold"] in (0.5, 0.6, 0.7, 0.8, 0.9) or row is result["chosen"]:
            prec = f"{row['precision']:.4f}" if row["precision"] is not None else "-"
            rec = f"{row['recall']:.4f}" if row["recall"] is not None else "-"
            mark = "  <-- chosen" if row is result["chosen"] else ""
            print(f"{row['threshold']:>6.2f} {row['asserted']:>9} {prec:>10} {rec:>8}{mark}")
    if result["chosen"]:
        c = result["chosen"]
        print(f"\nthreshold for >= {args.target_precision:.2f} precision: "
              f"{c['threshold']:.2f} (precision {c['precision']:.4f}, recall {c['recall']:.4f})")
    else:
        print(f"\nNO threshold reaches {args.target_precision:.2f} precision with "
              f">= {args.min_asserted} assertions")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
