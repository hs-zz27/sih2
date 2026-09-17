"""Why the CDVQA score is zero: format artifact, or real failure?

A benchmark number of exactly 0.0000 on every question type is suspicious
enough to check the measurement before believing it. CDVQA scores by exact
match against a closed vocabulary (`yes`/`no`, six SECOND semantic classes,
ten decile bins) while this system answers in prose, so exact match could in
principle be hiding answers that are right but differently worded.

This script computes a deliberately **lenient** upper bound on the same
predictions file:

* yes/no types (`change_or_not`, `increase_or_not`, `decrease_or_not`) score
  correct if the prose merely *starts* with the right yes or no;
* `change_ratio` scores correct if a percentage can be pulled out of the prose
  and it lands in the right decile bin, with the polarity of "how much has
  *not* changed" handled.

If the lenient number is also ~0, the exact-match zero is the system's real
score and not an artifact of the metric. That is the finding: it is.

This number is a **diagnostic, not a benchmark result.** CDVQA's published
protocol is exact match, and `docs/phase1-status.md` reports the exact-match
figure as the score.

Usage:
    python evaluation/cdvqa_diagnosis.py \
        --predictions artifacts/cdvqa/test_2900.json \
        --manifest data/cdvqa/cdvqa_test.json \
        --out artifacts/cdvqa/diagnosis.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

YES_NO_TYPES = {"change_or_not", "increase_or_not", "decrease_or_not"}

_YES_NO = re.compile(r"^(yes|no)\b")
_CHANGE_PCT = re.compile(r"about ([\d.]+)% of the scene changed")
# "how much has NOT changed" / "non-change ratio" invert the measured fraction.
_INVERTED = re.compile(r"\bnot\b|unchanged|non-change")


def lenient_prediction(answer_type: str, question: str, answer: str) -> str | None:
    """The most generous reading of a prose answer, or None if there is none."""
    text = answer.strip().lower()

    if answer_type in YES_NO_TYPES:
        match = _YES_NO.match(text)
        return match.group(1) if match else None

    if answer_type == "change_ratio":
        match = _CHANGE_PCT.match(text)
        if not match:
            return None
        pct = float(match.group(1))
        if _INVERTED.search(question.lower()):
            pct = 100.0 - pct
        low = min(int(pct // 10) * 10, 90)
        return f"{low}_to_{low + 10}"

    # Every other type needs a semantic change class, which no shipped tool
    # produces; there is no generous reading of prose that invents one.
    return None


def diagnose(predictions_path: Path, manifest_path: Path) -> dict[str, Any]:
    report = json.loads(predictions_path.read_text(encoding="utf-8"))
    predictions = report.get("predictions", report)
    items = {i["item_id"]: i for i in json.loads(manifest_path.read_text(encoding="utf-8"))}

    per_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "answered": 0, "lenient_correct": 0}
    )
    shapes: dict[str, int] = defaultdict(int)

    for prediction in predictions:
        item = items.get(prediction["item_id"])
        if item is None:
            continue
        answer_type = item["answer_type"]
        bucket = per_type[answer_type]
        bucket["n"] += 1

        text = str(prediction.get("answer", ""))
        shapes[_shape(text)] += 1

        if prediction.get("abstained"):
            continue
        bucket["answered"] += 1

        guess = lenient_prediction(answer_type, item["question"], text)
        if guess is not None and guess == str(item["answer"]).lower():
            bucket["lenient_correct"] += 1

    total = sum(b["n"] for b in per_type.values())
    correct = sum(b["lenient_correct"] for b in per_type.values())

    return {
        "n_items": total,
        "lenient_accuracy": round(correct / total, 6) if total else 0.0,
        "single_token_predictions": sum(
            1 for p in predictions if len(str(p.get("answer", "")).split()) <= 2
        ),
        "by_answer_type": {
            t: {
                **b,
                "lenient_accuracy": round(b["lenient_correct"] / b["n"], 6) if b["n"] else 0.0,
            }
            for t, b in sorted(per_type.items())
        },
        "answer_shapes": dict(sorted(shapes.items(), key=lambda kv: -kv[1])),
    }


def _shape(answer: str) -> str:
    """Which component produced this answer, read off its opening words."""
    text = answer.strip()
    if text.startswith("combined confidence"):
        return "abstained_on_confidence"
    if text.startswith("I cannot answer"):
        return "rs_vqa_refusal"
    if text.startswith("About "):
        return "change_mask_scene_percentage"
    return "other_prose"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    report = diagnose(args.predictions, args.manifest)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'type':<22}{'n':>6}{'answered':>10}{'lenient ok':>12}{'lenient acc':>13}")
    for kind, b in report["by_answer_type"].items():
        print(
            f"{kind:<22}{b['n']:>6}{b['answered']:>10}"
            f"{b['lenient_correct']:>12}{b['lenient_accuracy']:>13.4f}"
        )
    print(
        f"{'OVERALL':<22}{report['n_items']:>6}{'':>10}"
        f"{sum(b['lenient_correct'] for b in report['by_answer_type'].values()):>12}"
        f"{report['lenient_accuracy']:>13.4f}"
    )
    print(f"\nsingle-token predictions: {report['single_token_predictions']}")
    for shape, n in report["answer_shapes"].items():
        print(f"  {n:>5}  {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
