"""The baseline a CDVQA head has to beat.

CDVQA's answers are skewed: 31% of the test split is "no", and several
question types have a dominant class. A number that sounds like progress -
0.4439 against a previous 0.0000 - can still be worse than answering every
question of a given type with that type's most common answer. It was
(0.5084), which is the whole reason this file exists rather than living as a
one-off calculation in a commit message.

Two baselines, both **fitted on Train and applied to Test**, never reading
test answers:

* **global majority** - one answer for everything;
* **per-type majority** - one answer per question type. This is the one that
  matters, and the one to quote.

Usage:
    python evaluation/cdvqa_baseline.py --compare artifacts/cdvqa/head_test.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(annotations: Path, split: str) -> list[tuple[str, str]]:
    questions = json.loads(
        (annotations / f"{split}_questions.json").read_text(encoding="utf-8")
    )["questions"]
    answers = {
        a["question_id"]: a["answer"]
        for a in json.loads(
            (annotations / f"{split}_answers.json").read_text(encoding="utf-8")
        )["answers"]
    }
    return [(q["type"], answers[q["id"]]) for q in questions if q["id"] in answers]


def fit(train: list[tuple[str, str]]) -> tuple[str, dict[str, str]]:
    overall: Counter[str] = Counter()
    per_type: dict[str, Counter[str]] = defaultdict(Counter)
    for kind, answer in train:
        overall[answer] += 1
        per_type[kind][answer] += 1
    return (
        overall.most_common(1)[0][0],
        {kind: c.most_common(1)[0][0] for kind, c in per_type.items()},
    )


def score(test, global_answer: str, per_type: dict[str, str]) -> dict:
    total: Counter[str] = Counter()
    hit_global: Counter[str] = Counter()
    hit_type: Counter[str] = Counter()
    for kind, answer in test:
        total[kind] += 1
        if answer == global_answer:
            hit_global[kind] += 1
        if answer == per_type.get(kind):
            hit_type[kind] += 1

    n = sum(total.values())
    return {
        "n_questions": n,
        "global_answer": global_answer,
        "global_majority_accuracy": round(sum(hit_global.values()) / n, 6),
        "per_type_majority_accuracy": round(sum(hit_type.values()) / n, 6),
        "per_type_answer": per_type,
        "by_type": {
            kind: {
                "n": total[kind],
                "global_majority": round(hit_global[kind] / total[kind], 6),
                "per_type_majority": round(hit_type[kind] / total[kind], 6),
            }
            for kind in sorted(total)
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, default=Path("data/cdvqa"))
    p.add_argument("--split", default="Test", choices=["Val", "Test"])
    p.add_argument("--compare", type=Path, help="a cdvqa_predict report to compare")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    global_answer, per_type = fit(load(args.annotations, "Train"))
    report = score(load(args.annotations, args.split), global_answer, per_type)

    head = None
    if args.compare:
        head = json.loads(args.compare.read_text(encoding="utf-8"))
        report["head_accuracy"] = head["accuracy"]
        report["head_gain_over_per_type"] = round(
            head["accuracy"] - report["per_type_majority_accuracy"], 6
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    columns = f"{'type':<22}{'n':>7}{'global':>9}{'per-type':>10}"
    if head:
        columns += f"{'head':>9}{'gain':>9}"
    print(columns)
    for kind, stats in report["by_type"].items():
        line = (f"{kind:<22}{stats['n']:>7}{stats['global_majority']:>9.4f}"
                f"{stats['per_type_majority']:>10.4f}")
        if head:
            h = head["by_type"][kind]["accuracy"]
            line += f"{h:>9.4f}{h - stats['per_type_majority']:>+9.4f}"
        print(line)

    line = (f"{'OVERALL':<22}{report['n_questions']:>7}"
            f"{report['global_majority_accuracy']:>9.4f}"
            f"{report['per_type_majority_accuracy']:>10.4f}")
    if head:
        line += (f"{report['head_accuracy']:>9.4f}"
                 f"{report['head_gain_over_per_type']:>+9.4f}")
    print(line)

    if head and report["head_gain_over_per_type"] <= 0:
        print("\nThe head does NOT beat the per-type majority baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
