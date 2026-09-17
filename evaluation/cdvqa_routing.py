"""Does the router send CDVQA's questions to the tool that can answer them?

The agreement check between `evaluation/cdvqa_predict.py` (which calls the
tool directly) and `satquery eval` (which goes through the whole controller)
disagreed by 20 points on identical questions. This measures why.

The tool is only reached when the Tier-1 classifier picks
`TEMPORAL_CHANGE_VQA`. Anything else answers the question with a change mask's
prose, a change caption, or single-image VQA - all of which score zero against
CDVQA's closed vocabulary however good the segmenter is.

## The holdout, and why it is not optional

CDVQA ships **300 distinct question phrasings, and its train and test splits
use exactly the same 300**. A router trained on all of them routes perfectly
and demonstrates nothing: it has seen every test string verbatim. So
`satquery/synth/query_bank.py` embeds only the half selected by
`sha1(template)[0] % 2 == 0`, and this script recomputes that rule to report
two numbers separately:

* **seen templates** - the memorisation number, reported for completeness;
* **held-out templates** - the one that means anything, because those
  phrasings were never in the router's training data.

Usage:
    python evaluation/cdvqa_routing.py --split Test
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

TARGET_TASK = "TEMPORAL_CHANGE_VQA"


def is_trained_template(template: str) -> bool:
    """The split rule embedded in the query bank. Must not drift from it."""
    return hashlib.sha1(template.encode("utf-8")).digest()[0] % 2 == 0


def run(annotations: Path, split: str, per_type: int | None) -> dict:
    from satquery.controller.intent import default_classifier

    classifier = default_classifier()
    questions = json.loads(
        (annotations / f"{split}_questions.json").read_text(encoding="utf-8")
    )["questions"]

    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_type[q["type"]].append(q)

    hits: dict[tuple[str, str], int] = Counter()
    totals: dict[tuple[str, str], int] = Counter()
    elsewhere: dict[str, Counter[str]] = defaultdict(Counter)

    for kind, group in sorted(by_type.items()):
        if per_type:
            group = group[:per_type]
        for q in group:
            seen = "seen" if is_trained_template(q["question"]) else "held_out"
            totals[(kind, seen)] += 1
            task = classifier.predict(q["question"]).task
            if task == TARGET_TASK:
                hits[(kind, seen)] += 1
            else:
                elsewhere[kind][str(task)] += 1

    def rate(keys) -> float:
        t = sum(totals[k] for k in keys)
        return round(sum(hits[k] for k in keys) / t, 6) if t else 0.0

    kinds = sorted(by_type)
    return {
        "split": split,
        "n_questions": sum(totals.values()),
        "routed_to_change_vqa": rate(list(totals)),
        "seen_templates": rate([k for k in totals if k[1] == "seen"]),
        "held_out_templates": rate([k for k in totals if k[1] == "held_out"]),
        "by_type": {
            kind: {
                "n": totals[(kind, "seen")] + totals[(kind, "held_out")],
                "overall": rate([(kind, "seen"), (kind, "held_out")]),
                "seen": rate([(kind, "seen")]),
                "held_out": rate([(kind, "held_out")]),
                "n_held_out": totals[(kind, "held_out")],
                "goes_instead_to": dict(elsewhere[kind].most_common(3)),
            }
            for kind in kinds
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, default=Path("data/cdvqa"))
    p.add_argument("--split", default="Test", choices=["Train", "Val", "Test"])
    p.add_argument("--per-type", type=int, help="cap questions per type")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    report = run(args.annotations, args.split, args.per_type)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'type':<22}{'n':>7}{'overall':>9}{'seen':>8}{'held-out':>10}   goes instead to")
    for kind, stats in report["by_type"].items():
        instead = ", ".join(f"{k}:{v}" for k, v in stats["goes_instead_to"].items())
        print(f"{kind:<22}{stats['n']:>7}{stats['overall']:>9.3f}{stats['seen']:>8.3f}"
              f"{stats['held_out']:>10.3f}   {instead}")
    print(f"\n{'OVERALL':<22}{report['n_questions']:>7}"
          f"{report['routed_to_change_vqa']:>9.3f}{report['seen_templates']:>8.3f}"
          f"{report['held_out_templates']:>10.3f}")
    print("\nThe held-out column is the one that means anything: those phrasings")
    print("were never in the router's training data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
