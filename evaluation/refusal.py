"""Refusal evaluation for the Track B retrain (plan task 3.1).

The acceptance criterion is "model declines appropriately", and the obvious
way to measure it - refusal recall on the refusal examples - is close to
useless on its own. A model that refuses *everything* scores 100%.

So three numbers, and the second is the one that decides whether the training
worked:

* **refusal recall** - of the examples that should be declined, how many were?
* **false-refusal rate** - of the ANSWERABLE examples, how many were declined
  anyway? A model that learned "questions about SWIR get refused" rather than
  "this sensor has no SWIR" shows up here and nowhere else.
* **lexical-shortcut rate** - the sharpest probe. The `not_in_image` refusals
  in the mix use the *same question wording* as answerable examples and differ
  only in which tile they are asked about ("Is there water visible in this
  image?" on a tile with water and on one without). This measures accuracy on
  exactly those matched pairs. A model that has learned a lexical rule scores
  at chance here while looking fine on the other two.

Without the third number, "declines appropriately" can be satisfied by a model
that has learned nothing about images at all.

Usage:
    python evaluation/refusal.py --adapter checkpoints/track_b_v1/adapter_final \
        --base models/qwen25_vl_3b --data data/instruct_mix
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPORT = Path("docs/assets/refusal/refusal.json")

# A response counts as a refusal when it declines. Matched on content rather
# than exact string: the model will paraphrase, and requiring the canonical
# sentence verbatim would score paraphrases as failures to refuse.
REFUSAL_MARKERS = (
    "cannot answer",
    "can not answer",
    "unable to answer",
    "does not support",
    "not present in this scene",
    "i don't know",
    "i do not know",
)


def is_refusal(text: str) -> bool:
    lowered = str(text).lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def load_examples(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def matched_pairs(examples: list[dict]) -> list[tuple[dict, dict]]:
    """Answerable/refusal pairs sharing an identical question.

    These are the lexical-shortcut probe: the wording is byte-identical and
    only the image differs, so a model answering from the text alone cannot
    get both right.
    """
    by_question: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"answerable": [], "refusal": []}
    )
    for example in examples:
        bucket = "refusal" if example.get("kind") == "refusal" else "answerable"
        by_question[example["question"]][bucket].append(example)

    pairs = []
    for groups in by_question.values():
        for answerable, refusal in zip(
            groups["answerable"], groups["refusal"], strict=False
        ):
            pairs.append((answerable, refusal))
    return pairs


def evaluate(predict, examples: list[dict], pairs: list[tuple[dict, dict]]) -> dict:
    """Score a `predict(example) -> str` callable.

    Predictions are memoised per example. The metrics below each need the same
    outputs - recall, false-refusal, per-reason and the matched pairs all
    re-read them - and without this a real model regenerates every example
    about four times. Free for the degenerate baselines, four GPU passes for
    an adapter.
    """
    cache: dict[int, str] = {}
    # Bound to a separate name: `predict = cached` would make `cached` close
    # over itself and recurse until the stack dies.
    generate = predict

    def predict(example: dict) -> str:  # noqa: F811 - deliberate shadow
        key = id(example)
        if key not in cache:
            cache[key] = generate(example)
        return cache[key]

    refusals = [e for e in examples if e.get("kind") == "refusal"]
    answerable = [e for e in examples if e.get("kind") != "refusal"]

    declined_when_should = sum(is_refusal(predict(e)) for e in refusals)
    declined_when_should_not = sum(is_refusal(predict(e)) for e in answerable)

    by_reason = Counter()
    for example in refusals:
        if is_refusal(predict(example)):
            by_reason[example.get("refusal_reason", "unknown")] += 1

    pair_correct = 0
    for answerable_case, refusal_case in pairs:
        if not is_refusal(predict(answerable_case)) and is_refusal(
            predict(refusal_case)
        ):
            pair_correct += 1

    return {
        "n_refusal": len(refusals),
        "n_answerable": len(answerable),
        "refusal_recall": (
            declined_when_should / len(refusals) if refusals else None
        ),
        "false_refusal_rate": (
            declined_when_should_not / len(answerable) if answerable else None
        ),
        "refusal_recall_by_reason": {
            reason: count / max(
                1, sum(1 for e in refusals if e.get("refusal_reason") == reason)
            )
            for reason, count in by_reason.items()
        },
        "n_matched_pairs": len(pairs),
        "lexical_shortcut_probe": (
            pair_correct / len(pairs) if pairs else None
        ),
        "note": (
            "lexical_shortcut_probe scores matched pairs where the question "
            "wording is identical and only the image differs. A model that "
            "learned to refuse on phrasing scores ~0.5 here (it gets one of "
            "each pair right) while refusal_recall alone can still look "
            "excellent. Read it first."
        ),
    }


def always_refuse(_example) -> str:
    return "I cannot answer that from this image."


def never_refuse(_example) -> str:
    return "Yes."


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/instruct_mix"))
    p.add_argument("--split", default="val")
    p.add_argument("--adapter", type=Path)
    p.add_argument("--base", type=Path, default=Path("models/qwen25_vl_3b"))
    p.add_argument("--out", type=Path, default=REPORT)
    p.add_argument(
        "--baselines-only", action="store_true",
        help="score the degenerate baselines without loading a model",
    )
    args = p.parse_args()

    examples = load_examples(args.data / f"{args.split}.jsonl")
    pairs = matched_pairs(examples)
    print(f"{len(examples)} examples, {len(pairs)} matched pairs")

    results = {
        # Degenerate baselines are scored FIRST and always. They are what make
        # the model's numbers interpretable: "refuses everything" and "refuses
        # nothing" bracket the metric, and a model that does not beat both on
        # the matched-pair probe has learned nothing image-conditional.
        "always_refuse": evaluate(always_refuse, examples, pairs),
        "never_refuse": evaluate(never_refuse, examples, pairs),
    }

    if not args.baselines_only and args.adapter:
        if not args.adapter.exists():
            print(
                f"adapter {args.adapter} not found; run "
                f"training/track_b_vlm_qlora.py first",
                file=sys.stderr,
            )
        else:
            # Delegated rather than reimplemented. An earlier version called a
            # `load_handle_for_eval` helper that never existed, so this branch
            # raised ImportError the moment an adapter was passed - the tests
            # only exercised --baselines-only and never reached it.
            # `track_b_eval.Adapter` is the one loader, and it mirrors the
            # deployed tool's quantisation and decode settings.
            from evaluation.track_b_eval import Adapter

            adapter = Adapter(args.base, args.adapter)
            try:
                results["track_b"] = evaluate(
                    lambda e: adapter.answer(
                        args.data / e["image"], e["question"]
                    ),
                    examples,
                    pairs,
                )
            finally:
                adapter.close()

    for name, row in results.items():
        print(f"\n{name}:")
        for key in (
            "refusal_recall", "false_refusal_rate", "lexical_shortcut_probe"
        ):
            value = row[key]
            print(f"  {key:24s} "
                  f"{'n/a' if value is None else f'{value:.4f}'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"split": args.split, "n_examples": len(examples), "results": results},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
