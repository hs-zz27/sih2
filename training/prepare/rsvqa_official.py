"""Build the official RSVQA-LR test split that `rsvqa_official_eval.py` reads.

`evaluation/rsvqa_official_eval.py` loads `test_resolved.json` and
`train_majority.json` from its `--data` directory. Both were produced by hand
during Phase 4 and never committed, so the project's headline number - 0.8947
on the official split - could not be reproduced from the repository. With the
checkpoints lost, that gap became load-bearing: a retrained adapter has no
quotable accuracy without this step.

The release (Zenodo 10.5281/zenodo.6344334, CC-BY-4.0) ships, per split:

    LR_split_<split>_questions.json  {"questions": [{id, img_id, type,
                                      question, answers_ids, active}, ...]}
    LR_split_<split>_answers.json    {"answers": [{id, question_id, answer,
                                      active}, ...]}
    Images_LR.zip                    <img_id>.tif

Inactive rows carry only `id` and `active` and are dropped: the test split
holds 33,212 entries of which **10,004 are active**, which is the published
size (10,004 questions over 100 images). Verified against the real files on
2026-09-18; the types are `presence`, `comp`, `count` and `rural_urban`,
matching `PUBLISHED_TYPES` in the evaluator.

Usage:
    python training/prepare/rsvqa_official.py --src data/rsvqa_lr_official \\
        --out data/rsvqa_lr_official

Writes `test_resolved.json` and `train_majority.json` into --out. Images are
not touched: unzip Images_LR.zip so that `<out>/Images_LR/<img_id>.tif` exists.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.metrics.vqa import normalise_answer  # noqa: E402


def load_split(src: Path, split: str) -> list[dict]:
    """Active (question, answer) pairs of one split, as evaluator rows.

    Each row is {img, question, answer, type}: `img` is the image id the
    evaluator appends `.tif` to, and `type` is the official question type.
    """
    questions = json.loads(
        (src / f"LR_split_{split}_questions.json").read_text(encoding="utf-8")
    )["questions"]
    answers = json.loads(
        (src / f"LR_split_{split}_answers.json").read_text(encoding="utf-8")
    )["answers"]

    # Answers are keyed by question, and a question may list several answer
    # ids (different annotators). The first active one is the label, which is
    # what the published evaluation uses.
    by_question: dict[int, list[dict]] = collections.defaultdict(list)
    for answer in answers:
        if answer.get("active"):
            by_question[answer["question_id"]].append(answer)

    rows: list[dict] = []
    for question in questions:
        if not question.get("active"):
            continue
        candidates = by_question.get(question["id"], [])
        if not candidates:
            continue
        rows.append({
            "img": question["img_id"],
            "question": question["question"],
            "answer": candidates[0]["answer"],
            "type": question["type"],
        })
    return rows


def majority_by_type(rows: list[dict]) -> dict[str, str]:
    """Most common answer per question type - the train-fitted constant.

    Fitted on the TRAIN split only. Fitting it on the test split would be the
    optimistic baseline the VRSBench evaluator labels as such, and this one is
    quoted as an honest floor.
    """
    counters: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows:
        counters[row["type"]][normalise_answer(row["answer"])] += 1
    return {t: c.most_common(1)[0][0] for t, c in sorted(counters.items())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", type=Path, required=True,
                   help="directory holding the LR_split_*.json files")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-split", default="train",
                   help="split the constant baseline is fitted on")
    args = p.parse_args()

    test_rows = load_split(args.src, "test")
    if not test_rows:
        raise SystemExit(f"no active test questions under {args.src}")
    train_rows = load_split(args.src, args.train_split)
    if not train_rows:
        raise SystemExit(f"no active {args.train_split} questions under {args.src}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "test_resolved.json").write_text(
        json.dumps(test_rows, indent=1), encoding="utf-8")
    constant = majority_by_type(train_rows)
    (args.out / "train_majority.json").write_text(
        json.dumps(constant, indent=1), encoding="utf-8")

    counts = collections.Counter(r["type"] for r in test_rows)
    print(f"test_resolved.json : {len(test_rows)} questions over "
          f"{len({r['img'] for r in test_rows})} images")
    print(f"  types: {dict(counts.most_common())}")
    print(f"train_majority.json: fitted on {len(train_rows)} {args.train_split} "
          f"questions -> {constant}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
