"""Score the semantic change head on CDVQA's full test split.

Why this exists alongside `satquery eval`: the benchmark harness runs one
controller pass per item, and CDVQA's test split is **39,686 questions over
968 image pairs** - about 41 questions per pair. At the measured ~0.75 s per
controller item that is over eight hours to answer the same 968 image pairs
forty-one times each.

So this predicts each pair's class maps **once** and answers all of its
questions from them. It calls the *same* two functions the tool calls -
`change_vqa.predict_class_maps` and `semantic_change.answer` - so the numbers
describe the shipped code path, not a reimplementation of it.

What it therefore does **not** exercise, and what the controller run is for:
ingest, modality inference, the config gate, routing, plan validation and the
confidence combiner. Run `satquery eval` on a subset to confirm the wired path
agrees with this one; a disagreement means the tool is reached differently in
the pipeline than it is here, which is exactly the kind of gap that made the
task-3.5 entailment gate score identically to its own baseline.

Pair this with `evaluation/cdvqa_oracle.py`. The oracle is the ceiling this
answer layer allows (0.9975 on Test); the gap between the two numbers is the
segmenter's error and nothing else.

Usage:
    SATQUERY_CHANGE_VQA=checkpoints/change_vqa/best.pt \
    python evaluation/cdvqa_predict.py --split Test --out artifacts/cdvqa/head_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# The docstring above documents `python evaluation/cdvqa_predict.py ...`, and
# that invocation raised `ImportError: attempted relative import with no known
# parent package` before this line existed: run as a script, `evaluation` is
# not a package, so `from .cdvqa_oracle import ...` cannot resolve. Same
# pattern as evaluation/adversarial.py, which is the runnable sibling that got
# it right.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satquery.ingest import ingest  # noqa: E402
from satquery.tools import change_vqa  # noqa: E402
from satquery.verify.semantic_change import answer  # noqa: E402

from evaluation.cdvqa_oracle import load_split  # noqa: E402


def run(annotations: Path, second: Path, split: str, limit: int | None) -> dict:
    rows = load_split(annotations, split)
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_image[row["image"]].append(row)

    images = sorted(by_image)
    if limit:
        images = images[:limit]

    correct: Counter[str] = Counter()
    total: Counter[str] = Counter()
    deferred: Counter[str] = Counter()
    started = time.perf_counter()

    for n, name in enumerate(images, 1):
        manifest = ingest([second / "im1" / name, second / "im2" / name])
        t1, t2, checkpoint = change_vqa.predict_class_maps(*manifest.images)

        for row in by_image[name]:
            kind = row["type"]
            total[kind] += 1
            predicted = answer(row["question"], t1, t2)
            if predicted is None:
                deferred[kind] += 1
            elif predicted == row["answer"]:
                correct[kind] += 1

        if n % 100 == 0:
            print(f"  {n}/{len(images)} pairs ({time.perf_counter()-started:.0f}s)",
                  flush=True)

    n_questions = sum(total.values())
    n_correct = sum(correct.values())
    n_deferred = sum(deferred.values())
    answered = n_questions - n_deferred

    return {
        "split": split,
        "checkpoint": checkpoint,
        "n_images": len(images),
        "n_questions": n_questions,
        # Accuracy over ALL questions: a deferral counts against it, the same
        # way the harness counts an abstention.
        "accuracy": round(n_correct / n_questions, 6) if n_questions else 0.0,
        "accuracy_when_answered": round(n_correct / answered, 6) if answered else 0.0,
        "coverage": round(answered / n_questions, 6) if n_questions else 0.0,
        "runtime_s": round(time.perf_counter() - started, 1),
        "by_type": {
            kind: {
                "n": total[kind],
                "accuracy": round(correct[kind] / total[kind], 6),
                "deferred": deferred[kind],
            }
            for kind in sorted(total)
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, default=Path("data/cdvqa"))
    p.add_argument("--second", type=Path, default=Path("data/second"))
    p.add_argument("--split", default="Test", choices=["Train", "Val", "Test"])
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    ok, reason = change_vqa.semantic_available()
    if not ok:
        raise SystemExit(
            f"the semantic head is not available: {reason}\n"
            "  set SATQUERY_CHANGE_VQA to a trained checkpoint, e.g.\n"
            "  SATQUERY_CHANGE_VQA=checkpoints/change_vqa/best.pt"
        )

    report = run(args.annotations, args.second, args.split, args.limit)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{args.split}: {report['n_images']} pairs, {report['n_questions']} questions"
          f"  ({report['runtime_s']}s)")
    print(f"{'type':<22}{'n':>7}{'accuracy':>10}{'deferred':>10}")
    for kind, stats in report["by_type"].items():
        print(f"{kind:<22}{stats['n']:>7}{stats['accuracy']:>10.4f}{stats['deferred']:>10}")
    print(f"{'OVERALL':<22}{report['n_questions']:>7}{report['accuracy']:>10.4f}"
          f"{report['n_questions'] - int(report['coverage'] * report['n_questions']):>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
