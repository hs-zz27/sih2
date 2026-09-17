"""The CDVQA oracle: how much of the benchmark is arithmetic?

`satquery/verify/semantic_change.py` claims every CDVQA question type is
derivable in closed form from a pair of semantic change maps. This measures
that claim against ground-truth maps, which separates two things the
end-to-end number confuses:

* **the ceiling** - what a perfect semantic change segmenter would score
  through this answer layer, and
* **the model's error** - how far a trained segmenter falls below it.

Run it on `Train` while developing the derivation rules, and on `Val` to check
those rules were not fitted to the training images. **Do not develop against
`Test`**: CDVQA's splits partition SECOND's 2,968 labelled pairs (968 test,
1,600 train, 400 val), so the test labels are on disk and it would be easy to
tune against them by accident. The Test oracle is a single confirmation run,
reported once.

Usage:
    python evaluation/cdvqa_oracle.py --split Val --second data/second
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from satquery.verify.semantic_change import answer, decode_label


def load_split(annotations: Path, split: str) -> list[dict[str, Any]]:
    def read(kind: str) -> Any:
        return json.loads(
            (annotations / f"{split}_{kind}.json").read_text(encoding="utf-8")
        )

    questions = read("questions")["questions"]
    answers = {a["question_id"]: a["answer"] for a in read("answers")["answers"]}
    files = {i["id"]: i["file_name"] for i in read("images")["images"]}

    return [
        {
            "question": q["question"],
            "answer": answers[q["id"]],
            "type": q["type"],
            "image": files[q["img_id"]],
        }
        for q in questions
        if q.get("active", True) and q["id"] in answers
    ]


def run(annotations: Path, second: Path, split: str, limit: int | None) -> dict:
    from PIL import Image

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
    mistakes: list[dict] = []

    for name in images:
        t1 = decode_label(np.asarray(Image.open(second / "label1" / name).convert("RGB")))
        t2 = decode_label(np.asarray(Image.open(second / "label2" / name).convert("RGB")))

        for row in by_image[name]:
            kind = row["type"]
            total[kind] += 1
            predicted = answer(row["question"], t1, t2)
            if predicted is None:
                deferred[kind] += 1
            elif predicted == row["answer"]:
                correct[kind] += 1
            elif len(mistakes) < 25:
                mistakes.append(
                    {
                        "image": name,
                        "question": row["question"],
                        "truth": row["answer"],
                        "derived": predicted,
                        "type": kind,
                    }
                )

    n = sum(total.values())
    return {
        "split": split,
        "n_images": len(images),
        "n_questions": n,
        "oracle_accuracy": round(sum(correct.values()) / n, 6) if n else 0.0,
        "n_deferred": sum(deferred.values()),
        "by_type": {
            kind: {
                "n": total[kind],
                "accuracy": round(correct[kind] / total[kind], 6),
                "deferred": deferred[kind],
            }
            for kind in sorted(total)
        },
        "mistakes": mistakes,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, default=Path("data/cdvqa"))
    p.add_argument("--second", type=Path, default=Path("data/second"))
    p.add_argument("--split", default="Val", choices=["Train", "Val", "Test"])
    p.add_argument("--limit", type=int, help="first N images only")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    report = run(args.annotations, args.second, args.split, args.limit)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{args.split}: {report['n_images']} images, {report['n_questions']} questions")
    print(f"{'type':<22}{'n':>7}{'oracle':>10}{'deferred':>10}")
    for kind, stats in report["by_type"].items():
        print(f"{kind:<22}{stats['n']:>7}{stats['accuracy']:>10.4f}{stats['deferred']:>10}")
    print(f"{'OVERALL':<22}{report['n_questions']:>7}{report['oracle_accuracy']:>10.4f}"
          f"{report['n_deferred']:>10}")
    if report["mistakes"]:
        print("\nfirst disagreements:")
        for m in report["mistakes"][:8]:
            print(f"  [{m['type']}] {m['question'][:58]}")
            print(f"      truth {m['truth']!r}  derived {m['derived']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
