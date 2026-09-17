"""Build a CDVQA benchmark manifest from the official annotations (task 2.6 eval).

CDVQA is the PS-prescribed benchmark for multitemporal change VQA, and until
now it was the largest scoring gap: `change_vqa_v1` shipped and was never
measured against the split it will be graded on.

Two sources are combined, deliberately:

* **Questions, answers and question types come from the official release**
  (`github.com/YZHJessica/CDVQA`, Apache-2.0) - `Test_questions.json`,
  `Test_answers.json`, `Test_images.json`. That release is annotations only;
  it ships no imagery, exactly like VRSBench (docs/verification.md item 9).
* **Imagery comes from a webdataset mirror** (`ljx620/CDVQA` on HuggingFace),
  whose per-sample JSON carries the official `question_id` / `image_id`.
  The underlying pixels are the SECOND dataset's 512x512 bi-temporal pairs.

Because the imagery is a third-party mirror, every sample it supplies is
**checked against the official annotation** for the same `question_id`:
question text, answer and question type must all agree, or the sample is
dropped and counted. A mirror that has drifted therefore shrinks the manifest
rather than silently corrupting the benchmark.

The mirror stores one copy of the image pair per *question*, so the full test
split is ~32 GB of duplicated pixels for 968 unique pairs. This script
deduplicates by `image_id` on the way out, and works from however many shards
are on disk - a partial download yields a smaller manifest, and the report
records the coverage so the resulting number can be reported honestly.

Usage:
    python training/prepare/cdvqa.py \
        --annotations data/cdvqa \
        --shards data/cdvqa/webdataset/test \
        --split Test \
        --images data/cdvqa/images \
        --out data/cdvqa/cdvqa_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

# The mirror names sample members `<key>.0.img`, `<key>.1.img`, `<key>.json`.
_JSON_SUFFIX = ".json"
_T1_SUFFIX = ".0.img"
_T2_SUFFIX = ".1.img"


def load_official(annotations: Path, split: str) -> dict[int, dict[str, Any]]:
    """Index the official release by question id.

    Returns {question_id: {"question", "answer", "type", "image_id"}}.
    """

    def _read(kind: str) -> Any:
        path = annotations / f"{split}_{kind}.json"
        if not path.exists():
            raise SystemExit(
                f"Missing {path}. Fetch the official annotations first:\n"
                f"  curl -sfLO https://raw.githubusercontent.com/YZHJessica/"
                f"CDVQA/main/{split}_{kind}.json"
            )
        return json.loads(path.read_text(encoding="utf-8"))

    questions = _read("questions")["questions"]
    answers = _read("answers")["answers"]
    images = _read("images")["images"]

    # Answers are keyed to questions, not the other way round, and a question
    # may list several answer ids - take the first active one.
    answer_by_question: dict[int, str] = {}
    for answer in answers:
        if not answer.get("active", True):
            continue
        answer_by_question.setdefault(answer["question_id"], answer["answer"])

    file_by_record = {img["id"]: img["file_name"] for img in images}

    official: dict[int, dict[str, Any]] = {}
    for question in questions:
        if not question.get("active", True):
            continue
        qid = question["id"]
        if qid not in answer_by_question:
            continue
        official[qid] = {
            "question": question["question"],
            "answer": answer_by_question[qid],
            "type": question["type"],
            "image_id": file_by_record.get(question["img_id"]),
        }
    return official


def build_from_second(
    annotations: Path,
    second: Path,
    split: str,
    manifest_out: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build the manifest directly from SECOND, which is the better source.

    CDVQA's splits **partition SECOND's 2,968 labelled pairs** - 1,600 train,
    400 val, 968 test, and every CDVQA image id resolves in SECOND. So the
    imagery can come from the 2.4 GB SECOND archive instead of the ~32 GB
    webdataset mirror, at **100% coverage instead of whatever fraction of the
    shards finished downloading**, with the official annotations as the only
    source of questions and answers.

    That the splits partition SECOND also carries a warning worth stating
    here: SECOND's labels for the 968 *test* pairs are on the same disk.
    Training a semantic head on them would leak the benchmark. Train on the
    ids in CDVQA's Train split only.
    """
    official = load_official(annotations, split)
    if not (second / "im1").is_dir():
        raise SystemExit(
            f"No im1/ under {second}. Download SECOND first:\n"
            "  curl -L -o SECOND_train_set.rar 'https://drive.usercontent.google.com"
            "/download?id=1QlAdzrHpfBIOZ6SK78yHF2i1u6tikmBc&export=download&confirm=t'"
        )

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    root = manifest_out.parent.resolve()

    items: list[dict[str, Any]] = []
    pairs: set[str] = set()
    missing_images: set[str] = set()

    for qid, truth in sorted(official.items()):
        image_id = truth["image_id"]
        if image_id is None:
            continue
        t1 = (second / "im1" / image_id).resolve()
        t2 = (second / "im2" / image_id).resolve()
        if not (t1.exists() and t2.exists()):
            missing_images.add(image_id)
            continue
        pairs.add(image_id)
        items.append(
            {
                "item_id": f"cdvqa_{split.lower()}_{qid:06d}",
                "images": [_relative(t1, root), _relative(t2, root)],
                "question": truth["question"],
                "answer": str(truth["answer"]),
                "answer_type": truth["type"],
            }
        )
        if limit is not None and len(items) >= limit:
            break

    manifest_out.write_text(json.dumps(items, indent=1), encoding="utf-8")

    by_type: dict[str, int] = {}
    for item in items:
        by_type[item["answer_type"]] = by_type.get(item["answer_type"], 0) + 1

    return {
        "split": split,
        "source": "second",
        "n_items": len(items),
        "n_pairs": len(pairs),
        "official_questions": len(official),
        "official_pairs": len({v["image_id"] for v in official.values()}),
        "question_coverage": round(len(items) / len(official), 6) if official else 0.0,
        "by_type": dict(sorted(by_type.items())),
        "shards_read": 0,
        "shards_unreadable": [],
        "dropped_mismatched": 0,
        "dropped_unknown_question_id": 0,
        "dropped_incomplete_sample": len(missing_images),
    }


def _relative(path: Path, root: Path) -> str:
    """Path relative to the manifest's directory, or absolute if not under it."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def build(
    annotations: Path,
    shards: Path,
    split: str,
    images_out: Path,
    manifest_out: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    official = load_official(annotations, split)

    tars = sorted(shards.glob("*.tar"))
    if not tars:
        raise SystemExit(
            f"No .tar shards under {shards}. Download some first, e.g.\n"
            "  curl -sfLO https://huggingface.co/datasets/ljx620/CDVQA/"
            "resolve/main/test/test-00000.tar"
        )

    images_out.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    rel = images_out.resolve().relative_to(manifest_out.parent.resolve())

    items: list[dict[str, Any]] = []
    written_pairs: set[str] = set()
    seen_question_ids: set[int] = set()
    mismatched = 0
    unknown_qid = 0
    incomplete = 0
    unreadable_shards: list[str] = []

    for tar_path in tars:
        try:
            tar = tarfile.open(tar_path)
        except tarfile.TarError:
            # A partially downloaded shard is not an error worth stopping for;
            # it just contributes nothing. Recording it keeps coverage honest.
            unreadable_shards.append(tar_path.name)
            continue

        with tar:
            try:
                members = {m.name: m for m in tar.getmembers() if m.isfile()}
            except tarfile.TarError:
                unreadable_shards.append(tar_path.name)
                continue

            for name in sorted(members):
                if not name.endswith(_JSON_SUFFIX):
                    continue
                key = name[: -len(_JSON_SUFFIX)]
                t1_name, t2_name = key + _T1_SUFFIX, key + _T2_SUFFIX
                if t1_name not in members or t2_name not in members:
                    incomplete += 1
                    continue

                fh = tar.extractfile(members[name])
                if fh is None:
                    incomplete += 1
                    continue
                sample = json.loads(fh.read().decode("utf-8"))
                meta = sample.get("meta", {})
                qid = meta.get("question_id")

                truth = official.get(qid)
                if truth is None:
                    unknown_qid += 1
                    continue

                # Verify the mirror against the official release before trusting
                # its pixels. The question text is compared after stripping the
                # mirror's "Image 1: <image>\nImage 2: <image>" prompt prefix.
                turns = sample.get("conversations", [])
                asked = turns[0]["value"].split("\n")[-1].strip() if turns else ""
                given = turns[1]["value"].strip() if len(turns) > 1 else ""
                if (
                    asked != truth["question"].strip()
                    or given != str(truth["answer"]).strip()
                    or meta.get("question_type") != truth["type"]
                ):
                    mismatched += 1
                    continue

                image_id = truth["image_id"] or meta.get("image_id")
                stem = Path(str(image_id)).stem
                if stem not in written_pairs:
                    pair_ok = True
                    for member_name, suffix in ((t1_name, "t1"), (t2_name, "t2")):
                        src = tar.extractfile(members[member_name])
                        if src is None:
                            pair_ok = False
                            break
                        (images_out / f"{stem}_{suffix}.png").write_bytes(src.read())
                    if not pair_ok:
                        incomplete += 1
                        continue
                    written_pairs.add(stem)

                if qid in seen_question_ids:
                    continue
                seen_question_ids.add(qid)

                items.append(
                    {
                        "item_id": f"cdvqa_{split.lower()}_{qid:06d}",
                        "images": [
                            (rel / f"{stem}_t1.png").as_posix(),
                            (rel / f"{stem}_t2.png").as_posix(),
                        ],
                        "question": truth["question"],
                        "answer": str(truth["answer"]),
                        "answer_type": truth["type"],
                    }
                )
                if limit is not None and len(items) >= limit:
                    break
        if limit is not None and len(items) >= limit:
            break

    items.sort(key=lambda i: i["item_id"])
    manifest_out.write_text(json.dumps(items, indent=1), encoding="utf-8")

    by_type: dict[str, int] = {}
    for item in items:
        by_type[item["answer_type"]] = by_type.get(item["answer_type"], 0) + 1

    return {
        "split": split,
        "n_items": len(items),
        "n_pairs": len(written_pairs),
        "official_questions": len(official),
        "official_pairs": len({v["image_id"] for v in official.values()}),
        "question_coverage": round(len(items) / len(official), 6) if official else 0.0,
        "by_type": dict(sorted(by_type.items())),
        "shards_read": len(tars) - len(unreadable_shards),
        "shards_unreadable": unreadable_shards,
        "dropped_mismatched": mismatched,
        "dropped_unknown_question_id": unknown_qid,
        "dropped_incomplete_sample": incomplete,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, default=Path("data/cdvqa"))
    p.add_argument(
        "--second",
        type=Path,
        help="SECOND root (im1/, im2/). Preferred: it resolves every CDVQA "
        "image id, so the manifest reaches 100%% coverage.",
    )
    p.add_argument(
        "--shards",
        type=Path,
        help="webdataset mirror shards; the fallback when SECOND is not on disk",
    )
    p.add_argument("--split", default="Test", choices=["Train", "Val", "Test", "Test2"])
    p.add_argument("--images", type=Path, default=Path("data/cdvqa/images"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    if not (args.second or args.shards):
        p.error("supply --second (preferred) or --shards")

    if args.second:
        report = build_from_second(
            args.annotations, args.second, args.split, args.out, args.limit
        )
    else:
        report = build(
            args.annotations, args.shards, args.split, args.images, args.out, args.limit
        )
    if report["n_items"] == 0:
        print("No usable items produced.", file=sys.stderr)
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 1

    print(f"Wrote {report['n_items']} items to {args.out}")
    print(
        f"  {report['n_pairs']} unique image pairs of {report['official_pairs']} "
        f"official; question coverage {report['question_coverage']:.1%}"
    )
    for kind, n in report["by_type"].items():
        print(f"    {kind:<20} {n:>6}")
    for key in (
        "dropped_mismatched",
        "dropped_unknown_question_id",
        "dropped_incomplete_sample",
    ):
        if report[key]:
            print(f"  {key}: {report[key]}")
    if report["shards_unreadable"]:
        print(f"  unreadable shards: {len(report['shards_unreadable'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
