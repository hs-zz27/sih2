"""Convert VRSBench into the shared instruction JSONL format.

Every corpus gets one of these, so `track_b_vlm_qlora.py` stays identical no
matter which dataset it trains on. Output is one object per line:

    {"image": "<path relative to --out's parent>", "question": ..., "answer": ...,
     "source": "vrsbench", "kind": "vqa|caption|referring"}

ON THE INPUT SCHEMA: VRSBench's exact field names have not been confirmed
against a real download (docs/verification.md item 9 is still open), so this
reads defensively - it accepts the field spellings these datasets commonly use
and reports precisely what it found when none matches, rather than guessing and
silently producing an empty or wrong file.

Usage:
    python training/prepare/vrsbench.py --src data/vrsbench --out data/vrsbench/instruct.jsonl
    python training/prepare/vrsbench.py --src data/vrsbench --out ... --kinds vqa caption
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

# Field aliases, most specific first. Extend rather than rewrite when a real
# download reveals the actual spelling.
IMAGE_KEYS = ("image", "image_id", "img", "filename", "image_path", "file_name")
QUESTION_KEYS = ("question", "instruction", "query", "prompt", "text")
ANSWER_KEYS = ("answer", "response", "output", "caption", "gt", "ground_truth")
REFERRING_KEYS = ("referring", "expression", "referring_expression", "sent")


def _first(row: dict, keys: tuple[str, ...]) -> Any | None:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def iter_rows(path: Path) -> Iterator[dict]:
    """Yield rows from a .json (list or dict-of-lists) or .jsonl file."""
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        yield from (r for r in data if isinstance(r, dict))
    elif isinstance(data, dict):
        # Common shape: {"annotations": [...]} or {"images": [...], ...}
        for value in data.values():
            if isinstance(value, list):
                yield from (r for r in value if isinstance(r, dict))


def find_annotation_files(src: Path) -> list[Path]:
    files = sorted(
        p
        for p in src.rglob("*")
        if p.suffix in {".json", ".jsonl"} and p.name != "instruct.jsonl"
    )
    # Skip obvious non-annotation metadata.
    return [p for p in files if p.name not in {"dataset_infos.json", "config.json"}]


def convert(
    src: Path, out: Path, kinds: set[str], limit: int | None = None
) -> tuple[int, dict]:
    """Write instruct.jsonl. Returns (rows written, diagnostics)."""
    annotation_files = find_annotation_files(src)
    if not annotation_files:
        raise SystemExit(
            f"No .json/.jsonl annotation files under {src}.\n"
            "Download VRSBench first: python scripts/fetch_datasets.py --only vrsbench"
        )

    image_root = src
    seen_keys: set[str] = set()
    written = 0
    skipped_no_image = 0
    skipped_no_qa = 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for ann_file in annotation_files:
            try:
                rows = list(iter_rows(ann_file))
            except json.JSONDecodeError:
                continue
            for row in rows:
                seen_keys.update(row.keys())

                image = _first(row, IMAGE_KEYS)
                if image is None:
                    skipped_no_image += 1
                    continue

                question = _first(row, QUESTION_KEYS)
                answer = _first(row, ANSWER_KEYS)
                referring = _first(row, REFERRING_KEYS)

                if referring is not None and "referring" in kinds:
                    kind, question = "referring", f"Locate: {referring}"
                elif question is not None and answer is not None:
                    kind = "vqa"
                elif answer is not None:
                    kind, question = "caption", "Describe this image."
                else:
                    skipped_no_qa += 1
                    continue

                if kind not in kinds or answer is None:
                    continue

                fh.write(
                    json.dumps(
                        {
                            "image": str(Path(str(image))).replace("\\", "/"),
                            "question": str(question),
                            "answer": str(answer),
                            "source": "vrsbench",
                            "kind": kind,
                        }
                    )
                    + "\n"
                )
                written += 1
                if limit is not None and written >= limit:
                    break
            if limit is not None and written >= limit:
                break

    return written, {
        "annotation_files": [str(p.relative_to(src)) for p in annotation_files],
        "fields_seen": sorted(seen_keys),
        "skipped_no_image": skipped_no_image,
        "skipped_no_qa": skipped_no_qa,
        "image_root": str(image_root),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--kinds", nargs="*", default=["vqa", "caption", "referring"],
        choices=["vqa", "caption", "referring"],
    )
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    written, diag = convert(args.src, args.out, set(args.kinds), args.limit)

    if written == 0:
        print("No usable rows produced.", file=sys.stderr)
        print(f"Fields seen in the source: {diag['fields_seen']}", file=sys.stderr)
        print(
            "None matched the expected aliases. Add the real field names to "
            "IMAGE_KEYS / QUESTION_KEYS / ANSWER_KEYS at the top of this file.",
            file=sys.stderr,
        )
        return 1

    print(f"Wrote {written} examples to {args.out}")
    print(f"  from: {diag['annotation_files']}")
    if diag["skipped_no_image"] or diag["skipped_no_qa"]:
        print(
            f"  skipped: {diag['skipped_no_image']} without an image field, "
            f"{diag['skipped_no_qa']} without a question/answer pair"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
