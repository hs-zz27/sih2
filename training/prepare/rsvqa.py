"""Convert an RSVQA parquet release into the shared instruction JSONL format.

Chosen for Track B v0 because it is **self-contained**: the parquet embeds the
image bytes alongside the question and answer. VRSBench, by contrast, ships
annotations only - its 142k rows reference images that live in the separate
DOTA and DIOR datasets - so it cannot train anything on its own. See
docs/verification.md item 9.

Output matches `training/track_b_vlm_qlora.py`'s expected format:

    {"image": "images/000001.png", "question": ..., "answer": ...,
     "source": "rsvqa_lr", "kind": "vqa"}

Usage:
    python training/prepare/rsvqa.py --src data/rsvqa_lr_2k --out data/rsvqa_lr_2k/instruct.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMAGE_KEYS = ("image", "img", "picture")
QUESTION_KEYS = ("question", "query", "instruction")
ANSWER_KEYS = ("answer", "ground_truth", "response", "label")


def _pick(names: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {n.lower(): n for n in names}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    return None


def convert(src: Path, out: Path, limit: int | None = None) -> tuple[int, dict]:
    """Extract embedded images to disk and write the instruction JSONL."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pyarrow is required: pip install pyarrow") from exc

    parquets = sorted(src.rglob("*.parquet"))
    if not parquets:
        raise SystemExit(
            f"No .parquet files under {src}. Download first, e.g.\n"
            "  python -c \"from huggingface_hub import snapshot_download; "
            "snapshot_download('dmarsili/RSVQA-LR-2k', repo_type='dataset', "
            "local_dir='data/rsvqa_lr_2k')\""
        )

    image_dir = out.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    columns_seen: list[str] = []

    with out.open("w", encoding="utf-8") as fh:
        for pq_path in parquets:
            table = pq.read_table(pq_path)
            columns_seen = table.column_names
            img_col = _pick(columns_seen, IMAGE_KEYS)
            q_col = _pick(columns_seen, QUESTION_KEYS)
            a_col = _pick(columns_seen, ANSWER_KEYS)
            if not (img_col and q_col and a_col):
                continue

            data = table.to_pydict()
            for i in range(table.num_rows):
                image = data[img_col][i]
                question = data[q_col][i]
                answer = data[a_col][i]

                # HuggingFace image columns arrive as {"bytes": ..., "path": ...}.
                raw = image.get("bytes") if isinstance(image, dict) else image
                if not isinstance(raw, (bytes, bytearray)) or not question or answer in (None, ""):
                    skipped += 1
                    continue

                name = f"{written:06d}.png"
                (image_dir / name).write_bytes(raw)
                fh.write(
                    json.dumps(
                        {
                            "image": f"images/{name}",
                            "question": str(question).strip(),
                            "answer": str(answer).strip(),
                            "source": "rsvqa_lr",
                            "kind": "vqa",
                        }
                    )
                    + "\n"
                )
                written += 1
                if limit is not None and written >= limit:
                    break
            if limit is not None and written >= limit:
                break

    return written, {"columns": columns_seen, "skipped": skipped, "image_dir": str(image_dir)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    written, diag = convert(args.src, args.out, args.limit)
    if written == 0:
        print("No usable rows produced.", file=sys.stderr)
        print(f"Columns seen: {diag['columns']}", file=sys.stderr)
        print(
            "None matched the expected aliases; extend IMAGE_KEYS / "
            "QUESTION_KEYS / ANSWER_KEYS at the top of this file.",
            file=sys.stderr,
        )
        return 1

    print(f"Wrote {written} examples to {args.out}")
    print(f"  images -> {diag['image_dir']}")
    if diag["skipped"]:
        print(f"  skipped {diag['skipped']} rows without usable image/question/answer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
