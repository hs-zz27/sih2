"""Extract LEVIR-MCI into a change-captioning index (plan task 2.5).

Ships bitemporal pairs (A/B), change masks and five reference captions per
pair, with official train/val/test splits. Those splits are used unchanged:
LEVIR tiles are cut from a small number of large scenes, so a resplit would
put neighbouring tiles on both sides.

Only the FIRST caption is used as the training target while ALL five are kept
as references. BLEU against a single reference badly understates a correct
caption that happens to use different wording, and the dataset provides the
alternatives precisely so that does not happen.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

IMAGE_ROOT = "LEVIR-MCI-dataset/images"
CAPTIONS = "LEVIR-MCI-dataset/LevirCCcaptions.json"


def convert(archive: Path, dest: Path, limit: int | None = None) -> dict:
    z = zipfile.ZipFile(archive)
    dest.mkdir(parents=True, exist_ok=True)
    z.extract(CAPTIONS, dest)
    payload = json.loads((dest / CAPTIONS).read_text(encoding="utf-8"))

    members = set(z.namelist())
    index: dict = {"splits": {}}
    counts: dict = {}

    for row in payload["images"]:
        split = row["split"]
        name = row["filename"]
        entries = index["splits"].setdefault(split, [])
        if limit and len(entries) >= limit:
            continue

        paths = {}
        for kind, folder in (("a", "A"), ("b", "B"), ("label", "label")):
            member = f"{IMAGE_ROOT}/{split}/{folder}/{name}"
            if member in members:
                z.extract(member, dest)
                paths[kind] = str(dest / member)
        if "a" not in paths or "b" not in paths:
            continue

        sentences = [s.get("raw", "").strip() for s in row.get("sentences", [])]
        sentences = [s for s in sentences if s]
        if not sentences:
            continue

        entries.append({
            "id": f"{split}_{row['imgid']}",
            **paths,
            "caption": sentences[0],
            "captions": sentences,
            "changeflag": row.get("changeflag", 0),
        })
        counts[split] = counts.get(split, 0) + 1

    index["n_with_mask"] = sum(
        1 for rows in index["splits"].values() for r in rows if "label" in r
    )
    return index


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, help="cap rows per split")
    args = p.parse_args()

    index = convert(args.archive, args.dest, args.limit)
    counts = {k: len(v) for k, v in index["splits"].items()}
    if not any(counts.values()):
        print("no rows extracted", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(index), encoding="utf-8")
    print(f"Wrote {args.out}")
    for k, v in counts.items():
        print(f"  {k:<7} {v:>6} pairs")
    print(f"  with change mask: {index['n_with_mask']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
