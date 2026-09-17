"""Extract LEVIR-CD parquet shards into paired tiles + index (task 2.4).

The published parquet embeds imageA / imageB / label as bytes. Writing them
to disk keeps the training loop simple and lets the same index format serve
both training and the evaluation harness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def convert(src: Path, out: Path, limit: int | None = None) -> dict:
    import pyarrow.parquet as pq

    index: dict = {"splits": {}}
    for parquet in sorted(src.rglob("*.parquet")):
        split = parquet.stem.split("-")[0]  # train / test / val
        table = pq.read_table(parquet).to_pydict()
        cols = set(table)
        if not {"imageA", "imageB", "label"} <= cols:
            continue

        split_dir = out / split
        for kind in ("a", "b", "label"):
            (split_dir / kind).mkdir(parents=True, exist_ok=True)

        rows = []
        n = len(table["imageA"])
        for i in range(n if limit is None else min(n, limit)):
            paths = {}
            for key, kind in (("imageA", "a"), ("imageB", "b"), ("label", "label")):
                cell = table[key][i]
                raw = cell.get("bytes") if isinstance(cell, dict) else cell
                if not isinstance(raw, (bytes, bytearray)):
                    paths = {}
                    break
                target = split_dir / kind / f"{i:06d}.png"
                target.write_bytes(raw)
                paths[kind] = str(target)
            if len(paths) == 3:
                rows.append({"id": f"{split}_{i:06d}", **paths})
        index["splits"][split] = rows
    return index


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    index = convert(args.src, args.dest, args.limit)
    counts = {k: len(v) for k, v in index["splits"].items()}
    if not any(counts.values()):
        print("No usable rows found.", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(index), encoding="utf-8")
    print(f"Wrote {args.out}")
    for k, v in counts.items():
        print(f"  {k:<7} {v:>6} pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
