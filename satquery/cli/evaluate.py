"""`satquery eval` (plan task 1.8)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def add_parser(subparsers) -> None:
    p = subparsers.add_parser("eval", help="Run a benchmark through the pipeline")
    p.add_argument("--benchmark", required=True, help="benchmark name, e.g. rsvqa")
    p.add_argument(
        "--manifest", type=Path, required=True, help="benchmark manifest JSON"
    )
    p.add_argument(
        "--root", type=Path, default=Path("."), help="root directory for image paths"
    )
    p.add_argument(
        "--annotation-type",
        default="vqa",
        choices=["vqa", "caption", "grounding", "landcover"],
    )
    p.add_argument("--out", type=Path, help="write the JSON report here")
    p.add_argument("--limit", type=int, help="only run the first N items")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the manifest and inputs without running any tool",
    )


def run(args) -> int:
    from evaluation.harness import dry_run, evaluate, load_benchmark

    try:
        items = load_benchmark(args.manifest)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"Invalid benchmark manifest: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        report = dry_run(items, args.root)
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 1

    report = evaluate(
        benchmark_path=args.manifest,
        root=args.root,
        benchmark=args.benchmark,
        annotation_type=args.annotation_type,
        limit=args.limit,
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report to {args.out}")

    # Keep stdout readable: metrics summary, not every prediction.
    summary = {k: v for k, v in report.items() if k != "predictions"}
    print(json.dumps(summary, indent=2))

    # A benchmark run is the fastest way to fill a disk: one artifact
    # directory per item, none of which anything reads again. The API pruned
    # its uploads and this path pruned nothing, which is how `artifacts/`
    # reached 46 GB. Named directories are never touched - see
    # satquery/controller/retention.py.
    from satquery.controller.retention import auto_prune

    auto_prune()
    return 0
