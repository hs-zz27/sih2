import argparse
import sys
from pathlib import Path

from satquery.cli import evaluate as eval_cmd
from satquery.cli.matrix_validate import validate_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="satquery")
    subparsers = parser.add_subparsers(dest="command")

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument(
        "--validate", action="store_true", help="Validate the capability matrix"
    )
    matrix_parser.add_argument(
        "--matrix-path",
        type=Path,
        default=Path("configs/capability_matrix.yaml"),
        help="Path to matrix file",
    )

    eval_cmd.add_parser(subparsers)

    ask_parser = subparsers.add_parser("ask", help="Run one query against images")
    ask_parser.add_argument("images", type=Path, nargs="+", help="one or two rasters")
    ask_parser.add_argument("--query", required=True)
    ask_parser.add_argument("--trace", action="store_true", help="print the full trace")
    ask_parser.add_argument(
        "--evidence", type=Path,
        help="write an evidence pack (ZIP) to this directory",
    )

    prune_parser = subparsers.add_parser(
        "prune",
        help="delete old artifacts/run_* directories (keeps evidence directories)",
    )
    prune_parser.add_argument(
        "--root", type=Path, default=Path("artifacts"),
        help="artifact root to prune (default: artifacts)",
    )
    prune_parser.add_argument(
        "--keep", type=int, default=None,
        help="how many of the newest run directories to keep "
             "(default: SATQUERY_KEEP_RUN_ARTIFACTS or 20)",
    )
    prune_parser.add_argument(
        "--dry-run", action="store_true",
        help="report what would be deleted, and how much, without deleting it",
    )

    return parser


def _run_prune(args) -> int:
    from satquery.controller.retention import prune_run_artifacts

    report = prune_run_artifacts(
        args.root, args.keep, dry_run=args.dry_run, measure=True
    )
    verb = "would delete" if args.dry_run else "deleted"
    print(f"{report.root}: {report.considered} generated run directories")
    print(f"  kept      : {len(report.kept)}")
    print(f"  {verb:<10}: {len(report.deleted)} "
          f"({report.bytes_deleted / 1e9:.2f} GB)")
    print(f"  protected : {len(report.protected)} named directories "
          "(evidence, demo and rehearsal output - never deleted)")
    if report.failed:
        # Named rather than swallowed: on Windows an open preview handle
        # refuses the delete, and "nothing happened" would be misleading.
        print(f"  in use    : {len(report.failed)} could not be removed; "
              "they are retried on the next prune")
    return 0


def _run_ask(args) -> int:
    import json

    from satquery.controller.pipeline import Controller
    from satquery.controller.retention import auto_prune

    trace = Controller().run(args.images, args.query)

    if args.evidence:
        from satquery.report import export

        archive = export(trace, args.evidence, artifact_dir="artifacts")
        print(f"Evidence pack: {archive}")

    if args.trace:
        print(trace.model_dump_json(indent=2))
    else:
        print(f"Task      : {trace.routing.selected_task}")
        print(f"Answer    : {trace.answer}")
        print(
            f"Confidence: {trace.confidence.final:.3f} ({trace.confidence.band})"
        )
        if trace.abstained:
            print(f"Abstained : {trace.abstain_reason}")

    # Each full-scene run writes ~526 MB of index rasters under
    # artifacts/<run_id>. The API bounded its own uploads and nothing bounded
    # this: the tree reached 46 GB across 7,071 directories before anyone
    # noticed. Runs the user named are never touched - see retention.py.
    auto_prune()
    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "matrix":
        if args.validate:
            if validate_matrix(args.matrix_path):
                print("Matrix validation successful.")
                sys.exit(0)
            print("Matrix validation failed.")
            sys.exit(1)
        parser.parse_args([args.command, "--help"])
        sys.exit(1)

    if args.command == "eval":
        sys.exit(eval_cmd.run(args))

    if args.command == "ask":
        sys.exit(_run_ask(args))

    if args.command == "prune":
        sys.exit(_run_prune(args))

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
