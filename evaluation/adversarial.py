"""Run the adversarial routing suite and report it (plan task 3.8).

200 queries x 3 input configurations = 600 plans. Reports the illegal-plan
rate, and - the second half of the requirement - checks that every abstention
carries a named reason and a resolving input.

The routing half runs the real `Router` and `validate_plan`, which is fast
because no tool executes. The named-reason half needs the executor, so it runs
end to end on a stratified sample by default; `--execute-all` runs all 200
through the full controller, which takes minutes rather than seconds.

Usage:
    python evaluation/adversarial.py
    python evaluation/adversarial.py --execute-all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satquery.controller.matrix_loader import load_matrix  # noqa: E402
from satquery.controller.pipeline import Controller  # noqa: E402
from satquery.controller.router import Router  # noqa: E402
from satquery.controller.validator import validate_plan  # noqa: E402
from evaluation.scenes import build_configurations  # noqa: E402
from satquery.ingest import ingest  # noqa: E402
from satquery.synth.adversarial import (  # noqa: E402
    CATEGORIES,
    CATEGORY_OF,
    ADVERSARIAL_QUERIES,
)

REPORT = Path("docs/assets/adversarial/report.json")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute-all", action="store_true",
                   help="run every query through the full controller")
    p.add_argument("--sample-per-category", type=int, default=3)
    p.add_argument("--out", type=Path, default=REPORT)
    args = p.parse_args()

    import tempfile

    matrix = load_matrix(Path("configs/capability_matrix.yaml"))
    router = Router(matrix)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        fixtures = build_configurations(tmp)
        manifests = {name: ingest(paths) for name, paths in fixtures.items()}

        illegal: list[dict] = []
        by_category: dict[str, Counter] = defaultdict(Counter)
        total = 0

        for config, manifest in manifests.items():
            for query in ADVERSARIAL_QUERIES:
                total += 1
                plan = router.route(query, manifest)
                violations = validate_plan(plan, matrix)
                if violations:
                    illegal.append({
                        "config": config, "query": query,
                        "category": CATEGORY_OF[query],
                        "violations": violations,
                    })
                by_category[CATEGORY_OF[query]][plan.tasks[0]] += 1

        print(f"routed {total} plans across {len(manifests)} configurations")
        print(f"illegal plans: {len(illegal)}")
        for name in CATEGORIES:
            counts = dict(by_category[name].most_common(3))
            print(f"  {name:22s} -> {counts}")

        # Second requirement: every rejection names a reason.
        controller = Controller(matrix=matrix)
        if args.execute_all:
            to_execute = list(ADVERSARIAL_QUERIES)
        else:
            to_execute = [
                q
                for queries in CATEGORIES.values()
                for q in queries[: args.sample_per_category]
            ]

        unnamed: list[dict] = []
        abstentions = Counter()
        executed = 0
        for query in to_execute:
            trace = controller.run_on_manifest(manifests["SINGLE"], query)
            executed += 1
            if not trace.abstained:
                continue
            abstentions[trace.abstain_trigger or "unset"] += 1
            if not (trace.abstain_reason and trace.abstain_resolving_input):
                unnamed.append({
                    "query": query,
                    "category": CATEGORY_OF[query],
                    "reason": trace.abstain_reason,
                    "resolving_input": trace.abstain_resolving_input,
                })

        print(f"\nexecuted {executed} queries end to end")
        print(f"abstentions: {sum(abstentions.values())} {dict(abstentions)}")
        print(f"abstentions without a named reason: {len(unnamed)}")

        report = {
            "n_queries": len(ADVERSARIAL_QUERIES),
            "n_configs": len(manifests),
            "n_plans": total,
            "illegal_plans": len(illegal),
            "illegal_detail": illegal,
            "categories": {k: len(v) for k, v in CATEGORIES.items()},
            "selected_task_by_category": {
                k: dict(v) for k, v in by_category.items()
            },
            "executed": executed,
            "abstentions_by_trigger": dict(abstentions),
            "abstentions_without_named_reason": unnamed,
            "note": (
                "The illegal-plan guarantee is structural, not statistical. "
                "The legal task set is computed from the IMAGES, never from "
                "the query text, so no phrasing can widen it. A larger suite "
                "of cleverly-worded queries would not strengthen this; a new "
                "category attacking a different gate would."
            ),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")

    return 1 if illegal or unnamed else 0


if __name__ == "__main__":
    sys.exit(main())
