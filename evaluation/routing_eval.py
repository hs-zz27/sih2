"""Measure routing accuracy two ways: the raw classifier and the system.

The number the project has quoted for routing (0.5862, then 0.6552) is the
**raw** Tier-1 classifier over all nine classes. The system never acts on that
prediction. `Router.select_task` restricts it to the tasks the input
configuration makes legal, then replaces an unconfident pick with the
configuration default. A query that the raw classifier sends to
TEMPORAL_CHANGE_VQA can still route correctly on a single image, and a correct
but unconfident pick can still be overridden. Only the second number is what
a user experiences, so both are reported, side by side, on the same items.

Holdouts scored:

* `sealed`  - satquery/synth/holdout_sealed.py, n=63, carries configurations.
  **The headline.** Never used for template work.
* `clean`   - holdout.CLEAN_HOLDOUT, n=29. Its configuration is implied by
  the gold task (temporal -> BITEMPORAL_PAIR, cross-modal -> CROSSMODAL_PAIR,
  otherwise SINGLE). See its provenance note before quoting it.
* `tuned`   - holdout.TUNED_HOLDOUT, n=27. Optimistic by construction.

CPU only; fits the classifier once (about a second) and never loads a tool.

Usage:
    python evaluation/routing_eval.py --out docs/assets/routing/report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sklearn  # noqa: E402

from satquery.controller.intent import LOW_CONFIDENCE_TOP1, LOW_MARGIN, IntentClassifier  # noqa: E402
from satquery.controller.matrix_loader import load_matrix  # noqa: E402
from satquery.controller.router import Router  # noqa: E402
from satquery.synth.holdout import CLEAN_HOLDOUT, TUNED_HOLDOUT  # noqa: E402
from satquery.synth.holdout_sealed import SEALED_HOLDOUT  # noqa: E402
from satquery.synth.query_bank import generate  # noqa: E402

MATRIX_PATH = Path(__file__).resolve().parent.parent / "configs" / "capability_matrix.yaml"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def implied_config(task: str) -> str:
    if task.startswith("TEMPORAL_"):
        return "BITEMPORAL_PAIR"
    if task.startswith("XMODAL_"):
        return "CROSSMODAL_PAIR"
    return "SINGLE"


def items(name: str) -> list[tuple[str, str, str]]:
    """(text, config, gold task) for a named holdout."""
    if name == "sealed":
        return [(q.text, q.config, q.task) for q in SEALED_HOLDOUT]
    pairs = {"clean": CLEAN_HOLDOUT, "tuned": TUNED_HOLDOUT}[name]
    return [(text, implied_config(task), task) for text, task in pairs]


def score(router: Router, rows: list[tuple[str, str, str]]) -> dict:
    raw_hits = system_hits = gated = 0
    per_task: dict[str, Counter] = {}
    misroutes = []
    for text, config, gold in rows:
        raw = router.classifier.predict(text)
        legal = router.config_legal_tasks(config)
        task, constrained, _ = router.select_task(text, legal, config)
        was_gated = not constrained.is_confident and constrained.task != "CLARIFY_OR_ABSTAIN"
        gated += was_gated
        raw_ok, sys_ok = raw.task == gold, task == gold
        raw_hits += raw_ok
        system_hits += sys_ok
        c = per_task.setdefault(gold, Counter())
        c["n"] += 1
        c["raw"] += raw_ok
        c["system"] += sys_ok
        if not sys_ok or not raw_ok:
            misroutes.append({
                "query": text, "config": config, "gold": gold,
                "raw": raw.task, "raw_top1": raw.top1,
                "system": task, "constrained_top1": constrained.top1,
                "gated_to_default": was_gated,
            })
    n = len(rows)
    return {
        "n": n,
        "raw_accuracy": round(raw_hits / n, 4),
        "raw_ci95": wilson(raw_hits, n),
        "system_accuracy": round(system_hits / n, 4),
        "system_ci95": wilson(system_hits, n),
        "gated_to_config_default": gated,
        "per_task": {
            t: {"n": c["n"], "raw": round(c["raw"] / c["n"], 4),
                "system": round(c["system"] / c["n"], 4)}
            for t, c in sorted(per_task.items())
        },
        "misroutes": misroutes,
    }


def bank_fingerprint(examples) -> str:
    h = hashlib.sha256()
    for e in examples:
        h.update(f"{e.task}\t{e.text}\n".encode())
    return h.hexdigest()[:16]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path)
    p.add_argument("--label", default="", help="free text recorded in the report")
    args = p.parse_args()

    bank = generate()
    router = Router(load_matrix(MATRIX_PATH), classifier=IntentClassifier(bank))
    report = {
        "label": args.label,
        "classifier": "tfidf_logreg_v1",
        "gate": {"top1": LOW_CONFIDENCE_TOP1, "margin": LOW_MARGIN},
        "query_bank": {"n": len(bank), "sha256_16": bank_fingerprint(bank)},
        # The quoted 0.5862 did not reproduce here (0.6897 at the same
        # commit), and the classifier and its data were identical. A solver
        # version is the likely cause, so it is recorded with every number.
        "scikit_learn": sklearn.__version__,
        "holdouts": {name: score(router, items(name)) for name in ("sealed", "clean", "tuned")},
    }

    for name, r in report["holdouts"].items():
        print(f"{name:7s} n={r['n']:3d}  raw {r['raw_accuracy']:.4f} {r['raw_ci95']}  "
              f"system {r['system_accuracy']:.4f} {r['system_ci95']}  "
              f"gated {r['gated_to_config_default']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
