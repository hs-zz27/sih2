"""The routing evaluation measures the router, and the sealed set stays sealed."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from evaluation.routing_eval import implied_config, items, score
from satquery.controller.matrix_loader import load_matrix
from satquery.controller.router import CONFIG_TO_LEGAL_TASKS, Router
from satquery.synth.holdout_sealed import SEALED_HOLDOUT
from satquery.synth.query_bank import generate

MATRIX_PATH = Path(__file__).resolve().parent.parent / "configs" / "capability_matrix.yaml"


@pytest.fixture(scope="module")
def router():
    return Router(load_matrix(MATRIX_PATH))


def _norm(text: str) -> str:
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c == " ").split())


class TestSealedHoldout:
    def test_balanced_across_all_nine_tasks(self):
        counts = Counter(q.task for q in SEALED_HOLDOUT)
        assert len(counts) == 9
        assert set(counts.values()) == {7}

    def test_every_gold_task_is_legal_for_its_configuration(self, router):
        for q in SEALED_HOLDOUT:
            assert q.task in router.config_legal_tasks(q.config), q

    def test_no_substantive_query_is_in_the_training_bank(self):
        """A sealed query that the bank generates verbatim measures memorisation.

        Bare greetings ("hello", "analyse") are exempt: the abstain class is
        built from exactly such utterances, and a realistic holdout contains
        them too. Everything else must be absent - this is also the tripwire
        for a sealed query being copied into the templates.
        """
        bank = {_norm(e.text) for e in generate()}
        leaked = [
            q.text for q in SEALED_HOLDOUT
            if q.task != "CLARIFY_OR_ABSTAIN" and _norm(q.text) in bank
        ]
        assert not leaked


class TestEvaluator:
    def test_implied_configuration(self):
        assert implied_config("TEMPORAL_CHANGE_MAP") == "BITEMPORAL_PAIR"
        assert implied_config("XMODAL_JOINT_EXTRACT") == "CROSSMODAL_PAIR"
        assert implied_config("SINGLE_GROUND") == "SINGLE"

    @pytest.mark.parametrize("name", ["sealed", "clean", "tuned"])
    def test_report_is_internally_consistent(self, router, name):
        r = score(router, items(name))
        assert r["n"] == len(items(name))
        assert sum(t["n"] for t in r["per_task"].values()) == r["n"]
        assert 0.0 <= r["raw_accuracy"] <= 1.0
        lo, hi = r["system_ci95"]
        assert lo <= r["system_accuracy"] <= hi

    def test_system_routing_is_always_legal(self, router):
        """Whatever the classifier says, select_task never leaves the legal set."""
        for config in CONFIG_TO_LEGAL_TASKS:
            legal = router.config_legal_tasks(config)
            for text, _, _ in items("sealed"):
                task, _, _ = router.select_task(text, legal, config)
                assert task in legal
