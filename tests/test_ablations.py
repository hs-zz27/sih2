"""Ablation harness tests (plan task 3.7).

These assert the *structure* of the ablation report, not its numbers: the
numbers move when the models do, and pinning them would make the suite fail
for the right reasons at the wrong time.

The one number pinned is the agent arm's illegal-plan rate, because 0 is a
structural guarantee rather than a measurement.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from evaluation.run_ablations import (
    ablation_agent_monolith,
    ablation_triad,
    ablation_two_track,
)
from evaluation.scenes import build_configurations
from satquery.controller.matrix_loader import load_matrix
from satquery.ingest import ingest


@pytest.fixture(scope="module")
def manifests():
    with tempfile.TemporaryDirectory() as tmp:
        configs = build_configurations(Path(tmp))
        yield {k: ingest(v) for k, v in configs.items()}


@pytest.fixture(scope="module")
def matrix():
    return load_matrix(Path("configs/capability_matrix.yaml"))


class TestAgentVsMonolith:
    @pytest.fixture(scope="class")
    def result(self, manifests, matrix):
        return ablation_agent_monolith(manifests, matrix)

    def test_the_agent_arm_is_zero(self, result):
        agent = result.arms["agent (gated + validated)"]
        assert agent["illegal_plans"] == 0
        assert agent["plans"] == 600

    def test_the_monolith_arm_is_not_zero(self, result):
        """A comparison where both arms score 0 would prove nothing.

        If this ever fails it means the ungated classifier stopped selecting
        impossible tasks, which would be a real finding - and would make the
        ablation vacuous, so it must be noticed rather than silently passing.
        """
        monolith = result.arms["monolith (classifier alone)"]
        assert monolith["illegal_plans"] > 0
        assert monolith["illegal_plan_rate"] > 0.05

    def test_the_gap_is_attributed_to_structure_not_the_model(self, result):
        """Both arms use the SAME classifier; only the guards differ."""
        assert "same classifier" in result.verdict

    def test_the_caveat_limits_the_claim_to_legality(self, result):
        assert "not answer quality" in result.caveat.lower()


class TestTriad:
    def test_reports_the_negative_result_without_softening_it(self):
        result = ablation_triad()
        if result.status == "not_run":
            pytest.skip("no fusion checkpoint on disk")
        assert result.arms["complementarity"]["gain"] < 0
        assert "does NOT beat" in result.verdict


class TestTwoTrack:
    def test_declines_to_compare_incomparable_numbers(self):
        result = ablation_two_track()
        assert result.status == "not_comparable"
        assert "DIFFERENT" in result.verdict

    def test_names_the_run_that_would_make_it_comparable(self):
        """A blocked ablation must say what unblocks it."""
        result = ablation_two_track()
        assert "BigEarthNet" in result.caveat
        assert "has not happened" in result.caveat


class TestReportShape:
    def test_every_ablation_states_its_own_status(self, manifests, matrix):
        results = [
            ablation_agent_monolith(manifests, matrix),
            ablation_triad(),
            ablation_two_track(),
        ]
        for result in results:
            assert result.status in {
                "measured", "measured_offline", "not_comparable", "not_run"
            }
            assert result.question.endswith("?")
            assert result.to_dict()["name"]
