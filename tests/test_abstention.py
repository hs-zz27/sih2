"""Abstention and selective-prediction tests (plan task 3.6).

Two halves: the risk-coverage maths in `evaluation/abstention.py`, and the
runtime policy in `satquery/controller/abstention.py` that decides when to
decline and what to tell the user.

The policy tests assert one property above all others: **every abstention
names a resolving input**. An abstention nobody can act on is just a refusal
with extra steps.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.abstention import (
    aurc,
    coverage_at_risk,
    evaluate_selective,
    optimal_aurc,
    risk_coverage_curve,
    risk_coverage_svg,
)
from satquery.controller.abstention import (
    AbstentionPolicy,
    decide,
)


def policy(**overrides) -> AbstentionPolicy:
    base = {
        "min_final_confidence": 0.25,
        "min_input_quality": 0.50,
        "abstain_when_all_sentences_flagged": True,
    }
    base.update(overrides)
    return AbstentionPolicy(**base)


GOOD = {"model": 0.9, "agreement": 0.9, "input_quality": 0.9}


def call(**overrides):
    kwargs = {
        "policy": policy(),
        "routed_to_abstain": False,
        "blocking_failures": [],
        "final_confidence": 0.9,
        "components": dict(GOOD),
        "failing_checks": [],
        "conflicts": [],
        "gate_sentences": 3,
        "gate_flagged": 0,
    }
    kwargs.update(overrides)
    return decide(**kwargs)


class TestRiskCoverage:
    def test_perfect_confidence_ranking_has_zero_excess_aurc(self):
        """E-AURC is the metric precisely because it is zero here."""
        correct = np.array([1.0] * 70 + [0.0] * 30)
        result = evaluate_selective(correct, correct, "perfect")
        assert result.e_aurc == pytest.approx(0.0, abs=1e-12)
        assert result.base_error == pytest.approx(0.30)

    def test_constant_confidence_gives_a_flat_curve_at_the_base_error(self):
        rng = np.random.default_rng(0)
        correct = (rng.random(2000) < 0.7).astype("float64")
        confidence = np.full(2000, 0.8)
        result = evaluate_selective(confidence, correct, "useless")
        # Every coverage answers a random subset, so risk sits at the base
        # error throughout and the excess is the whole gap to optimal.
        assert result.curve[-1].risk == pytest.approx(result.base_error)
        assert result.e_aurc > 0.05

    def test_aurc_is_unchanged_by_a_monotone_calibration(self):
        """Calibration changes what a number claims, not what it ranks.

        Both shipped transforms in task 3.3 are monotone, so if this ever
        moved it would mean a calibration had started reordering predictions.
        """
        rng = np.random.default_rng(1)
        logits = rng.normal(size=5000)
        correct = (rng.random(5000) < 1 / (1 + np.exp(-logits))).astype("float64")
        raw = 1 / (1 + np.exp(-logits))
        calibrated = 1 / (1 + np.exp(-(0.35 * logits - 0.85)))
        assert aurc(risk_coverage_curve(raw, correct)) == pytest.approx(
            aurc(risk_coverage_curve(calibrated, correct))
        )

    def test_optimal_aurc_depends_only_on_accuracy(self):
        correct = np.array([1.0] * 80 + [0.0] * 20)
        rng = np.random.default_rng(2)
        assert optimal_aurc(correct) == pytest.approx(
            optimal_aurc(rng.permutation(correct))
        )

    def test_unreachable_risk_target_reports_none_not_zero(self):
        """"Impossible" and "we can answer nothing" are different claims."""
        correct = np.zeros(100)
        curve = risk_coverage_curve(np.linspace(0, 1, 100), correct)
        assert coverage_at_risk(curve, (0.05,))["risk<=0.05"] is None

    def test_curve_covers_every_prediction_at_full_coverage(self):
        correct = np.array([1.0, 0.0, 1.0, 1.0])
        curve = risk_coverage_curve(np.array([0.9, 0.8, 0.7, 0.6]), correct)
        assert curve[-1].coverage == 1.0
        assert curve[-1].n_answered == 4
        assert curve[-1].risk == pytest.approx(0.25)

    def test_svg_draws_the_base_error_reference_line(self):
        correct = np.array([1.0] * 60 + [0.0] * 40)
        svg = risk_coverage_svg(evaluate_selective(correct, correct, "x"))
        assert svg.startswith("<svg") and "base error" in svg


class TestPolicyLoading:
    def test_missing_file_falls_back_to_documented_defaults(self, tmp_path):
        loaded = AbstentionPolicy.load(tmp_path / "absent.yaml")
        assert loaded == AbstentionPolicy()

    def test_malformed_file_degrades_instead_of_raising(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("abstention: [not, a, mapping]", encoding="utf-8")
        assert AbstentionPolicy.load(path) == AbstentionPolicy()

    def test_shipped_thresholds_file_parses(self):
        loaded = AbstentionPolicy.load()
        assert 0.0 <= loaded.min_final_confidence <= 1.0
        assert 0.0 <= loaded.min_input_quality <= 1.0


class TestPolicyDecisions:
    def test_a_healthy_run_does_not_abstain(self):
        assert call().abstained is False

    def test_blocking_failures_abstain_and_name_the_checks(self):
        decision = call(blocking_failures=["crs_present", "band_count"])
        assert decision.abstained
        assert decision.trigger == "input_validation"
        assert "crs_present" in decision.reason

    def test_input_validation_outranks_low_confidence(self):
        """The actionable cause wins over the vague one.

        Both conditions hold here; "your file has no CRS" is something a user
        can fix, "confidence was 0.10" is not.
        """
        decision = call(
            blocking_failures=["crs_present"],
            final_confidence=0.10,
            components={"model": 0.1, "agreement": 0.1, "input_quality": 0.1},
        )
        assert decision.trigger == "input_validation"

    def test_routing_abstention_is_reported_as_routing(self):
        decision = call(routed_to_abstain=True)
        assert decision.trigger == "routing"
        assert "rephrase" in decision.resolving_input

    def test_a_fully_flagged_answer_abstains(self):
        decision = call(gate_sentences=2, gate_flagged=2,
                        conflicts=["ndwi: claimed 90%, measures 5%"])
        assert decision.abstained
        assert decision.trigger == "no_supported_content"
        assert "ndwi" in decision.reason

    def test_a_partly_flagged_answer_does_not_abstain(self):
        """The gate already removed the bad sentence; the rest still stands."""
        assert call(gate_sentences=3, gate_flagged=1).abstained is False

    def test_low_confidence_names_the_limiting_component(self):
        decision = call(
            final_confidence=0.10,
            components={"model": 0.9, "agreement": 0.9, "input_quality": 0.05},
        )
        assert decision.trigger == "low_confidence"
        assert decision.limiting_component == "input_quality"
        assert "input_quality" in decision.reason

    def test_the_limiting_component_is_the_smallest_not_the_first(self):
        decision = call(
            final_confidence=0.10,
            components={"model": 0.02, "agreement": 0.9, "input_quality": 0.9},
        )
        assert decision.limiting_component == "model"

    def test_bad_input_quality_abstains_even_when_the_mean_survives(self):
        """A geometric mean can stay above threshold on a suspect input."""
        decision = call(
            final_confidence=0.60,
            components={"model": 0.95, "agreement": 0.95, "input_quality": 0.30},
        )
        assert decision.abstained
        assert decision.limiting_component == "input_quality"

    def test_failing_check_names_reach_the_resolving_input(self):
        decision = call(
            final_confidence=0.10,
            components={"model": 0.9, "agreement": 0.9, "input_quality": 0.05},
            failing_checks=["nodata_fraction", "gsd_mismatch"],
        )
        assert "nodata_fraction" in decision.resolving_input

    @pytest.mark.parametrize(
        "overrides",
        [
            {"blocking_failures": ["crs_present"]},
            {"routed_to_abstain": True},
            {"gate_sentences": 2, "gate_flagged": 2},
            {"final_confidence": 0.05,
             "components": {"model": 0.05, "agreement": 0.9, "input_quality": 0.9}},
            {"final_confidence": 0.6,
             "components": {"model": 0.9, "agreement": 0.9, "input_quality": 0.2}},
        ],
    )
    def test_every_abstention_names_a_resolving_input(self, overrides):
        """The rule the whole module exists to enforce."""
        decision = call(**overrides)
        assert decision.abstained
        assert decision.reason
        assert decision.resolving_input
        assert len(decision.resolving_input) > 20
