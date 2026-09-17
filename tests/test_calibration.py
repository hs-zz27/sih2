"""Calibration tests (plan task 3.3).

Two halves, matching the two halves of the feature: the offline fitting
maths in `evaluation/calibration.py`, and the runtime registry in
`satquery/controller/calibration.py` that decides whether a fitted parameter
is allowed anywhere near a reported confidence.

The fitting tests all inject a KNOWN miscalibration and check it is
recovered, rather than asserting that some number went down. A test that only
checks ECE improved would pass for a transform that collapses every
probability onto the base rate, which is the exact failure the acceptance
rule exists to reject.
"""

from __future__ import annotations

import json
from typing import get_args

import numpy as np
import pytest

from satquery.contracts.tool_result import ToolResult

from evaluation.calibration import (
    apply_affine,
    apply_temperature,
    calibrate_head,
    fit_affine,
    fit_temperature,
    multiclass_curve,
    multilabel_curve,
    reliability_svg,
    sigmoid,
    softmax,
)
from satquery.controller import calibration as runtime
from satquery.controller.calibration import (
    CALIBRATABLE_CONFIDENCE_METHODS,
    CalibrationEntry,
    load_registry,
    method_label,
    reset_cache,
)


@pytest.fixture(autouse=True)
def _clear_registry_cache():
    reset_cache()
    yield
    reset_cache()


def perfect_multilabel(n=4000, c=8, seed=0):
    """Probabilities that are true by construction: labels drawn FROM them."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, c))
    labels = (rng.random((n, c)) < sigmoid(logits)).astype("float64")
    return logits, labels


class TestCurves:
    def test_perfectly_calibrated_scores_near_zero_ece(self):
        logits, labels = perfect_multilabel()
        curve = multilabel_curve(sigmoid(logits), labels)
        # Sampling noise keeps this off exactly zero; anything above a couple
        # of percent would mean the estimator itself is biased.
        assert curve.ece < 0.02

    def test_overconfident_scores_large_ece(self):
        logits, labels = perfect_multilabel()
        curve = multilabel_curve(sigmoid(logits * 4.0), labels)
        assert curve.ece > 0.10

    def test_bins_partition_every_sample(self):
        logits, labels = perfect_multilabel(n=500, c=4)
        curve = multilabel_curve(sigmoid(logits), labels, n_bins=15)
        assert sum(b.count for b in curve.bins) == curve.n == 500 * 4

    def test_probability_of_one_lands_in_the_last_bin(self):
        curve = multilabel_curve(np.array([[1.0]]), np.array([[1.0]]), n_bins=10)
        assert curve.bins[-1].count == 1

    def test_multiclass_uses_top1_confidence_and_correctness(self):
        # Two samples: one confident and right, one confident and wrong.
        probs = np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])
        curve = multiclass_curve(probs, np.array([0, 1]), n_bins=10)
        assert curve.n == 2
        assert curve.positive_rate == 0.5

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            multilabel_curve(np.zeros((4, 3)), np.zeros((4, 2)))


class TestTemperatureFit:
    @pytest.mark.parametrize("true_t", [0.5, 2.0, 3.5])
    def test_recovers_an_injected_temperature_multilabel(self, true_t):
        logits, labels = perfect_multilabel(n=6000)
        fit = fit_temperature(logits * (1.0 / true_t), labels, "multilabel")
        assert fit.T == pytest.approx(1.0 / true_t, rel=0.12)

    def test_recovers_an_injected_temperature_multiclass(self):
        rng = np.random.default_rng(3)
        logits = rng.normal(size=(5000, 7))
        probs = softmax(logits)
        labels = np.array([rng.choice(7, p=row) for row in probs])
        fit = fit_temperature(logits * 2.5, labels, "multiclass")
        assert fit.T == pytest.approx(2.5, rel=0.12)

    def test_nll_never_increases(self):
        logits, labels = perfect_multilabel()
        fit = fit_temperature(logits * 3.0, labels, "multilabel")
        assert fit.nll_after <= fit.nll_before


class TestAffineFit:
    def test_recovers_an_injected_shift(self):
        logits, labels = perfect_multilabel(n=6000)
        fit = fit_affine(logits + 1.7, labels)
        assert fit.a == pytest.approx(1.0, rel=0.15)
        assert fit.b == pytest.approx(-1.7, abs=0.2)

    def test_temperature_alone_cannot_undo_a_pure_shift(self):
        """The claim that justifies affine scaling existing at all.

        A `pos_weight` adds a roughly constant offset to every logit.
        Temperature scaling multiplies, so it can only trade one end of the
        range against the other; the intercept is what removes an offset.
        """
        logits, labels = perfect_multilabel(n=6000)
        shifted = logits + 1.7

        temp = calibrate_head(
            shifted, labels, head="h", mode="multilabel",
            dataset="synthetic", split_note="synthetic", method="temperature",
        )
        aff = calibrate_head(
            shifted, labels, head="h", mode="multilabel",
            dataset="synthetic", split_note="synthetic", method="affine",
        )
        assert aff.after["ece"] < temp.after["ece"] / 2


class TestAcceptanceRules:
    def test_too_few_samples_is_rejected(self):
        logits, labels = perfect_multilabel(n=40, c=2)
        report = calibrate_head(
            logits * 3, labels, head="tiny", mode="multilabel",
            dataset="synthetic", split_note="synthetic",
        )
        assert not report.accepted
        assert "fitting samples" in report.rejection_reason

    def test_a_saturated_temperature_is_rejected_despite_improving_ece(self):
        """The case where a lower ECE is the wrong thing to reward.

        Logits carrying no information about labels are best "calibrated" by
        flattening every prediction onto the base rate, so the search runs to
        its bound and ECE genuinely improves - by discarding the model
        entirely. Both the ECE and the Brier guard are satisfied here, and
        only the bound guard catches it, which is why it exists.
        """
        rng = np.random.default_rng(11)
        logits = rng.normal(size=(4000, 4))
        labels = rng.integers(0, 2, size=(4000, 4)).astype("float64")
        report = calibrate_head(
            logits, labels, head="noise", mode="multilabel",
            dataset="synthetic", split_note="synthetic",
        )
        assert report.ece_improvement > 0
        assert not report.accepted
        assert "saturated" in report.rejection_reason

    def test_fit_and_eval_splits_are_disjoint_and_cover_everything(self):
        logits, labels = perfect_multilabel(n=1000)
        report = calibrate_head(
            logits, labels, head="h", mode="multilabel",
            dataset="synthetic", split_note="synthetic",
        )
        assert report.n_fit + report.n_eval == 1000
        assert report.n_fit > 0 and report.n_eval > 0

    def test_accepted_fit_improves_both_ece_and_brier(self):
        """Guards against a transform that games ECE by collapsing spread."""
        logits, labels = perfect_multilabel(n=6000)
        report = calibrate_head(
            logits * 3.0, labels, head="h", mode="multilabel",
            dataset="synthetic", split_note="synthetic",
        )
        assert report.accepted
        assert report.after["ece"] < report.before["ece"]
        assert report.after["brier"] <= report.before["brier"]


class TestMonotonicity:
    def test_calibration_never_reorders_predictions(self):
        """Rankings must survive, or every AP and mAP already reported moves."""
        logits = np.linspace(-6, 6, 500)
        order = np.argsort(logits)
        for transformed in (
            apply_temperature(logits, 2.7, "multilabel"),
            apply_affine(logits, 0.35, -0.85),
        ):
            assert np.array_equal(np.argsort(transformed), order)


class TestReliabilitySvg:
    def test_emits_wellformed_standalone_svg(self):
        logits, labels = perfect_multilabel(n=800, c=4)
        curve = multilabel_curve(sigmoid(logits), labels)
        svg = reliability_svg(curve, "test head")
        assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
        assert "test head" in svg
        # The bin populations must be drawn, not just the bars: a bar over
        # four samples is noise and the diagram has to show that.
        assert svg.count("<rect") > len([b for b in curve.bins if b.count])


class TestRuntimeRegistry:
    def test_missing_registry_degrades_to_uncalibrated(self, tmp_path):
        registry = load_registry(tmp_path / "absent.json")
        assert registry.status == "missing"
        assert registry.lookup("SINGLE_LANDCOVER") is None
        assert "no calibration registry" in method_label(None, registry, "X")

    def test_malformed_registry_degrades_instead_of_raising(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        registry = load_registry(path)
        assert registry.status == "invalid"
        assert registry.entries == {}

    def test_lookup_returns_the_entry_and_applies_it(self, tmp_path):
        path = tmp_path / "calibration.json"
        path.write_text(
            json.dumps({
                "heads": {
                    "SINGLE_LANDCOVER": {
                        "method": "affine", "T": 2.87, "a": 0.35, "b": -0.85,
                        "ece_before": 0.064, "ece_after": 0.047,
                        "n_fit": 2934, "n_eval": 2933, "dataset": "BigEarthNet-19",
                        "split_note": "official test shard",
                    }
                }
            }),
            encoding="utf-8",
        )
        registry = load_registry(path)
        entry = registry.lookup("SINGLE_LANDCOVER")
        assert entry is not None
        assert method_label(entry, registry) == "affine:SINGLE_LANDCOVER"
        # a < 1 with a negative intercept must pull an overconfident score down.
        assert entry.apply(0.9) < 0.9

    def test_identity_parameters_leave_a_probability_alone(self):
        entry = CalibrationEntry(
            head="h", method="affine", T=1.0, a=1.0, b=0.0,
            ece_before=0.0, ece_after=0.0, n_fit=1, n_eval=1,
            dataset="", split_note="",
        )
        for p in (0.01, 0.3, 0.5, 0.99):
            assert entry.apply(p) == pytest.approx(p, abs=1e-6)

    def test_extreme_probabilities_stay_in_range(self):
        entry = CalibrationEntry(
            head="h", method="affine", T=1.0, a=4.0, b=2.0,
            ece_before=0.0, ece_after=0.0, n_fit=1, n_eval=1,
            dataset="", split_note="",
        )
        for p in (0.0, 1.0):
            assert 0.0 <= entry.apply(p) <= 1.0

    def test_no_current_confidence_method_is_calibratable(self):
        """The gate that keeps a fitted transform off a non-probability.

        Every value the contract allows is checked, so adding a new
        `confidence_method` to `ToolResult` without deciding whether it is a
        P(correct) fails here rather than silently defaulting either way.
        """
        allowed = get_args(ToolResult.model_fields["confidence_method"].annotation)
        assert set(allowed) == {
            "logprob", "sharpness", "mean_asserted_probability",
            "threshold_rule", "deterministic",
            # Added with change_vqa_v1's semantic path. Decided NOT
            # calibratable: the arithmetic over the change maps is exact and
            # the value is a fixed conservative constant, so there is no
            # per-answer probability to fit. What describes that head's
            # reliability is the segmenter's mIoU.
            "segmentation_derived",
            # Added 2026-09-01 for the placeholder tools. Decided NOT
            # calibratable, and the strongest case of the six: a stub runs no
            # model at all, so there is nothing to fit a transform to. It
            # reports 0.0, is excluded from the model component, and caps the
            # final score at STUB_CONFIDENCE_CAP.
            "stub",
            # Added 2026-09-12 for the selective heads (landcover, optsar)
            # when every class falls in their abstention band. Decided NOT
            # calibratable: the tool made no claim, so there is no P(correct)
            # of any answer to transform. Excluded from the model-confidence
            # minimum - its 0.0 was vetoing answers other tools gave - and,
            # when no learned tool scored at all, the final is capped like a
            # stub, because the combined number would otherwise describe
            # input quality alone while reading as a model result.
            "no_assertion",
        }
        for method in allowed:
            assert method not in CALIBRATABLE_CONFIDENCE_METHODS, method

    def test_the_retired_softmax_temp_scaled_label_is_gone(self):
        """It claimed a temperature scaling that never happened."""
        allowed = get_args(ToolResult.model_fields["confidence_method"].annotation)
        assert "softmax_temp_scaled" not in allowed

    def test_a_calibratable_method_would_actually_reach_the_registry(self):
        """The gate is closed today, but it must not be closed by accident.

        With the set empty the executor never passes a head, so nothing else
        in this file would notice if `lookup` stopped working. This asserts
        the wiring behind the gate is live.
        """
        entry = CalibrationEntry(
            head="SINGLE_LANDCOVER", method="affine", T=2.87, a=0.35, b=-0.85,
            ece_before=0.064, ece_after=0.047, n_fit=2934, n_eval=2933,
            dataset="BigEarthNet-19", split_note="official test shard",
        )
        registry = runtime.Registry({"SINGLE_LANDCOVER": entry}, "test", "loaded")
        assert registry.lookup("SINGLE_LANDCOVER") is entry
        assert method_label(entry, registry) == "affine:SINGLE_LANDCOVER"

    def test_shipped_registry_is_loadable_and_self_consistent(self):
        """The committed registry must parse and must not ship a rejected fit."""
        registry = load_registry(runtime.DEFAULT_PATH)
        assert registry.status == "loaded"
        blob = json.loads(runtime.DEFAULT_PATH.read_text(encoding="utf-8"))
        assert set(blob["heads"]).isdisjoint(blob["rejected"])
        for head, row in blob["heads"].items():
            assert row["ece_after"] < row["ece_before"], head
            assert row["n_fit"] > 0 and row["n_eval"] > 0, head
            assert row["split_note"], f"{head} ships without a split note"


# --- apply_logit: the lossless path ------------------------------------------


def _entry(method="affine", a=0.35, b=-0.85, T=2.9):
    from satquery.controller.calibration import CalibrationEntry

    return CalibrationEntry(head="h", method=method, T=T, a=a, b=b,
                            ece_before=0.1, ece_after=0.05, n_fit=100,
                            n_eval=100, dataset="d", split_note="s")


@pytest.mark.parametrize("method", ["affine", "temperature"])
@pytest.mark.parametrize("z", [-12.0, -3.0, -0.5, 0.0, 0.5, 3.0, 12.0])
def test_apply_logit_equals_apply_of_sigmoid_inside_the_clamp(method, z):
    """Every v1 head lives in this range, so v1 behaviour is bit-for-bit kept."""
    import math

    entry = _entry(method)
    via_probability = entry.apply(1.0 / (1.0 + math.exp(-z)))
    assert entry.apply_logit(z) == pytest.approx(via_probability, abs=1e-9)


def test_apply_logit_keeps_ranking_beyond_the_clamp():
    """The Phase 5 failure: logits of +44 and +71 must not calibrate equal.

    Through `apply` they do - float32 sigmoid saturates, the clamp caps the
    recovered logit at 13.8, and both land on the same served probability.
    That merged the 89%-precise top 56 decisions into a bucket of 33,620 at
    40%, and the tool could assert nothing.
    """
    entry = _entry("affine", a=0.045, b=-1.18)      # the fitted v2 transform
    top, next_ = entry.apply_logit(71.0), entry.apply_logit(44.0)
    assert top > next_ > entry.apply_logit(13.8)
    assert top > 0.85, "the most confident decision must be servable as confident"


def test_apply_still_clamps_for_callers_that_only_have_a_probability():
    """The confidence combiner passes probabilities; log(0) must stay guarded."""
    entry = _entry("affine")
    assert 0.0 < entry.apply(0.0) < 1.0
    assert 0.0 < entry.apply(1.0) < 1.0
