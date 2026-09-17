"""Weighted confidence combiner and stress response (plan task 3.4)."""

from __future__ import annotations

import math

from pathlib import Path

import pytest

from evaluation.confidence_stress import STRESSORS
from satquery.controller.confidence import (
    DEFAULT_WEIGHTS,
    geometric_mean,
    load_weights,
)


class TestWeightedGeometricMean:
    def test_equal_weights_match_the_unweighted_mean(self):
        assert geometric_mean(0.5, 0.8, 1.0) == pytest.approx(
            (0.5 * 0.8 * 1.0) ** (1 / 3)
        )

    def test_a_heavier_weight_pulls_toward_that_component(self):
        light = geometric_mean(0.2, 0.9, 0.9, weights=(1, 1, 1))
        heavy = geometric_mean(0.2, 0.9, 0.9, weights=(5, 1, 1))
        assert heavy < light

    def test_any_zero_component_collapses_the_score(self):
        """The reason for a geometric rather than arithmetic mean."""
        assert geometric_mean(0.99, 0.99, 0.0) == 0.0
        assert geometric_mean(0.99, 0.99, 0.0, weights=(9, 9, 1)) == 0.0

    def test_all_ones_is_one(self):
        assert geometric_mean(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_mismatched_weight_count_is_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            geometric_mean(0.5, 0.5, weights=(1.0,))

    def test_zero_total_weight_is_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            geometric_mean(0.5, 0.5, weights=(0.0, 0.0))

    def test_matches_the_closed_form(self):
        values, weights = (0.4, 0.7, 0.9), (2.0, 3.0, 5.0)
        expected = math.exp(
            sum(w * math.log(v) for v, w in zip(values, weights, strict=True))
            / sum(weights)
        )
        assert geometric_mean(*values, weights=weights) == pytest.approx(expected)


class TestWeightLoading:
    def test_shipped_weights_are_equal(self):
        """They must stay equal until there is data to fit them on.

        Fitting needs (components -> was the answer correct) pairs, and no
        learned tool reports a probability of correctness yet - the same gap
        recorded under tasks 3.3 and 3.6. A fitted weight would be fitted to
        nothing. If this fails, the fit must be documented.
        """
        assert load_weights() == DEFAULT_WEIGHTS

    def test_missing_file_falls_back_to_equal(self, tmp_path):
        assert load_weights(tmp_path / "absent.yaml") == DEFAULT_WEIGHTS

    def test_malformed_file_falls_back_to_equal(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("confidence: [not, a, mapping]", encoding="utf-8")
        assert load_weights(path) == DEFAULT_WEIGHTS

    def test_negative_weights_are_rejected(self, tmp_path):
        path = tmp_path / "neg.yaml"
        path.write_text(
            "confidence:\n  weights:\n    model: -1.0\n", encoding="utf-8"
        )
        assert load_weights(path) == DEFAULT_WEIGHTS

    def test_valid_weights_are_honoured(self, tmp_path):
        path = tmp_path / "w.yaml"
        path.write_text(
            "confidence:\n  weights:\n    model: 2.0\n    agreement: 1.0\n"
            "    input_quality: 3.0\n",
            encoding="utf-8",
        )
        assert load_weights(path) == (2.0, 1.0, 3.0)


class TestStressorDefinitions:
    def test_every_stressor_declares_what_it_targets(self):
        for stressor in STRESSORS:
            assert stressor.targets in {"model", "agreement", "input_quality"}
            assert stressor.description

    def test_expected_collateral_carries_a_reason(self):
        """A declared exception must say why, or it is just an excuse."""
        for stressor in STRESSORS:
            for component, why in stressor.expected_collateral.items():
                assert component in {"model", "agreement", "input_quality"}
                assert len(why) > 60, f"{stressor.name}/{component} needs a reason"

    def test_the_suite_covers_more_than_one_target(self):
        assert len({s.targets for s in STRESSORS}) >= 2


def _manifest():
    """A minimal clean manifest, so component arithmetic is the only variable."""
    from satquery.contracts.input_manifest import (
        ImageMeta, IngestMode, InputManifest,
    )

    return InputManifest(
        run_id="run_conf",
        ingest_mode=IngestMode.OPERATIONAL,
        images=[
            ImageMeta(
                role="single",
                path=Path("scene.tif"),
                modality="MSI",
                modality_evidence={},
                crs="EPSG:32643",
                gsd_m=10.0,
                width=256,
                height=256,
                bands=["BLUE", "GREEN", "RED", "NIR"],
                band_presence=[True, True, True, True],
                dtype="uint16",
                effective_bits=12,
                acquisition_dt=None,
                nodata_pct=0.0,
                cloud_pct=0.0,
                sensor_guess=None,
                polarisations=None,
                look_count_est=None,
            )
        ],
        config="SINGLE",
        checks=[],
        index_availability={},
        artifacts={},
        blocking_failures=[],
    )


class TestStubConfidenceIsNeverHigh:
    """A placeholder answer must not be presentable as a confident result.

    The stubs reported 0.80-0.95 under `threshold_rule`, indistinguishable to
    the combiner from a real head, so a stubbed VQA answer reached the user as
    "0.9473 HIGH" beside the text "[STUB - no model loaded]".

    Fixed by making the stub report 0.0 under a `stub` method, excluding it
    from the model component, and capping the FINAL score at
    STUB_CONFIDENCE_CAP - which sits below MEDIUM_BAND, so the band is LOW,
    and above the abstention threshold, so the run still answers.
    """

    def _run(self, paths, query):
        from satquery.controller.pipeline import Controller

        return Controller().run(paths, query)

    def test_a_stubbed_run_is_never_high(self, msi_6band):
        trace = self._run([msi_6band], "Is there a stadium in this image?")
        assert trace.confidence.band != "HIGH"
        assert trace.confidence.band == "LOW"

    def test_a_stubbed_run_is_capped_at_the_documented_ceiling(self, msi_6band):
        from satquery.controller.confidence import STUB_CONFIDENCE_CAP

        trace = self._run([msi_6band], "Is there a stadium in this image?")
        assert trace.confidence.final <= STUB_CONFIDENCE_CAP

    def test_the_cap_sits_below_medium_and_above_abstention(self):
        """The two properties that make the cap the right number."""
        from satquery.controller.abstention import AbstentionPolicy
        from satquery.controller.confidence import (
            MEDIUM_BAND, STUB_CONFIDENCE_CAP, band,
        )

        assert STUB_CONFIDENCE_CAP < MEDIUM_BAND
        assert band(STUB_CONFIDENCE_CAP) == "LOW"
        assert STUB_CONFIDENCE_CAP > AbstentionPolicy().min_final_confidence

    def test_a_stubbed_run_still_answers(self, msi_6band):
        """Capping must not turn every CI run into a refusal.

        A stub reporting 0.0 straight into the geometric mean collapsed the
        score to 0.0 and tripped abstention; 36 tests failed that way on
        2026-09-01. The cap exists so the abstention policy is untouched.
        """
        trace = self._run([msi_6band], "Is there a stadium in this image?")
        assert trace.abstained is False
        assert trace.answer

    def test_the_stub_is_named_in_the_trace(self, msi_6band):
        trace = self._run([msi_6band], "Is there a stadium in this image?")
        steps = [s for s in trace.execution if s.tool == "rs_vqa_v1"]
        assert steps, "the VQA step should be recorded"
        assert steps[0].confidence_method == "stub"
        assert steps[0].confidence == 0.0
        assert steps[0].version.endswith("-stub")

    def test_the_stub_is_marked_in_the_answer(self, msi_6band):
        from satquery.tools.stubs import STUB_NOTICE

        trace = self._run([msi_6band], "Is there a stadium in this image?")
        assert STUB_NOTICE in trace.answer

    def test_no_stub_method_is_calibratable(self):
        from satquery.controller.calibration import CALIBRATABLE_CONFIDENCE_METHODS

        assert "stub" not in CALIBRATABLE_CONFIDENCE_METHODS

    def test_a_stub_contributes_no_model_confidence(self):
        """Excluded from the minimum, rather than collapsing it."""
        from satquery.controller.confidence import compute_confidence

        capped = compute_confidence(
            model_confidence=1.0, manifest=_manifest(), agreements={}, stubbed=True
        )
        uncapped = compute_confidence(
            model_confidence=1.0, manifest=_manifest(), agreements={}, stubbed=False
        )
        assert capped.final < uncapped.final
        assert capped.band == "LOW"


class TestRealModelConfidenceIsUnchanged:
    """The cap must apply to placeholders only."""

    def test_an_unstubbed_run_is_not_capped(self):
        from satquery.controller.confidence import (
            STUB_CONFIDENCE_CAP, compute_confidence,
        )

        result = compute_confidence(
            model_confidence=0.85, manifest=_manifest(), agreements={}
        )
        assert result.final > STUB_CONFIDENCE_CAP
        assert result.band == "HIGH"

    def test_the_arithmetic_is_identical_when_not_stubbed(self):
        """Same inputs, same number as before the parameter existed."""
        from satquery.controller.confidence import (
            compute_confidence, geometric_mean, input_quality, load_weights,
        )

        manifest = _manifest()
        result = compute_confidence(
            model_confidence=0.85, manifest=manifest, agreements={}
        )
        expected = geometric_mean(
            0.85, 1.0, input_quality(manifest), weights=load_weights()
        )
        assert result.final == round(expected, 6)

    def test_default_is_not_stubbed(self):
        """A caller that does not pass the flag gets the old behaviour."""
        from satquery.controller.confidence import compute_confidence

        assert compute_confidence(
            model_confidence=0.9, manifest=_manifest(), agreements={}
        ).band == "HIGH"


class TestAbstentionBehaviourIsUnchanged:
    def test_the_abstention_threshold_is_untouched(self):
        from satquery.controller.abstention import AbstentionPolicy

        assert AbstentionPolicy().min_final_confidence == 0.25

    def test_a_capped_run_does_not_cross_the_threshold(self):
        from satquery.controller.abstention import AbstentionPolicy
        from satquery.controller.confidence import STUB_CONFIDENCE_CAP

        assert STUB_CONFIDENCE_CAP > AbstentionPolicy().min_final_confidence

    def test_a_genuinely_low_score_still_abstains(self, no_crs_raster):
        """Input-validation abstention is unaffected by the stub cap."""
        from satquery.controller.pipeline import Controller

        trace = Controller().run([no_crs_raster], "Describe this image.")
        assert trace.abstained is True
        assert trace.abstain_reason
