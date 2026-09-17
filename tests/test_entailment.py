"""Entailment gate tests (plan task 3.5).

The deterministic backend is always exercised. The NLI backend needs a local
MNLI checkpoint, which is gitignored and absent in CI, so those tests skip
rather than fail - the same pattern `rs_vqa_v1` and `change_mask_v1` use for
their models.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from satquery.contracts.trace import EntailmentGateTrace
from satquery.verify.entailment import (
    DeterministicBackend,
    NLIBackend,
    build_premises,
    run_gate,
    split_sentences,
)

NLI_PATH = os.environ.get("SATQUERY_NLI") or "models/nli_deberta_mnli"


def _nli_runnable() -> tuple[bool, str]:
    """The backend needs a checkpoint AND the stack that loads it.

    Gating on the checkpoint alone was wrong: CI passes only because
    `models/` is gitignored so the path is absent. A machine that HAS the
    checkpoint but no torch - a broken or partial install - ran these and
    failed on an import, which is a confusing way to learn the environment is
    incomplete.
    """
    if not Path(NLI_PATH).exists():
        return False, f"no MNLI checkpoint at {NLI_PATH}"
    for module in ("torch", "transformers"):
        try:
            __import__(module)
        except ImportError:
            return False, f"{module} is not installed"
    return True, ""


_runnable, _why = _nli_runnable()
nli_available = pytest.mark.skipif(not _runnable, reason=_why or "runnable")


def payload(**fractions: float) -> dict:
    return {
        "indices": {
            k: {"fraction_above_threshold": v} for k, v in fractions.items()
        }
    }


VEG = payload(ndvi=0.62, ndwi=0.05, ndbi=0.11)


class TestSentenceSplitting:
    def test_splits_on_terminal_punctuation(self):
        assert len(split_sentences("First one here. Second one here.")) == 2

    def test_drops_punctuation_fragments(self):
        # A trailing "." must not become a sentence with nothing in it.
        assert split_sentences("A real sentence here. .") == [
            "A real sentence here."
        ]

    def test_empty_answer_yields_nothing(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []


class TestPremises:
    def test_one_premise_per_measured_index(self):
        premises = build_premises(VEG)
        assert len(premises) == 3
        assert any("62%" in p.text for p in premises)

    def test_premise_states_index_independence(self):
        """Without this an NLI model infers a partition that does not exist.

        The indices are thresholded independently and overlap, so "8%
        vegetation" does not imply "not mostly water" - but a model will
        happily reason that it does.
        """
        for premise in build_premises(VEG):
            assert "independently" in premise.text

    def test_indices_without_a_measured_fraction_are_skipped(self):
        assert build_premises({"indices": {"ndvi": {}}}) == []


class TestDeterministicBackend:
    def test_matching_percentage_is_retained_and_strong(self):
        verdict = DeterministicBackend().judge(
            "Vegetation covers 62% of the scene.", build_premises(VEG), VEG
        )
        assert verdict.status == "retained"
        assert verdict.strength == "strong"

    def test_conflicting_percentage_is_flagged(self):
        verdict = DeterministicBackend().judge(
            "The scene is 90% water.", build_premises(VEG), VEG
        )
        assert verdict.status == "flagged"
        assert "5%" in verdict.reason

    def test_presence_support_is_marked_weak(self):
        """The finding that made the hybrid worth building.

        A presence check establishes only that a class exists. "Almost
        entirely covered by water" against a measured 5% NDWI passes it, for
        a plainly false sentence, so the verdict must not be allowed to block
        a backend that can read magnitude.
        """
        verdict = DeterministicBackend().judge(
            "The scene is almost entirely covered by water.",
            build_premises(VEG), VEG,
        )
        assert verdict.status == "retained"
        assert verdict.strength == "weak"
        assert "cannot read magnitude" in verdict.reason

    def test_sentence_with_no_measurable_subject_is_unverifiable(self):
        verdict = DeterministicBackend().judge(
            "A large airport dominates the northern half.",
            build_premises(VEG), VEG,
        )
        assert verdict.status == "unverifiable"


class TestGateOutcomes:
    def test_three_statuses_always_sum_to_the_sentence_count(self):
        """`retained` must never absorb sentences nothing could check."""
        answer = (
            "Vegetation covers 62% of the scene. The scene is 90% water. "
            "A large airport dominates the northern half."
        )
        result = run_gate(answer, VEG)
        assert result.sentences == 3
        assert result.counts_are_consistent()
        assert (result.retained, result.flagged, result.unverifiable) == (1, 1, 1)

    def test_flagged_sentence_is_dropped_from_the_answer(self):
        result = run_gate(
            "Vegetation covers 62% of the scene. The scene is 90% water.", VEG
        )
        assert "90% water" not in result.answer
        assert "62%" in result.answer
        assert result.modified

    def test_the_original_answer_is_preserved_verbatim(self):
        original = "The scene is 90% water."
        result = run_gate(original, VEG)
        assert result.original_answer == original

    def test_dropping_everything_explains_itself_instead_of_returning_empty(self):
        result = run_gate("The scene is 90% water.", VEG)
        assert result.flagged == 1
        assert result.answer.strip()
        assert "contradicted the measured indices" in result.answer

    def test_annotate_keeps_the_sentence_and_marks_it(self):
        result = run_gate(
            "The scene is 90% water.", VEG, action="annotate"
        )
        assert "90% water" in result.answer
        assert "unsupported" in result.answer

    def test_disabled_gate_is_distinguishable_from_a_gate_that_found_nothing(self):
        """The off arm of the verifier ablation must not look like a pass."""
        result = run_gate("The scene is 90% water.", VEG, enabled=False)
        assert result.backend == "disabled"
        assert result.answer == result.original_answer
        assert (result.sentences, result.flagged) == (0, 0)

    def test_gate_with_no_indices_marks_everything_unverifiable(self):
        result = run_gate("Vegetation covers 62% of the scene.", {})
        assert result.unverifiable == 1
        assert result.retained == 0


class TestTraceContract:
    def test_default_trace_reports_not_run_rather_than_a_clean_pass(self):
        trace = EntailmentGateTrace(sentences=0, retained=0, flagged=0)
        assert trace.backend == "not_run"
        assert trace.unverifiable == 0


@nli_available
class TestNLIBackend:
    @pytest.fixture(scope="class")
    def backend(self):
        return NLIBackend(NLI_PATH)

    def test_label_order_is_read_from_the_config(self, backend):
        """Assuming it would silently invert entailment and contradiction."""
        backend._load()
        assert any("entail" in label for label in backend._label_order)

    def test_flags_a_magnitude_overstatement_the_parser_cannot_see(self, backend):
        verdict = backend.judge(
            "The scene is almost entirely covered by water.",
            build_premises(VEG), VEG,
        )
        assert verdict.status == "flagged"

    def test_leaves_an_off_topic_sentence_unverifiable(self, backend):
        verdict = backend.judge(
            "A large airport dominates the northern half.",
            build_premises(VEG), VEG,
        )
        assert verdict.status == "unverifiable"

    def test_entailment_outweighs_an_inferred_contradiction(self, backend):
        """A directly measured entailment beats another index's inference.

        "Most of this scene is under water" is entailed by a 71% NDWI premise
        and was previously flagged against the 8% NDVI premise, because the
        model reasoned that little vegetation excludes mostly-water.
        """
        water = payload(ndvi=0.08, ndwi=0.71, ndbi=0.03)
        verdict = backend.judge(
            "Most of this scene is under water.", build_premises(water), water
        )
        assert verdict.status == "retained"

    def test_hybrid_overturns_a_weak_deterministic_retain(self, backend):
        result = run_gate(
            "The scene is almost entirely covered by water.", VEG,
            backends=[DeterministicBackend(), backend],
        )
        assert result.flagged == 1

    def test_hybrid_does_not_let_nli_overturn_a_measured_verdict(self, backend):
        """A strong deterministic verdict is not up for a second opinion."""
        result = run_gate(
            "Vegetation covers 62% of the scene.", VEG,
            backends=[DeterministicBackend(), backend],
        )
        assert result.retained == 1
        assert result.verdicts[0].backend == "deterministic"
