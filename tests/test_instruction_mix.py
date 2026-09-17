"""Instruction mix and refusal evaluation (plan tasks 3.1).

The mix's whole design claim is that refusals are *image-conditional* rather
than lexical. These tests assert that property holds in the generated data,
because if it does not, the refusal metrics measure nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.refusal import (
    always_refuse,
    evaluate,
    is_refusal,
    matched_pairs,
    never_refuse,
)
from training.prepare.instruction_mix import (
    REFUSALS,
    Example,
    build_mix,
    stratified_split,
)

MIX_DIR = Path("data/instruct_mix")


class TestRefusalDetection:
    @pytest.mark.parametrize("text", list(REFUSALS.values()))
    def test_every_canonical_refusal_is_detected(self, text):
        assert is_refusal(text)

    def test_paraphrases_are_detected(self):
        """Requiring the exact string would score a paraphrase as a failure."""
        assert is_refusal("I am unable to answer that from this imagery.")
        assert is_refusal("Sorry, I don't know.")

    def test_an_answer_is_not_a_refusal(self):
        assert not is_refusal("Yes, forest covers about 78% of the scene.")
        assert not is_refusal("5")


class TestMatchedPairs:
    def test_pairs_share_an_identical_question(self):
        examples = [
            {"question": "Is there water visible in this image?",
             "kind": "vqa", "image": "a.tif"},
            {"question": "Is there water visible in this image?",
             "kind": "refusal", "image": "b.tif",
             "refusal_reason": "not_in_image"},
        ]
        pairs = matched_pairs(examples)
        assert len(pairs) == 1
        assert pairs[0][0]["question"] == pairs[0][1]["question"]
        assert pairs[0][0]["image"] != pairs[0][1]["image"]

    def test_unmatched_questions_produce_no_pair(self):
        examples = [
            {"question": "Is there water here?", "kind": "vqa"},
            {"question": "Who owns this land?", "kind": "refusal"},
        ]
        assert matched_pairs(examples) == []


class TestDegenerateBaselines:
    """The baselines are what make the model's numbers interpretable."""

    @pytest.fixture
    def examples(self):
        return [
            {"question": "Is there water visible in this image?", "kind": "vqa"},
            {"question": "Is there water visible in this image?",
             "kind": "refusal", "refusal_reason": "not_in_image"},
            {"question": "Who owns this land?", "kind": "refusal",
             "refusal_reason": "out_of_scope"},
        ]

    def test_always_refuse_gets_perfect_recall_and_fails_the_probe(self, examples):
        """Exactly why refusal recall alone is not a measurement."""
        result = evaluate(always_refuse, examples, matched_pairs(examples))
        assert result["refusal_recall"] == 1.0
        assert result["false_refusal_rate"] == 1.0
        assert result["lexical_shortcut_probe"] == 0.0

    def test_never_refuse_gets_zero_recall(self, examples):
        result = evaluate(never_refuse, examples, matched_pairs(examples))
        assert result["refusal_recall"] == 0.0
        assert result["false_refusal_rate"] == 0.0

    def test_an_ideal_model_scores_one_on_the_probe(self, examples):
        """Answers the answerable one, refuses the identically-worded one."""
        def ideal(example):
            return (
                "I cannot answer that from this image."
                if example["kind"] == "refusal"
                else "Yes."
            )

        result = evaluate(ideal, examples, matched_pairs(examples))
        assert result["lexical_shortcut_probe"] == 1.0
        assert result["false_refusal_rate"] == 0.0


class TestStratifiedSplit:
    """The split must not let a refusal reason vanish from the held-out set.

    A plain random split did exactly that: of 244 refusals, val received 17
    and `sensor_cannot_measure` received ZERO, so one of four reasons was
    unmeasurable and refusal recall moved 5.9 points per item.
    """

    @pytest.fixture
    def examples(self):
        rows = []
        for i in range(200):
            rows.append(Example(f"{i}.tif", "q", "a", "src", "vqa"))
        for reason, count in (
            ("not_in_image", 20), ("sensor_cannot_measure", 6),
            ("single_temporal", 6), ("out_of_scope", 3),
        ):
            for i in range(count):
                rows.append(
                    Example(f"{reason}_{i}.tif", "q", REFUSALS["not_in_image"],
                            "src", "refusal", refusal_reason=reason)
                )
        return rows

    def test_every_refusal_reason_reaches_the_val_side(self, examples):
        _, val = stratified_split(examples, 0.1, seed=0)
        reasons = {e.refusal_reason for e in val if e.refusal_reason}
        assert reasons == {
            "not_in_image", "sensor_cannot_measure", "single_temporal",
            "out_of_scope",
        }

    def test_nothing_is_lost_or_duplicated(self, examples):
        train, val = stratified_split(examples, 0.1, seed=0)
        assert len(train) + len(val) == len(examples)
        assert not {id(e) for e in train} & {id(e) for e in val}

    def test_the_val_fraction_is_approximately_honoured(self, examples):
        _, val = stratified_split(examples, 0.1, seed=0)
        assert 0.08 <= len(val) / len(examples) <= 0.16

    def test_it_is_deterministic_for_a_seed(self, examples):
        a = stratified_split(examples, 0.1, seed=7)[1]
        b = stratified_split(examples, 0.1, seed=7)[1]
        assert [e.image for e in a] == [e.image for e in b]

    def test_a_singleton_stratum_stays_in_train(self, examples):
        """One example cannot be both trained on and held out."""
        rows = examples + [
            Example("only.tif", "q", "a", "src", "refusal",
                    refusal_reason="unique_reason")
        ]
        train, val = stratified_split(rows, 0.1, seed=0)
        assert "only.tif" in {e.image for e in train}
        assert "only.tif" not in {e.image for e in val}


@pytest.mark.skipif(
    not (MIX_DIR / "stats.json").exists(),
    reason="instruction mix not built; run training/prepare/instruction_mix.py",
)
class TestGeneratedMix:
    @pytest.fixture(scope="class")
    def stats(self):
        return json.loads((MIX_DIR / "stats.json").read_text(encoding="utf-8"))

    def test_refusal_fraction_is_about_five_percent(self, stats):
        assert 0.03 <= stats["refusal_fraction"] <= 0.07

    def test_sar_examples_are_present(self, stats):
        """Task 3.1 asks for SAR samples specifically."""
        assert stats["by_modality"].get("sar", 0) > 0

    def test_more_than_one_refusal_reason(self, stats):
        """A single-reason refusal set is a lexical rule waiting to happen."""
        assert len(stats["by_refusal_reason"]) >= 3

    def test_image_conditional_refusals_dominate(self, stats):
        """`not_in_image` is the only category grounded in a specific tile.

        `out_of_scope` is genuinely lexical, so if it dominated the model
        would learn phrasing. It is capped for that reason.
        """
        reasons = stats["by_refusal_reason"]
        assert reasons.get("not_in_image", 0) > reasons.get("out_of_scope", 0)

    def test_the_mix_is_larger_than_the_v0_set(self, stats):
        """v0 trained on RSVQA-LR alone: 2,000 examples."""
        assert stats["total"] > 2000
