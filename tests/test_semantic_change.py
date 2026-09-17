"""Deterministic CDVQA answers from semantic change maps.

The end-to-end oracle (`evaluation/cdvqa_oracle.py`) reports 0.9981 on the
held-out Val split, which says the derivation is right on average. These tests
say *why* it is right, on hand-built maps small enough to count by eye - so a
regression names the rule it broke instead of moving an aggregate by 0.3%.

The three rules that were measured rather than assumed each get a test that
would fail under the plausible wrong version.
"""

from __future__ import annotations

import numpy as np
import pytest

from satquery.verify.semantic_change import (
    CLASSES,
    PALETTE,
    answer,
    class_areas,
    decile_bin,
    decode_label,
    parse_question,
)

IDX = {name: i for i, name in enumerate(CLASSES)}


def maps(t1_spec: dict[str, int], t2_spec: dict[str, int], size: int = 100):
    """Two 10x10 class maps with the requested pixel counts per class."""
    def build(spec):
        flat = np.zeros(size, dtype=np.int8)
        cursor = 0
        for name, count in spec.items():
            flat[cursor : cursor + count] = IDX[name]
            cursor += count
        return flat.reshape(10, 10)

    return build(t1_spec), build(t2_spec)


class TestLabelDecoding:
    def test_every_palette_colour_round_trips(self):
        rgb = np.array([[c for c, _ in PALETTE]], dtype=np.uint8)
        assert decode_label(rgb).tolist() == [list(range(len(PALETTE)))]

    def test_an_unknown_colour_reads_as_unchanged(self):
        """A resampled or recompressed label must not crash the derivation."""
        rgb = np.array([[(7, 7, 7)]], dtype=np.uint8)
        assert decode_label(rgb).tolist() == [[0]]

    def test_class_areas_counts_every_class(self):
        t1, _ = maps({"buildings": 30, "water": 20}, {})
        areas = class_areas(t1)
        assert areas["buildings"] == 30
        assert areas["water"] == 20
        assert areas["unchanged"] == 50


class TestQuestionParsing:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Did the areas of trees change?", "both"),
            ("Have the regions of water changed in the first image?", "t1"),
            ("Did the areas of buildings change in the pre-event image?", "t1"),
            ("Have the areas of trees changed in the second image?", "t2"),
            ("Did the regions of water change in the post-change image?", "t2"),
        ],
    )
    def test_the_date_qualifier_is_read(self, text, expected):
        assert parse_question(text).scope == expected

    def test_low_vegetation_is_not_swallowed_by_a_shorter_phrase(self):
        assert parse_question("Did the regions of low vegetation change?").subject == (
            "low_vegetation"
        )

    @pytest.mark.parametrize(
        "text,kind",
        [
            ("What is the non-change ratio of the imagery?", "change_ratio"),
            ("How much of the area has changed?", "change_ratio"),
            ("What is the percentage of unchanged areas?", "change_ratio"),
            ("What is the change proportion of buildings in the first image?",
             "change_ratio_types"),
            ("What is the change ratio of water in the post-event image?",
             "change_ratio_types"),
            ("What have the areas of trees mainly changed to?", "change_to_what"),
            ("What is the largest change?", "largest_change"),
            ("What type of change is the smallest?", "smallest_change"),
            ("Have the regions of buildings increased?", "increase_or_not"),
            ("Did the areas of trees decrease?", "decrease_or_not"),
            ("Did the areas of trees change?", "change_or_not"),
        ],
    )
    def test_question_shapes(self, text, kind):
        assert parse_question(text).kind == kind

    def test_a_scene_ratio_question_is_not_read_as_a_per_class_one(self):
        """"non-change ratio of the imagery" contains "change ratio of the",
        which is the per-class pattern. Only the absent subject separates
        them."""
        parsed = parse_question("What is the non-change ratio of the imagery?")
        assert parsed.kind == "change_ratio"
        assert parsed.subject is None


class TestDecileBins:
    def test_zero_is_its_own_answer_not_the_first_bin(self):
        assert decile_bin(0.0) == "0"

    @pytest.mark.parametrize(
        "pct,expected",
        [(0.5, "0_to_10"), (10.0, "10_to_20"), (19.9, "10_to_20"), (100.0, "90_to_100")],
    )
    def test_binning(self, pct, expected):
        assert decile_bin(pct) == expected


class TestDateScope:
    def test_a_class_absent_on_the_named_date_answers_no(self):
        """The rule that took change_or_not from 0.935 to 1.0000. Trees are
        absent at t1 and present at t2; asked about the first image, the
        answer is no."""
        t1, t2 = maps({}, {"trees": 20})
        assert answer("Have the areas of trees changed in the first image?", t1, t2) == "no"
        assert answer("Have the areas of trees changed in the second image?", t1, t2) == "yes"
        assert answer("Have the areas of trees changed?", t1, t2) == "yes"


class TestRatios:
    def test_change_ratio_is_the_changed_fraction_of_the_scene(self):
        t1, t2 = maps({"buildings": 25}, {"water": 25})
        assert answer("How much of the area has changed?", t1, t2) == "20_to_30"

    def test_the_inverted_phrasing_reports_the_unchanged_fraction(self):
        t1, t2 = maps({"buildings": 25}, {"water": 25})
        assert answer("How much of the area has not changed?", t1, t2) == "70_to_80"
        assert answer("What is the non-change ratio of the imagery?", t1, t2) == "70_to_80"

    def test_per_class_ratio_is_of_the_scene_not_of_the_changed_area(self):
        """Measured: the scene denominator scores 1.0000 and the changed-area
        denominator 0.6199. Here 20 building pixels are 20% of the scene but
        50% of the 40 changed pixels."""
        t1, t2 = maps({"buildings": 20, "water": 20}, {"trees": 40})
        assert answer(
            "What is the change ratio of buildings in the first image?", t1, t2
        ) == "20_to_30"


class TestDirection:
    def test_increase_and_decrease_compare_the_two_dates(self):
        t1, t2 = maps({"buildings": 10}, {"buildings": 30})
        assert answer("Have the areas of buildings increased?", t1, t2) == "yes"
        assert answer("Have the areas of buildings decreased?", t1, t2) == "no"

    def test_equal_areas_are_neither_an_increase_nor_a_decrease(self):
        t1, t2 = maps({"water": 10}, {"water": 10})
        assert answer("Did the regions of water increase?", t1, t2) == "no"
        assert answer("Did the regions of water decrease?", t1, t2) == "no"


class TestExtremes:
    def test_largest_and_smallest_ignore_classes_that_did_not_change(self):
        t1, t2 = maps({"buildings": 30, "water": 5}, {"trees": 35})
        assert answer("What is the largest change?", t1, t2) == "trees"
        assert answer("What type of change is the smallest?", t1, t2) == "water"

    def test_an_unchanged_scene_defers_rather_than_naming_a_class(self):
        """CDVQA answers here by taking an argmax over a row of zeros. That is
        a property of its generator, not of the image."""
        t1, t2 = maps({}, {})
        assert answer("What is the largest change?", t1, t2) is None

    def test_an_exact_tie_defers(self):
        t1, t2 = maps({"water": 20, "trees": 20}, {})
        assert answer("What is the largest change?", t1, t2) is None


class TestChangeToWhat:
    def test_the_majority_destination_class_is_returned(self):
        t1, t2 = maps(
            {"buildings": 30},
            {"trees": 20, "water": 10},
        )
        assert answer(
            "What have the areas of buildings mainly changed to?", t1, t2
        ) == "trees"

    def test_a_subject_absent_at_the_source_date_defers(self):
        t1, t2 = maps({"water": 10}, {"trees": 10})
        assert answer(
            "What have the areas of buildings mainly changed to?", t1, t2
        ) is None


class TestContract:
    def test_an_unrecognised_question_defers_rather_than_guessing(self):
        t1, t2 = maps({"water": 10}, {"trees": 10})
        assert answer("How many aircraft are parked on the apron?", t1, t2) is None

    def test_every_derived_answer_is_in_cdvqa_s_vocabulary(self):
        """A derived answer that is not a CDVQA token scores zero however
        right it is - that is exactly how the benchmark scored 0.0000 before
        this module existed."""
        vocabulary = set(CLASSES[1:]) | {"yes", "no", "0"} | {
            f"{low}_to_{low + 10}" for low in range(0, 100, 10)
        }
        t1, t2 = maps({"buildings": 30, "water": 5}, {"trees": 35})
        questions = [
            "Did the areas of trees change?",
            "How much of the area has changed?",
            "What is the change ratio of buildings in the first image?",
            "What have the areas of buildings mainly changed to?",
            "What is the largest change?",
            "What type of change is the smallest?",
            "Have the regions of buildings increased?",
            "Did the areas of trees decrease?",
        ]
        for question in questions:
            result = answer(question, t1, t2)
            assert result is None or result in vocabulary, question
