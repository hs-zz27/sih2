"""Track B evaluation scoring (plan task 3.1).

Tests the scoring maths, not the model: loading a 3B VLM is not a unit test.
The metric that needs pinning is `token_f1`, because exact match alone would
score a correct-but-verbose answer as wrong and the whole comparison would
then measure verbosity.
"""

from __future__ import annotations

import pytest

from evaluation.track_b_eval import normalise, token_f1


class TestNormalise:
    def test_case_and_punctuation_are_ignored(self):
        assert normalise("Yes, forest!") == normalise("yes forest")

    def test_numbers_survive(self):
        assert normalise("about 78% of the scene") == [
            "about", "78", "of", "the", "scene"
        ]


class TestTokenF1:
    def test_identical_text_scores_one(self):
        assert token_f1("yes", "yes") == 1.0

    def test_disjoint_text_scores_zero(self):
        assert token_f1("water", "aircraft") == 0.0

    def test_a_verbose_correct_answer_is_not_scored_zero(self):
        """Exact match would score this 0, which is why F1 is reported too."""
        score = token_f1(
            "Yes, forest covers about 78% of the scene.", "yes"
        )
        assert 0.0 < score < 1.0

    def test_it_is_symmetric_in_the_harmonic_sense(self):
        assert token_f1("a b c", "a b") == pytest.approx(token_f1("a b", "a b c"))

    def test_padding_an_answer_with_junk_lowers_precision(self):
        """So a model cannot win by emitting every plausible token."""
        tight = token_f1("water", "water")
        padded = token_f1("water forest city road ships aircraft", "water")
        assert padded < tight

    def test_empty_prediction_scores_zero_against_content(self):
        assert token_f1("", "water") == 0.0

    def test_two_empties_match(self):
        assert token_f1("", "") == 1.0
