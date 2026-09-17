"""Physics verifier and deterministic change-VQA tests (tasks 2.9, 2.6)."""

from __future__ import annotations

import pytest

from satquery.ingest import ingest
from satquery.tools.change_vqa import (
    ChangeVQATemplate,
    measure_change,
    subject_of,
)
from satquery.verify.verifier import extract_claims, verify, verify_claim


def indices(**fractions) -> dict:
    return {"indices": {
        name: {"fraction_above_threshold": value} for name, value in fractions.items()
    }}


class TestClaimExtraction:
    def test_extracts_percentage_claims(self):
        claims = extract_claims("Index thresholds indicate 65% vegetation.")
        assert len(claims) == 1
        assert claims[0].subject == "vegetation"
        assert claims[0].value == pytest.approx(0.65)

    def test_extracts_multiple_subjects(self):
        claims = extract_claims("About 20% water and 40% built-up land.")
        subjects = {c.subject for c in claims}
        assert subjects == {"water", "built_up"}

    def test_presence_claim_without_a_number(self):
        claims = extract_claims("There is a river running through the scene.")
        assert claims[0].kind == "presence"
        assert claims[0].subject == "water"

    def test_ignores_unmeasurable_subjects(self):
        """Only claims the index engine can check are extracted."""
        assert extract_claims("The image was taken on a Tuesday.") == []

    def test_empty_answer_yields_nothing(self):
        assert extract_claims("") == []


class TestVerification:
    def test_accurate_claim_agrees(self):
        result = verify("The scene is 60% vegetation.", indices(ndvi=0.62))
        assert result["agreements"]["vegetation:fraction"] == 1.0
        assert result["conflicts"] == []

    def test_wildly_wrong_claim_conflicts(self):
        result = verify("The scene is 90% water.", indices(ndwi=0.05))
        assert result["agreements"]["water:fraction"] == 0.0
        assert result["conflicts"]

    def test_borderline_claim_scores_between(self):
        result = verify("The scene is 50% vegetation.", indices(ndvi=0.33))
        score = result["agreements"]["vegetation:fraction"]
        assert 0.0 < score < 1.0

    def test_unverifiable_claim_is_neutral_not_confirmed(self):
        """No index means unknown, not agreement."""
        result = verify("The scene is 30% water.", indices(ndvi=0.5))
        assert result["agreements"]["water:fraction"] == 0.5
        assert "no index available" in result["verdicts"][0]["note"]

    def test_mndwi_preferred_over_ndwi(self):
        result = verify("40% water", indices(mndwi=0.40, ndwi=0.90))
        assert result["verdicts"][0]["index"] == "mndwi"
        assert result["agreements"]["water:fraction"] == 1.0

    def test_presence_claim_rejected_when_class_absent(self):
        result = verify("There is water here.", indices(ndwi=0.001))
        assert result["agreements"]["water:presence"] < 0.5
        assert result["conflicts"]


class TestSwirFreePath:
    def test_ndbi_used_when_available(self):
        result = verify("30% built-up", indices(ndbi=0.30))
        assert result["built_up_path"] == "ndbi"
        assert result["agreements"]["built_up:fraction"] == 1.0

    def test_proxy_used_when_ndbi_absent(self):
        result = verify("30% built-up", indices(builtup_proxy=0.30))
        assert result["built_up_path"] == "swir_free_proxy"

    def test_proxy_verdict_is_capped_below_certainty(self):
        """A SWIR-free proxy must not certify a claim as fully as NDBI."""
        exact = verify("30% built-up", indices(ndbi=0.30))
        proxy = verify("30% built-up", indices(builtup_proxy=0.30))
        assert proxy["agreements"]["built_up:fraction"] < exact["agreements"]["built_up:fraction"]
        assert proxy["agreements"]["built_up:fraction"] == 0.7

    def test_proxy_note_states_the_substitution(self):
        result = verify("30% built-up", indices(builtup_proxy=0.30))
        assert "SWIR-free proxy" in result["verdicts"][0]["note"]


class TestChangeVQASubject:
    @pytest.mark.parametrize("q,expected", [
        ("How much did the water area change?", "water"),
        ("Did the forest decrease?", "vegetation"),
        ("By how much did the urban area grow?", "built_up"),
        ("What colour is the sky?", None),
    ])
    def test_subject_detection(self, q, expected):
        assert subject_of(q) == expected


class TestChangeVQATemplate:
    @pytest.fixture
    def tool(self):
        return ChangeVQATemplate()

    def test_measures_vegetation_change(self, tool, msi_6band, msi_6band_t2):
        manifest = ingest([msi_6band, msi_6band_t2])
        result = tool.run(manifest, {"_query": "How much did the vegetation change?"})
        data = result.payload.data
        assert data["path"] == "deterministic_template"
        assert data["subject"] == "vegetation"
        assert "measurement" in data
        assert result.confidence_method == "deterministic"

    def test_answer_states_direction_and_magnitude(self, tool, msi_6band, msi_6band_t2):
        manifest = ingest([msi_6band, msi_6band_t2])
        answer = tool.run(manifest, {"_query": "Did the vegetation decrease?"}).payload.data["answer"]
        assert any(w in answer for w in ("increased", "decreased", "did not change"))
        assert "km2" in answer or "did not change" in answer

    def test_same_threshold_applied_to_both_dates(self, tool, msi_6band, msi_6band_t2):
        """Re-deriving the threshold per date would let a threshold shift
        masquerade as real change."""
        manifest = ingest([msi_6band, msi_6band_t2])
        m = tool.run(manifest, {"_query": "How much vegetation changed?"}).payload.data
        assert "threshold" in m["measurement"]
        assert m["measurement"]["fraction_t1"] != m["measurement"]["fraction_t2"]

    def test_defers_on_single_image(self, tool, msi_6band):
        manifest = ingest([msi_6band])
        data = tool.run(manifest, {"_query": "How much did water change?"}).payload.data
        assert data["deferred"] is True
        assert "two images" in data["reason"]

    def test_defers_on_non_measurement_question(self, tool, msi_6band, msi_6band_t2):
        """Anything the arithmetic cannot answer is handed on, not guessed."""
        manifest = ingest([msi_6band, msi_6band_t2])
        data = tool.run(manifest, {"_query": "What buildings are these?"}).payload.data
        assert data["deferred"] is True

    def test_deferral_has_zero_confidence(self, tool, msi_6band):
        result = ChangeVQATemplate().run(ingest([msi_6band]), {"_query": "how much water"})
        assert result.confidence == 0.0

    def test_measure_change_returns_none_without_bands(self, msi_4band, msi_6band):
        """4-band VNIR cannot measure a SWIR-only subject."""
        manifest = ingest([msi_4band])
        meta = manifest.images[0]
        assert measure_change("water", meta, meta) is not None  # NDWI fallback works

    def test_relative_change_none_when_baseline_empty(self, msi_6band):
        manifest = ingest([msi_6band])
        m = measure_change("vegetation", manifest.images[0], manifest.images[0])
        assert m["delta_fraction"] == 0.0


class TestPercentageAttribution:
    """Regression: "N% <class>" must bind the number to the class after it."""

    def test_three_classes_in_one_sentence(self):
        claims = extract_claims(
            "Index thresholds indicate 65% vegetation, 28% water and 14% built-up land."
        )
        got = {c.subject: c.value for c in claims}
        assert got["vegetation"] == pytest.approx(0.65)
        assert got["water"] == pytest.approx(0.28)
        assert got["built_up"] == pytest.approx(0.14)

    def test_subject_following_the_number_wins_over_equidistant_earlier_one(self):
        claims = extract_claims("water and 14% built-up")
        assert claims[0].subject == "built_up"

    def test_falls_back_to_a_preceding_subject(self):
        claims = extract_claims("The vegetation covers 42%")
        assert claims[0].subject == "vegetation"


class TestEverySubjectIsChecked:
    """A sentence naming several classes must be verified on all of them.

    `extract_claims` used `subject_of`, which returns only the FIRST subject,
    so a presence sentence got exactly one claim however many classes it
    asserted. The trained captioner exposed it at once: "a bridge is over a
    river with some green trees on both sides" produced a single vegetation
    claim and the water assertion was never checked.
    """

    def test_a_multi_subject_sentence_yields_a_claim_per_subject(self):
        from satquery.verify.verifier import extract_claims

        claims = extract_claims(
            "a bridge is over a river with some green trees on both sides"
        )
        assert {c.subject for c in claims} == {"water", "vegetation"}
        assert all(c.kind == "presence" for c in claims)

    def test_an_unchecked_subject_can_now_be_flagged(self):
        """The behaviour the bug suppressed."""
        from satquery.verify.entailment import DeterministicBackend, build_premises

        payload = {
            "indices": {
                "ndvi": {"fraction_above_threshold": 0.18},
                "ndwi": {"fraction_above_threshold": 0.01},
            }
        }
        verdict = DeterministicBackend().judge(
            "a bridge is over a river with some green trees on both sides",
            build_premises(payload),
            payload,
        )
        assert verdict.status == "flagged"
        assert "water" in verdict.reason

    def test_a_single_subject_sentence_is_unchanged(self):
        """The fix must not alter the single-subject path the bench measures."""
        from satquery.verify.verifier import extract_claims

        claims = extract_claims("The scene is dominated by vegetation.")
        assert [c.subject for c in claims] == ["vegetation"]

    def test_percentage_attribution_is_untouched(self):
        """Fraction claims already handled multiple subjects correctly."""
        from satquery.verify.verifier import extract_claims

        claims = extract_claims("65% vegetation, 28% water and 14% built-up land")
        assert {c.subject for c in claims} == {"vegetation", "water", "built_up"}
        assert all(c.kind == "fraction" for c in claims)
