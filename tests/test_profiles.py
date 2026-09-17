"""Runtime profiles, and the lite profile actually working (task 3.10).

The requirement is "every task answers in `lite`, degraded but never
failing". Two things are asserted separately, because they are different
claims:

* **never failing** - no exception, no empty answer, no traceback;
* **degraded** - and honestly so. A lite answer that claimed to have exported
  a raster it did not produce would satisfy "never failing" while being
  worse than a crash.

Both bugs in that second category existed until this test was written:
`TEMPORAL_CHANGE_MAP` asserted "see the exported raster artifact"
unconditionally, and `XMODAL_JOINT_EXTRACT` returned an empty string.
"""

from __future__ import annotations

import pytest

from satquery.controller.pipeline import Controller
from satquery.controller.profiles import BUILTIN, Profile, load_profile
from satquery.ingest import ingest

# (config fixture names, query) covering all nine tasks.
LITE_CASES = [
    (["msi_6band"], "How many buildings are visible?"),
    (["msi_6band"], "Describe this image."),
    (["msi_6band"], "Classify the land cover."),
    (["msi_6band"], "Show me where the roads are."),
    (["msi_6band", "msi_6band_t2"], "Describe what changed between the images."),
    (["msi_6band", "msi_6band_t2"], "Produce a change mask."),
    (["msi_6band", "msi_6band_t2"], "How much did the built-up area change?"),
    (["msi_6band", "sar_dualpol"],
     "Combine the optical and radar images to find buildings."),
    (["msi_6band"], "hmm"),
]


@pytest.fixture(scope="module")
def lite():
    return Controller(profile="lite")


def resolve(request, names):
    return [request.getfixturevalue(n) for n in names]


class TestProfileLoading:
    def test_both_profiles_load(self):
        assert load_profile("full").vram_budget_mb is None
        assert load_profile("lite").vram_budget_mb == 0

    def test_lite_is_cpu_only_and_skips_the_nli_download(self):
        profile = load_profile("lite")
        assert profile.is_cpu_only
        assert profile.enable_nli is False

    def test_lite_keeps_the_verifier_on(self):
        """A degraded answer is the one most in need of verification."""
        assert load_profile("lite").verifier_enabled is True

    def test_unknown_profile_is_rejected_loudly(self):
        with pytest.raises(ValueError, match="unknown profile"):
            load_profile("turbo")

    def test_malformed_profile_file_falls_back_to_the_builtin(
        self, monkeypatch, tmp_path
    ):
        from satquery.controller import profiles

        bad = tmp_path / "profiles"
        bad.mkdir()
        (bad / "lite.yaml").write_text("::: not yaml [", encoding="utf-8")
        monkeypatch.setattr(profiles, "PROFILE_DIR", bad)
        assert load_profile("lite") == BUILTIN["lite"]

    def test_env_var_selects_the_profile(self, monkeypatch):
        from satquery.controller.profiles import ENV_PROFILE

        monkeypatch.setenv(ENV_PROFILE, "lite")
        assert load_profile().name == "lite"

    def test_explicit_arguments_beat_the_profile(self):
        """So the 3.7 ablation can force the verifier off inside lite."""
        controller = Controller(profile="lite", verifier_enabled=False)
        assert controller.executor.verifier_enabled is False


class TestLiteAnswersEveryTask:
    @pytest.mark.parametrize(
        "fixtures,query", LITE_CASES, ids=[q[:28] for _, q in LITE_CASES]
    )
    def test_lite_never_fails_and_never_returns_empty(
        self, lite, request, fixtures, query
    ):
        manifest = ingest(resolve(request, fixtures))
        trace = lite.run_on_manifest(manifest, query)
        assert trace.answer.strip(), f"empty answer for {query!r}"
        assert "Traceback" not in trace.answer
        if trace.abstained:
            assert trace.abstain_reason and trace.abstain_resolving_input

    def test_learned_tools_are_shed_but_the_index_engine_survives(
        self, lite, msi_6band
    ):
        trace = lite.run_on_manifest(ingest([msi_6band]), "Classify the land cover.")
        tools = {step.tool for step in trace.execution}
        assert tools == {"index_engine_v1"}

    def test_lite_answers_are_grounded_in_measured_indices(self, lite, msi_6band):
        """The point of lite: the physics half never needed a GPU."""
        trace = lite.run_on_manifest(ingest([msi_6band]), "Classify the land cover.")
        assert "Index thresholds indicate" in trace.answer
        assert trace.verification.physics_agreement

    def test_a_shed_task_abstains_with_profile_degraded_not_tool_failure(
        self, lite, msi_6band
    ):
        """Nothing broke; the tool was never loaded. The message must say so."""
        trace = lite.run_on_manifest(
            ingest([msi_6band]), "How many buildings are visible?"
        )
        assert trace.abstained
        assert trace.abstain_trigger == "profile_degraded"
        assert "full profile" in trace.abstain_resolving_input
        assert "retry" not in trace.abstain_resolving_input.lower()


class TestLiteDoesNotOverclaim:
    def test_change_map_does_not_claim_a_raster_it_did_not_produce(
        self, lite, msi_6band, msi_6band_t2
    ):
        """Regression: this asserted an exported raster unconditionally.

        The index engine writes its own COGs, so a bare `if artifacts:` was
        true in lite even though no change mask existed.
        """
        trace = lite.run_on_manifest(
            ingest([msi_6band, msi_6band_t2]), "Produce a change mask."
        )
        assert "No change raster was produced" in trace.answer
        assert "see the exported raster artifact" not in trace.answer

    def test_fusion_says_it_did_not_fuse(self, lite, msi_6band, sar_dualpol):
        """Regression: this returned an empty string in lite."""
        trace = lite.run_on_manifest(
            ingest([msi_6band, sar_dualpol]),
            "Combine the optical and radar images to find buildings.",
        )
        assert trace.answer.strip()
        assert "did not run in this profile" in trace.answer

    def test_full_profile_still_claims_the_raster_it_does_produce(
        self, msi_6band, msi_6band_t2
    ):
        """The lite fix must not have broken the honest claim in full."""
        trace = Controller(profile="full").run_on_manifest(
            ingest([msi_6band, msi_6band_t2]), "Produce a change mask."
        )
        assert "exported raster artifact" in trace.answer


class TestProfileCannotWidenLegality:
    def test_a_profile_does_not_change_the_legal_task_set(
        self, lite, msi_6band
    ):
        """Profiles govern resources; the matrix governs legality.

        A profile that could widen the legal set would be a second, quieter
        authority in front of the guarantee task 3.8 measures.
        """
        full = Controller(profile="full")
        manifest = ingest([msi_6band])
        assert set(lite.router.legal_tasks(manifest)) == set(
            full.router.legal_tasks(manifest)
        )

    def test_lite_cannot_reach_a_bitemporal_tool_on_one_image(
        self, lite, msi_6band
    ):
        trace = lite.run_on_manifest(
            ingest([msi_6band]), "Produce a change mask for these images."
        )
        assert not trace.routing.selected_task.startswith("TEMPORAL_")


class TestProfileDataclass:
    def test_profiles_are_immutable(self):
        with pytest.raises(Exception):
            load_profile("lite").vram_budget_mb = 4096  # type: ignore[misc]

    def test_builtin_profiles_are_complete(self):
        assert set(BUILTIN) == {"full", "lite"}
        for profile in BUILTIN.values():
            assert isinstance(profile, Profile)
            assert profile.description
