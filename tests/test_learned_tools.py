"""The three learned tools wired in for tasks 2.5, 2.7 and 2.8.

Those tasks say `caption_v1`, `grounding_v1` and `change_caption_v1` should be
REAL tools. The models were trained and their metrics reported, but no tool
module existed - the registry kept the stub, so the pipeline could not reach
any of them. These tests cover the wiring.

The real-model tests skip without a checkpoint, matching every other learned
tool. The fallback tests do not skip: "the registry degrades to a stub when
nothing is configured" is what keeps CI green and must always hold.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from satquery.tools import caption as caption_mod
from satquery.tools import change_caption as change_caption_mod
from satquery.tools import grounding as grounding_mod
from satquery.tools.stubs import REGISTRY


def has_ckpt(path: str, needs: str = "vocab.json") -> bool:
    """A checkpoint the tool can actually load.

    `needs` differs per tool: the caption/grounding models ship a vocab.json,
    the Track A head ships band_stats.json. Requiring vocab.json for all of
    them silently skipped the land-cover tests.

    The sidecar must be **readable**, not merely present. Restoring
    `checkpoints/` from a volume shadow copy on 2026-08-31 returned
    `caption/vocab.json` and `grounding/vocab.json` as NUL bytes - the file
    sizes reached the volume, the contents did not - and this gate let three
    tests through to die inside the loader with a `JSONDecodeError`. A model
    whose vocabulary cannot be read is not a model these tests can evaluate,
    so they skip and name the reason, the same way they skip when the
    checkpoint is absent entirely. `satquery/tools/sidecars.py` applies the
    identical rule to `is_available()`, which is what stops the registry
    selecting such a tool at runtime.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    if not Path(path).exists():
        return False
    from satquery.tools.sidecars import readable_json

    return readable_json(Path(path) / needs, expect=dict)[0]


def has_track_a() -> bool:
    return has_ckpt("checkpoints/track_a_full_base", needs="band_stats.json")


class TestAvailabilityGates:
    @pytest.mark.parametrize(
        "module,env",
        [
            (caption_mod, "SATQUERY_CAPTION"),
            (grounding_mod, "SATQUERY_GROUNDING"),
            (change_caption_mod, "SATQUERY_CHANGE_CAPTION"),
        ],
        ids=["caption", "grounding", "change_caption"],
    )
    def test_unset_env_reports_why(self, module, env, monkeypatch):
        monkeypatch.delenv(env, raising=False)
        available, reason = module.is_available()
        assert available is False
        assert env in reason

    @pytest.mark.parametrize(
        "module,env",
        [
            (caption_mod, "SATQUERY_CAPTION"),
            (grounding_mod, "SATQUERY_GROUNDING"),
            (change_caption_mod, "SATQUERY_CHANGE_CAPTION"),
        ],
        ids=["caption", "grounding", "change_caption"],
    )
    def test_missing_checkpoint_reports_the_path(self, module, env, monkeypatch, tmp_path):
        monkeypatch.setenv(env, str(tmp_path / "absent"))
        available, reason = module.is_available()
        assert available is False
        assert "not found" in reason

    def test_a_checkpoint_without_a_vocab_is_refused(self, monkeypatch, tmp_path):
        """Without vocab.json the ids decode to nothing and the caption is
        silently empty, which is a worse failure than refusing to load."""
        (tmp_path / "ckpt").mkdir()
        monkeypatch.setenv("SATQUERY_CAPTION", str(tmp_path / "ckpt"))
        available, reason = caption_mod.is_available()
        assert available is False
        assert "vocab.json" in reason


class TestRegistryFallback:
    @pytest.mark.parametrize(
        "name", ["caption_v1", "grounding_v1", "change_caption_v1"]
    )
    def test_registry_holds_a_working_tool_either_way(self, name):
        tool = REGISTRY[name]
        assert hasattr(tool, "run") and hasattr(tool, "run_batch")

    def test_no_env_means_stubs_so_ci_stays_green(self, monkeypatch):
        for env in (
            "SATQUERY_CAPTION", "SATQUERY_GROUNDING", "SATQUERY_CHANGE_CAPTION"
        ):
            monkeypatch.delenv(env, raising=False)
        from satquery.tools.stubs import (
            _caption_tool, _change_caption_tool, _grounding_tool,
        )
        for factory in (_caption_tool, _grounding_tool, _change_caption_tool):
            assert type(factory()).__module__.endswith("stubs")


@pytest.mark.skipif(
    not has_ckpt("checkpoints/caption"), reason="no caption checkpoint"
)
class TestCaptionTool:
    def test_it_produces_a_non_empty_caption(self, monkeypatch, msi_6band):
        from satquery.ingest import ingest

        monkeypatch.setenv("SATQUERY_CAPTION", "checkpoints/caption")
        result = caption_mod.CaptionTool().run(ingest([msi_6band]), {})
        assert result.payload.data["caption"].strip()
        assert result.confidence_method == "logprob"
        assert 0.0 <= result.confidence <= 1.0


@pytest.mark.skipif(
    not has_ckpt("checkpoints/grounding"), reason="no grounding checkpoint"
)
class TestGroundingTool:
    def test_the_box_is_inside_the_image(self, monkeypatch, msi_6band):
        """The head sigmoids centre and size, so this must hold by
        construction - and the pixel conversion must use the ACTUAL image
        size, not the 224px model input, or every box on a non-square scene
        is wrong."""
        from satquery.ingest import ingest

        monkeypatch.setenv("SATQUERY_GROUNDING", "checkpoints/grounding")
        manifest = ingest([msi_6band])
        result = grounding_mod.GroundingTool().run(
            manifest, {"_query": "show me the roads"}
        )
        box = result.payload.data["bounding_boxes"][0]
        width, height = manifest.images[0].width, manifest.images[0].height
        assert 0 <= box["x0"] <= box["x1"] <= width
        assert 0 <= box["y0"] <= box["y1"] <= height

    def test_it_warns_about_its_own_weakness(self, monkeypatch, msi_6band):
        """Acc@0.5 is 0.0762. A box from this model must not be presented
        as a finding without that travelling with it."""
        from satquery.ingest import ingest

        monkeypatch.setenv("SATQUERY_GROUNDING", "checkpoints/grounding")
        result = grounding_mod.GroundingTool().run(
            ingest([msi_6band]), {"_query": "roads"}
        )
        assert any("0.0762" in w for w in result.warnings)


@pytest.mark.skipif(
    not has_ckpt("checkpoints/change_caption"),
    reason="no change_caption checkpoint",
)
class TestChangeCaptionTool:
    def test_a_missing_mask_is_warned_about_not_hidden(
        self, monkeypatch, msi_6band, msi_6band_t2
    ):
        """It is a MASK-conditioned model. Feeding zeros runs it outside its
        training distribution, so that has to be stated."""
        from satquery.ingest import ingest

        monkeypatch.setenv("SATQUERY_CHANGE_CAPTION", "checkpoints/change_caption")
        monkeypatch.delenv("SATQUERY_CHANGE_MASK", raising=False)
        result = change_caption_mod.ChangeCaptionTool().run(
            ingest([msi_6band, msi_6band_t2]), {}
        )
        assert result.payload.data["mask_conditioned"] is False
        assert any("zero mask" in w for w in result.warnings)

    def test_a_single_image_is_rejected(self, monkeypatch, msi_6band):
        from satquery.ingest import ingest

        monkeypatch.setenv("SATQUERY_CHANGE_CAPTION", "checkpoints/change_caption")
        with pytest.raises(ValueError, match="bi-temporal"):
            change_caption_mod.ChangeCaptionTool().run(ingest([msi_6band]), {})


class TestLandcoverTool:
    """Track A wired with selective prediction (tasks 2.1, 3.3, 3.6).

    Task 3.6 measured that this head is WORSE than always predicting negative
    at threshold 0.5, so the tool must not threshold at 0.5, and it must not
    run at all without the band statistics the head was normalised with.
    """

    def test_the_threshold_is_never_naive(self):
        """0.5 reproduces the worse-than-trivial behaviour."""
        from satquery.tools.landcover import decision_confidence

        assert decision_confidence() > 0.5

    def test_a_bad_configured_threshold_falls_back_to_the_measured_one(
        self, tmp_path, monkeypatch
    ):
        from satquery.controller.abstention import ENV_THRESHOLDS
        from satquery.tools.landcover import (
            DEFAULT_DECISION_CONFIDENCE,
            decision_confidence,
        )

        bad = tmp_path / "t.yaml"
        bad.write_text("landcover:\n  decision_confidence: 0.4\n", encoding="utf-8")
        monkeypatch.setenv(ENV_THRESHOLDS, str(bad))
        assert decision_confidence() == DEFAULT_DECISION_CONFIDENCE

    def test_class_names_match_the_training_order(self):
        """An invented ordering printed the wrong label on every assertion."""
        from satquery.tools.landcover import _FALLBACK_CLASS_NAMES, class_names

        names = class_names()
        assert len(names) == 19
        assert names[18] == "Urban fabric"
        assert names[0] == "Agro-forestry areas"
        assert _FALLBACK_CLASS_NAMES == names

    def test_missing_band_stats_refuses_to_run(self, tmp_path, monkeypatch):
        """Running without them produced confident nonsense - see the source."""
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        monkeypatch.setenv("SATQUERY_LANDCOVER", str(ckpt))
        from satquery.tools import landcover

        available, reason = landcover.is_available()
        assert available is False
        assert "band_stats.json" in reason

    @pytest.mark.skipif(
        not has_track_a(), reason="no Track A checkpoint with band stats"
    )
    def test_it_applies_the_training_normalisation(self, monkeypatch, msi_6band):
        from satquery.ingest import ingest
        from satquery.tools import landcover

        monkeypatch.setenv("SATQUERY_LANDCOVER", "checkpoints/track_a_full_base")
        result = landcover.LandcoverTool().run(ingest([msi_6band]), {})
        data = result.payload.data
        assert data["calibration"].startswith("affine:")
        # Assertions + denials + abstentions must account for all 19 classes.
        assert (
            len(data["asserted"]) + len(data["abstained"]) + data["n_denied"] == 19
        )
        assert result.confidence_method == "mean_asserted_probability"

    @pytest.mark.skipif(
        not has_track_a(), reason="no Track A checkpoint with band stats"
    )
    def test_an_abstaining_run_still_answers(self, monkeypatch, msi_6band):
        """Recall is 0.25%, so it usually asserts nothing - and must say so."""
        from satquery.ingest import ingest
        from satquery.tools import landcover

        monkeypatch.setenv("SATQUERY_LANDCOVER", "checkpoints/track_a_full_base")
        result = landcover.LandcoverTool().run(ingest([msi_6band]), {})
        assert result.payload.data["answer"].strip()
        if not result.payload.data["asserted"]:
            assert any("abstained on" in w for w in result.warnings)
