"""W16: re-measuring a head must not destroy the number being checked.

Six of the seven specialist heads scored themselves at the end of training and
then wrote `metrics.json` into the checkpoint directory unconditionally, so
the documented way to reach the eval block also overwrote the published
result. `--eval-only` is the fix, and these tests pin the property that makes
it a fix: **under `--eval-only` nothing inside the checkpoint directory is
written.** Not the metrics, not the run metadata, not the vocabulary.

The guards live in one module rather than at eighteen call sites, so this is
where they are tested. The last test walks the six trainers and asserts none
of them has grown a new unguarded write.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytest

from training.common.eval_only import (
    add_eval_only_args,
    epochs_for,
    is_eval_only,
    resolve_metrics_path,
    vocab_for,
    write_metrics,
    write_run_metadata_unless_eval,
    write_vocab_unless_eval,
)

TRAINERS = [
    "train_caption",
    "train_grounding",
    "train_change_mask",
    "track_a_full",
    "train_optsar_fusion",
    "train_change_caption",
]


def _args(ckpt_dir, *, eval_only=False, out=None, epochs=5):
    return argparse.Namespace(
        ckpt_dir=Path(ckpt_dir), eval_only=eval_only, out=out,
        epochs=epochs, resume=False,
    )


class TestMetricsDestination:
    def test_training_run_keeps_writing_metrics_json(self, tmp_path):
        # The existing behaviour is unchanged for a real training run.
        args = _args(tmp_path)
        assert resolve_metrics_path(args) == tmp_path / "metrics.json"

    def test_eval_only_writes_where_told(self, tmp_path):
        out = tmp_path / "elsewhere" / "score.json"
        args = _args(tmp_path / "ckpt", eval_only=True, out=out)
        assert resolve_metrics_path(args) == out
        assert out.parent.is_dir(), "the parent directory should be created"

    def test_eval_only_without_out_exits(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            resolve_metrics_path(_args(tmp_path, eval_only=True))
        assert "--out" in str(exc.value)

    def test_out_inside_ckpt_dir_is_refused(self, tmp_path):
        # The failure this whole flag exists to prevent, spelled correctly.
        ckpt = tmp_path / "caption"
        ckpt.mkdir()
        with pytest.raises(SystemExit) as exc:
            resolve_metrics_path(
                _args(ckpt, eval_only=True, out=ckpt / "metrics.json")
            )
        assert "inside" in str(exc.value)

    def test_out_nested_deeper_inside_ckpt_dir_is_also_refused(self, tmp_path):
        ckpt = tmp_path / "caption"
        (ckpt / "sub").mkdir(parents=True)
        with pytest.raises(SystemExit):
            resolve_metrics_path(
                _args(ckpt, eval_only=True, out=ckpt / "sub" / "m.json")
            )


class TestNothingIsWrittenIntoCkptDir:
    """The property that makes --eval-only safe, asserted directly."""

    def test_published_artifacts_survive_an_eval_only_run(self, tmp_path):
        ckpt = tmp_path / "caption"
        ckpt.mkdir()
        published = {
            "metrics.json": '{"bleu4_sentence_mean": 0.2446}',
            "run_metadata.json": '{"task": "scene_caption_rsicd"}',
            "vocab.json": '{"a": 1}',
        }
        for name, body in published.items():
            (ckpt / name).write_text(body, encoding="utf-8")
        before = {p.name: p.read_text(encoding="utf-8") for p in ckpt.iterdir()}

        args = _args(ckpt, eval_only=True, out=tmp_path / "out.json")
        write_run_metadata_unless_eval(args, {"task": "OVERWRITTEN"})
        write_vocab_unless_eval(args, {"OVERWRITTEN": 0})
        write_metrics(args, {"bleu4_sentence_mean": 0.0})

        after = {p.name: p.read_text(encoding="utf-8") for p in ckpt.iterdir()}
        assert after == before, "an --eval-only run modified the checkpoint dir"
        assert json.loads((tmp_path / "out.json").read_text())[
            "bleu4_sentence_mean"
        ] == 0.0

    def test_a_training_run_still_writes_all_three(self, tmp_path):
        # The guards must not disable the normal path. `write_run_metadata`
        # lives in the torch-backed checkpointing module, so this arm needs
        # the training environment; the read-only arm above deliberately
        # does not, which is the arm that matters in CI.
        pytest.importorskip("torch")
        ckpt = tmp_path / "caption"
        ckpt.mkdir()
        args = _args(ckpt)
        write_run_metadata_unless_eval(args, {"task": "scene_caption_rsicd"})
        write_vocab_unless_eval(args, {"a": 1})
        write_metrics(args, {"bleu4_sentence_mean": 0.24})
        for name in ("run_metadata.json", "vocab.json", "metrics.json"):
            assert (ckpt / name).exists(), f"{name} was not written"


class TestTrainingIsSkipped:
    def test_epochs_are_zero_under_eval_only(self, tmp_path):
        assert epochs_for(_args(tmp_path, eval_only=True, epochs=40)) == 0
        assert epochs_for(_args(tmp_path, epochs=40)) == 40

    def test_is_eval_only_defaults_false_for_older_namespaces(self, tmp_path):
        # A trainer that has not been given the flag must not crash.
        assert is_eval_only(argparse.Namespace(ckpt_dir=tmp_path)) is False


class TestVocabulary:
    def test_saved_vocabulary_is_read_not_rebuilt(self, tmp_path):
        # The checkpoint's embedding rows are sized to the SAVED vocabulary.
        ckpt = tmp_path / "caption"
        ckpt.mkdir()
        (ckpt / "vocab.json").write_text('{"saved": 1}', encoding="utf-8")
        args = _args(ckpt, eval_only=True, out=tmp_path / "o.json")
        assert vocab_for(args, lambda: {"rebuilt": 1}) == {"saved": 1}

    def test_training_run_rebuilds(self, tmp_path):
        ckpt = tmp_path / "caption"
        ckpt.mkdir()
        (ckpt / "vocab.json").write_text('{"saved": 1}', encoding="utf-8")
        assert vocab_for(_args(ckpt), lambda: {"rebuilt": 1}) == {"rebuilt": 1}

    def test_missing_vocabulary_falls_back_and_warns(self, tmp_path, capsys):
        args = _args(tmp_path, eval_only=True, out=tmp_path / "o.json")
        assert vocab_for(args, lambda: {"rebuilt": 1}) == {"rebuilt": 1}
        assert "no vocab.json" in capsys.readouterr().err


class TestEveryTrainerIsWired:
    """A new unguarded write in any of the six would defeat the flag."""

    @pytest.mark.parametrize("name", TRAINERS)
    def test_no_unguarded_writes_into_ckpt_dir(self, name):
        src = Path("training") / f"{name}.py"
        text = src.read_text(encoding="utf-8")
        for forbidden in (
            "write_run_metadata(args.ckpt_dir",
            "save_checkpoint(args.ckpt_dir",
        ):
            assert forbidden not in text, (
                f"{name}.py writes into the checkpoint directory without a "
                f"guard: {forbidden!r}. Use the training.common.eval_only "
                "wrapper so --eval-only stays read-only."
            )

        # Generic, not a list of known filenames. `band_stats.json` was
        # missed by an earlier name-by-name version of this check, and it is
        # the sidecar whose loss makes landcover_v1 assert class 0 on every
        # patch - exactly the file a re-measurement must not touch.
        stray = re.findall(r"\(\s*args\.ckpt_dir\s*/[^)]*\)\s*\.write_text", text)
        assert not stray, (
            f"{name}.py writes a file into the checkpoint directory "
            f"directly: {stray}. Route it through "
            "training.common.eval_only.write_sidecar_unless_eval so "
            "--eval-only stays read-only."
        )

    @pytest.mark.parametrize("name", TRAINERS)
    def test_parser_offers_the_flag(self, name):
        text = (Path("training") / f"{name}.py").read_text(encoding="utf-8")
        assert "add_eval_only_args(p)" in text


class TestDestinationIsValidatedEarly:
    """A bad --out must fail before the scoring pass, not after it."""

    def test_resume_or_load_validates_before_looking_for_weights(self, tmp_path):
        from training.common.eval_only import resume_or_load_for_eval

        ckpt = tmp_path / "caption"
        ckpt.mkdir()
        # No checkpoint present either, but the --out error must win: it is
        # the one the caller can fix, and it costs nothing to reach.
        with pytest.raises(SystemExit) as exc:
            resume_or_load_for_eval(
                _args(ckpt, eval_only=True, out=ckpt / "metrics.json"),
                model=None,
            )
        assert "inside" in str(exc.value)

    def test_missing_checkpoint_exits_rather_than_scoring_noise(self, tmp_path):
        # Reaching the "is there a checkpoint?" question needs the torch-backed
        # helpers; the validation test above deliberately does not.
        pytest.importorskip("torch")
        from training.common.eval_only import resume_or_load_for_eval

        ckpt = tmp_path / "caption"
        ckpt.mkdir()
        with pytest.raises(SystemExit) as exc:
            resume_or_load_for_eval(
                _args(ckpt, eval_only=True, out=tmp_path / "o.json"),
                model=None,
            )
        assert "nothing to score" in str(exc.value)


def test_add_eval_only_args_parses():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/x"))
    add_eval_only_args(p)
    args = p.parse_args(["--eval-only", "--out", "score.json"])
    assert args.eval_only is True
    assert args.out == Path("score.json")
    assert p.parse_args([]).eval_only is False
