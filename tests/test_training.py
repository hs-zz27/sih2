"""Tests for training infrastructure and fetch scripts.

The GPU training loop itself cannot run here. Everything around it can, and
that is where the failures that waste GPU sessions actually live: broken
resume, malformed data, wrong paths. Those are tested properly.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# torch is a TRAINING dependency, not a runtime one: the ingest pipeline,
# index engine, controller, API and evidence pack all work without it. CI
# therefore does not install it, and these tests skip there rather than
# adding an ~800 MB download to every run. They do execute locally, where
# the training environment exists.
pytest.importorskip("torch")
import torch
import torch.nn as nn

from training.common.checkpointing import (
    TrainingState,
    find_latest_checkpoint,
    load_checkpoint,
    maybe_resume,
    save_checkpoint,
    set_seed,
    write_run_metadata,
)


def tiny_model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


class TestCheckpointing:
    def test_save_and_find(self, tmp_path):
        model = tiny_model()
        save_checkpoint(tmp_path, 10, model)
        save_checkpoint(tmp_path, 20, model)
        latest = find_latest_checkpoint(tmp_path)
        assert latest is not None
        assert latest.name == "ckpt_step_20.pt"

    def test_find_returns_none_when_empty(self, tmp_path):
        assert find_latest_checkpoint(tmp_path) is None

    def test_step_ordering_is_numeric_not_lexical(self, tmp_path):
        """Step 100 must beat step 20; string sorting would get this wrong."""
        model = tiny_model()
        save_checkpoint(tmp_path, 20, model, keep_last=0)
        save_checkpoint(tmp_path, 100, model, keep_last=0)
        assert find_latest_checkpoint(tmp_path).name == "ckpt_step_100.pt"

    def test_model_weights_restored_exactly(self, tmp_path):
        model = tiny_model()
        save_checkpoint(tmp_path, 1, model)
        original = {k: v.clone() for k, v in model.state_dict().items()}

        for p in model.parameters():
            p.data.add_(1.0)  # corrupt

        load_checkpoint(tmp_path / "ckpt_step_1.pt", model)
        for k, v in model.state_dict().items():
            assert torch.allclose(v, original[k])

    def test_optimizer_state_restored(self, tmp_path):
        model = tiny_model()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
        save_checkpoint(tmp_path, 1, model, opt)

        fresh_model, fresh_opt = tiny_model(), None
        fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=0.01)
        load_checkpoint(tmp_path / "ckpt_step_1.pt", fresh_model, fresh_opt)
        assert fresh_opt.state_dict()["state"]

    def test_rng_state_restored_so_resume_is_a_continuation(self, tmp_path):
        """The point of saving RNG state: the resumed run must replay identically."""
        set_seed(123)
        model = tiny_model()
        save_checkpoint(tmp_path, 1, model)

        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

        # Advance all three generators.
        random.random(); np.random.rand(); torch.rand(1)

        load_checkpoint(tmp_path / "ckpt_step_1.pt", model)
        got = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        assert got == pytest.approx(expected)

    def test_training_state_roundtrips(self, tmp_path):
        state = TrainingState(step=42, epoch=3, best_metric=0.87)
        state.metrics_history.append({"step": 42, "loss": 0.5})
        save_checkpoint(tmp_path, 42, tiny_model(), state=state)

        restored, _ = load_checkpoint(tmp_path / "ckpt_step_42.pt")
        assert restored.step == 42
        assert restored.epoch == 3
        assert restored.best_metric == pytest.approx(0.87)
        assert restored.metrics_history == [{"step": 42, "loss": 0.5}]

    def test_extra_payload_roundtrips(self, tmp_path):
        save_checkpoint(tmp_path, 1, tiny_model(), extra={"note": "hello"})
        _, extra = load_checkpoint(tmp_path / "ckpt_step_1.pt")
        assert extra["note"] == "hello"

    def test_keep_last_prunes_old_checkpoints(self, tmp_path):
        model = tiny_model()
        for step in range(1, 7):
            save_checkpoint(tmp_path, step, model, keep_last=2)
        remaining = sorted(p.name for p in tmp_path.glob("ckpt_step_*.pt"))
        assert remaining == ["ckpt_step_5.pt", "ckpt_step_6.pt"]

    def test_keep_last_zero_keeps_everything(self, tmp_path):
        model = tiny_model()
        for step in range(1, 5):
            save_checkpoint(tmp_path, step, model, keep_last=0)
        assert len(list(tmp_path.glob("ckpt_step_*.pt"))) == 4

    def test_no_temp_files_left_behind(self, tmp_path):
        """Atomic write must not leave .tmp files on success."""
        save_checkpoint(tmp_path, 1, tiny_model())
        assert list(tmp_path.glob(".*tmp")) == []

    def test_truncated_temp_file_is_not_picked_up_as_latest(self, tmp_path):
        """A kill mid-save leaves a .tmp; the previous good ckpt must still win."""
        save_checkpoint(tmp_path, 5, tiny_model())
        (tmp_path / ".ckpt_step_9.pt.tmp").write_bytes(b"truncated garbage")
        assert find_latest_checkpoint(tmp_path).name == "ckpt_step_5.pt"

    def test_maybe_resume_disabled_starts_fresh(self, tmp_path):
        save_checkpoint(tmp_path, 7, tiny_model())
        state, _ = maybe_resume(tmp_path, enabled=False)
        assert state.step == 0

    def test_maybe_resume_with_no_checkpoint(self, tmp_path):
        state, _ = maybe_resume(tmp_path, enabled=True)
        assert state.step == 0

    def test_maybe_resume_picks_up_step(self, tmp_path):
        model = tiny_model()
        save_checkpoint(tmp_path, 33, model, state=TrainingState(step=33))
        state, _ = maybe_resume(tmp_path, model, enabled=True)
        assert state.step == 33

    def test_run_metadata_written(self, tmp_path):
        write_run_metadata(tmp_path, {"task": "x", "lr": 1e-4})
        data = json.loads((tmp_path / "run_metadata.json").read_text())
        assert data["task"] == "x"


class TestKillAndResume:
    """The failure mode the plan warns about, end to end."""

    def test_resumed_run_matches_uninterrupted_run(self, tmp_path):
        def train(steps, ckpt_dir, resume=False, start_state=None):
            set_seed(7)
            model = tiny_model()
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            state = start_state or TrainingState()
            if resume:
                state, _ = maybe_resume(ckpt_dir, model, opt, enabled=True)
            for step in range(state.step, steps):
                x = torch.randn(4, 4)
                loss = model(x).sum()
                opt.zero_grad(); loss.backward(); opt.step()
                state.step = step + 1
                save_checkpoint(ckpt_dir, state.step, model, opt, state=state, keep_last=2)
            return model

        # Uninterrupted 10 steps.
        full = train(10, tmp_path / "full")

        # Killed at 5, resumed to 10.
        train(5, tmp_path / "split")
        resumed = train(10, tmp_path / "split", resume=True)

        for a, b in zip(full.state_dict().values(), resumed.state_dict().values()):
            assert torch.allclose(a, b, atol=1e-6), (
                "resumed run diverged from the uninterrupted run"
            )


class TestTrackBDataLoading:
    def _write(self, tmp_path, rows):
        (tmp_path / "instruct.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        return tmp_path

    def test_loads_examples(self, tmp_path):
        from training.track_b_vlm_qlora import load_examples

        d = self._write(tmp_path, [
            {"image": "a.jpg", "question": "How many?", "answer": "3"},
            {"image": "b.jpg", "question": "What is this?", "answer": "a field"},
        ])
        examples = load_examples(d)
        assert len(examples) == 2
        assert examples[0].answer == "3"
        assert examples[0].image_path.endswith("a.jpg")

    def test_limit_respected(self, tmp_path):
        from training.track_b_vlm_qlora import load_examples

        d = self._write(tmp_path, [
            {"image": f"{i}.jpg", "question": "q", "answer": "a"} for i in range(10)
        ])
        assert len(load_examples(d, limit=3)) == 3

    def test_missing_file_names_the_prepare_command(self, tmp_path):
        from training.track_b_vlm_qlora import load_examples

        with pytest.raises(FileNotFoundError, match="prepare/vrsbench.py"):
            load_examples(tmp_path)

    def test_missing_field_reports_line_number(self, tmp_path):
        from training.track_b_vlm_qlora import load_examples

        d = self._write(tmp_path, [
            {"image": "a.jpg", "question": "q", "answer": "a"},
            {"image": "b.jpg", "question": "q"},  # no answer
        ])
        with pytest.raises(ValueError, match="line|:2"):
            load_examples(d)

    def test_malformed_json_reports_line_number(self, tmp_path):
        from training.track_b_vlm_qlora import load_examples

        # The malformed line must come first, otherwise the missing-fields
        # check on an earlier valid-but-wrong line fires before the parse error.
        (tmp_path / "instruct.jsonl").write_text("not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_examples(tmp_path)

    def test_empty_file_rejected(self, tmp_path):
        from training.track_b_vlm_qlora import load_examples

        (tmp_path / "instruct.jsonl").write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="no examples"):
            load_examples(tmp_path)

    def test_chat_format_has_image_and_answer(self, tmp_path):
        from training.track_b_vlm_qlora import Example, build_chat

        chat = build_chat(Example("img.jpg", "How many?", "3"))
        assert [t["role"] for t in chat] == ["system", "user", "assistant"]
        assert any(c["type"] == "image" for c in chat[1]["content"])
        assert chat[2]["content"][0]["text"] == "3"

    def test_label_masking_hides_the_prompt(self):
        from training.track_b_vlm_qlora import mask_prompt_labels

        ids = torch.tensor([1, 2, 3, 4, 5, 0, 0])
        labels = mask_prompt_labels(ids, assistant_start=3, pad_token_id=0)
        assert labels[:3].tolist() == [-100, -100, -100]   # prompt hidden
        assert labels[3:5].tolist() == [4, 5]              # answer supervised
        assert labels[5:].tolist() == [-100, -100]         # padding hidden


class TestTrackBDryRun:
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "training/track_b_vlm_qlora.py", *args],
            capture_output=True, text=True,
        )

    def test_dry_run_reports_problems_and_exits_nonzero(self, tmp_path):
        r = self._run(
            "--model", str(tmp_path / "nope"), "--data", str(tmp_path), "--dry-run"
        )
        assert r.returncode == 1
        assert "model dir missing" in r.stdout

    def test_dry_run_succeeds_with_valid_inputs(self, tmp_path):
        model_dir = tmp_path / "model"; model_dir.mkdir()
        data_dir = tmp_path / "data"; data_dir.mkdir()
        (data_dir / "img.jpg").write_bytes(b"x")
        (data_dir / "instruct.jsonl").write_text(
            json.dumps({"image": "img.jpg", "question": "q", "answer": "a"}),
            encoding="utf-8",
        )
        r = self._run(
            "--model", str(model_dir), "--data", str(data_dir), "--dry-run"
        )
        assert r.returncode == 0, r.stdout + r.stderr
        assert "Ready to train on a GPU box" in r.stdout

    def test_dry_run_does_not_require_gpu_stack(self, tmp_path):
        """--dry-run must not import bitsandbytes, which needs CUDA."""
        r = self._run("--model", str(tmp_path), "--data", str(tmp_path), "--dry-run")
        assert "bitsandbytes" not in r.stderr


class TestFetchScripts:
    def _run(self, script, *args):
        return subprocess.run(
            [sys.executable, f"scripts/{script}", *args],
            capture_output=True, text=True,
        )

    def test_datasets_list(self):
        r = self._run("fetch_datasets.py", "--list")
        assert r.returncode == 0, r.stderr
        assert "bigearthnet_txt" in r.stdout
        assert "vrsbench" in r.stdout

    def test_datasets_list_flags_unverified_entries(self):
        """Unverified identifiers must be visibly marked, not implied correct."""
        r = self._run("fetch_datasets.py", "--list")
        assert "NO" in r.stdout
        assert "best-effort" in r.stdout

    def test_datasets_unknown_key_rejected(self):
        r = self._run("fetch_datasets.py", "--only", "not_a_dataset")
        assert r.returncode == 2
        assert "Unknown dataset keys" in r.stderr

    def test_models_list(self):
        r = self._run("fetch_models.py", "--list")
        assert r.returncode == 0, r.stderr
        assert "qwen25_vl_3b" in r.stdout
        assert "florence2_large" in r.stdout

    def test_no_fabricated_checksums(self):
        """A digest we did not obtain from a publisher must not be pinned."""
        from scripts.fetch_datasets import EXPECTED_SHA256 as ds
        from scripts.fetch_models import EXPECTED_SHA256 as ms

        assert ds == {}, "populate only with publisher-provided digests"
        assert ms == {}, "populate only with publisher-provided digests"

    def test_tree_digest_is_order_independent(self, tmp_path):
        from scripts.fetch_datasets import sha256_tree

        for name in ("b.txt", "a.txt", "c.txt"):
            (tmp_path / name).write_text(name, encoding="utf-8")
        first = sha256_tree(tmp_path)

        other = tmp_path / "copy"; other.mkdir()
        for name in ("c.txt", "a.txt", "b.txt"):
            (other / name).write_text(name, encoding="utf-8")
        assert sha256_tree(other) == first

    def test_tree_digest_changes_with_content(self, tmp_path):
        from scripts.fetch_datasets import sha256_tree

        (tmp_path / "a.txt").write_text("one", encoding="utf-8")
        before = sha256_tree(tmp_path)
        (tmp_path / "a.txt").write_text("two", encoding="utf-8")
        assert sha256_tree(tmp_path) != before

    def test_model_weight_patterns_exclude_pickle_by_default(self):
        """Default download must not pull .bin/.pt, which are pickles."""
        from scripts.fetch_models import MODELS

        for model in MODELS:
            joined = " ".join(model.allow_patterns)
            assert "*.bin" not in joined
            assert "*.pt" not in joined


class TestVRSBenchPrepare:
    def test_converts_vqa_rows(self, tmp_path):
        from training.prepare.vrsbench import convert

        src = tmp_path / "src"; src.mkdir()
        (src / "ann.json").write_text(json.dumps([
            {"image": "a.jpg", "question": "How many?", "answer": "3"},
            {"image": "b.jpg", "question": "What?", "answer": "field"},
        ]), encoding="utf-8")

        out = tmp_path / "instruct.jsonl"
        written, diag = convert(src, out, {"vqa", "caption", "referring"})
        assert written == 2
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert rows[0]["kind"] == "vqa"
        assert rows[0]["source"] == "vrsbench"

    def test_caption_rows_get_a_synthetic_question(self, tmp_path):
        from training.prepare.vrsbench import convert

        src = tmp_path / "src"; src.mkdir()
        (src / "cap.json").write_text(
            json.dumps([{"image": "a.jpg", "caption": "a river"}]), encoding="utf-8"
        )
        out = tmp_path / "o.jsonl"
        written, _ = convert(src, out, {"caption"})
        assert written == 1
        row = json.loads(out.read_text().splitlines()[0])
        assert row["kind"] == "caption"
        assert row["question"] == "Describe this image."

    def test_kind_filter_respected(self, tmp_path):
        from training.prepare.vrsbench import convert

        src = tmp_path / "src"; src.mkdir()
        (src / "a.json").write_text(json.dumps([
            {"image": "a.jpg", "question": "q", "answer": "a"},
            {"image": "b.jpg", "caption": "c"},
        ]), encoding="utf-8")
        out = tmp_path / "o.jsonl"
        written, _ = convert(src, out, {"vqa"})
        assert written == 1

    def test_unrecognised_schema_reports_fields_seen(self, tmp_path):
        """Guessing silently would be worse than saying what was found."""
        from training.prepare.vrsbench import convert

        src = tmp_path / "src"; src.mkdir()
        (src / "weird.json").write_text(
            json.dumps([{"totally": "different", "shape": 1}]), encoding="utf-8"
        )
        written, diag = convert(src, tmp_path / "o.jsonl", {"vqa"})
        assert written == 0
        assert "totally" in diag["fields_seen"]

    def test_missing_source_dir_is_actionable(self, tmp_path):
        from training.prepare.vrsbench import convert

        empty = tmp_path / "empty"; empty.mkdir()
        with pytest.raises(SystemExit, match="fetch_datasets"):
            convert(empty, tmp_path / "o.jsonl", {"vqa"})


class TestCheckpointDeviceHandling:
    """Regression: loading with map_location='cuda' must still restore RNG.

    torch.set_rng_state requires a CPU ByteTensor, but map_location='cuda'
    moves every stored tensor to the GPU including the RNG state. The
    kill/resume test did not catch this because it loads on CPU by default;
    the cross-sensor evaluation did.
    """

    def test_load_with_cuda_map_location(self, tmp_path):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        model = tiny_model().cuda()
        save_checkpoint(tmp_path, 1, model)
        fresh = tiny_model().cuda()
        state, _ = load_checkpoint(
            tmp_path / "ckpt_step_1.pt", fresh, map_location="cuda"
        )
        assert state.step == 1

    def test_rng_state_restored_on_cpu_tensor(self, tmp_path):
        """The restored generator state must be usable regardless of device."""
        save_checkpoint(tmp_path, 1, tiny_model())
        load_checkpoint(tmp_path / "ckpt_step_1.pt", map_location="cpu")
        assert torch.get_rng_state().device.type == "cpu"
