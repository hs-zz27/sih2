"""HF Jobs launcher: the generated command, no bad account defaults, install-back."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "training" / "hf_jobs"))

from install_retrained import install_from_hub  # noqa: E402
from training.hf_jobs import launch  # noqa: E402

MODELS = ["vqa", "change_mask", "caption"]


@pytest.mark.parametrize("model", MODELS)
def test_command_clones_the_real_repo_and_calls_the_kaggle_driver(model):
    cmd = launch.build_command(model, smoke=False, fp32_compute=False, push_to="x/y")
    assert launch.REPO_URL in cmd
    assert f"retrain.py --model {model}" in cmd
    # The script lives under training/kaggle/, but it must be pointed at
    # generic paths here, not Kaggle's runtime-only /kaggle/tmp or /kaggle/working.
    assert "/kaggle/tmp" not in cmd and "/kaggle/working" not in cmd
    assert "--work /tmp/satquery" in cmd and "--out /tmp/output" in cmd


def test_smoke_flag_reaches_the_kaggle_driver():
    cmd = launch.build_command("caption", smoke=True, fp32_compute=False, push_to="x/y")
    assert "--smoke" in cmd


def test_fp32_flag_only_matters_for_vqa_but_is_passed_through():
    cmd = launch.build_command("vqa", smoke=False, fp32_compute=True, push_to="x/y")
    assert "--fp32-compute" in cmd


def test_result_is_pushed_to_the_named_private_repo():
    cmd = launch.build_command("change_mask", smoke=False, fp32_compute=False,
                               push_to="friend/his-repo")
    assert "friend/his-repo" in cmd
    assert "private=True" in cmd
    assert "/tmp/output" in cmd  # what retrain.py wrote is what gets pushed


def test_no_torch_or_dependency_install_in_the_wrapper_command():
    """retrain.py installs its own training packages; the wrapper must not
    reinstall torch/torchvision, which would replace the image's CUDA build."""
    cmd = launch.build_command("vqa", smoke=False, fp32_compute=False, push_to="x/y")
    before_driver = cmd.split("retrain.py")[0]
    assert "pip install" not in before_driver
    assert "torch" not in before_driver


@pytest.mark.parametrize("model", MODELS)
def test_default_timeout_exists_for_every_model(model):
    assert model in launch.DEFAULT_TIMEOUT
    assert launch.DEFAULT_TIMEOUT[model].endswith("h")


def test_dry_run_never_submits_without_login(monkeypatch, capsys):
    """Exercises the real main() with no HF token in the environment."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["launch.py", "--model", "caption", "--dry-run"])
    code = launch.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "dry run" in out.lower()
    assert "retrain.py --model caption" in out


def test_real_submission_without_login_is_refused(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["launch.py", "--model", "caption"])
    code = launch.main()
    assert code == 1
    assert "not logged in" in capsys.readouterr().out.lower()


# --- installing an HF Job's pushed result -----------------------------------

def test_install_from_hub_downloads_then_installs_like_a_local_folder(tmp_path, monkeypatch):
    pushed = tmp_path / "pushed"
    weights = pushed / "checkpoints" / "v2" / "change_mask"
    weights.mkdir(parents=True)
    (weights / "model.pt").write_text("x")
    (pushed / "retrain_manifest_change_mask.json").write_text(
        json.dumps({"model": "change_mask", "status": "complete", "metrics.json": {"f1": 0.7}}))

    calls = {}

    def fake_snapshot_download(repo_id, repo_type, local_dir):
        calls["repo_id"] = repo_id
        calls["repo_type"] = repo_type
        # Real snapshot_download populates local_dir; mirror that.
        import shutil
        shutil.copytree(pushed, local_dir, dirs_exist_ok=True)
        return local_dir

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    repo = tmp_path / "repo"
    messages = install_from_hub("friend/satquery-retrain-change_mask", repo)
    assert calls == {"repo_id": "friend/satquery-retrain-change_mask", "repo_type": "model"}
    assert (repo / "checkpoints" / "v2" / "change_mask" / "model.pt").exists()
    assert any("complete" in m for m in messages)


def test_submission_passes_the_token_as_a_secret(monkeypatch, capsys):
    """The job must be able to push its result; the token goes as a secret."""
    import huggingface_hub

    seen = {}

    class FakeApi:
        def whoami(self):
            return {"name": "friend"}

    class FakeJob:
        url, id = "https://huggingface.co/jobs/friend/1", "1"

    def fake_run_job(**kwargs):
        seen.update(kwargs)
        return FakeJob()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_secret")
    monkeypatch.setattr(huggingface_hub, "run_job", fake_run_job)
    monkeypatch.setattr(sys, "argv", ["launch.py", "--model", "change_mask"])
    assert launch.main() == 0
    assert seen["secrets"] == {"HF_TOKEN": "hf_secret"}
    assert seen["flavor"] == "t4-small" and seen["timeout"] == "8h"
    out = capsys.readouterr().out
    assert "hf_secret" not in out            # never printed
    assert "friend/satquery-retrain-change_mask" in out


def test_vqa_scores_itself_in_the_same_job_after_training():
    cmd = launch.build_command("vqa", smoke=False, fp32_compute=False, push_to="x/y")
    train = cmd.index("retrain.py --model vqa")
    score = cmd.index("retrain.py --model eval_vqa")
    push = cmd.index("upload_folder")
    assert train < score < push
    assert "--adapter /tmp/output/checkpoints/v2/track_b_vqa/adapter_final" in cmd
    assert 'if [ "$TRAIN" = 0 ]' in cmd            # only score a successful training run


@pytest.mark.parametrize("model", ["change_mask", "caption"])
def test_other_models_do_not_score_vqa(model):
    assert "eval_vqa" not in launch.build_command(model, False, False, "x/y")


def test_push_runs_even_when_training_fails_and_exit_code_is_kept():
    """No `set -e`: a failure must still push logs/manifest, then exit non-zero."""
    cmd = launch.build_command("change_mask", smoke=False, fp32_compute=False, push_to="x/y")
    assert "set -e" not in cmd and "set -euo" not in cmd
    assert cmd.strip().endswith("exit $TRAIN")
    assert cmd.index("TRAIN=$?") < cmd.index("upload_folder")


def test_failed_scoring_cannot_block_the_push():
    cmd = launch.build_command("vqa", smoke=False, fp32_compute=False, push_to="x/y")
    assert "|| echo '!! official scoring failed" in cmd
    assert cmd.index("official scoring failed") < cmd.index("upload_folder")


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("model", MODELS)
def test_generated_script_is_valid_bash(model, smoke):
    """Piped to bash on stdin rather than written to a path: a Windows tmp
    path is not resolvable by the Git-Bash child, which used to fail this test
    for a reason unrelated to the script."""
    import subprocess
    script = launch.build_command(model, smoke, True, "x/y")
    # Bytes, not text: in text mode on Windows, Python rewrites the line
    # endings on the way to stdin and bash then rejects the stray carriage
    # returns. The real script reaches the API as the str built here.
    result = subprocess.run(["bash", "-n"], input=script.encode(), capture_output=True)
    assert result.returncode == 0, (model, smoke, result.stderr.decode())


# --- bounding what a failure can cost ---------------------------------------

@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("model", MODELS)
def test_the_driver_budget_always_fits_inside_the_job_timeout(model, smoke):
    """The regression this exists for: change_mask ran with retrain.py's
    default 10.5 h budget under a 9 h job timeout, so the budget could never
    fire. The container was killed first and the push - the only way the
    result leaves an ephemeral job - never ran."""
    prof = launch.PROFILE[(model, smoke)]
    cap = launch.hours(prof["timeout"])
    assert cap is not None
    assert prof["budget"] <= prof["deadline"], "training may not outlast the shared deadline"
    assert prof["deadline"] < cap, "the deadline must fall inside the billable window"
    assert prof["hard"] < cap, "the backstop must fire before HF kills the container"
    assert cap - prof["deadline"] >= 0.5, "leave at least 30 min to push the result"


@pytest.mark.parametrize("model", MODELS)
def test_push_is_preflighted_before_any_gpu_time_is_spent(model):
    """Repo-write used to be exercised for the first time at the very end, so
    a token that could not write turned hours of paid training into nothing."""
    cmd = launch.build_command(model, False, False, "x/y")
    assert cmd.index("preflight.py") < cmd.index("retrain.py")
    assert "job_started.json" in cmd
    assert "aborting now, before any GPU time is billed" in cmd


def test_training_and_scoring_share_one_deadline_from_job_start():
    cmd = launch.build_command("vqa", False, False, "x/y")
    prof = launch.PROFILE[("vqa", False)]
    assert cmd.count("--session-start $START") == 2
    assert f"--budget-hours {prof['budget']}" in cmd
    assert f"--budget-hours {prof['deadline']}" in cmd


def test_scoring_is_skipped_when_the_deadline_is_nearly_spent():
    """An adapter pushed unscored beats a container killed mid-upload."""
    cmd = launch.build_command("vqa", False, False, "x/y")
    assert 'if [ "$LEFT" -gt 600 ]' in cmd
    assert "no time left in the budget for official scoring" in cmd
    assert cmd.index("LEFT=") < cmd.index("upload_folder")


@pytest.mark.parametrize("model", MODELS)
def test_every_driver_call_has_a_hard_time_limit(model):
    """retrain.py budgets only the training step, so a stalled dataset or
    base-model download is the one hang it cannot interrupt by itself."""
    cmd = launch.build_command(model, False, False, "x/y")
    hard = int(launch.PROFILE[(model, False)]["hard"] * 3600)
    assert f"timeout --signal=TERM --kill-after=300 {hard}s" in cmd
    assert f"{hard}s python training/kaggle/retrain.py --model {model}" in cmd


@pytest.mark.parametrize("model", MODELS)
def test_push_is_retried(model):
    cmd = launch.build_command(model, False, False, "x/y")
    assert "for attempt in 1 2 3; do" in cmd


def test_hours_parses_the_timeout_strings_it_prices():
    assert launch.hours("12h") == 12.0
    assert launch.hours("90m") == 1.5
    assert launch.hours("2d") is None      # unpriced unit -> no false confidence


def _fake_hub(monkeypatch, submitted):
    import huggingface_hub

    class FakeApi:
        def whoami(self):
            return {"name": "me"}

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "tok")
    monkeypatch.setattr(huggingface_hub, "run_job",
                        lambda **k: submitted.append(k) or type("J", (), {"url": "u", "id": "i"}))


def test_credit_guard_refuses_a_job_it_cannot_afford(monkeypatch, capsys):
    submitted = []
    _fake_hub(monkeypatch, submitted)
    monkeypatch.setattr(sys, "argv", ["launch.py", "--model", "vqa", "--credits", "3.00"])
    assert launch.main() == 1
    assert submitted == [], "nothing may be submitted once the guard trips"
    assert "REFUSING TO SUBMIT" in capsys.readouterr().out


def test_credit_guard_allows_a_job_that_fits(monkeypatch, capsys):
    submitted = []
    _fake_hub(monkeypatch, submitted)
    monkeypatch.setattr(sys, "argv", ["launch.py", "--model", "vqa", "--credits", "11.62"])
    assert launch.main() == 0
    assert len(submitted) == 1
    assert "$4.80" in capsys.readouterr().out
