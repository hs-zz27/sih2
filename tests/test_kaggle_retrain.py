"""The Kaggle retrain driver: paths, smoke isolation, budget, NaN guard, install."""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from demo_assets import ASSETS  # noqa: E402
from install_retrained import install  # noqa: E402
from training.kaggle import envcheck, retrain  # noqa: E402

DEMO_PATHS = {a.path for a in ASSETS}
MODELS = ["vqa", "change_mask", "caption"]


def flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def train_step(plan):
    return next(s for s in plan.steps if s.budgeted)


# --- plans -----------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS)
def test_packaged_weights_land_where_the_demo_reads_them(model, tmp_path):
    assert f"checkpoints/{retrain.build_plan(model, tmp_path).deploy_path}" in DEMO_PATHS


@pytest.mark.parametrize("model", MODELS)
def test_plans_write_only_under_work_and_budget_only_training(model, tmp_path):
    plan = retrain.build_plan(model, tmp_path)
    for step in plan.steps:
        for arg in step.cmd:
            if arg.startswith("/") and "python" not in arg:
                assert arg.startswith(str(tmp_path)), (step.name, arg)
    assert sum(s.budgeted for s in plan.steps) == 1


def test_vqa_recipe_is_honest_about_examples_per_step(tmp_path):
    cmd = train_step(retrain.build_plan("vqa", tmp_path)).cmd
    # The trainer ignores --batch-size in its loop; grad-accum is the real count.
    assert flag(cmd, "--batch-size") == "1" and flag(cmd, "--grad-accum") == "8"
    assert flag(cmd, "--max-steps") == "1500" and flag(cmd, "--patience") == "3"


@pytest.mark.parametrize("model", MODELS)
def test_smoke_uses_its_own_checkpoints_so_the_real_run_cannot_resume_from_it(model, tmp_path):
    real = retrain.build_plan(model, tmp_path)
    smoke = retrain.build_plan(model, tmp_path, smoke=True)
    assert smoke.smoke and not real.smoke
    assert smoke.ckpt_dir != real.ckpt_dir
    assert "ckpt_smoke" in smoke.ckpt_dir.parts


def test_smoke_limits_are_tiny(tmp_path):
    vqa = train_step(retrain.build_plan("vqa", tmp_path, smoke=True)).cmd
    assert int(flag(vqa, "--max-steps")) <= 10 and int(flag(vqa, "--limit")) <= 64
    mask = train_step(retrain.build_plan("change_mask", tmp_path, smoke=True)).cmd
    assert flag(mask, "--epochs") == "1" and int(flag(mask, "--limit-train")) <= 64
    cap = train_step(retrain.build_plan("caption", tmp_path, smoke=True)).cmd
    assert flag(cap, "--epochs") == "1"


def test_fp32_fallback_reaches_the_trainer_environment(tmp_path):
    step = train_step(retrain.build_plan("vqa", tmp_path, fp32_compute=True))
    assert step.env == {"SATQUERY_BNB_COMPUTE_DTYPE": "float32"}
    assert train_step(retrain.build_plan("vqa", tmp_path)).env == {}


def test_install_list_never_touches_torch():
    assert not any(p.split("=")[0].split("<")[0].split(">")[0] in ("torch", "torchvision")
                   for p in retrain.PIP_PACKAGES)


def test_unknown_model_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        retrain.build_plan("grounding", tmp_path)


# --- running steps ----------------------------------------------------------

def test_budget_stops_a_long_training_step():
    step = retrain.Step("train", [sys.executable, "-c", "import time; print('go', flush=True); time.sleep(60)"],
                        budgeted=True)
    log = io.StringIO()
    _, stopped = retrain.run_step(step, log, deadline=0.0)
    assert stopped and "budget reached" in log.getvalue()


def test_quiet_slow_step_output_is_not_lost():
    step = retrain.Step("slow", [sys.executable, "-c",
                                 "import time\nprint('a', flush=True)\ntime.sleep(1.5)\nprint('b')"],
                        budgeted=True)
    log = io.StringIO()
    code, stopped = retrain.run_step(step, log, deadline=None)
    assert (code, stopped) == (0, False)
    assert "a" in log.getvalue() and "b" in log.getvalue()


def test_steps_see_one_gpu_and_their_own_env():
    step = retrain.Step("env", [sys.executable, "-c",
                                "import os; print(os.environ['CUDA_VISIBLE_DEVICES'], os.environ['X_TEST'])"],
                        env={"X_TEST": "yes"})
    log = io.StringIO()
    retrain.run_step(step, log, deadline=None)
    assert "0 yes" in log.getvalue()


# --- the NaN guard -----------------------------------------------------------

def write_history(ckpt: Path, losses: list[float], best_adapter: bool = True):
    ckpt.mkdir(parents=True, exist_ok=True)
    history = [{"step": 250 * (i + 1), "train_loss": x, "val_loss": x} for i, x in enumerate(losses)]
    # json.dumps writes NaN as NaN, and json.loads reads it back - as Python does.
    (ckpt / "val_history.json").write_text(json.dumps({"history": history}))
    if best_adapter:
        (ckpt / "adapter_best").mkdir(exist_ok=True)
        (ckpt / "adapter_best" / "adapter_config.json").write_text("{}")


def test_vqa_with_finite_losses_passes(tmp_path):
    plan = retrain.build_plan("vqa", tmp_path)
    write_history(plan.ckpt_dir, [0.16, 0.14, 0.13])
    ok, reason = retrain.check_trained(plan)
    assert ok and "0.1300" in reason


def test_vqa_nan_first_validation_is_refused_even_with_an_adapter_best(tmp_path):
    """min() picks a leading NaN as best, so a NaN adapter_best exists."""
    plan = retrain.build_plan("vqa", tmp_path)
    write_history(plan.ckpt_dir, [math.nan, 0.2, 0.3])
    ok, reason = retrain.check_trained(plan)
    assert not ok and "fp32" in reason


def test_vqa_without_any_validation_is_refused(tmp_path):
    plan = retrain.build_plan("vqa", tmp_path)
    ok, _ = retrain.check_trained(plan)
    assert not ok


def test_vqa_without_adapter_best_is_refused(tmp_path):
    plan = retrain.build_plan("vqa", tmp_path)
    write_history(plan.ckpt_dir, [0.2], best_adapter=False)
    assert retrain.check_trained(plan)[0] is False


@pytest.mark.parametrize("metrics,ok", [
    ({"f1": 0.8, "iou": 0.7}, True),
    ({"f1": math.nan}, False),
    ({"nested": {"bleu4": math.inf}}, False),
])
def test_head_metrics_must_be_finite(tmp_path, metrics, ok):
    plan = retrain.build_plan("change_mask", tmp_path)
    plan.ckpt_dir.mkdir(parents=True)
    (plan.ckpt_dir / "metrics.json").write_text(json.dumps(metrics))
    assert retrain.check_trained(plan)[0] is ok


# --- packaging and manifests --------------------------------------------------

def test_package_prefers_the_best_adapter(tmp_path):
    plan = retrain.build_plan("vqa", tmp_path / "work")
    for name, marker in (("adapter_final", "final"), ("adapter_best", "best")):
        d = plan.ckpt_dir / name
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text(marker)
    out = tmp_path / "out"
    assert retrain.package(plan, out) == "adapter_best"
    assert (out / "checkpoints" / plan.deploy_path / "adapter_config.json").read_text() == "best"


def run_main(monkeypatch, plan, argv):
    monkeypatch.setattr(retrain, "build_plan", lambda *a, **k: plan)
    monkeypatch.setattr(sys, "argv", ["retrain.py", *argv])
    return retrain.main()


def test_a_failing_step_is_recorded_in_the_manifest(tmp_path, monkeypatch):
    plan = retrain.Plan(
        model="caption", datasets={"x": "y"}, deploy_path="v2/caption_pre",
        steps=[retrain.Step("broken", [sys.executable, "-c", "import sys; sys.exit(3)"])],
        ckpt_dir=tmp_path / "ckpt", package_from=["."], eval_cmd=None,
    )
    orig = retrain.run_step
    monkeypatch.setattr(retrain, "run_step",
                        lambda step, log, deadline: (0, False) if step.name == "environment check"
                        else orig(step, log, deadline))
    code = run_main(monkeypatch, plan, ["--model", "caption", "--work", str(tmp_path / "w"),
                                        "--out", str(tmp_path / "out"), "--skip-install"])
    assert code == 1
    manifest = json.loads((tmp_path / "out" / "retrain_manifest_caption.json").read_text())
    assert manifest["status"] == "failed at: broken"


def test_nan_training_is_not_packaged(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt"
    plan = retrain.Plan(
        model="caption", datasets={"x": "y"}, deploy_path="v2/caption_pre",
        steps=[retrain.Step("train", [sys.executable, "-c",
               f"import json,pathlib; p=pathlib.Path({str(ckpt)!r}); p.mkdir(parents=True); "
               "(p/'metrics.json').write_text('{\"bleu4_sentence_mean\": NaN}'); (p/'w.pt').write_text('x')"],
               budgeted=True)],
        ckpt_dir=ckpt, package_from=["."], eval_cmd=None,
    )
    # Skip the real environment check: it would demand CUDA on this machine.
    orig = retrain.run_step
    monkeypatch.setattr(retrain, "run_step",
                        lambda step, log, deadline: (0, False) if step.name == "environment check"
                        else orig(step, log, deadline))
    code = run_main(monkeypatch, plan, ["--model", "caption", "--work", str(tmp_path / "w"),
                                        "--out", str(tmp_path / "out"), "--skip-install"])
    assert code == 1
    manifest = json.loads((tmp_path / "out" / "retrain_manifest_caption.json").read_text())
    assert manifest["status"].startswith("failed check:")
    assert not (tmp_path / "out" / "checkpoints").exists()


def test_session_start_makes_the_budget_count_earlier_work(tmp_path, monkeypatch):
    """A session that started 11 h ago has no training time left under a 10.5 h budget."""
    import time
    ckpt = tmp_path / "ckpt"
    plan = retrain.Plan(
        model="caption", datasets={"x": "y"}, deploy_path="v2/caption_pre",
        steps=[retrain.Step("train", [sys.executable, "-c", "import time; time.sleep(30)"], budgeted=True)],
        ckpt_dir=ckpt, package_from=["."], eval_cmd=None,
    )
    orig = retrain.run_step
    seen = {}

    def spy(step, log, deadline):
        if step.name == "environment check":
            return 0, False
        seen["deadline"] = deadline
        return orig(step, log, deadline)

    monkeypatch.setattr(retrain, "run_step", spy)
    start = time.time() - 11 * 3600
    run_main(monkeypatch, plan, ["--model", "caption", "--work", str(tmp_path / "w"),
                                 "--out", str(tmp_path / "out"), "--skip-install",
                                 "--session-start", str(start), "--budget-hours", "10.5"])
    assert seen["deadline"] < time.time()
    manifest = json.loads((tmp_path / "out" / "retrain_manifest_caption.json").read_text())
    assert manifest["stopped_by_budget"] is True


# --- envcheck ------------------------------------------------------------------

def test_envcheck_report_is_parsed_from_output():
    out = 'noise\nENVCHECK {"cuda": true, "problems": []}\nmore'
    assert retrain.envcheck_report(out) == {"cuda": True, "problems": []}


def test_envcheck_flags_no_cuda_and_low_disk(tmp_path, monkeypatch):
    monkeypatch.setitem(envcheck.DISK_NEEDED_GB, "caption", 10**9)
    monkeypatch.setattr(envcheck.urllib.request, "urlopen", lambda *a, **k: None)
    report, problems = envcheck.check("caption", tmp_path)
    assert any("GB free" in p for p in problems)


# --- installer -------------------------------------------------------------------

def make_output(root: Path, folder: str, manifest: dict):
    out = root / folder
    weights = out / "checkpoints" / "v2" / "caption_pre"
    weights.mkdir(parents=True)
    (weights / "vocab.json").write_text("{}")
    (out / "retrain_manifest_caption.json").write_text(json.dumps(manifest))


def test_install_copies_a_finished_run(tmp_path):
    make_output(tmp_path / "dl", "retrained", {"model": "caption", "status": "complete",
                                               "metrics.json": {"bleu4_sentence_mean": 0.25}})
    repo = tmp_path / "repo"
    messages = install(tmp_path / "dl", repo)
    assert (repo / "checkpoints" / "v2" / "caption_pre" / "vocab.json").exists()
    assert any("installed v2/caption_pre" in m for m in messages)
    assert any("skipped v2/caption_pre" in m for m in install(tmp_path / "dl", repo))


def test_install_ignores_smoke_output(tmp_path):
    make_output(tmp_path / "dl", "retrained_smoke", {"model": "caption", "smoke": True,
                                                     "status": "smoke complete"})
    repo = tmp_path / "repo"
    messages = install(tmp_path / "dl", repo)
    assert not (repo / "checkpoints" / "v2" / "caption_pre").exists()
    assert "smoke output is not" in messages[0]


# --- notebooks ------------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS)
def test_notebook_runs_smoke_before_the_real_run(model):
    nb = json.loads((ROOT / "notebooks" / "kaggle" / f"retrain_{model}.ipynb").read_text())
    source = "".join("".join(c["source"]) for c in nb["cells"])
    smoke = source.index(f"retrain.py --model {model} --smoke")
    real = source.index("--session-start {SESSION_START}")
    assert smoke < real
    assert "github.com/hs-zz27/sih2" in source


# --- the official evaluation -----------------------------------------------

def test_eval_plan_scores_an_adapter_and_produces_a_result(tmp_path):
    plan = retrain.build_plan("eval_vqa", tmp_path, adapter=tmp_path / "ad")
    assert plan.result_file is not None and plan.result_file.name.endswith(".json")
    names = [s.name for s in plan.steps]
    assert "resolve the official test split" in names
    scoring = train_step(plan)
    assert f"retrained={tmp_path / 'ad'}" in scoring.cmd
    assert "--limit" not in scoring.cmd        # full 10,004-question split


def test_eval_smoke_limits_the_question_count(tmp_path):
    cmd = train_step(retrain.build_plan("eval_vqa", tmp_path, smoke=True)).cmd
    assert flag(cmd, "--limit") == "40"


def test_eval_result_is_checked_and_packaged(tmp_path):
    plan = retrain.build_plan("eval_vqa", tmp_path / "work")
    plan.result_file.parent.mkdir(parents=True)
    plan.result_file.write_text(json.dumps(
        {"arms": {"retrained": {"published_convention": {"micro_accuracy": 0.87}}}}))
    ok, reason = retrain.check_trained(plan)
    assert ok and "0.87" in reason
    out = tmp_path / "out"
    assert retrain.package(plan, out) == plan.result_file.name
    assert (out / "results" / plan.result_file.name).exists()


def test_eval_without_a_scored_arm_is_refused(tmp_path):
    plan = retrain.build_plan("eval_vqa", tmp_path / "work")
    plan.result_file.parent.mkdir(parents=True)
    plan.result_file.write_text(json.dumps({"arms": {}}))
    assert retrain.check_trained(plan)[0] is False


def test_eval_with_a_non_finite_score_is_refused(tmp_path):
    plan = retrain.build_plan("eval_vqa", tmp_path / "work")
    plan.result_file.parent.mkdir(parents=True)
    plan.result_file.write_text('{"arms": {"retrained": {"published_convention": '
                                '{"micro_accuracy": NaN}}}}')
    assert retrain.check_trained(plan)[0] is False


# --- a non-blocking pipe that has no data yet --------------------------------

class _Raises:
    """What a non-blocking TextIOWrapper does when the pipe is empty: it does
    not return None, it raises from inside its decoder."""

    def read(self):
        raise TypeError("can't concat NoneType to bytes")

    def readline(self):
        raise TypeError("can't concat NoneType to bytes")


class _Returns:
    def __init__(self, value):
        self.value = value

    def read(self):
        return self.value

    def readline(self):
        return self.value


@pytest.mark.parametrize("method", ["read", "readline"])
def test_reading_an_empty_nonblocking_stream_yields_no_output(method):
    """The regression: an HF Job died here fourteen seconds in, and the same
    race would have killed a seven-hour run just as easily."""
    assert retrain._read(_Raises(), method) == ""
    assert retrain._read(_Returns(None), method) == ""


@pytest.mark.parametrize("method", ["read", "readline"])
def test_reading_a_nonblocking_stream_still_returns_real_output(method):
    assert retrain._read(_Returns("tail\n"), method) == "tail\n"


def test_run_step_captures_every_line_and_the_exit_code(tmp_path):
    """Drives the real subprocess path, which is where the race lives."""
    log_path = tmp_path / "step.log"
    step = retrain.Step("chatty", [sys.executable, "-c",
                                   "for i in range(500): print('line', i)"])
    with open(log_path, "w", encoding="utf-8") as log:
        code, stopped = retrain.run_step(step, log, None)

    assert (code, stopped) == (0, False)
    text = log_path.read_text(encoding="utf-8")
    assert "line 0" in text and "line 499" in text


def test_run_step_keeps_a_failing_step_exit_code(tmp_path):
    log_path = tmp_path / "step.log"
    step = retrain.Step("doomed", [sys.executable, "-c",
                                   "import sys; print('nope'); sys.exit(3)"])
    with open(log_path, "w", encoding="utf-8") as log:
        code, stopped = retrain.run_step(step, log, None)

    assert code == 3 and stopped is False
    assert "nope" in log_path.read_text(encoding="utf-8")
