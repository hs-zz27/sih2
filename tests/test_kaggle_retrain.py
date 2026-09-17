"""The Kaggle retrain driver: right paths, budget stop, packaging, failure record."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from demo_assets import ASSETS  # noqa: E402
from training.kaggle import retrain  # noqa: E402

DEMO_PATHS = {a.path for a in ASSETS}


@pytest.mark.parametrize("model", ["vqa", "change_mask", "caption"])
def test_packaged_weights_land_where_the_demo_reads_them(model, tmp_path):
    plan = retrain.build_plan(model, tmp_path)
    assert f"checkpoints/{plan.deploy_path}" in DEMO_PATHS


@pytest.mark.parametrize("model", ["vqa", "change_mask", "caption"])
def test_every_plan_downloads_verified_ids_and_writes_only_under_work(model, tmp_path):
    plan = retrain.build_plan(model, tmp_path)
    assert plan.datasets
    for step in plan.steps:
        for arg in step.cmd:
            if arg.startswith("/") and "python" not in arg:
                assert arg.startswith(str(tmp_path)), (step.name, arg)
    assert sum(s.budgeted for s in plan.steps) == 1, "exactly the training step is budgeted"


def test_vqa_uses_the_early_stopping_recipe(tmp_path):
    train = next(s for s in retrain.build_plan("vqa", tmp_path).steps if s.budgeted).cmd
    value = lambda flag: train[train.index(flag) + 1]  # noqa: E731
    assert value("--max-steps") == "1500" and value("--patience") == "3"
    assert int(value("--batch-size")) * int(value("--grad-accum")) == 16
    assert "--resume" in train


def test_unknown_model_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        retrain.build_plan("grounding", tmp_path)


def test_budget_stops_a_long_training_step():
    step = retrain.Step("train", [sys.executable, "-c", "import time; print('go', flush=True); time.sleep(60)"],
                        budgeted=True)
    log = io.StringIO()
    code, stopped = retrain.run_step(step, log, deadline=0.0)  # already past the deadline
    assert stopped is True
    assert "budget reached" in log.getvalue()


def test_unbudgeted_step_runs_to_completion():
    step = retrain.Step("prep", [sys.executable, "-c", "print('ok')"])
    log = io.StringIO()
    code, stopped = retrain.run_step(step, log, deadline=0.0)
    assert (code, stopped) == (0, False)
    assert "ok" in log.getvalue()


def test_package_prefers_the_best_adapter(tmp_path):
    plan = retrain.build_plan("vqa", tmp_path / "work")
    for name, marker in (("adapter_final", "final"), ("adapter_best", "best")):
        d = plan.ckpt_dir / name
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text(marker)
    out = tmp_path / "out"
    assert retrain.package(plan, out) == "adapter_best"
    assert (out / "checkpoints" / plan.deploy_path / "adapter_config.json").read_text() == "best"


def test_package_reports_nothing_when_training_produced_nothing(tmp_path):
    plan = retrain.build_plan("caption", tmp_path / "work")
    assert retrain.package(plan, tmp_path / "out") is None


def test_a_failing_step_is_recorded_in_the_manifest(tmp_path, monkeypatch):
    plan = retrain.Plan(
        model="caption", datasets={"x": "y"}, deploy_path="v2/caption_pre",
        steps=[retrain.Step("broken", [sys.executable, "-c", "import sys; sys.exit(3)"])],
        ckpt_dir=tmp_path / "ckpt", package_from=["."], eval_cmd=None,
    )
    monkeypatch.setattr(retrain, "build_plan", lambda model, work: plan)
    monkeypatch.setattr(sys, "argv", ["retrain.py", "--model", "caption", "--work", str(tmp_path / "w"),
                                      "--out", str(tmp_path / "out"), "--skip-install"])
    assert retrain.main() == 1
    manifest = json.loads((tmp_path / "out" / "retrain_manifest_caption.json").read_text())
    assert manifest["status"] == "failed at: broken"
    assert manifest["steps"][0]["exit_code"] == 3


def test_install_copies_weights_and_manifests(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from install_retrained import install

    out = tmp_path / "download" / "retrained"
    weights = out / "checkpoints" / "v2" / "caption_pre"
    weights.mkdir(parents=True)
    (weights / "vocab.json").write_text("{}")
    (out / "retrain_manifest_caption.json").write_text(json.dumps(
        {"model": "caption", "status": "complete", "metrics.json": {"bleu4_sentence_mean": 0.25},
         "deviations": ["none"]}))

    repo = tmp_path / "repo"
    messages = install(tmp_path / "download", repo)
    assert (repo / "checkpoints" / "v2" / "caption_pre" / "vocab.json").exists()
    assert (repo / "checkpoints" / "retrain_manifests" / "retrain_manifest_caption.json").exists()
    assert any("installed v2/caption_pre" in m for m in messages)
    # A second install does not overwrite without --force.
    assert any("skipped v2/caption_pre" in m for m in install(tmp_path / "download", repo))


@pytest.mark.parametrize("model", ["vqa", "change_mask", "caption"])
def test_notebook_runs_the_driver_for_its_model(model):
    nb = json.loads((ROOT / "notebooks" / "kaggle" / f"retrain_{model}.ipynb").read_text())
    source = "".join("".join(c["source"]) for c in nb["cells"])
    assert f"retrain.py --model {model}" in source
    assert "github.com/hs-zz27/sih2" in source
