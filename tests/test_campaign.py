"""The Phase 5 campaign harness: config validation, scheduling, and staging.

Everything here is torch-free on purpose. The failures these tests guard
against - a dependency cycle, a budget that never starts anything, a stale run
resumed twice, a corrupt shard that passes verification - all happen in path
and JSON logic, and all of them would otherwise be discovered on a cluster at
the cost of GPU hours. They belong in the no-torch CI run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from training.cluster import stage_data
from training.cluster.campaign import (
    DONE, FAILED, INTERRUPTED, PENDING, RUNNING, Campaign, RunState,
    load_campaign,
)
from training.cluster.env_probe import (
    BASE_RECIPES, GpuProfile, probe, recipe_for,
)

REPO = Path(__file__).resolve().parent.parent


def minimal_config(**overrides) -> dict:
    """A config whose scripts really exist, so validation has something to pass."""
    config = {
        "version": "test",
        "name": "test-campaign",
        "checkpoint_root": "checkpoints/test",
        "runs": [
            {"id": "a", "tool": "a_v1", "family": "encoder",
             "script": "training/track_a_full.py", "est_hours": 10},
            {"id": "b", "tool": "b_v1", "family": "change",
             "script": "training/train_change_mask.py", "depends_on": ["a"],
             "est_hours": 2},
            {"id": "c", "tool": "c_v1", "family": "grounding",
             "script": "training/train_grounding.py", "est_hours": 1},
        ],
    }
    config.update(overrides)
    return config


# --- Config validation ------------------------------------------------------


def test_real_campaign_config_is_valid(tmp_path):
    """The shipped config must load. It names ten runs and eight scripts."""
    campaign = load_campaign("configs/campaign.yaml", tmp_path)
    assert campaign.runs
    for run in campaign.runs.values():
        assert (REPO / run.script).is_file()
        # Every checkpoint directory is under v2. A run writing into a
        # published checkpoint directory would overwrite a frozen number.
        assert run.ckpt_dir.startswith("checkpoints/v2/"), run.id


def test_campaign_config_args_are_real_flags():
    """Every `--flag` in the config exists in the script it is passed to.

    A typo here costs a scheduled run that dies instantly at argparse, which
    on a shared cluster means a wasted queue slot rather than an error you see.
    """
    import re

    import yaml

    config = yaml.safe_load((REPO / "configs/campaign.yaml").read_text(encoding="utf-8"))
    for run in config["runs"]:
        source = (REPO / run["script"]).read_text(encoding="utf-8")
        known = set(re.findall(r'add_argument\(\s*\n?\s*"(--[a-z0-9-]+)"', source))
        used = {a for a in run["args"] if a.startswith("--")}
        assert used <= known, f"{run['id']}: unknown flags {sorted(used - known)}"


def test_campaign_config_supplies_every_required_flag():
    """A run missing a `required=True` argument dies instantly at argparse.

    The regression this caught: three runs - `change_mask`, `change_caption`
    and `optsar_fusion` - were configured without `--index`, which every one
    of them requires. On a cluster that is a queue slot spent producing an
    argparse usage message. Checking that the flags used are *valid* does not
    catch it; only checking that the required ones are *present* does.
    """
    import re

    import yaml

    config = yaml.safe_load((REPO / "configs/campaign.yaml").read_text(encoding="utf-8"))
    for run in config["runs"]:
        source = (REPO / run["script"]).read_text(encoding="utf-8")
        required = set(re.findall(
            r'add_argument\(\s*"(--[a-z0-9-]+)"[^)]*required=True', source
        ))
        used = {a for a in run["args"] if a.startswith("--")}
        assert required <= used, \
            f"{run['id']}: missing required {sorted(required - used)}"


def test_unknown_dependency_is_rejected(tmp_path):
    config = minimal_config()
    config["runs"][1]["depends_on"] = ["nonexistent"]
    with pytest.raises(ValueError, match="unknown run"):
        Campaign(config, tmp_path)


def test_dependency_cycle_is_rejected(tmp_path):
    config = minimal_config()
    config["runs"][0]["depends_on"] = ["b"]
    with pytest.raises(ValueError, match="cycle"):
        Campaign(config, tmp_path)


def test_missing_script_is_rejected(tmp_path):
    config = minimal_config()
    config["runs"][0]["script"] = "training/does_not_exist.py"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Campaign(config, tmp_path)


# --- Scheduling -------------------------------------------------------------


def test_dependencies_gate_readiness(tmp_path):
    campaign = Campaign(minimal_config(), tmp_path)
    assert "b" not in campaign.ready(), "b depends on a and a is not done"
    campaign.state["a"].status = DONE
    assert "b" in campaign.ready()


def test_failed_runs_are_not_picked_up_again(tmp_path):
    """A run that raised needs a human, not another GPU hour."""
    campaign = Campaign(minimal_config(), tmp_path)
    campaign.state["a"].status = FAILED
    campaign.state["c"].status = DONE
    assert campaign.next_run() not in ("a", "c")


def test_interrupted_runs_are_resumed(tmp_path):
    campaign = Campaign(minimal_config(), tmp_path)
    campaign.state["a"].status = INTERRUPTED
    assert campaign.next_run() == "a"


def test_budget_prefers_a_run_that_fits(tmp_path):
    campaign = Campaign(minimal_config(), tmp_path)
    # a=10h, c=1h, both ready. A 90-minute session should take c.
    assert campaign.next_run(budget_minutes=90) == "c"


def test_budget_starts_a_long_run_when_nothing_fits(tmp_path):
    """The regression this exists for.

    Sessions are 4-12 hours and runs are up to 20. A budget that refused
    anything longer than the session would report "nothing ready" forever and
    train nothing at all. Every trainer checkpoints, so starting it is right.
    """
    config = minimal_config()
    for run in config["runs"]:
        run["est_hours"] = 20
    campaign = Campaign(config, tmp_path)
    assert campaign.next_run(budget_minutes=60) is not None


def test_strict_budget_refuses_instead(tmp_path):
    config = minimal_config()
    for run in config["runs"]:
        run["est_hours"] = 20
    campaign = Campaign(config, tmp_path)
    assert campaign.next_run(budget_minutes=60, strict=True) is None


def test_partly_trained_run_is_estimated_by_what_remains(tmp_path):
    """A run 90% done is a short run, not a fresh ten-hour commitment."""
    campaign = Campaign(minimal_config(), tmp_path)
    campaign.state["a"].seconds = 9.5 * 3600      # 30 min left of 10 h
    campaign.state["c"].status = DONE
    assert campaign.remaining_minutes("a") == pytest.approx(30, abs=1)
    assert campaign.next_run(budget_minutes=60) == "a"


# --- Crash recovery ---------------------------------------------------------


def test_stale_running_run_is_reclaimed(tmp_path):
    """The session-death case: marked running, nothing running it."""
    campaign = Campaign(minimal_config(), tmp_path)
    state = campaign.state["a"]
    state.status = RUNNING
    state.owner_host = "some-dead-node"
    state.owner_pid = 999999
    state.heartbeat = time.time() - 3600
    assert campaign.reclaim_stale(verbose=False) == ["a"]
    assert campaign.state["a"].status == INTERRUPTED
    assert campaign.next_run() == "a"


def test_fresh_heartbeat_is_left_alone(tmp_path):
    """A live run on another node must not be reclaimed and started twice."""
    campaign = Campaign(minimal_config(), tmp_path)
    state = campaign.state["a"]
    state.status = RUNNING
    state.owner_host = "another-node"
    state.owner_pid = 12345
    state.heartbeat = time.time()
    assert campaign.reclaim_stale(verbose=False) == []
    assert campaign.state["a"].status == RUNNING


def test_run_owned_by_this_live_process_is_left_alone(tmp_path):
    import os

    campaign = Campaign(minimal_config(), tmp_path)
    state = campaign.state["a"]
    state.status = RUNNING
    state.owner_host = None
    state.owner_pid = os.getpid()
    state.heartbeat = time.time() - 3600
    assert state.owner_alive()
    assert campaign.reclaim_stale(verbose=False) == []


def test_state_survives_a_reload(tmp_path):
    campaign = Campaign(minimal_config(), tmp_path)
    campaign.state["a"].status = DONE
    campaign.state["a"].seconds = 1234.5
    campaign.save()

    reloaded = Campaign(minimal_config(), tmp_path)
    assert reloaded.state["a"].status == DONE
    assert reloaded.state["a"].seconds == 1234.5


def test_state_file_is_written_atomically(tmp_path):
    """No `.tmp` left behind, and the file parses after every save."""
    campaign = Campaign(minimal_config(), tmp_path)
    campaign.save()
    assert not list(tmp_path.glob("*.tmp"))
    json.loads(campaign.state_path.read_text(encoding="utf-8"))


def test_resume_flag_is_added_only_after_the_first_attempt(tmp_path):
    campaign = Campaign(minimal_config(), tmp_path)
    run = campaign.runs["a"]
    assert "--resume" not in run.command("python", resume=False)
    assert "--resume" in run.command("python", resume=True)


# --- GPU profile and batch shapes -------------------------------------------


def test_probe_never_raises_without_a_gpu():
    profile = probe(".")
    assert isinstance(profile.dtype_name, str)
    assert profile.attn_implementation in ("sdpa", "flash_attention_2")


@pytest.mark.parametrize("vram", [6.4, 16.0, 24.0, 40.0, 80.0])
@pytest.mark.parametrize("family", sorted(BASE_RECIPES))
def test_effective_batch_survives_the_card(family, vram):
    """A bigger GPU must make a run faster, not different.

    If the effective batch moved with the hardware, two runs of the same
    config on two nodes would be solving different optimisation problems and
    their metrics would not be comparable - which is exactly the thing a
    campaign spread over whatever node is free cannot afford.
    """
    profile = GpuProfile(available=True, device_count=1, name="test",
                         capability=(8, 0), vram_gb=vram, bf16=True)
    recipe = recipe_for(family, profile)
    target = BASE_RECIPES[family]["micro_batch"] * BASE_RECIPES[family]["grad_accum"]
    assert recipe["micro_batch"] >= 1
    # At or above the reference, never below it: rounding up costs a fraction
    # of a batch, rounding down silently weakens the run.
    assert target <= recipe["effective_batch"] < target * 2


def test_small_card_shrinks_the_micro_batch():
    """The bug this caught: scaling only upwards OOMs a 6 GB laptop."""
    small = GpuProfile(available=True, device_count=1, name="laptop",
                       capability=(8, 9), vram_gb=6.4, bf16=True)
    assert recipe_for("encoder", small)["micro_batch"] < \
        BASE_RECIPES["encoder"]["micro_batch"]


def test_batches_are_sized_from_free_vram_not_total():
    """The AI Lab L40S: 46.1 GB total, 11.4 GB free, two other students' jobs.

    Sizing from the total would ask for roughly four times the memory the
    driver will actually hand out, and the run dies on its first step - after
    the data has loaded, which is the expensive way to find out.
    """
    shared = GpuProfile(available=True, device_count=1, name="NVIDIA L40S",
                        capability=(8, 9), vram_gb=46.1, free_vram_gb=11.4,
                        bf16=True)
    idle = GpuProfile(available=True, device_count=1, name="NVIDIA L40S",
                      capability=(8, 9), vram_gb=46.1, free_vram_gb=46.1,
                      bf16=True)
    assert (recipe_for("encoder", shared)["micro_batch"]
            < recipe_for("encoder", idle)["micro_batch"])


def test_free_vram_falls_back_to_total_when_unreadable():
    """`mem_get_info` can fail on an odd driver; that must not mean zero batch."""
    profile = GpuProfile(available=True, device_count=1, name="card",
                         capability=(8, 0), vram_gb=24.0, free_vram_gb=0.0,
                         bf16=True)
    assert profile.usable_vram_gb == 24.0
    assert recipe_for("encoder", profile)["micro_batch"] >= 1


def test_pre_ampere_gets_fp16_and_no_flash_attention():
    turing = GpuProfile(available=True, device_count=1, name="Tesla T4",
                        capability=(7, 5), vram_gb=16.0, bf16=False)
    assert turing.dtype_name == "float16"
    assert turing.needs_grad_scaler
    assert turing.attn_implementation == "sdpa"


def test_unknown_job_family_raises():
    profile = GpuProfile()
    with pytest.raises(KeyError):
        recipe_for("not-a-family", profile)


# --- Data staging -----------------------------------------------------------


def make_dataset(root: Path, key: str = "levircd", content: bytes = b"payload") -> Path:
    subdir = root / stage_data.BY_KEY[key].subdir
    (subdir / "nested").mkdir(parents=True, exist_ok=True)
    (subdir / "a.png").write_bytes(content)
    (subdir / "nested" / "b.png").write_bytes(content + b"2")
    return subdir


def test_manifest_round_trips(tmp_path):
    make_dataset(tmp_path)
    manifest = stage_data.build_manifest(tmp_path, ["levircd"], progress=False)
    verdicts = stage_data.verify(tmp_path, manifest, ["levircd"])
    assert verdicts["levircd"].ok
    assert verdicts["levircd"].checked == 2


def test_verify_detects_a_missing_file(tmp_path):
    subdir = make_dataset(tmp_path)
    manifest = stage_data.build_manifest(tmp_path, ["levircd"], progress=False)
    (subdir / "a.png").unlink()
    verdict = stage_data.verify(tmp_path, manifest, ["levircd"])["levircd"]
    assert not verdict.ok
    assert verdict.missing


def test_verify_detects_content_changed_at_the_same_size(tmp_path):
    """The failure size-and-mtime cannot see, and the reason digests are used.

    A zero-filled file keeps its size. `docs/model-cards.md` records twelve
    sidecars that came back from a restore as NUL bytes and a verification
    that hashed without opening them.
    """
    subdir = make_dataset(tmp_path)
    manifest = stage_data.build_manifest(tmp_path, ["levircd"], progress=False)
    (subdir / "a.png").write_bytes(b"\x00" * len(b"payload"))
    verdict = stage_data.verify(tmp_path, manifest, ["levircd"])["levircd"]
    assert not verdict.ok
    assert any("digest" in c for c in verdict.corrupt)


def test_quick_manifest_is_refused_as_a_gate(tmp_path):
    make_dataset(tmp_path)
    manifest = stage_data.build_manifest(tmp_path, ["levircd"], quick=True,
                                         progress=False)
    with pytest.raises(ValueError, match="no digests"):
        stage_data.verify(tmp_path, manifest, ["levircd"])


def test_absent_dataset_is_not_reported_ok(tmp_path):
    """An empty directory must fail, not pass vacuously."""
    manifest = stage_data.build_manifest(tmp_path, ["levircd"], progress=False)
    assert not stage_data.verify(tmp_path, manifest, ["levircd"])["levircd"].ok


def test_partial_staging_reports_which_runs_are_unblocked(tmp_path):
    make_dataset(tmp_path, "levircd")
    manifest = stage_data.build_manifest(tmp_path, progress=False)
    verdicts = stage_data.verify(tmp_path, manifest)
    ready, blocked, unchecked = stage_data.runs_unblocked(verdicts)
    assert "change_mask" in ready
    assert "track_a" in blocked


def test_datasets_not_checked_are_not_reported_as_blocked(tmp_path):
    """`--only` checks a subset; the rest are unknown, not failed.

    Running `verify --only second rsvqa_lr_2k` on the lab box reported all ten
    runs blocked immediately after both datasets had *passed*. A gate that
    reports a false failure gets ignored, which costs more than no gate.
    """
    make_dataset(tmp_path, "levircd")
    manifest = stage_data.build_manifest(tmp_path, ["levircd"], progress=False)
    verdicts = stage_data.verify(tmp_path, manifest, ["levircd"])

    ready, blocked, unchecked = stage_data.runs_unblocked(verdicts)
    assert "change_mask" in ready
    assert blocked == [], "nothing failed, so nothing may be reported as blocked"
    assert "track_a" in unchecked


def test_every_campaign_run_maps_to_a_known_dataset():
    """`needs_data` must name datasets staging knows how to verify."""
    import yaml

    config = yaml.safe_load((REPO / "configs/campaign.yaml").read_text(encoding="utf-8"))
    for run in config["runs"]:
        for key in run.get("needs_data", []):
            assert key in stage_data.BY_KEY, f"{run['id']} needs unknown data '{key}'"


def test_every_required_by_names_a_real_run():
    """The other direction, which was wrong and reported runs that do not exist.

    `runs_unblocked` is read by a human deciding whether a partial transfer is
    enough to start something. Naming `stage_a2` and `caption` when neither was
    a run in the campaign made that output actively misleading.
    """
    import yaml

    config = yaml.safe_load((REPO / "configs/campaign.yaml").read_text(encoding="utf-8"))
    run_ids = {run["id"] for run in config["runs"]}
    for dataset in stage_data.DATASETS:
        for run_id in dataset.required_by:
            assert run_id in run_ids, \
                f"dataset '{dataset.key}' claims run '{run_id}', which does not exist"


def test_data_requirements_agree_in_both_directions():
    """A run that needs data X must be listed in X's `required_by`, and vice versa."""
    import yaml

    config = yaml.safe_load((REPO / "configs/campaign.yaml").read_text(encoding="utf-8"))
    for run in config["runs"]:
        for key in run.get("needs_data", []):
            assert run["id"] in stage_data.BY_KEY[key].required_by, \
                f"run '{run['id']}' needs '{key}' but is not in its required_by"


# --- Notebook survival: output volume and detached monitoring ---------------


def _noisy_script(tmp_path: Path, lines: int = 200) -> Path:
    """A stand-in trainer that prints a lot and one alarming thing."""
    script = tmp_path / "noisy.py"
    script.write_text(
        "import argparse, time\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--n', type=int, default=10)\n"
        "p.add_argument('--resume', action='store_true')\n"
        "a = p.parse_args()\n"
        "for i in range(a.n):\n"
        "    print(f'step {i} loss 0.1', flush=True)\n"
        "    if i == a.n // 2:\n"
        "        print('RuntimeError: CUDA out of memory', flush=True)\n",
        encoding="utf-8",
    )
    return script


def _campaign_with(script: Path, state_dir: Path, lines: int) -> Campaign:
    config = {
        "version": "t", "name": "t", "checkpoint_root": "checkpoints/v2",
        "runs": [{"id": "noisy", "tool": "t", "family": "head",
                  "script": "training/cluster/campaign.py", "est_hours": 0.1}],
    }
    campaign = Campaign(config, state_dir)
    # Point at the stand-in after validation, which requires a repo-relative path.
    campaign.runs["noisy"].script = str(script)
    campaign.runs["noisy"].args = ["--n", str(lines)]
    return campaign


def test_throttled_echo_collapses_output_but_keeps_the_error(tmp_path, capsys):
    """The failure this prevents is a notebook that cannot be opened.

    A 14-hour run at full verbosity writes tens of thousands of lines into the
    .ipynb as saved cell output; the file reaches hundreds of MB and the tab
    becomes unusable. Throttling has to cut the volume *without* dropping the
    one line that explains a death.
    """
    script = _noisy_script(tmp_path)
    campaign = _campaign_with(script, tmp_path / "state", 200)

    campaign.launch("noisy", echo="throttled", echo_every_s=3600)
    printed = capsys.readouterr().out.splitlines()

    assert len(printed) < 30, f"throttling let {len(printed)} lines through"
    assert any("out of memory" in line.lower() for line in printed), \
        "throttling swallowed the error line"


def test_full_echo_prints_everything(tmp_path, capsys):
    script = _noisy_script(tmp_path)
    campaign = _campaign_with(script, tmp_path / "state", 50)
    campaign.launch("noisy", echo="full")
    assert len([x for x in capsys.readouterr().out.splitlines()
                if x.startswith("step ")]) == 50


def test_the_log_file_keeps_every_line_whatever_echo_does(tmp_path, capsys):
    """Throttling is about display only. The log is the record."""
    script = _noisy_script(tmp_path)
    state_dir = tmp_path / "state"
    campaign = _campaign_with(script, state_dir, 200)
    campaign.launch("noisy", echo="none", echo_every_s=3600)
    capsys.readouterr()

    log = (state_dir / "logs" / "noisy.log").read_text(encoding="utf-8")
    assert log.count("step ") == 200


def test_tail_reads_the_end_without_reading_the_whole_file(tmp_path, capsys):
    """A monitoring cell must not load a 500 MB log to show 40 lines."""
    script = _noisy_script(tmp_path)
    state_dir = tmp_path / "state"
    campaign = _campaign_with(script, state_dir, 500)
    campaign.launch("noisy", echo="none", echo_every_s=3600)
    capsys.readouterr()

    tail = campaign.tail("noisy", lines=10)
    assert tail.count("\n") == 9
    assert "step 499" in tail
    assert "step 0 " not in tail


def test_tail_of_a_run_that_has_not_started_is_not_an_error(tmp_path):
    """The monitor cell is re-run constantly, including before anything runs."""
    campaign = Campaign(minimal_config(), tmp_path)
    assert "no log yet" in campaign.tail("a")


def test_watch_reports_from_disk_not_from_being_the_parent(tmp_path, capsys):
    """The detached mode depends on this.

    When the campaign is started with nohup from a terminal, the notebook is
    not its parent and never sees its stdout. Monitoring has to work purely
    from the state file and the logs.
    """
    script = _noisy_script(tmp_path)
    state_dir = tmp_path / "state"
    runner = _campaign_with(script, state_dir, 30)
    runner.launch("noisy", echo="none")
    capsys.readouterr()

    # A completely separate Campaign object, as a monitoring cell would build.
    observer = _campaign_with(script, state_dir, 30)
    assert observer.state["noisy"].status == DONE
    assert "step 29" in observer.tail("noisy", lines=5)


# --- Data-aware scheduling --------------------------------------------------


def _campaign_with_data(tmp_path, monkeypatch, present: list[str]):
    """A campaign whose runs need real dataset keys, with `present` on disk."""
    import training.cluster.campaign as campaign_module

    root = tmp_path / "data"
    for key in present:
        subdir = root / stage_data.BY_KEY[key].subdir
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / "file.bin").write_bytes(b"x")
    monkeypatch.setenv("SATQUERY_DATA_ROOT", str(root))
    monkeypatch.setattr(campaign_module, "REPO_ROOT", REPO, raising=False)

    config = minimal_config()
    config["runs"][0]["needs_data"] = ["ben_full"]
    config["runs"][2]["needs_data"] = ["levircd"]
    return Campaign(config, tmp_path / "state")


def test_runs_without_their_data_are_not_scheduled(tmp_path, monkeypatch):
    """The regression this exists for.

    While 66 GB is still transferring, the scheduler must not start a run
    whose data has not landed. The trainer would die at the first file open
    and the run would be recorded `failed` - a status the queue never retries,
    so ten minutes of missing data would cost the whole run.
    """
    campaign = _campaign_with_data(tmp_path, monkeypatch, present=["levircd"])
    ready = campaign.ready()
    assert "c" in ready, "levircd is present, so run c is startable"
    assert "a" not in ready, "ben_full is absent, so run a must not start"


def test_data_arriving_unblocks_the_run(tmp_path, monkeypatch):
    campaign = _campaign_with_data(tmp_path, monkeypatch, present=["levircd"])
    assert "a" not in campaign.ready()

    subdir = tmp_path / "data" / stage_data.BY_KEY["ben_full"].subdir
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / "shard.h5").write_bytes(b"x")
    assert "a" in campaign.ready()


def test_an_empty_dataset_directory_counts_as_missing(tmp_path, monkeypatch):
    """`mkdir` happens before the bytes arrive; a directory is not the data."""
    campaign = _campaign_with_data(tmp_path, monkeypatch, present=["levircd"])
    (tmp_path / "data" / stage_data.BY_KEY["ben_full"].subdir).mkdir(
        parents=True, exist_ok=True)
    assert campaign.data_missing("a") == ["ben_full"]


def test_launch_refuses_a_run_whose_data_is_absent(tmp_path, monkeypatch):
    campaign = _campaign_with_data(tmp_path, monkeypatch, present=["levircd"])
    assert campaign.launch("a", dry_run=True) == 1
    # And crucially it stays pending rather than being recorded as failed.
    assert campaign.state["a"].status == PENDING


# --- Concurrent edits to the state file -------------------------------------


def test_a_reset_from_another_shell_survives_a_running_driver(tmp_path):
    """The lost update measured on the lab box.

    `run_all` loads state once and keeps it for hours. While it ran,
    `campaign.py reset caption` was used from another shell to re-queue two
    runs that had failed on a missing dependency. The reset reached disk; the
    long-running driver then wrote its stale in-memory copy over it, and both
    runs stayed `failed` and were never retried - while the operator had every
    reason to think they had been re-queued.
    """
    driver = Campaign(minimal_config(), tmp_path)
    driver.state["a"].status = RUNNING          # the long-running driver
    driver.state["c"].status = FAILED
    driver.save()

    # Another shell re-queues 'c' after fixing what broke it.
    other = Campaign(minimal_config(), tmp_path)
    other.state["c"].status = PENDING
    other.state["c"].attempts = 0
    other.save()

    # The driver finishes its own run and reports on it.
    driver.state["a"].status = DONE
    driver.save(only="a")

    reloaded = Campaign(minimal_config(), tmp_path)
    assert reloaded.state["a"].status == DONE, "the driver's own result must land"
    assert reloaded.state["c"].status == PENDING, \
        "the external reset must survive the driver's save"


def test_step_picks_up_an_external_reset(tmp_path):
    """A driver mid-loop must see a reset performed while it was working."""
    driver = Campaign(minimal_config(), tmp_path)
    driver.state["a"].status = FAILED
    driver.state["b"].status = DONE
    driver.state["c"].status = DONE
    driver.save()
    assert driver.next_run() is None

    other = Campaign(minimal_config(), tmp_path)
    other.state["a"].status = PENDING
    other.save()

    # `step` reloads before scheduling, so the reset is visible.
    assert driver.step(dry_run=True) == "a"


def test_a_corrupt_state_file_does_not_stop_a_save(tmp_path):
    """A half-written file must not take the campaign down with it."""
    campaign = Campaign(minimal_config(), tmp_path)
    campaign.save()
    campaign.state_path.write_text("{ this is not json", encoding="utf-8")

    campaign.state["a"].status = DONE
    campaign.save(only="a")            # must not raise
    assert Campaign(minimal_config(), tmp_path).state["a"].status == DONE


def test_a_dead_owner_on_this_host_is_reclaimed_immediately(tmp_path):
    """No ten-minute wait when the pid is provably gone.

    After the driver was restarted to pick up a fix, the run it had been
    executing sat `running` with a fresh heartbeat and a dead pid. Waiting out
    the heartbeat window would have idled the GPU for ten minutes to confirm
    something already known.
    """
    campaign = Campaign(minimal_config(), tmp_path)
    state = campaign.state["a"]
    state.status = RUNNING
    state.owner_host = None          # i.e. this host
    state.owner_pid = 999999         # not a live pid
    state.heartbeat = time.time()    # fresh
    assert state.is_stale()
    assert campaign.reclaim_stale(verbose=False) == ["a"]


def test_a_live_owner_is_never_reclaimed_however_old_the_heartbeat(tmp_path):
    import os

    campaign = Campaign(minimal_config(), tmp_path)
    state = campaign.state["a"]
    state.status = RUNNING
    state.owner_host = None
    state.owner_pid = os.getpid()
    state.heartbeat = time.time() - 100_000
    assert not state.is_stale()


# --- Batch size comes from the card, not the config -------------------------


def test_command_appends_the_measured_batch_size(tmp_path):
    """The gap this closes: env_probe computed a size nothing ever used.

    `configs/campaign.yaml` pinned `--batch-size 64`, so the probe was an
    elaborate way of printing a number. On the lab GPU with ~10 GB free and a
    stem that reshapes to `batch x 12 bands`, that 64 became 768 images in one
    forward pass and track_a OOMed after loading 43 GB of shards.
    """
    campaign = Campaign(minimal_config(), tmp_path)
    cmd = campaign.runs["a"].command("python", resume=False, batch_size=8)
    assert "--batch-size" in cmd
    assert cmd[cmd.index("--batch-size") + 1] == "8"


def test_a_config_that_pins_its_batch_size_wins(tmp_path):
    """An explicit 64 in the config must not be silently overridden."""
    config = minimal_config()
    config["runs"][0]["args"] = ["--batch-size", "64"]
    campaign = Campaign(config, tmp_path)
    cmd = campaign.runs["a"].command("python", resume=False, batch_size=8)
    assert cmd.count("--batch-size") == 1
    assert cmd[cmd.index("--batch-size") + 1] == "64"


def test_no_batch_size_is_appended_without_a_gpu(tmp_path):
    """On CPU the trainer's own default is the right answer."""
    campaign = Campaign(minimal_config(), tmp_path)
    cmd = campaign.runs["a"].command("python", resume=False, batch_size=None)
    assert "--batch-size" not in cmd


def test_multiband_family_is_smaller_than_the_plain_encoder():
    """Track A's activation memory scales with batch x bands, not batch."""
    from training.cluster.env_probe import BASE_RECIPES

    plain = BASE_RECIPES["encoder"]
    multiband = BASE_RECIPES["encoder_multiband"]
    assert multiband["micro_batch"] < plain["micro_batch"]
    # ...but the effective batch is unchanged, so the two stay comparable.
    assert (multiband["micro_batch"] * multiband["grad_accum"]
            == plain["micro_batch"] * plain["grad_accum"])


def test_every_campaign_family_has_a_recipe():
    """A family with no recipe silently falls back to the trainer's default."""
    import yaml

    from training.cluster.env_probe import BASE_RECIPES

    config = yaml.safe_load((REPO / "configs/campaign.yaml").read_text(encoding="utf-8"))
    for run in config["runs"]:
        assert run["family"] in BASE_RECIPES, \
            f"{run['id']}: family '{run['family']}' has no batch recipe"


def test_batch_size_is_not_injected_into_a_script_that_lacks_the_flag(tmp_path):
    """Injecting an undeclared flag turns a working run into an argparse error.

    Caught by the echo tests, whose synthetic script takes no arguments: every
    one of them started failing the moment auto-sizing was added.
    """
    script = tmp_path / "plain.py"
    script.write_text("import sys\nprint('hi')\n", encoding="utf-8")
    config = minimal_config()
    config["runs"][0]["script"] = str(script)
    campaign = Campaign(config, tmp_path / "state")
    cmd = campaign.runs["a"].command("python", resume=False, batch_size=8)
    assert "--batch-size" not in cmd


def test_all_campaign_scripts_accept_the_flag_they_will_be_given(tmp_path):
    campaign = load_campaign("configs/campaign.yaml", tmp_path)
    for run in campaign.runs.values():
        assert run.accepts_batch_size(), \
            f"{run.id}: {run.script} does not declare --batch-size"


# --- An orphaned trainer must never be double-started -----------------------


def test_a_live_trainer_keeps_the_run_from_being_reclaimed(tmp_path):
    """The near-miss on the lab box.

    The driver was killed while `track_a` kept training as an orphan. The
    queue tracked only the DRIVER's pid, so it reported the run stale and
    ready to restart - which would have put two trainers in one checkpoint
    directory, the exact corruption this file exists to prevent.
    """
    import os

    campaign = Campaign(minimal_config(), tmp_path)
    state = campaign.state["a"]
    state.status = RUNNING
    state.owner_host = None
    state.owner_pid = 999999          # driver: dead
    state.child_pid = os.getpid()     # trainer: alive
    state.child_cmdline = None
    state.heartbeat = time.time() - 10_000

    assert state.trainer_alive()
    assert not state.is_stale()
    assert campaign.reclaim_stale(verbose=False) == []
    assert "a" not in campaign.ready()


def test_a_dead_trainer_with_a_dead_driver_is_reclaimed(tmp_path):
    campaign = Campaign(minimal_config(), tmp_path)
    state = campaign.state["a"]
    state.status = RUNNING
    state.owner_host = None
    state.owner_pid = 999999
    state.child_pid = 999998
    state.heartbeat = time.time() - 10_000
    assert not state.trainer_alive()
    assert state.is_stale()


def test_pid_reuse_does_not_make_a_stranger_look_like_our_trainer(tmp_path):
    """A recycled pid running something else must not hang the queue forever."""
    import os
    import sys

    campaign = Campaign(minimal_config(), tmp_path)
    state = campaign.state["a"]
    state.status = RUNNING
    state.child_pid = os.getpid()
    state.child_cmdline = "definitely_not_this_process_xyzzy.py"

    if sys.platform.startswith("linux"):
        assert not state.trainer_alive()
    else:
        # No /proc: the pid check stands alone and errs towards "alive",
        # which is the safe direction - it refuses to start a duplicate.
        assert state.trainer_alive()


def test_no_child_pid_recorded_means_not_alive(tmp_path):
    campaign = Campaign(minimal_config(), tmp_path)
    assert not campaign.state["a"].trainer_alive()


def test_runs_compute_every_metric_the_comparison_expects():
    """A run must produce the numbers it will later be judged on.

    `track_a` finished having written only `map_all_bands`, because the config
    never passed `--ablation` - the flag that computes 4-band mAP and
    retention. The v1 published run passed it, so the v2 result was not
    comparable on the Cartosat claim, which is the one that matters most; and
    `track_a_nodropout`, whose entire purpose is measuring retention, was
    running under the same omission.

    Nothing in the config or the trainer objected. This does.
    """
    import re

    import yaml

    from evaluation.compare_v2 import COMPARISONS

    config = yaml.safe_load((REPO / "configs/campaign.yaml").read_text(encoding="utf-8"))
    by_id = {run["id"]: run for run in config["runs"]}

    # Metrics only produced under a flag, and the flag that produces them.
    GATED = {"map_cartosat_4band": "--ablation", "retention": "--ablation"}

    for entry in COMPARISONS:
        flag = GATED.get(entry.metric)
        if flag is None:
            continue
        run = by_id.get(entry.run_id)
        assert run is not None, f"comparison names unknown run {entry.run_id}"
        assert flag in run["args"], (
            f"{entry.run_id} is compared on '{entry.metric}', which is only "
            f"computed when {flag} is passed - and it is not"
        )
        source = (REPO / run["script"]).read_text(encoding="utf-8")
        assert re.search(r'add_argument\(\s*"' + re.escape(flag) + '"', source), \
            f"{run['script']} does not define {flag}"


# --- Pruning must not delete the best checkpoint ----------------------------


def test_prune_keeps_the_protected_checkpoint(tmp_path):
    """The bug that cost `track_b_vqa` its best adapter.

    Validation loss bottomed at step 4000 (0.1649) and rose to 0.2529 by step
    6000. With `keep_last=3` only 5500/5750/6000 survived, so the adapter the
    pipeline loads was measurably worse than one that had been deleted - and
    `val_history.json` still named `ckpt_step_4000.pt` as best, pointing at a
    file that no longer existed.
    """
    pytest.importorskip("torch")
    from training.common.checkpointing import _prune_old_checkpoints

    for step in (4000, 5500, 5750, 6000):
        (tmp_path / f"ckpt_step_{step}.pt").write_bytes(b"x")
        (tmp_path / f"adapter_step_{step}").mkdir()

    _prune_old_checkpoints(tmp_path, keep_last=3, protect=4000)

    surviving = {p.name for p in tmp_path.glob("ckpt_step_*.pt")}
    assert surviving == {"ckpt_step_4000.pt", "ckpt_step_5500.pt",
                         "ckpt_step_5750.pt", "ckpt_step_6000.pt"}
    # The adapter directory beside it must survive too - the .pt alone is
    # useless for a PEFT run, which stores the weights next to it.
    assert (tmp_path / "adapter_step_4000").is_dir()


def test_prune_without_protection_is_unchanged(tmp_path):
    """The default must still behave exactly as it always did."""
    pytest.importorskip("torch")
    from training.common.checkpointing import _prune_old_checkpoints

    for step in (1000, 2000, 3000, 4000):
        (tmp_path / f"ckpt_step_{step}.pt").write_bytes(b"x")

    _prune_old_checkpoints(tmp_path, keep_last=3)
    assert {p.name for p in tmp_path.glob("ckpt_step_*.pt")} == {
        "ckpt_step_2000.pt", "ckpt_step_3000.pt", "ckpt_step_4000.pt"}


def test_protecting_a_recent_checkpoint_deletes_nothing_extra(tmp_path):
    pytest.importorskip("torch")
    from training.common.checkpointing import _prune_old_checkpoints

    for step in (1000, 2000, 3000, 4000):
        (tmp_path / f"ckpt_step_{step}.pt").write_bytes(b"x")

    _prune_old_checkpoints(tmp_path, keep_last=3, protect=4000)
    assert len(list(tmp_path.glob("ckpt_step_*.pt"))) == 3
