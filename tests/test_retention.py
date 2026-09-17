"""Artifact retention: bounded growth that cannot eat the evidence.

`artifacts/` was unbounded. Each full-scene run writes ~526 MB of index
rasters into `artifacts/<run_id>/`, the API pruned only its *upload*
directories under the system temp directory, and the CLI and evaluation
harness pruned nothing at all. It reached 46 GB across 7,071 directories
before being cleared by hand, and 23 GB across 1,133 by the time of this
audit.

The dangerous half of a fix like this is not failing to delete. It is
deleting the wrong thing: `artifacts/calibration/logits` is cited by
`configs/thresholds.yaml` as the provenance of a published threshold, and
`artifacts/demo_*` and `artifacts/rehearsal_*` are the evidence for Phase-4
tasks 4.1 and 4.2. Most of what follows tests that those survive.
"""

from __future__ import annotations

import os
import time

import pytest

from satquery.controller.retention import (
    DEFAULT_KEEP,
    ENV_KEEP,
    ENV_NO_AUTO_PRUNE,
    GENERATED_RUN_ID,
    keep_default,
    prune_run_artifacts,
)


def make_run(root, name: str, *, age_seconds: float = 0.0, size: int = 32):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ndvi.tif").write_bytes(b"\x00" * size)
    if age_seconds:
        stamp = time.time() - age_seconds
        os.utime(directory, (stamp, stamp))
    return directory


class TestWhatCountsAsDisposable:
    @pytest.mark.parametrize(
        "name",
        ["run_0123456789ab", "run_deadbeefcafe01", "run_abcdef12"],
    )
    def test_generated_ids_match(self, name):
        assert GENERATED_RUN_ID.match(name)

    @pytest.mark.parametrize(
        "name",
        [
            "calibration",           # cited by configs/thresholds.yaml
            "cdvqa",                 # the CDVQA correction's artifacts
            "demo_single_optical",   # task 4.1 evidence
            "rehearsal_7_large_scene",  # task 4.2 evidence
            "fixed_run_id",
            "run_final_demo",        # human-named, deliberately not matched
            "reports",
            "run_",
        ],
    )
    def test_named_directories_do_not_match(self, name):
        assert GENERATED_RUN_ID.match(name) is None


class TestPruning:
    def test_the_newest_runs_are_kept_and_the_rest_deleted(self, tmp_path):
        for i in range(10):
            make_run(tmp_path, f"run_{i:012x}", age_seconds=i * 10)

        report = prune_run_artifacts(tmp_path, keep=3)

        assert len(report.kept) == 3
        assert len(report.deleted) == 7
        # Newest first: ages 0, 10, 20 seconds.
        assert set(report.kept) == {"run_000000000000", "run_000000000001", "run_000000000002"}
        assert not (tmp_path / "run_000000000009").exists()

    def test_evidence_directories_are_never_deleted(self, tmp_path):
        protected = [
            "calibration",
            "cdvqa",
            "demo_single_optical",
            "rehearsal_7_large_scene",
            "reports",
            "run_final_demo",
        ]
        for name in protected:
            make_run(tmp_path, name, age_seconds=99999)  # oldest of all
        for i in range(5):
            make_run(tmp_path, f"run_{i:012x}")

        report = prune_run_artifacts(tmp_path, keep=0)

        assert sorted(report.protected) == sorted(protected)
        for name in protected:
            assert (tmp_path / name).is_dir(), f"{name} was deleted"
        assert len(report.deleted) == 5

    def test_keeping_more_than_exist_deletes_nothing(self, tmp_path):
        make_run(tmp_path, "run_0123456789ab")

        report = prune_run_artifacts(tmp_path, keep=20)

        assert report.deleted == []
        assert (tmp_path / "run_0123456789ab").exists()

    def test_a_dry_run_deletes_nothing_and_reports_the_size(self, tmp_path):
        make_run(tmp_path, "run_0123456789ab", size=1024)
        make_run(tmp_path, "run_ba9876543210", size=1024, age_seconds=100)

        report = prune_run_artifacts(tmp_path, keep=1, dry_run=True)

        assert report.deleted == ["run_ba9876543210"]
        assert report.bytes_deleted == 1024
        assert (tmp_path / "run_ba9876543210").exists()

    def test_a_missing_root_is_not_an_error(self, tmp_path):
        report = prune_run_artifacts(tmp_path / "absent", keep=1)

        assert report.considered == 0
        assert report.deleted == []

    def test_a_loose_file_in_the_root_is_left_alone(self, tmp_path):
        """`artifacts/soak_120.json` lives here. It is not a directory."""
        (tmp_path / "soak_120.json").write_text("{}", encoding="utf-8")
        make_run(tmp_path, "run_0123456789ab")

        prune_run_artifacts(tmp_path, keep=0)

        assert (tmp_path / "soak_120.json").exists()


class TestConfiguration:
    def test_the_default_is_twenty_runs(self, monkeypatch):
        monkeypatch.delenv(ENV_KEEP, raising=False)
        assert keep_default() == DEFAULT_KEEP == 20

    def test_the_environment_overrides_it(self, monkeypatch):
        monkeypatch.setenv(ENV_KEEP, "5")
        assert keep_default() == 5

    @pytest.mark.parametrize("value", ["", "many", "-3"])
    def test_a_nonsense_value_falls_back_rather_than_raising(self, value, monkeypatch):
        """Housekeeping must never be the thing that takes a run down."""
        monkeypatch.setenv(ENV_KEEP, value)
        assert keep_default() == DEFAULT_KEEP


class TestAutomaticPruneGuard:
    """The implicit prune can be switched off; the explicit one cannot.

    `tests/conftest.py` sets `SATQUERY_NO_AUTO_PRUNE` for the whole session,
    because several tests run `satquery ask` and `satquery eval` for real from
    the repository root and the first suite run after retention landed
    reclaimed 12.29 GB of a developer's `artifacts/` tree. Nothing protected
    was deleted - the guarantee held - but a test suite must not surprise the
    person running it.
    """

    def test_auto_prune_does_nothing_while_the_guard_is_set(self, tmp_path, monkeypatch):
        from satquery.controller.retention import ENV_NO_AUTO_PRUNE, auto_prune

        make_run(tmp_path, "run_0123456789ab")
        make_run(tmp_path, "run_ba9876543210", age_seconds=100)
        monkeypatch.setenv(ENV_NO_AUTO_PRUNE, "1")

        assert auto_prune(tmp_path, keep=0) is None
        assert len(list(tmp_path.iterdir())) == 2

    def test_auto_prune_runs_when_the_guard_is_unset(self, tmp_path, monkeypatch):
        from satquery.controller.retention import ENV_NO_AUTO_PRUNE, auto_prune

        make_run(tmp_path, "run_0123456789ab")
        monkeypatch.delenv(ENV_NO_AUTO_PRUNE, raising=False)

        report = auto_prune(tmp_path, keep=0)

        assert report is not None and report.deleted == ["run_0123456789ab"]

    def test_the_explicit_command_ignores_the_guard(self, tmp_path, monkeypatch):
        """A human who typed `satquery prune` meant it."""
        from satquery.controller.retention import ENV_NO_AUTO_PRUNE

        make_run(tmp_path, "run_0123456789ab")
        monkeypatch.setenv(ENV_NO_AUTO_PRUNE, "1")

        report = prune_run_artifacts(tmp_path, keep=0)

        assert report.deleted == ["run_0123456789ab"]


class TestCallers:
    """The three paths that write run artifacts all prune afterwards."""

    def test_the_ask_command_prunes(self, msi_4band, tmp_path, monkeypatch):
        import shutil
        from pathlib import Path

        from satquery import __main__ as cli

        # The controller loads `configs/capability_matrix.yaml` relative to
        # the working directory, so the working directory has to look like a
        # deployment root rather than an empty temp folder.
        shutil.copytree(
            Path(__file__).resolve().parents[1] / "configs", tmp_path / "configs"
        )
        monkeypatch.delenv(ENV_NO_AUTO_PRUNE, raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(ENV_KEEP, "2")
        for i in range(5):
            make_run(tmp_path / "artifacts", f"run_{i:012x}", age_seconds=100 + i)

        args = cli.build_parser().parse_args(
            ["ask", str(msi_4band), "--query", "Describe this image."]
        )
        assert cli._run_ask(args) == 0

        remaining = sorted(
            p.name for p in (tmp_path / "artifacts").iterdir()
            if p.is_dir() and GENERATED_RUN_ID.match(p.name)
        )
        assert len(remaining) == 2

    def test_the_api_prune_covers_the_artifact_tree(self, tmp_path, monkeypatch):
        """`_prune_run_dirs` bounded uploads only; the rasters outlived them."""
        import satquery.api.main as api

        monkeypatch.delenv(ENV_NO_AUTO_PRUNE, raising=False)
        monkeypatch.chdir(tmp_path)
        for i in range(4):
            make_run(tmp_path / "artifacts", f"run_{i:012x}", age_seconds=100 + i)

        api._prune_run_dirs(keep=1)

        remaining = [
            p.name for p in (tmp_path / "artifacts").iterdir() if p.is_dir()
        ]
        assert remaining == ["run_000000000000"]
