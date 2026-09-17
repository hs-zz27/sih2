"""Every runnable script must parse arguments before it does anything.

Two defects, found together, and the second one bit hard.

**`evaluation/cdvqa_predict.py` could not be run at all.** Its docstring
documents `python evaluation/cdvqa_predict.py --split Test --out ...`, and
that exact command raised `ImportError: attempted relative import with no
known parent package`: run as a script, `evaluation` is not a package. It had
a `__main__` guard, an argument parser, and no way to reach either. Nothing
noticed, because the tests import these modules rather than executing them.

**`training/run_checkpoint_test.py` had no argument parser, and deleted
`checkpoints/` on line 9.** Passing `--help` to a script with no parser does
not print help - the argument is ignored and the program runs. Running it to
check that it *could* run destroyed every trained model in the project. That
is what the `HAS_ARGPARSE` gate below exists to prevent, in both directions:
a script without a parser is never executed here, and a `__main__` script
without a parser is a failure in its own right.

So the rule this file enforces is: **if a module can be run, it must take
arguments, and `--help` must be reachable without side effects.**

Modules with no `__main__` guard are libraries - `evaluation/harness.py` is
the one - and are excluded rather than "fixed" into scripts.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = ("evaluation", "training", "scripts")


def _scripts() -> list[Path]:
    found = []
    for directory in SEARCH_DIRS:
        for path in sorted((ROOT / directory).rglob("*.py")):
            if path.name == "__init__.py":
                continue
            if '__name__ == "__main__"' in path.read_text(encoding="utf-8"):
                found.append(path)
    return found


def _uses_argparse(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    # `parse_known_args` counts: scripts/ci_no_torch_sim.py forwards its
    # unrecognised arguments to pytest, and `--help` still reaches argparse.
    return "argparse" in source and (
        "parse_args" in source or "parse_known_args" in source
    )


SCRIPTS = _scripts()
PARSING = [p for p in SCRIPTS if _uses_argparse(p)]
NOT_PARSING = [p for p in SCRIPTS if not _uses_argparse(p)]


def _ids(paths):
    return [str(p.relative_to(ROOT)).replace("\\", "/") for p in paths]


def test_the_discovery_found_scripts():
    """Guards the guard: an empty list would make everything below vacuous."""
    assert len(SCRIPTS) >= 10


def test_every_runnable_script_parses_arguments():
    """A `__main__` script with no parser executes on `--help`.

    `training/run_checkpoint_test.py` was one, and what it executed was
    `shutil.rmtree("checkpoints")`.
    """
    assert _ids(NOT_PARSING) == []


# The `train` extra. CI installs `.[dev]` and not this, so on a runner these
# imports fail before argparse is ever reached - which is the environment
# working as designed, not the entry point being broken. Measured: 14 of the
# 44 runnable scripts import one of these at module scope, and the first
# version of this test failed all 14 under scripts/ci_no_torch_sim.py.
OPTIONAL_DEPENDENCIES = {
    "torch", "peft", "transformers", "bitsandbytes", "accelerate", "datasets",
    "h5py", "pandas", "pyarrow", "psutil",
}


def _missing_optional_dependency(output: str) -> str | None:
    """The optional package a `ModuleNotFoundError` is about, if it is one."""
    match = re.search(r"No module named '([A-Za-z0-9_]+)'", output)
    if match and match.group(1) in OPTIONAL_DEPENDENCIES:
        return match.group(1)
    return None


@pytest.mark.parametrize("script", PARSING, ids=_ids(PARSING))
def test_the_script_runs_as_a_script(script):
    """`python <path> --help` must exit 0, or say which extra is missing.

    Only for scripts that parse arguments: `--help` reaches argparse, which
    prints and exits before the program body. Executing a script that does
    NOT parse arguments would run it, which is the accident this suite is
    named after.

    A script that cannot start because the `train` extra is not installed
    **skips**. That is the CI environment and it is deliberate; failing there
    would assert that a machine has a GPU stack it was never meant to have.
    Any other non-zero exit is a real failure - including the relative-import
    error that made `evaluation/cdvqa_predict.py` unrunnable everywhere.
    """
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.returncode != 0:
        missing = _missing_optional_dependency(result.stdout + result.stderr)
        if missing:
            pytest.skip(
                f"{script.relative_to(ROOT)} needs the train extra "
                f"({missing} is not installed)"
            )

    assert result.returncode == 0, (
        f"{script.relative_to(ROOT)} cannot be executed:\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )


class TestResumeHarnessCannotDestroyCheckpoints:
    """The specific guard, tested specifically. See training/run_checkpoint_test.py."""

    def test_the_default_directory_is_not_the_checkpoints_directory(self):
        from training.run_checkpoint_test import DEFAULT_CKPT_DIR

        assert Path(DEFAULT_CKPT_DIR).name != "checkpoints"
        assert "artifacts" in Path(DEFAULT_CKPT_DIR).parts

    def test_pointing_it_at_checkpoints_is_refused(self):
        from training.run_checkpoint_test import clear_scratch

        with pytest.raises(SystemExit) as excinfo:
            clear_scratch(ROOT / "checkpoints")
        assert "trained models" in str(excinfo.value)

    def test_a_directory_holding_foreign_files_is_refused(self, tmp_path):
        """A trained model directory holds metrics.json. Nothing else does."""
        (tmp_path / "ckpt_step_5.pt").write_bytes(b"scratch")
        (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")

        from training.run_checkpoint_test import clear_scratch

        with pytest.raises(SystemExit) as excinfo:
            clear_scratch(tmp_path)
        assert "metrics.json" in str(excinfo.value)
        assert (tmp_path / "metrics.json").exists()

    def test_a_scratch_directory_is_cleared(self, tmp_path):
        target = tmp_path / "scratch"
        target.mkdir()
        (target / "ckpt_step_5.pt").write_bytes(b"scratch")

        from training.run_checkpoint_test import clear_scratch

        clear_scratch(target)

        assert target.is_dir()
        assert list(target.iterdir()) == []

    def test_a_missing_directory_is_created_not_refused(self, tmp_path):
        from training.run_checkpoint_test import clear_scratch

        target = tmp_path / "absent" / "nested"
        clear_scratch(target)

        assert target.is_dir()

class TestNoTorchSimulationReporting:
    """The gate must be diagnosable, not only countable.

    `docs/code-freeze.md` names the no-torch simulation in the bug-fix bar, so
    a red run has to say what broke. It recorded counts only, and a failure on
    2026-08-31 could not be identified from `docs/assets/ci/no_torch.json` -
    the whole simulation had to be re-run to find out what had failed.
    """

    @staticmethod
    def _parse(*output_lines: str):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "ci_no_torch_sim", Path("scripts/ci_no_torch_sim.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.parse_summary(chr(10).join(output_lines))

    def test_records_which_tests_failed(self):
        parsed = self._parse(
            "FAILED tests/test_a.py::TestX::test_one - AssertionError: boom",
            "1 failed, 5 passed in 3.21s",
        )
        assert parsed["failed"] == 1
        assert parsed["failed_tests"] == ["tests/test_a.py::TestX::test_one"]

    def test_records_errors_separately_from_failures(self):
        parsed = self._parse(
            "ERROR tests/test_b.py::test_two",
            "1 error, 2 passed in 1.0s",
        )
        assert parsed["error_tests"] == ["tests/test_b.py::test_two"]

    def test_a_green_run_lists_no_failures(self):
        parsed = self._parse("879 passed, 32 skipped in 149.12s")
        assert parsed["passed"] == 879
        assert parsed["failed_tests"] == []
