"""Crash-and-resume harness: kill a training run, restart it, prove it resumed.

Written in Phase 0 against free-tier Colab, where the session dies without
warning and a run that cannot resume has to start from zero.

## What changed on 2026-08-30, and why it matters

This script used to begin:

```python
ckpt_dir = "checkpoints"
if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)
```

`checkpoints/` is where **every trained model in this project lives** - Track
A, the captioner, grounding, the change mask, the change captioner, the
semantic change head, the optical-SAR fusion head, and the `metrics.json` and
`run_metadata.json` beside each of them that the `/models` registry page
reads. The script took no arguments, so `--help` did not print help: it was
ignored, the module ran, and the directory was destroyed. `make test-resume`
did the same thing.

That was correct in Phase 0, when `checkpoints/` held nothing but this
harness's own `ckpt_step_*.pt` files. It became a landmine the moment the
first real model was trained, and it stayed one until it went off.

Three changes, in order of how much they matter:

1. The default directory is a **scratch path under `artifacts/`**, not
   `checkpoints/`. The harness needs somewhere to write eight small tensors;
   it never needed the project's model directory.
2. Deletion is **refused** when the target holds anything this harness did
   not write. A directory with a `metrics.json` in it is somebody's trained
   model, whatever its name.
3. There is an **argument parser**, so `--help` prints help instead of
   executing the program.

Usage:

    python training/run_checkpoint_test.py            # scratch directory
    python training/run_checkpoint_test.py --ckpt-dir /tmp/resume-check
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Scratch, and deliberately not `checkpoints/`. `artifacts/` is gitignored
# runtime output; this name is not a generated run id, so the retention
# sweep in satquery/controller/retention.py leaves it alone.
DEFAULT_CKPT_DIR = Path("artifacts") / "checkpoint_resume_test"

# The only files this harness creates. Anything else in the target directory
# belongs to someone else.
OWNED_PREFIX = "ckpt_step_"


def clear_scratch(ckpt_dir: Path) -> None:
    """Empty the scratch directory, refusing anything that is not scratch.

    The refusal is the point. `checkpoints/` is named explicitly because it
    is the specific directory this script destroyed, and the general rule -
    "contains files I did not write" - catches the next one.
    """
    resolved = ckpt_dir.resolve()
    if resolved == (REPO_ROOT / "checkpoints").resolve():
        raise SystemExit(
            "refusing to use checkpoints/ as the resume-test directory: it "
            "holds the project's trained models. This script deleted them "
            "once. Pass --ckpt-dir with a scratch path, or use the default "
            f"({DEFAULT_CKPT_DIR})."
        )

    if not resolved.exists():
        resolved.mkdir(parents=True, exist_ok=True)
        return

    foreign = sorted(
        p.name for p in resolved.iterdir() if not p.name.startswith(OWNED_PREFIX)
    )
    if foreign:
        raise SystemExit(
            f"refusing to clear {resolved}: it contains {len(foreign)} file(s) "
            f"this harness did not write ({', '.join(foreign[:5])}"
            f"{', ...' if len(foreign) > 5 else ''}). Point --ckpt-dir at an "
            "empty or dedicated directory."
        )

    shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def run_test(ckpt_dir: Path) -> int:
    clear_scratch(ckpt_dir)

    print(f"--- Starting initial training run ({ckpt_dir}) ---")
    script_path = os.path.join("training", "checkpoint_resume_test.py")

    proc = subprocess.Popen(
        [
            sys.executable, script_path, "--ckpt-dir", str(ckpt_dir),
            "--save-every", "5", "--total-steps", "100",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    saved_checkpoint = False

    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(line, end="")

        if "Saved checkpoint to" in line:
            saved_checkpoint = True
            time.sleep(0.5)
            print("--- Simulating free-tier disconnect (SIGKILL) ---")
            proc.kill()
            break

    proc.wait()

    if not saved_checkpoint:
        print("ERROR: Process exited before saving a checkpoint.")
        return 1

    print("--- Resuming training run ---")
    proc2 = subprocess.Popen(
        [
            sys.executable, script_path, "--ckpt-dir", str(ckpt_dir),
            "--resume", "--save-every", "5", "--total-steps", "15",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    resume_step = -1
    for line in proc2.stdout:
        print(line, end="")
        if "RESUMING TRAINING FROM STEP" in line:
            resume_step = int(line.strip().split()[-1])

    proc2.wait()

    if resume_step > 0:
        print(
            f"\nSUCCESS: Training resumed correctly from step {resume_step} "
            "rather than 0."
        )
        return 0

    print("\nERROR: Failed to resume correctly.")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=DEFAULT_CKPT_DIR,
        help=(
            "scratch directory for the harness's own checkpoints "
            f"(default: {DEFAULT_CKPT_DIR}). Never point this at "
            "checkpoints/ - it is cleared before the run."
        ),
    )
    args = parser.parse_args(argv)
    return run_test(args.ckpt_dir)


if __name__ == "__main__":
    sys.exit(main())
