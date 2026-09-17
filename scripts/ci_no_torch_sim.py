"""Run the test suite as CI runs it: with no GPU stack installed.

CI installs `pip install -e ".[dev]"`. The `train` extra - torch, peft,
transformers, bitsandbytes, accelerate - is **not** installed there, so every
learned tool falls back to its stub and every torch-dependent test is meant
to skip. A developer machine has torch, so the ordinary suite exercises a
different code path from the one CI runs, and the difference is not
cosmetic: it caught a module-scope import that made the whole package
unimportable, which the local suite could not see.

`docs/code-freeze.md` makes this simulation part of the bar a bug fix has to
clear, and until now it had **no script**. It was run by hand, its result was
quoted in four documents, and nobody could reproduce it. That is the gap this
file closes.

Usage:

    python scripts/ci_no_torch_sim.py               # the whole suite
    python scripts/ci_no_torch_sim.py -k ingest     # extra args go to pytest
    python scripts/ci_no_torch_sim.py --json out.json

The mechanism is a `sitecustomize.py` on `PYTHONPATH` that installs an import
hook rejecting the GPU-stack modules, so the block is in place before pytest
collects anything - a conftest fixture would be too late for a module-scope
import. The subprocess is otherwise the same interpreter and the same
working directory.

Exit code is pytest's own, so this is usable as a CI step directly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# What CI does not have. `bitsandbytes` and `accelerate` are included even
# though little imports them directly: the point is to reproduce the absence
# of the whole extra, not of one package.
BLOCKED = ("torch", "peft", "transformers", "bitsandbytes", "accelerate", "datasets")

SITECUSTOMIZE = '''\
"""Injected by scripts/ci_no_torch_sim.py - not part of the repository."""

import sys
from importlib.abc import MetaPathFinder

BLOCKED = {blocked!r}


class _Blocked(MetaPathFinder):
    """Reject the GPU stack the way an uninstalled package is rejected.

    `find_spec` returning None would let the next finder resolve it; raising
    ModuleNotFoundError is what an absent distribution actually looks like to
    `import torch`, including to a bare `except ImportError`.
    """

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            raise ModuleNotFoundError(
                f"No module named {{root!r}} (blocked by the no-torch CI simulation)",
                name=root,
            )
        return None


sys.meta_path.insert(0, _Blocked())

# Anything imported before the hook - by another sitecustomize, or by the
# interpreter itself - would defeat it silently.
for _name in list(sys.modules):
    if _name.split(".")[0] in BLOCKED:
        del sys.modules[_name]
'''

SUMMARY_RE = re.compile(
    r"(?:(?P<passed>\d+) passed)?"
    r"(?:.*?(?P<failed>\d+) failed)?"
    r"(?:.*?(?P<skipped>\d+) skipped)?"
    r"(?:.*?(?P<errors>\d+) error)?"
)


def parse_summary(output: str) -> dict:
    """Pull the counts out of pytest's last summary line.

    Reported as None when the line cannot be found, rather than as zeros: a
    zero failure count that came from a parse miss is exactly the kind of
    number this project does not publish.
    """
    counts = {"passed": None, "failed": None, "skipped": None, "errors": None}
    for line in reversed(output.strip().splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            for key in counts:
                match = re.search(rf"(\d+) {key[:-1] if key == 'errors' else key}", line)
                if match:
                    counts[key] = int(match.group(1))
            counts["summary_line"] = line.strip()
            break
    # Which tests failed, not only how many. A red run recorded as a bare
    # count cannot be diagnosed afterwards - the JSON says "failed: 1" and the
    # only way to learn what broke is to run the whole simulation again, which
    # is how this gap was found.
    counts["failed_tests"] = re.findall(r"^FAILED (\S+)", output, re.MULTILINE)
    counts["error_tests"] = re.findall(r"^ERROR (\S+)", output, re.MULTILINE)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", type=Path, help="write the parsed result to this JSON file"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="only print the summary line"
    )
    args, pytest_args = parser.parse_known_args(argv)

    with tempfile.TemporaryDirectory(prefix="satquery_no_torch_") as shim_dir:
        (Path(shim_dir) / "sitecustomize.py").write_text(
            SITECUSTOMIZE.format(blocked=set(BLOCKED)), encoding="utf-8"
        )

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [shim_dir, *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
        )
        # The learned tools are opt-in by environment variable. If the machine
        # running this has them set, the simulation would test a different
        # configuration from CI, which has none of them.
        for name in list(env):
            if name.startswith("SATQUERY_") and name not in {
                "SATQUERY_PROFILE",
                "SATQUERY_KEEP_RUN_ARTIFACTS",
            }:
                env.pop(name)

        command = [sys.executable, "-m", "pytest", "tests/", "-q", *pytest_args]
        started = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.perf_counter() - started

    output = result.stdout + result.stderr
    if not args.quiet:
        print(output)

    parsed = parse_summary(output)
    parsed["duration_s"] = round(elapsed, 2)
    parsed["exit_code"] = result.returncode
    parsed["blocked_modules"] = list(BLOCKED)

    print(
        f"no-torch simulation: {parsed.get('summary_line', 'no summary line found')} "
        f"[{parsed['duration_s']} s, exit {result.returncode}]"
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
