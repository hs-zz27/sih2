"""Bounded retention for the index rasters a run writes to `artifacts/`.

Every full-scene run writes ~526 MB of index rasters into
`artifacts/<run_id>/`, and until now nothing ever removed them. The API
already prunes its *upload* directories under the system temp directory
(`_prune_run_dirs` in `satquery/api/main.py`), but the artifact tree it wrote
alongside them was untouched, and the CLI and the evaluation harness prune
neither. It reached **46 GB across 7,071 directories** before being cleared
by hand on 2026-08-30, and **23 GB across 1,133 directories** again by the
audit on the same day. That is a disk-exhaustion failure with a stopwatch on
it, not a tidiness problem.

## What may be deleted, and what may never be

Only directories whose name matches the **generated** run-id shape:
`run_` followed by hex, which is what `satquery.api.main` mints
(`run_{uuid4().hex[:12]}`) and what `satquery.ingest` generates when no id is
supplied. Those are disposable by construction - one query's scratch output,
reproducible by re-running the query.

Everything else under `artifacts/` is left alone, unconditionally:

* `calibration/logits/` - the cached logits behind the fitted calibration and
  the selective-prediction curves. `configs/thresholds.yaml` cites them as
  the provenance of the land-cover assertion threshold; deleting them would
  orphan a published number.
* `cdvqa/` - the CDVQA prediction and diagnosis artifacts behind the 0.5380
  correction.
* `demo_*` and `rehearsal_*` - the demo bundle's and the rehearsal harness's
  named outputs, which are evidence for tasks 4.1 and 4.2.
* anything else a human named, `reports/`, `soak_120.json`, and so on.

The rule is deliberately whitelist-shaped: an unrecognised directory is
**kept**. Deleting evidence to save disk would trade a recoverable problem
for an unrecoverable one.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

# `run_` plus at least eight hex characters. Deliberately narrow: it matches
# the generated ids and not a human-chosen name like `run_final_demo`, which
# would be someone's kept output. `fixed_run_id`, `psq`, `q5` and the rest of
# the hand-named directories fail it too.
GENERATED_RUN_ID = re.compile(r"^run_[0-9a-f]{8,}$")

DEFAULT_ROOT = Path("artifacts")
ENV_KEEP = "SATQUERY_KEEP_RUN_ARTIFACTS"
DEFAULT_KEEP = 20

# Set to disable the *automatic* prune that the API, `satquery ask` and
# `satquery eval` perform after a run. `tests/conftest.py` sets it for the
# whole session: the suite exercises those entry points from the repository
# root, and the first run of it reclaimed 12.29 GB of a developer's real
# `artifacts/` tree as a side effect of running the tests. Nothing protected
# was touched - the 78 named evidence directories all survived, which is the
# guarantee working - but a test suite silently deleting a developer's output
# is a surprise, and surprises are how the checkpoint incident happened.
#
# It does NOT disable `prune_run_artifacts` itself, so `satquery prune` and
# the retention tests still work while the variable is set. Only the implicit
# call sites consult it.
ENV_NO_AUTO_PRUNE = "SATQUERY_NO_AUTO_PRUNE"


def keep_default() -> int:
    """How many run directories to retain, overridable by environment.

    A non-numeric or negative value falls back to the default rather than
    raising: retention is housekeeping, and housekeeping must not be able to
    take a run down.
    """
    try:
        value = int(os.environ.get(ENV_KEEP, DEFAULT_KEEP))
    except (TypeError, ValueError):
        return DEFAULT_KEEP
    return value if value >= 0 else DEFAULT_KEEP


@dataclass
class PruneReport:
    """What a prune did, or would do under `dry_run`."""

    root: Path
    kept: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    protected: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    bytes_deleted: int = 0

    @property
    def considered(self) -> int:
        return len(self.kept) + len(self.deleted) + len(self.failed)


def auto_prune(root: Path | str | None = None, keep: int | None = None) -> PruneReport | None:
    """The implicit prune the entry points perform after a run.

    Returns None - having done nothing - when `SATQUERY_NO_AUTO_PRUNE` is set.
    Explicit pruning (`satquery prune`) ignores the variable, because a user
    who typed the command means it.
    """
    if os.environ.get(ENV_NO_AUTO_PRUNE):
        return None
    return prune_run_artifacts(root, keep)


def _directory_size(path: Path) -> int:
    total = 0
    for file in path.rglob("*"):
        try:
            if file.is_file():
                total += file.stat().st_size
        except OSError:
            continue
    return total


def prune_run_artifacts(
    root: Path | str | None = None,
    keep: int | None = None,
    *,
    dry_run: bool = False,
    measure: bool = False,
) -> PruneReport:
    """Delete generated run directories beyond the newest `keep`.

    Ordering is by modification time, newest first, so the most recent runs -
    the ones a `/runs/{id}` permalink or an open preview might still be
    reading - are the ones kept.

    Best effort throughout. A directory that will not delete (an open file
    handle on Windows, a permission problem) is recorded in `failed` and
    retried on the next call rather than raising: reclaiming disk must never
    be the thing that fails a run which has already succeeded.
    """
    root = Path(root or DEFAULT_ROOT)
    keep = keep_default() if keep is None else keep
    report = PruneReport(root=root)

    if not root.is_dir():
        return report

    candidates = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if not GENERATED_RUN_ID.match(entry.name):
            report.protected.append(entry.name)
            continue
        try:
            candidates.append((entry.stat().st_mtime, entry))
        except OSError:
            report.protected.append(entry.name)

    candidates.sort(key=lambda pair: pair[0], reverse=True)

    for _, entry in candidates[:keep]:
        report.kept.append(entry.name)

    for _, entry in candidates[keep:]:
        size = _directory_size(entry) if (measure or dry_run) else 0
        if dry_run:
            report.deleted.append(entry.name)
            report.bytes_deleted += size
            continue
        shutil.rmtree(entry, ignore_errors=True)
        if entry.exists():
            report.failed.append(entry.name)
        else:
            report.deleted.append(entry.name)
            report.bytes_deleted += size

    return report
