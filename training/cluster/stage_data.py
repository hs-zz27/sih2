"""Getting the prepared datasets onto a cluster, and proving they arrived.

`scripts/fetch_datasets.py` downloads the *raw* corpora. It is not what a
campaign run consumes. Every trainer in this repository reads a **prepared**
artifact produced by `training/prepare/*.py` - the HDF5 shards under
`data/ben_full`, the `index.json` files, `instruct.jsonl` - and those take
hours of CPU to rebuild. On a notebook-only cluster where a session can be
killed at any moment, rebuilding them per session is not viable.

So there are two routes onto the cluster, and this module supports both
because which one is available is a property of the cluster, not a decision:

* **ship the prepared artifacts** (rsync/scp from a machine that already
  built them), then verify here;
* **rebuild on the cluster** from raw downloads, then verify against the same
  manifest to confirm the rebuild produced the same bytes.

The manifest is what makes those two routes interchangeable. Both end with
the same digests or the campaign refuses to start.

WHY VERIFY AT ALL

Because the failure this guards against has already happened to this project
once. `docs/model-cards.md` records twelve JSON sidecars that came back from a
restore as NUL bytes, and the verification that missed it *hashed* files
without opening them. A truncated HDF5 shard behaves the same way: `h5py`
opens it, the first shard reads fine, and the run dies four hours in - or
worse, does not die and trains on a short epoch. A digest check before the
first step costs minutes and converts that into an immediate, named failure.

WHY FULL SHA-256 AND NOT SIZE+MTIME

mtime does not survive rsync, scp, a zip round-trip or a shared filesystem
with a skewed clock, and size does not move when a transfer zero-fills. The
one failure mode that matters here is *content that changed without the size
changing*, which is precisely what size+mtime cannot see. `--quick` exists for
triage and says so in its output; it is never what `verify` runs by default,
and the campaign driver refuses a quick verification as a gate.

Usage:
    # on the machine that has prepared data
    python training/cluster/stage_data.py build --root data --out configs/data_manifest.json

    # on the cluster, after rsync
    python training/cluster/stage_data.py verify --root /scratch/$USER/data \
        --manifest configs/data_manifest.json

    # what is missing, before spending a transfer window on it
    python training/cluster/stage_data.py plan --root /scratch/$USER/data \
        --manifest configs/data_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.common.paths import index_path  # noqa: E402

CHUNK = 1024 * 1024
MANIFEST_VERSION = 2

# The prepared artifacts each campaign run needs, keyed by the dataset
# directory under the data root. `required_by` names the campaign runs that
# cannot start without it, so `plan` can tell you which runs a partial
# transfer has already unblocked instead of reporting one undifferentiated
# "incomplete".
@dataclass(frozen=True)
class DataSet:
    key: str
    subdir: str
    required_by: tuple[str, ...]
    patterns: tuple[str, ...] = ("**/*",)
    note: str = ""


DATASETS: tuple[DataSet, ...] = (
    DataSet(
        key="ben_full",
        subdir="ben_full",
        required_by=("track_a", "track_a_nodropout"),
        patterns=("*.h5", "*.hdf5", "*.json"),
        note="BigEarthNet v2 HDF5 shards: images (N,12,120,120) + labels19",
    ),
    DataSet(
        key="bigearthnet_14k",
        subdir="bigearthnet_14k",
        required_by=(),
        note="the v0 subset; kept so the published Track A v0 numbers stay reproducible",
    ),
    DataSet(
        key="whu_opt_sar",
        subdir="whu_opt_sar",
        required_by=("optsar_fusion", "track_b_vqa"),
        note="~5 m optical+SAR, the middle rung of the resolution ladder",
    ),
    DataSet(
        key="instruct_mix",
        subdir="instruct_mix",
        required_by=("track_b_vqa",),
        patterns=("*.jsonl", "*.json"),
        note="instruction mix: POINTERS ONLY (~5 MB). The pixels are in "
             "whu_opt_sar and rsvqa_lr_2k - staging this alone trains nothing",
    ),
    DataSet(
        key="rsvqa_lr_2k",
        subdir="rsvqa_lr_2k",
        required_by=("track_b_vqa",),
        note="RSVQA-LR imagery; 2,043 of the instruction mix's rows point here",
    ),
    DataSet(
        key="second",
        subdir="second",
        required_by=("change_vqa", "change_vqa_scratch_v2"),
        note="SECOND semantic-change pairs - the imagery behind CDVQA. "
             "Weights trained on it are unpublishable: it states no licence",
    ),
    DataSet(
        key="dior_rsvg",
        subdir="dior_rsvg",
        required_by=("grounding",),
        note="referring-expression grounding",
    ),
    DataSet(
        key="levircd",
        subdir="levircd",
        required_by=("change_mask",),
        note="LEVIR-CD bitemporal pairs + change masks",
    ),
    DataSet(
        key="levir_mci",
        subdir="levir_mci",
        required_by=("change_caption",),
        note="LEVIR-MCI: masks and captions from one corpus",
    ),
    DataSet(
        key="cdvqa",
        subdir="cdvqa",
        required_by=("change_vqa", "change_vqa_scratch_v2"),
        note="change VQA",
    ),
    DataSet(
        key="rsicd",
        subdir="rsicd",
        required_by=("caption",),
        note="RSICD captions",
    ),
)

BY_KEY = {d.key: d for d in DATASETS}


def sha256_file(path: Path, chunk: int = CHUNK) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def iter_files(root: Path, dataset: DataSet):
    """Every file of `dataset` under `root`, sorted, relative to `root`.

    Sorted so that two machines hashing the same data produce byte-identical
    manifests, and a diff between them reflects a change in the data rather
    than the order the filesystem happened to enumerate it.
    """
    base = root / dataset.subdir
    if not base.is_dir():
        return
    seen: set[Path] = set()
    for pattern in dataset.patterns:
        for path in base.glob(pattern):
            if path.is_file():
                seen.add(path)
    for path in sorted(seen):
        yield path.relative_to(root)


def build_manifest(root: Path, keys: list[str] | None = None,
                   quick: bool = False, progress: bool = True) -> dict:
    """Hash everything and record it. This is the slow, authoritative pass."""
    root = Path(root)
    chosen = [BY_KEY[k] for k in keys] if keys else list(DATASETS)
    entries: dict[str, dict] = {}
    totals: dict[str, dict] = {}

    for dataset in chosen:
        files, nbytes = {}, 0
        started = time.time()
        for rel in iter_files(root, dataset):
            absolute = root / rel
            size = absolute.stat().st_size
            record = {"size": size}
            if not quick:
                record["sha256"] = sha256_file(absolute)
            files[rel.as_posix()] = record
            nbytes += size
        entries[dataset.key] = files
        totals[dataset.key] = {
            "files": len(files),
            "bytes": nbytes,
            "seconds": round(time.time() - started, 1),
            "required_by": list(dataset.required_by),
            "note": dataset.note,
            "present": bool(files),
        }
        if progress:
            state = f"{len(files):>6} files  {nbytes / 1e9:>8.2f} GB"
            print(f"  {dataset.key:<18} {state}"
                  f"{'  (absent)' if not files else ''}", flush=True)

    return {
        "manifest_version": MANIFEST_VERSION,
        "quick": quick,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "built_on": os.environ.get("HOSTNAME") or os.environ.get("COMPUTERNAME", "?"),
        "root": str(root),
        "totals": totals,
        "files": entries,
    }


@dataclass
class Verdict:
    """The outcome for one dataset. `ok` is the only thing the gate reads."""

    key: str
    ok: bool
    checked: int = 0
    missing: list[str] = None
    corrupt: list[str] = None
    extra: int = 0

    def __post_init__(self):
        self.missing = self.missing or []
        self.corrupt = self.corrupt or []

    def summary(self) -> str:
        if self.ok:
            return f"{self.key:<18} OK       {self.checked} files"
        parts = []
        if self.missing:
            parts.append(f"{len(self.missing)} missing")
        if self.corrupt:
            parts.append(f"{len(self.corrupt)} corrupt")
        return f"{self.key:<18} FAILED   {', '.join(parts) or 'absent'}"


def verify(root: Path, manifest: dict, keys: list[str] | None = None,
           quick: bool = False, limit_report: int = 5) -> dict[str, Verdict]:
    """Check the data on disk against the manifest.

    `quick` compares sizes only and is for triage. The campaign gate rejects a
    quick pass, because a zero-filled file is exactly the failure that keeps
    its size.
    """
    root = Path(root)
    if manifest.get("quick") and not quick:
        raise ValueError(
            "manifest was built with --quick and carries no digests; "
            "rebuild it without --quick before using it as a gate"
        )

    chosen = keys or list(manifest["files"].keys())
    verdicts: dict[str, Verdict] = {}

    for key in chosen:
        recorded = manifest["files"].get(key)
        if recorded is None:
            verdicts[key] = Verdict(key, ok=False)
            continue

        missing, corrupt, checked = [], [], 0
        for rel, expected in recorded.items():
            # Manifests may be built on Windows; `index_path` is the repo's
            # existing rule for reading a stored path on either OS.
            absolute = root / index_path(rel)
            if not absolute.is_file():
                missing.append(rel)
                continue
            if absolute.stat().st_size != expected["size"]:
                corrupt.append(f"{rel} (size)")
                continue
            if not quick and "sha256" in expected:
                if sha256_file(absolute) != expected["sha256"]:
                    corrupt.append(f"{rel} (digest)")
                    continue
            checked += 1

        verdicts[key] = Verdict(
            key=key,
            ok=not missing and not corrupt and checked > 0,
            checked=checked,
            missing=missing[:limit_report],
            corrupt=corrupt[:limit_report],
        )

    return verdicts


def runs_unblocked(
    verdicts: dict[str, Verdict],
) -> tuple[list[str], list[str], list[str]]:
    """Which campaign runs the verified data allows: ready, blocked, unchecked.

    A partial transfer is the normal state during a staging window, and the
    useful question then is not "is the data complete" but "can I start
    anything yet".

    THREE OUTCOMES, NOT TWO. "Not checked" is not the same as "failed", and
    conflating them made a `--only second rsvqa_lr_2k` run report all ten runs
    as blocked when both datasets had in fact just passed. A gate that reports
    a false failure gets ignored, which costs more than having no gate.

    A run is `ready` only when every dataset it needs was checked AND passed -
    never on the strength of an unchecked one.
    """
    ready, blocked, unchecked = set(), set(), set()
    needs: dict[str, list[DataSet]] = {}
    for dataset in DATASETS:
        for run in dataset.required_by:
            needs.setdefault(run, []).append(dataset)

    for run, datasets in needs.items():
        results = [verdicts.get(d.key) for d in datasets]
        if any(v is not None and not v.ok for v in results):
            blocked.add(run)
        elif any(v is None for v in results):
            unchecked.add(run)
        else:
            ready.add(run)

    return sorted(ready), sorted(blocked), sorted(unchecked)


def _load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description="Stage and verify prepared datasets.")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path,
                        default=Path(os.environ.get("SATQUERY_DATA_ROOT", "data")))
    common.add_argument("--only", nargs="*", metavar="KEY",
                        help=f"restrict to these datasets: {', '.join(BY_KEY)}")
    common.add_argument("--quick", action="store_true",
                        help="size only, no digests - triage, never a gate")

    b = sub.add_parser("build", parents=[common], help="hash the prepared data")
    b.add_argument("--out", type=Path, default=Path("configs/data_manifest.json"))

    v = sub.add_parser("verify", parents=[common], help="check data against a manifest")
    v.add_argument("--manifest", type=Path, default=Path("configs/data_manifest.json"))

    pl = sub.add_parser("plan", parents=[common],
                        help="what is present, and which runs it unblocks")
    pl.add_argument("--manifest", type=Path, default=Path("configs/data_manifest.json"))

    sub.add_parser("list", help="the datasets and the runs that need them")

    args = p.parse_args()

    if args.command == "list":
        print(f"{'key':<18} {'required_by':<34} note")
        for d in DATASETS:
            req = ", ".join(d.required_by) or "-"
            print(f"{d.key:<18} {req:<34} {d.note}")
        return 0

    if args.only:
        unknown = [k for k in args.only if k not in BY_KEY]
        if unknown:
            print(f"unknown dataset key(s): {', '.join(unknown)}", file=sys.stderr)
            return 2

    if args.command == "build":
        print(f"hashing {args.root} ...")
        manifest = build_manifest(args.root, args.only, quick=args.quick)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        total = sum(t["bytes"] for t in manifest["totals"].values())
        print(f"\nwrote {args.out}  ({total / 1e9:.2f} GB)")
        if args.quick:
            print("NOTE: --quick manifest carries no digests and cannot gate a campaign.")
        return 0

    manifest = _load_manifest(args.manifest)
    verdicts = verify(args.root, manifest, args.only, quick=args.quick)

    for verdict in verdicts.values():
        print(verdict.summary())
        for item in verdict.missing:
            print(f"    missing  {item}")
        for item in verdict.corrupt:
            print(f"    corrupt  {item}")

    ready, blocked, unchecked = runs_unblocked(verdicts)
    print()
    print(f"runs unblocked: {', '.join(ready) or '(none)'}")
    if blocked:
        print(f"runs blocked:   {', '.join(blocked)}")
    if unchecked:
        # Normal with --only. Distinct from blocked: nothing is known to be
        # wrong, these runs simply were not part of this check.
        print(f"not checked:    {', '.join(unchecked)}")

    if args.command == "plan":
        return 0
    return 0 if all(v.ok for v in verdicts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
