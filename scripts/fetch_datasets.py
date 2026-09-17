"""Download and mirror the P0 datasets (plan task 0.7).

Free-tier notebook filesystems are ephemeral, so everything is fetched once
into a shared location and verified on every later use. Re-downloading 60 GB in
week 9 is a wasted day.

INTEGRITY MODEL - read this before trusting the checksums.

Published SHA-256 digests do not exist for most of these datasets, and
inventing them would be worse than useless: a fabricated digest either fails
forever or, if it is never checked, provides false assurance. This script
therefore uses **trust on first use**:

* The first successful download records the digest in `configs/dataset_lock.json`.
* Every later download of that entry is verified against the recorded digest.
* A mismatch is a hard failure - the file changed upstream, or is corrupt.
* `--verify-only` re-checks what is already on disk without downloading.

That gives reproducibility across the team and across machines from the moment
the first person runs it, which is the property that actually matters here. It
does **not** protect against a compromised first download; where a publisher
does provide a digest, put it in `EXPECTED_SHA256` below and it will be
enforced from the very first fetch instead.

Usage:
    python scripts/fetch_datasets.py --list
    python scripts/fetch_datasets.py --dest /mnt/data --priority P0
    python scripts/fetch_datasets.py --dest /mnt/data --only vrsbench
    python scripts/fetch_datasets.py --dest /mnt/data --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

LOCKFILE = Path("configs/dataset_lock.json")
CHUNK = 1024 * 1024


@dataclass
class Dataset:
    """One dataset entry.

    `hf_repo` is a HuggingFace dataset id; `urls` are direct downloads. Exactly
    one of the two is used. `verified` records whether the identifier itself
    has been confirmed against the source, per docs/verification.md - entries
    marked False are best-effort and must be checked before a real run.
    """

    key: str
    name: str
    priority: str
    used_for: str
    hf_repo: str | None = None
    hf_config: str | None = None
    urls: list[str] = field(default_factory=list)
    approx_size: str = "unknown"
    licence: str = "unknown"
    verified: bool = False
    notes: str = ""


# Identifiers come from docs/03-Models-and-Datasets.md section 4.2. Only
# BigEarthNet.txt is marked verified, because it is the one entry whose paper
# and HF card were actually read (docs/verification.md items 1 and 2).
DATASETS: list[Dataset] = [
    Dataset(
        key="bigearthnet_txt",
        name="BigEarthNet.txt",
        priority="P0",
        used_for="Track A + Track B backbone; the adaptation mandate",
        hf_repo="BIFOLD-BigEarthNetTextual/BigEarthNet-Textual",
        approx_size="467 MB Parquet (text only)",
        licence="CDLA-Permissive-1.0",
        verified=True,
        notes=(
            "Text only: 9,553,962 rows, one per annotation. The S1/S2 imagery "
            "is a separate reBEN download - see bigearthnet_v2."
        ),
    ),
    Dataset(
        key="bigearthnet_v2",
        name="BigEarthNet v2 (reBEN imagery)",
        priority="P0",
        used_for="Track A land-cover head; imagery for BigEarthNet.txt",
        urls=["https://bigearth.net/"],
        approx_size="~66 GB (S2)",
        licence="CDLA-Permissive-1.0",
        notes="Large. Subset before mirroring; see docs/03 section 4.2.",
    ),
    Dataset(
        key="vrsbench",
        name="VRSBench",
        priority="P0",
        used_for="Track B; prescribed eval split",
        hf_repo="xiang709/VRSBench",
        approx_size="moderate",
        notes="Prescribed benchmark - use the official test split, not a resplit.",
    ),
    Dataset(
        key="rsvqa_lr",
        name="RSVQA-LR",
        priority="P0",
        used_for="Track B; prescribed eval split",
        urls=["https://rsvqa.sylvainlobry.com/"],
        approx_size="moderate",
    ),
    Dataset(
        key="rsvqa_hr",
        name="RSVQA-HR",
        priority="P0",
        used_for="Track B; prescribed eval split",
        urls=["https://rsvqa.sylvainlobry.com/"],
        approx_size="moderate",
    ),
    Dataset(
        key="cdvqa",
        name="CDVQA",
        priority="P0",
        used_for="change_vqa_v1; prescribed eval split",
        urls=[
            "https://github.com/YZHJessica/CDVQA",
            "https://huggingface.co/datasets/ljx620/CDVQA",
        ],
        hf_repo="ljx620/CDVQA",
        verified=True,
        approx_size="small annotations; ~32 GB imagery mirror",
        notes=(
            "Annotations only from the GitHub repo (Apache-2.0) - "
            "curl the {Train,Val,Test,Test2}_{questions,answers,images}.json "
            "files directly. Imagery lives in the SECOND dataset; the "
            "ljx620/CDVQA webdataset mirror carries it keyed by the official "
            "question_id, one copy of the pair per question (~32 GB for 968 "
            "test pairs). training/prepare/cdvqa.py verifies the mirror "
            "against the official annotations and works from a partial "
            "download."
        ),
    ),
    Dataset(
        key="levir_cc",
        name="LEVIR-CC",
        priority="P0",
        used_for="change_caption_v1",
        urls=["https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset"],
        approx_size="moderate",
    ),
    Dataset(
        key="levir_cd",
        name="LEVIR-CD",
        priority="P0",
        used_for="change_mask_v1",
        urls=["https://chenhao.in/LEVIR/"],
        approx_size="moderate",
    ),
    Dataset(
        key="dior_rsvg",
        name="DIOR-RSVG",
        priority="P0",
        used_for="grounding_v1",
        urls=["https://github.com/ZhanYang-nwpu/RSVG-pytorch"],
        approx_size="moderate",
    ),
    Dataset(
        key="whu_opt_sar",
        name="WHU-OPT-SAR",
        priority="P0",
        used_for="Stage A2 - the resolution bridge",
        urls=["https://github.com/AmberHen/WHU-OPT-SAR-dataset"],
        approx_size="~10 GB",
    ),
    Dataset(
        key="geochat_instruct",
        name="GeoChat-Instruct",
        priority="P1",
        used_for="Track B instruction diversity",
        hf_repo="MBZUAI/GeoChat_Instruct",
        approx_size="moderate",
    ),
]

BY_KEY = {d.key: d for d in DATASETS}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(root: Path) -> str:
    """Digest of a directory: hashes relative paths and contents, sorted.

    Sorting makes the result independent of filesystem iteration order, so the
    same tree hashes identically on Windows and Linux.
    """
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(path.relative_to(root).as_posix().encode("utf-8"))
        h.update(sha256_file(path).encode("ascii"))
    return h.hexdigest()


def load_lock() -> dict:
    if LOCKFILE.exists():
        return json.loads(LOCKFILE.read_text(encoding="utf-8"))
    return {}


def save_lock(lock: dict) -> None:
    LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    LOCKFILE.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")


# Publisher-provided digests go here and are enforced from the first fetch.
# Empty by design: none of these publishers documents one. Do not populate
# this with digests computed from your own download - that is what the
# lockfile is for, and conflating the two would misrepresent their provenance.
EXPECTED_SHA256: dict[str, str] = {}


def fetch_hf(dataset: Dataset, dest: Path) -> Path:
    """Download a HuggingFace dataset repo into `dest`."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "huggingface_hub is required for HF datasets: pip install huggingface_hub"
        ) from exc

    target = dest / dataset.key
    print(f"  downloading {dataset.hf_repo} -> {target}")
    snapshot_download(
        repo_id=dataset.hf_repo,
        repo_type="dataset",
        local_dir=str(target),
        resume_download=True,
    )
    return target


def verify(dataset: Dataset, path: Path, lock: dict, update: bool) -> bool:
    """Check the digest against EXPECTED_SHA256, then the lockfile."""
    if not path.exists():
        print(f"  MISSING: {path}")
        return False

    digest = sha256_tree(path) if path.is_dir() else sha256_file(path)

    expected = EXPECTED_SHA256.get(dataset.key)
    if expected:
        if digest == expected:
            print("  OK (publisher digest)")
            return True
        print(f"  FAIL: publisher digest mismatch\n    expected {expected}\n    got      {digest}")
        return False

    recorded = lock.get(dataset.key, {}).get("sha256")
    if recorded is None:
        if update:
            lock[dataset.key] = {
                "sha256": digest,
                "name": dataset.name,
                "note": "trust-on-first-use; not a publisher-provided digest",
            }
            print(f"  recorded digest {digest[:16]}... (first use)")
            return True
        print(f"  no recorded digest (run without --verify-only to record); got {digest[:16]}...")
        return False

    if digest == recorded:
        print("  OK (matches lockfile)")
        return True
    print(f"  FAIL: digest changed\n    locked {recorded}\n    got    {digest}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data"))
    parser.add_argument("--priority", choices=["P0", "P1", "P2", "all"], default="P0")
    parser.add_argument("--only", nargs="*", help="specific dataset keys")
    parser.add_argument("--list", action="store_true", help="list datasets and exit")
    parser.add_argument(
        "--verify-only", action="store_true", help="verify what is on disk, download nothing"
    )
    args = parser.parse_args()

    if args.list:
        print(f"{'key':<20} {'pri':<4} {'verified':<9} {'size':<28} name")
        for d in DATASETS:
            flag = "yes" if d.verified else "NO"
            print(f"{d.key:<20} {d.priority:<4} {flag:<9} {d.approx_size:<28} {d.name}")
        print(
            "\n'verified' means the identifier itself was confirmed against the "
            "source (docs/verification.md). Entries marked NO are best-effort "
            "and must be checked before a real run."
        )
        return 0

    if args.only:
        unknown = [k for k in args.only if k not in BY_KEY]
        if unknown:
            print(f"Unknown dataset keys: {unknown}", file=sys.stderr)
            print(f"Known: {sorted(BY_KEY)}", file=sys.stderr)
            return 2
        selected = [BY_KEY[k] for k in args.only]
    elif args.priority == "all":
        selected = list(DATASETS)
    else:
        selected = [d for d in DATASETS if d.priority == args.priority]

    lock = load_lock()
    args.dest.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    manual: list[Dataset] = []

    for dataset in selected:
        print(f"\n[{dataset.key}] {dataset.name} ({dataset.priority})")
        target = args.dest / dataset.key

        if not args.verify_only and dataset.hf_repo:
            try:
                target = fetch_hf(dataset, args.dest)
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001 - reported per dataset
                print(f"  download failed: {exc}")
                failures.append(dataset.key)
                continue
        elif not args.verify_only:
            # No programmatic download available: these publishers gate behind
            # a form, a licence click-through or a GitHub release page. Saying
            # so is more useful than a broken URL guess.
            manual.append(dataset)
            if not target.exists():
                continue

        if not verify(dataset, target, lock, update=not args.verify_only):
            failures.append(dataset.key)

    if not args.verify_only:
        save_lock(lock)
        print(f"\nLockfile written to {LOCKFILE}")

    if manual:
        print("\nManual download required (no public direct URL):")
        for d in manual:
            print(f"  {d.key:<20} {d.urls[0] if d.urls else '(see docs/03)'}")
            print(f"  {'':<20} extract into {args.dest / d.key}, then re-run to record its digest")

    if failures:
        print(f"\nFAILED or unverified: {failures}")
        return 1

    print("\nAll selected datasets verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
