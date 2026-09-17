"""What is this cluster, and what will it let us do?

Run this FIRST, in a notebook cell or a terminal on the cluster, before
transferring 66 GB or spending a GPU-hour. It answers the four questions that
change the plan and cannot be answered from anywhere else:

1. **Is there actually a GPU on this kernel?** JupyterHub commonly puts you on
   a CPU node unless a GPU profile was chosen at spawn, and the failure looks
   like "training is very slow" rather than like an error.
2. **How much disk, and where?** Home directories on shared clusters are
   routinely 10-50 GB. The full corpus is ~66 GB, so a quota check decides
   whether everything is staged at once or in the tiers this prints.
3. **Is there outbound network?** It decides whether the code arrives by
   `git clone` or by upload, and whether pip can install anything.
4. **What is already installed?** torch, h5py and peft are the three that
   stop a run dead.

Nothing here writes, downloads, or installs. It only looks.

Usage, on the cluster:
    python training/cluster/recon.py

or, from a notebook cell in a checkout:
    !python training/cluster/recon.py
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

# The staging tiers this prints. Ordered by GB-per-run-unblocked, which is the
# right order when disk is the binding constraint: the first 0.7 GB unblocks a
# real training run, while the last 45.6 GB unblocks one.
TIERS: list[tuple[str, list[str], float, str]] = [
    ("1", ["levircd"], 0.69, "change_mask - the cheapest real run"),
    ("2", ["dior_rsvg"], 2.01, "grounding - the largest expected gain"),
    ("3", ["rsicd"], 0.53, "caption"),
    ("4", ["levir_mci"], 5.67, "change_caption"),
    ("5", ["second", "cdvqa"], 2.54, "change_vqa (both arms)"),
    ("6", ["whu_opt_sar", "rsvqa_lr_2k", "instruct_mix"], 3.40,
     "optsar_fusion + track_b_vqa (also needs models/qwen25_vl_3b, ~7 GB)"),
    ("7", ["ben_full"], 45.56, "track_a + track_a_nodropout"),
]

PACKAGES = [
    ("torch", "training - nothing runs without it"),
    ("h5py", "BigEarthNet shards (track_a)"),
    ("numpy", "everything"),
    ("yaml", "the campaign config"),
    ("PIL", "every image-based trainer"),
    ("peft", "track_b_vqa (QLoRA)"),
    ("bitsandbytes", "track_b_vqa (4-bit)"),
    ("transformers", "track_b_vqa"),
    ("accelerate", "track_b_vqa"),
]


def rule(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 58 - len(title)))


def check_gpu() -> None:
    rule("GPU")
    try:
        import torch
    except ImportError:
        print("  torch NOT INSTALLED - cannot tell. See the packages section.")
        return

    print(f"  torch {torch.__version__}   cuda build {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("  !! NO GPU VISIBLE ON THIS KERNEL")
        print("     On JupyterHub this usually means the session was spawned on a")
        print("     CPU profile. Stop the server and pick a GPU profile, or ask")
        print("     which profile has one. Training here would be unusably slow")
        print("     rather than broken, which is why this is worth checking now.")
        return

    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        bf16 = (p.major, p.minor) >= (8, 0)
        print(f"  [{i}] {p.name}  {p.total_memory / 1e9:.1f} GB  "
              f"sm{p.major}{p.minor}  {'bf16' if bf16 else 'fp16 only (pre-Ampere)'}")


def check_disk() -> None:
    rule("DISK")
    home = Path.home()
    candidates = [("home", home)]
    for env in ("SCRATCH", "WORK", "TMPDIR"):
        if os.environ.get(env):
            candidates.append((env.lower(), Path(os.environ[env])))
    for name in ("/scratch", "/data", "/mnt/data", "/tmp"):
        path = Path(name)
        if path.is_dir():
            user_path = path / os.environ.get("USER", "")
            candidates.append((name, user_path if user_path.is_dir() else path))

    seen = set()
    for label, path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            usage = shutil.disk_usage(resolved)
        except OSError as exc:
            print(f"  {label:<10} {resolved}  (unreadable: {exc})")
            continue
        print(f"  {label:<10} {str(resolved):<34} "
              f"{usage.free / 1e9:>8.1f} GB free of {usage.total / 1e9:.1f}")

    # A filesystem with terabytes free can still refuse the 51st gigabyte.
    print("\n  Free space is not the same as your quota. If these exist, they")
    print("  are the authority:")
    for cmd in (["quota", "-s"], ["lfs", "quota", "-h", str(home)]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0 and out.stdout.strip():
            print(f"\n  $ {' '.join(cmd)}")
            for line in out.stdout.strip().splitlines()[:8]:
                print(f"    {line}")


def check_network() -> None:
    rule("NETWORK")
    reachable = 0
    for label, host, port in (("github.com", "github.com", 443),
                              ("pypi.org", "pypi.org", 443),
                              ("huggingface.co", "huggingface.co", 443)):
        try:
            with socket.create_connection((host, port), timeout=6):
                print(f"  {label:<16} reachable")
                reachable += 1
        except OSError as exc:
            print(f"  {label:<16} BLOCKED ({type(exc).__name__})")

    if reachable == 0:
        print("\n  Nothing reachable -> the code arrives by upload, not `git clone`,")
        print("  and pip cannot install anything. Everything must be staged.")
    elif reachable < 3:
        print("\n  Partial egress - likely a proxy or an allowlist. Whatever is")
        print("  blocked has to be staged by hand instead.")


def check_packages() -> None:
    rule("PACKAGES")
    missing = []
    for name, why in PACKAGES:
        try:
            __import__(name)
            print(f"  {name:<16} present")
        except ImportError:
            missing.append(name)
            print(f"  {name:<16} MISSING     ({why})")
    if missing:
        print(f"\n  pip install {' '.join(missing)}")
        print("  If pip is blocked, only the runs whose dependencies are present")
        print("  can go ahead - which is most of them; peft/bitsandbytes/")
        print("  transformers are needed by track_b_vqa alone.")


def check_scheduler() -> None:
    rule("SCHEDULER / SESSION")
    print(f"  host      {socket.gethostname()}")
    print(f"  python    {sys.version.split()[0]}  ({sys.executable})")
    slurm = {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}
    if slurm:
        print("  SLURM detected - this notebook is inside a job:")
        for key in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_CPUS_ON_NODE",
                    "SLURM_MEM_PER_NODE", "SLURM_JOB_GPUS", "SLURM_TIMELIMIT"):
            if key in slurm:
                print(f"    {key:<22} {slurm[key]}")
        print("  The job's time limit is the number to put in SESSION_MINUTES.")
    else:
        print("  No SLURM variables - a plain JupyterHub spawn. Find the session")
        print("  time limit from your admin or the Hub's control panel; it is")
        print("  what SESSION_MINUTES should be set to.")


def print_tiers() -> None:
    rule("STAGING TIERS (if disk is tight)")
    print("  You do NOT need all 66 GB to start. In this order, each tier")
    print("  unblocks a real training run:\n")
    cumulative = 0.0
    for tier, keys, gb, unblocks in TIERS:
        cumulative += gb
        print(f"  {tier}  {gb:>6.2f} GB  (total {cumulative:>6.2f})  "
              f"{', '.join(keys)}")
        print(f"     -> {unblocks}")
    print("\n  Tiers 1-6 are 14.8 GB and unblock 8 of the 10 runs. The 45.6 GB")
    print("  BigEarthNet shard set is needed by track_a alone - so if disk is")
    print("  the constraint, do track_a last and everything else first.")


SECTIONS = {
    "scheduler": check_scheduler,
    "gpu": check_gpu,
    "disk": check_disk,
    "network": check_network,
    "packages": check_packages,
    "tiers": print_tiers,
}


def main() -> int:
    # A parser even though every argument is optional, because
    # `tests/test_script_entrypoints.py` requires one of every runnable script
    # - a rule that exists because `training/run_checkpoint_test.py` had no
    # parser, so `--help` did not print help, it ran the program, and what it
    # ran was `shutil.rmtree("checkpoints")`.
    import argparse

    p = argparse.ArgumentParser(
        description="Report what this cluster is, before staging data or "
                    "spending a GPU-hour. Reads only; writes nothing."
    )
    p.add_argument("--only", nargs="*", choices=sorted(SECTIONS),
                   help="run just these sections (default: all)")
    p.add_argument("--no-network", action="store_true",
                   help="skip the egress probe, which opens outbound sockets")
    args = p.parse_args()

    chosen = args.only or list(SECTIONS)
    if args.no_network and "network" in chosen:
        chosen = [s for s in chosen if s != "network"]

    print("SatQuery AI - cluster reconnaissance")
    for name in SECTIONS:          # fixed order, not the order given
        if name in chosen:
            SECTIONS[name]()
    print("\nSend this whole output back before staging data or starting a run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
