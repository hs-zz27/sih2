"""Go / no-go before recording or presenting the demo.

Run it against the API you are about to demo, from the machine that will be on
screen. It answers the questions that otherwise get answered on camera:

* Is the API up, and is the web UI up?
* Is every learned tool serving a trained model, or is anything a stub whose
  answers will read "[STUB - no model loaded]"? Each stub names its reason.
* Is there a CUDA GPU behind the API?
* Does this machine have the disk and memory the demo needs?
* Has the demo bundle been built?

Standard library only, and read-only: it makes GET requests and reads the
filesystem. `--verify-bundle` additionally runs every demo beat through the
controller (`scripts/make_demo_bundle.py --verify`), which is the heavier step.

Usage:
    python scripts/demo_preflight.py
    python scripts/demo_preflight.py --api http://localhost:8000 --web http://localhost:3000
    python scripts/demo_preflight.py --allow-stubs      # layout rehearsal only

Exit code 0 means GO, 1 means NO-GO.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# From docs: checkpoints 8.56 GB + base models 7.36 GB on disk, and the
# largest single tool peaks around 4.2 GB of VRAM (router.TOOL_VRAM_MB).
MIN_FREE_DISK_GB = 20.0
MIN_RAM_GB = 16.0


def fetch_json(url: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - local URLs only
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def reachable(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return resp.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def total_ram_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        return None


def evaluate(
    *,
    health: dict | None,
    readiness: dict | None,
    web_up: bool | None,
    free_disk_gb: float,
    ram_gb: float | None,
    bundle_built: bool,
    allow_stubs: bool = False,
) -> tuple[bool, list[tuple[str, str]]]:
    """(go, [(level, message)]) where level is OK, WARN or FAIL."""
    out: list[tuple[str, str]] = []

    if health is None:
        out.append(("FAIL", "API not reachable - start it: docker compose up -d (or make dev)"))
    else:
        out.append(("OK", f"API up (version {health.get('version', '?')})"))

    if web_up is None:
        pass
    elif web_up:
        out.append(("OK", "web UI up"))
    else:
        out.append(("FAIL", "web UI not reachable - the demo is recorded in the browser"))

    if readiness is None:
        if health is not None:
            out.append(("FAIL", "API has no /readiness - it is older than this script; rebuild the image"))
    else:
        live, total = readiness["learned_live"], readiness["learned_total"]
        if readiness["demo_ready"]:
            out.append(("OK", f"models: {live}/{total} learned tools live, no stubs"))
        else:
            level = "WARN" if allow_stubs else "FAIL"
            out.append((level, f"models: {live}/{total} live - these answer with placeholder text:"))
            for t in readiness["tools"]:
                if t["status"] == "stub":
                    out.append((level, f"    {t['tool']}: {t['reason']}"))
        gpu = readiness.get("gpu") or {}
        if gpu.get("cuda"):
            out.append(("OK", f"GPU: {gpu.get('name')}"))
        else:
            out.append(("WARN", f"no CUDA GPU behind the API ({gpu.get('note', 'unknown')}) - rs_vqa_v1 needs one"))

    if free_disk_gb < MIN_FREE_DISK_GB:
        out.append(("WARN", f"free disk {free_disk_gb:.1f} GB (< {MIN_FREE_DISK_GB:.0f} GB for checkpoints + base models)"))
    else:
        out.append(("OK", f"free disk {free_disk_gb:.1f} GB"))

    if ram_gb is None:
        out.append(("WARN", "could not read total RAM"))
    elif ram_gb < MIN_RAM_GB - 0.5:
        out.append(("WARN", f"RAM {ram_gb:.0f} GB (< {MIN_RAM_GB:.0f} GB recommended)"))
    else:
        out.append(("OK", f"RAM {ram_gb:.0f} GB"))

    if bundle_built:
        out.append(("OK", "demo bundle built (data/demo_bundle/manifest.json)"))
    else:
        out.append(("WARN", "demo bundle not built - python scripts/make_demo_bundle.py --out data/demo_bundle --verify"))

    go = not any(level == "FAIL" for level, _ in out)
    return go, out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--api", default=os.getenv("NEXT_PUBLIC_API_URL", "http://localhost:8000"))
    p.add_argument("--web", default="http://localhost:3000", help="'' to skip the UI check")
    p.add_argument("--allow-stubs", action="store_true",
                   help="downgrade stubs to warnings (rehearsing layout, not recording)")
    p.add_argument("--verify-bundle", action="store_true",
                   help="also run every demo beat through the controller (heavier)")
    args = p.parse_args()

    api = args.api.rstrip("/")
    health = fetch_json(f"{api}/health")
    readiness = fetch_json(f"{api}/readiness") if health else None
    web_up = reachable(args.web) if args.web else None
    bundle = ROOT / "data" / "demo_bundle" / "manifest.json"

    go, lines = evaluate(
        health=health,
        readiness=readiness,
        web_up=web_up,
        free_disk_gb=shutil.disk_usage(ROOT).free / 1024**3,
        ram_gb=total_ram_gb(),
        bundle_built=bundle.exists(),
        allow_stubs=args.allow_stubs,
    )

    mark = {"OK": "  ok  ", "WARN": " warn ", "FAIL": " FAIL "}
    print(f"SatQuery demo pre-flight  (api {api})")
    for level, message in lines:
        print(f"[{mark[level]}] {message}")

    if args.verify_bundle:
        print("\nrunning every demo beat through the controller ...", flush=True)
        rc = subprocess.call([sys.executable, str(ROOT / "scripts" / "make_demo_bundle.py"),
                              "--out", str(ROOT / "data" / "demo_bundle"), "--verify"])
        if rc != 0:
            print("[ FAIL ] demo bundle: a beat did not behave as scripted")
            go = False

    print("\nGO - ready to record." if go else "\nNO-GO - fix the FAIL lines first.")
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
