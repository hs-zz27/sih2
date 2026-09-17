"""What the demo needs on disk, what is missing, and the environment to use it.

The learned tools find their weights through SATQUERY_* environment variables.
docker-compose.yml sets them for the container; running natively on a laptop
nothing did, so every tool fell back to its stub. This is the one list of
those paths for a native run, kept in step with docker-compose.yml by
tests/test_demo_assets.py.

Usage:
    python scripts/demo_assets.py            # report: present / missing, sizes
    python scripts/demo_assets.py --env      # `export` lines for the launcher
    python scripts/demo_assets.py --json

Nothing is downloaded or modified.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Asset:
    env: str
    path: str          # relative to the repository root
    what: str
    source: str


ASSETS: list[Asset] = [
    Asset("SATQUERY_VQA_BASE", "models/qwen25_vl_3b", "Qwen2.5-VL-3B base model (~7 GB)",
          "python scripts/fetch_models.py --dest models --only qwen25_vl_3b"),
    Asset("SATQUERY_VQA_ADAPTER", "checkpoints/v2/track_b_vqa/adapter_final", "VQA adapter",
          "team checkpoints (Phase 5)"),
    Asset("SATQUERY_CAPTION", "checkpoints/v2/caption_pre", "caption model", "team checkpoints (Phase 5)"),
    Asset("SATQUERY_LANDCOVER", "checkpoints/v2/track_a", "land-cover head", "team checkpoints (Phase 5)"),
    Asset("SATQUERY_GROUNDING", "checkpoints/v2/grounding_pre", "grounding model", "team checkpoints (Phase 5)"),
    Asset("SATQUERY_CHANGE_MASK", "checkpoints/v2/change_mask", "change-mask model", "team checkpoints (Phase 5)"),
    # v1 on purpose: the v2 change captioner regressed on changed pairs.
    Asset("SATQUERY_CHANGE_CAPTION", "checkpoints/change_caption", "change captioner (v1)", "team checkpoints (v1)"),
    Asset("SATQUERY_CHANGE_VQA", "checkpoints/v2/change_vqa/best.pt", "change-VQA semantic head",
          "team checkpoints (Phase 5)"),
    Asset("SATQUERY_FUSION", "checkpoints/v2/optsar_fusion", "optical-SAR fusion model", "team checkpoints (Phase 5)"),
    # In the repository, so always present.
    Asset("SATQUERY_CALIBRATION", "configs/calibration.v2.json", "calibration for the Phase 5 heads", "repository"),
    Asset("SATQUERY_THRESHOLDS", "configs/thresholds.v2.yaml", "thresholds for the Phase 5 heads", "repository"),
]

# Set for every native demo run.
FIXED_ENV = {
    "SATQUERY_PROFILE": "cpu",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "SATQUERY_CORS_ORIGINS": "http://localhost:3000,http://127.0.0.1:3000",
}


def size_gb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1024**3
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024**3


def status(root: Path = ROOT) -> list[dict]:
    out = []
    for a in ASSETS:
        full = root / a.path
        present = full.exists()
        out.append({
            "env": a.env, "path": a.path, "what": a.what, "source": a.source,
            "present": present, "size_gb": round(size_gb(full), 2) if present else None,
        })
    return out


def env_dict(root: Path = ROOT) -> dict[str, str]:
    """Environment for a native demo run. Missing paths are set too, so each
    stub's reason reads "checkpoint not found: <path>" instead of "... is not
    set"."""
    env = dict(FIXED_ENV)
    env.update({a.env: str(root / a.path) for a in ASSETS})
    nli = root / "models" / "nli_deberta_mnli"
    if nli.exists():
        env["SATQUERY_NLI"] = str(nli)
    return env


def env_lines(root: Path = ROOT) -> list[str]:
    """The same environment as POSIX `export` lines."""
    return [f"export {k}={shlex.quote(v)}" for k, v in env_dict(root).items()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--env", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if args.env:
        print("\n".join(env_lines()))
        return 0
    rows = status()
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    missing = [r for r in rows if not r["present"]]
    print("Demo assets (paths relative to the repository root)")
    for r in rows:
        mark = "  ok  " if r["present"] else "MISSING"
        size = f"{r['size_gb']:.2f} GB" if r["present"] else ""
        print(f"[{mark}] {r['path']:45s} {size:>9s}  {r['what']}")
    if missing:
        print("\nMissing items make their tools answer as stubs. Where to get them:")
        for source in sorted({r["source"] for r in missing}):
            print(f"  - {source}: " + ", ".join(r["path"] for r in missing if r["source"] == source))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
