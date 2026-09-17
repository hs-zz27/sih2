"""Can every tool actually load its Phase 5 checkpoint through the real loader?

The Phase 5 cards name a checkpoint per tool. This is the check that the
named checkpoint is loadable by the code that will load it in production -
`satquery/tools/<tool>.py`, not the trainer - with the sidecars it needs, the
architecture it recorded, and the weights it wrote.

Why this is a separate step from training finishing. A trainer writing
`metrics.json` proves the weights exist and were scored. It does not prove
the tool's `is_available()` accepts the directory (it parses sidecars, and a
missing `vocab.json` or `band_stats.json` fails there), that `_Handle`
rebuilds the right architecture from `extra`, or that the state dict loads
without a key mismatch. The 2026-08-31 recovery found a whole model reported
as "recovered" by a check that hashed files without opening them; this opens
them.

Read-only: sets environment variables for the current process only, loads,
and reports. Nothing is written.

Usage, on the machine that holds checkpoints/v2:
    python scripts/verify_v2_deploy.py
    python scripts/verify_v2_deploy.py --root /scratch/home/<cluster-user>/satquery
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# tool module -> (env var, checkpoint path relative to root, loader attribute)
# The checkpoint per tool is the one its Phase 5 card recommends.
DEPLOY = {
    "landcover":      ("SATQUERY_LANDCOVER",      "checkpoints/v2/track_a",         "_Handle"),
    "change_mask":    ("SATQUERY_CHANGE_MASK",    "checkpoints/v2/change_mask",     "_Handle"),
    "grounding":      ("SATQUERY_GROUNDING",      "checkpoints/v2/grounding_pre",   "_Handle"),
    "caption":        ("SATQUERY_CAPTION",        "checkpoints/v2/caption_pre",     "_Handle"),
    # v1, deliberately: the v2 change captioner regressed on the changed half
    # (0.1641 vs 0.3063) once the split was reinstated. See its Phase 5 card.
    "change_caption": ("SATQUERY_CHANGE_CAPTION", "checkpoints/change_caption",     "_Handle"),
    "optsar_fusion":  ("SATQUERY_FUSION",         "checkpoints/v2/optsar_fusion",   "_Handle"),
    "change_vqa":     ("SATQUERY_CHANGE_VQA",     "checkpoints/v2/change_vqa/best.pt", "_SemanticHandle"),
}

# rs_vqa needs two variables and a different constructor; handled separately.
RS_VQA = {
    "base":    ("SATQUERY_VQA_BASE",    "models/qwen25_vl_3b"),
    "adapter": ("SATQUERY_VQA_ADAPTER", "checkpoints/v2/track_b_vqa/adapter_final"),
}


def check(tool: str, env: str, rel: str, loader: str, root: Path) -> tuple[bool, str]:
    path = root / rel
    if not path.exists():
        return False, f"{rel} does not exist"
    os.environ[env] = str(path)
    module = importlib.import_module(f"satquery.tools.{tool}")

    if hasattr(module, "is_available"):
        ok, reason = module.is_available()
        if not ok:
            return False, f"is_available(): {reason}"

    started = time.time()
    handle = getattr(module, loader)(path)
    model = getattr(handle, "model", None)
    n = sum(p.numel() for p in model.parameters()) / 1e6 if model is not None else 0.0
    arch = type(model).__name__ if model is not None else "?"
    return True, f"{arch}, {n:.1f}M params, {time.time() - started:.1f}s"


def check_rs_vqa(root: Path) -> tuple[bool, str]:
    for env, rel in RS_VQA.values():
        if not (root / rel).exists():
            return False, f"{rel} does not exist"
        os.environ[env] = str(root / rel)
    module = importlib.import_module("satquery.tools.rs_vqa")
    ok, reason = module.is_available()
    if not ok:
        return False, f"is_available(): {reason}"
    started = time.time()
    handle = module._ModelHandle.get(
        Path(os.environ["SATQUERY_VQA_BASE"]), Path(os.environ["SATQUERY_VQA_ADAPTER"])
    )
    return True, f"{type(handle.model).__name__}, adapter loaded, {time.time() - started:.1f}s"


def main() -> int:
    p = argparse.ArgumentParser(description="Load every Phase 5 checkpoint through its tool.")
    p.add_argument("--root", type=Path, default=Path("."),
                   help="repo root holding checkpoints/v2 and models/")
    p.add_argument("--skip-vqa", action="store_true",
                   help="skip rs_vqa (loads a 3B model; slow without a GPU)")
    args = p.parse_args()
    root = args.root.resolve()
    os.chdir(root)

    failures = 0
    print(f"{'tool':<16} {'result':<6} detail")
    print("-" * 78)
    for tool, (env, rel, loader) in DEPLOY.items():
        try:
            ok, detail = check(tool, env, rel, loader, root)
        except Exception as exc:  # the point is to SEE the failure
            ok, detail = False, f"{type(exc).__name__}: {exc}"
            traceback.print_exc(limit=3, file=sys.stderr)
        failures += not ok
        print(f"{tool:<16} {'OK' if ok else 'FAIL':<6} {detail}")

    if not args.skip_vqa:
        try:
            ok, detail = check_rs_vqa(root)
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
            traceback.print_exc(limit=3, file=sys.stderr)
        failures += not ok
        print(f"{'rs_vqa':<16} {'OK' if ok else 'FAIL':<6} {detail}")

    total = len(DEPLOY) + (0 if args.skip_vqa else 1)
    print(f"\n{total - failures}/{total} tools load their Phase 5 checkpoint")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
