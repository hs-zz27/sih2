"""Put retrained weights where the demo reads them.

From a finished Kaggle notebook's output (Output tab -> Download):

    python scripts/install_retrained.py ~/Downloads/output.zip
    python scripts/install_retrained.py path/to/extracted/folder

From a finished HF Job (training/hf_jobs/launch.py pushed its result to a
private Hub repo when it completed):

    python scripts/install_retrained.py --hf-repo <name>/satquery-retrain-vqa

Either way it copies every `checkpoints/...` folder into `./checkpoints/`,
keeps the manifests in `checkpoints/retrain_manifests/`, prints each run's
status, metrics and deviations, and then shows what is still missing.
Existing checkpoint folders are not overwritten unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from demo_assets import ASSETS, status  # noqa: E402

DEPLOY_DIRS = [a.path.removeprefix("checkpoints/") for a in ASSETS if a.path.startswith("checkpoints/")]


def find_retrained_root(folder: Path) -> Path | None:
    """The real-run output folder. Smoke-pass output is never installed: its
    weights come from a few steps on a few dozen examples."""
    for manifest in folder.rglob("retrain_manifest_*.json"):
        if json.loads(manifest.read_text(encoding="utf-8")).get("smoke"):
            continue
        return manifest.parent
    return None


def install(source: Path, dest_root: Path, force: bool = False) -> list[str]:
    """Copy packaged weights from an extracted output folder. Returns messages."""
    messages: list[str] = []
    retrained = find_retrained_root(source)
    if retrained is None:
        return [f"no real-run retrain_manifest_*.json under {source} (smoke output is not "
                "installed) - download the retrained/ folder, not retrained_smoke/"]

    for rel in DEPLOY_DIRS:
        src = retrained / "checkpoints" / rel
        if not src.exists():
            continue
        dst = dest_root / "checkpoints" / rel
        if dst.exists() and not force:
            messages.append(f"skipped {rel}: already present (use --force to replace)")
            continue
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
        messages.append(f"installed {rel}")

    keep = dest_root / "checkpoints" / "retrain_manifests"
    keep.mkdir(parents=True, exist_ok=True)
    for manifest in retrained.glob("retrain_manifest_*.json"):
        d = json.loads(manifest.read_text(encoding="utf-8"))
        if not str(d.get("status", "")).startswith(("complete", "stopped by budget")):
            messages.append(f"{d.get('model')}: NOT a finished run ({d.get('status')}) - "
                            "its weights were not packaged; see its log")
            continue
        shutil.copy2(manifest, keep / manifest.name)
        messages.append(f"{d.get('model')}: {d.get('status')} ({d.get('elapsed_hours')} h on {d.get('gpu')})")
        for key in ("metrics.json", "metrics_after_budget.json"):
            if key in d:
                messages.append(f"    {key}: {json.dumps(d[key])[:300]}")
        for deviation in d.get("deviations", []):
            messages.append(f"    deviation: {deviation}")
    return messages


def install_from_hub(repo_id: str, dest_root: Path, force: bool = False) -> list[str]:
    """Download an HF Job's pushed result and install it, same as a local folder."""
    from huggingface_hub import snapshot_download

    with tempfile.TemporaryDirectory() as tmp:
        local = snapshot_download(repo_id=repo_id, repo_type="model", local_dir=tmp)
        return install(Path(local), dest_root, force)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source", type=Path, nargs="?",
                   help="Kaggle output .zip or an extracted folder")
    p.add_argument("--hf-repo", help="Hub repo id an HF Job pushed its result to")
    p.add_argument("--force", action="store_true", help="replace existing checkpoint folders")
    args = p.parse_args()

    if bool(args.source) == bool(args.hf_repo):
        p.error("pass exactly one of a source path/zip, or --hf-repo")

    if args.hf_repo:
        messages = install_from_hub(args.hf_repo, ROOT, args.force)
    elif args.source.suffix == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(args.source) as zf:
                zf.extractall(tmp)
            messages = install(Path(tmp), ROOT, args.force)
    else:
        messages = install(args.source, ROOT, args.force)
    print("\n".join(messages))

    missing = [r["path"] for r in status() if not r["present"]]
    print("\nStill missing:" if missing else "\nAll demo assets present.")
    for path in missing:
        print("  -", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
