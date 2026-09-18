"""Check a Kaggle session can run a retrain, before any hours are spent.

Run after the driver's pip install, in a fresh process, so it sees the
versions that were actually installed. Prints one JSON line prefixed with
`ENVCHECK ` (the driver copies it into the manifest) and exits non-zero on a
problem that would otherwise surface hours into training:

* no CUDA device
* a training import that fails (torch, torchvision, transformers, peft,
  bitsandbytes, accelerate, pyarrow, huggingface_hub)
* too little free disk under the work directory
* HuggingFace unreachable (internet switched off in the notebook settings)
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

# GB of free disk each model needs under --work (downloads + extraction +
# working checkpoints). The VQA base model alone is ~7.5 GB.
# eval_vqa re-uses the base model the vqa run already downloaded, so its
# real need is the official split plus working room - but it is given the same
# headroom as vqa because it may also be run in a fresh session.
DISK_NEEDED_GB = {"vqa": 30, "change_mask": 8, "caption": 6, "eval_vqa": 20}

IMPORTS = {
    "vqa": ["torch", "torchvision", "transformers", "peft", "bitsandbytes",
            "accelerate", "pyarrow", "huggingface_hub", "PIL"],
    "change_mask": ["torch", "torchvision", "pyarrow", "huggingface_hub", "PIL", "numpy"],
    "caption": ["torch", "torchvision", "pyarrow", "huggingface_hub", "PIL", "numpy"],
    # Scoring imports evaluation.track_b_eval, which reaches satquery.tools -
    # and that package's __init__ pulls in the index engine, so rasterio,
    # scikit-image, scipy, scikit-learn, pydantic and yaml are all on the path
    # to a single SYSTEM_PROMPT constant. Checked here so a missing one fails
    # in minute one rather than after the model has loaded.
    "eval_vqa": ["torch", "torchvision", "transformers", "peft", "bitsandbytes",
                 "huggingface_hub", "PIL", "numpy", "rasterio", "skimage",
                 "scipy", "sklearn", "pydantic", "yaml"],
}


def check(model: str, work: Path) -> tuple[dict, list[str]]:
    report: dict = {"python": sys.version.split()[0], "versions": {}}
    problems: list[str] = []

    for name in IMPORTS[model]:
        try:
            module = importlib.import_module(name)
            report["versions"][name] = getattr(module, "__version__", "?")
        except Exception as exc:  # noqa: BLE001 - any import failure is the finding
            problems.append(f"import {name} failed: {type(exc).__name__}: {exc}")

    try:
        import torch

        report["cuda"] = bool(torch.cuda.is_available())
        if report["cuda"]:
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_count"] = torch.cuda.device_count()
            report["bf16"] = bool(torch.cuda.is_bf16_supported())
        else:
            problems.append("no CUDA device: set Accelerator to GPU T4 x2 in the notebook settings")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"torch unusable: {exc}")

    work.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(work).free / 1024**3
    report["free_disk_gb"] = round(free_gb, 1)
    if free_gb < DISK_NEEDED_GB[model]:
        problems.append(f"only {free_gb:.1f} GB free under {work}; {model} needs ~{DISK_NEEDED_GB[model]} GB")

    try:
        urllib.request.urlopen("https://huggingface.co", timeout=15)
        report["internet"] = True
    except Exception as exc:  # noqa: BLE001
        report["internet"] = False
        problems.append(f"HuggingFace unreachable ({exc}): switch Internet On in the notebook settings")

    return report, problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, choices=sorted(IMPORTS))
    p.add_argument("--work", type=Path, required=True)
    args = p.parse_args()

    report, problems = check(args.model, args.work)
    report["problems"] = problems
    print("ENVCHECK " + json.dumps(report))
    for problem in problems:
        print(f"!! {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
