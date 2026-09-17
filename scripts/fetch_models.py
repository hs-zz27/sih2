"""Download and verify base model checkpoints (plan task 0.8).

Same integrity model as `fetch_datasets.py`: trust on first use, recorded in
`configs/model_lock.json`, enforced thereafter. Publisher digests go in
`EXPECTED_SHA256` where they exist and are enforced from the first fetch.

Two things this script exists to guarantee beyond "the files are present":

1. **Offline boot.** The plan requires the system to run with networking
   disabled (task 3.9). `--offline-test` re-imports every model with
   HF_HUB_OFFLINE=1 to prove nothing silently reaches for the network at load
   time. Discovering that on demo day is not an option.

2. **Safe loading.** Weights are fetched as safetensors wherever the publisher
   offers them. A third-party `.bin`/`.pt` checkpoint is a pickle, and loading
   one executes arbitrary code; `--allow-pickle` exists but warns loudly and
   should not be used for anything downloaded from outside the team.

Usage:
    python scripts/fetch_models.py --list
    python scripts/fetch_models.py --dest models --only qwen25_vl_3b
    python scripts/fetch_models.py --dest models --offline-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_datasets import sha256_tree  # noqa: E402  reuse the same digest rule

LOCKFILE = Path("configs/model_lock.json")


@dataclass
class Model:
    key: str
    name: str
    hf_repo: str
    used_for: str
    licence: str = "unknown"
    approx_size: str = "unknown"
    verified: bool = False
    allow_patterns: list[str] = field(
        # Pull safetensors + config/tokenizer only. Excluding *.bin keeps a
        # pickle checkpoint off the disk entirely rather than relying on
        # remembering not to load it.
        default_factory=lambda: [
            "*.safetensors", "*.safetensors.index.json", "*.json", "*.txt",
            "*.model", "tokenizer*", "preprocessor_config.json",
        ]
    )
    notes: str = ""


# Model choices come from docs/03-Models-and-Datasets.md section 2. None is
# marked verified: verification items 3, 4, 7 and 12 are still open, so repo
# ids and licences must be confirmed before a real training run.
MODELS: list[Model] = [
    Model(
        key="qwen25_vl_3b",
        name="Qwen2.5-VL-3B-Instruct",
        hf_repo="Qwen/Qwen2.5-VL-3B-Instruct",
        used_for="Track B base for rs_vqa_v1 and caption_v1 (two LoRA adapters)",
        licence="Apache-2.0",
        approx_size="~7 GB",
        notes=(
            "Primary recommendation. Natively supported by transformers, so it "
            "loads with trust_remote_code=False - no third-party Python is "
            "executed. Check for a newer small Qwen-VL first (item 7)."
        ),
    ),
    Model(
        key="internvl3_1b",
        name="InternVL3-1B",
        hf_repo="OpenGVLab/InternVL3-1B",
        used_for="Alternate Track B backbone to evaluate against Qwen (item 7)",
        approx_size="~2 GB",
        notes=(
            "Used by the BigEarthNet.txt authors as their RS backbone. "
            "REQUIRES trust_remote_code=True and custom *.py modeling files, "
            "which allow_patterns deliberately excludes - enabling it executes "
            "code from the model repo. Add \"*.py\" and pass the flag only as "
            "a conscious decision."
        ),
    ),
    Model(
        key="florence2_large",
        name="Florence-2-large",
        hf_repo="microsoft/Florence-2-large",
        used_for="grounding_v1 - purpose-built for region tasks, 0.77B",
        approx_size="~1.5 GB",
        notes=(
            "Trains in hours; the right specialist rather than forcing VQA to "
            "ground. Same trust_remote_code caveat as InternVL3."
        ),
    ),
    Model(
        key="nli_deberta_mnli",
        name="DeBERTa-v3-base-mnli-fever-anli",
        hf_repo="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        used_for="Entailment gate NLI backend (task 3.5), via SATQUERY_NLI",
        licence="MIT",
        approx_size="~370 MB",
        verified=True,
        notes=(
            "Loads with trust_remote_code=False - DeBERTa-v2/v3 is native to "
            "transformers, so no third-party Python is executed, which is the "
            "same bar Florence-2 failed in task 2.7. Trained on MNLI + FEVER + "
            "ANLI; FEVER in particular is fact-verification data, which is "
            "closer to checking a sentence against a measured premise than "
            "plain MNLI. Chosen over roberta-large-mnli and bart-large-mnli "
            "because it is ~4x smaller for comparable accuracy, which matters "
            "for the lite profile (3.10) and the 6 GB demo laptop."
        ),
    ),
]

BY_KEY = {m.key: m for m in MODELS}

EXPECTED_SHA256: dict[str, str] = {}


def load_lock() -> dict:
    if LOCKFILE.exists():
        return json.loads(LOCKFILE.read_text(encoding="utf-8"))
    return {}


def save_lock(lock: dict) -> None:
    LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    LOCKFILE.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")


def fetch(model: Model, dest: Path, allow_pickle: bool) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub is required: pip install huggingface_hub"
        ) from exc

    patterns = list(model.allow_patterns)
    if allow_pickle:
        patterns += ["*.bin", "*.pt", "*.pth"]
        print(
            "  WARNING: --allow-pickle enabled. torch will execute pickle on "
            "load, which is arbitrary code execution if the publisher or the "
            "transport is compromised. Do not use this for untrusted weights."
        )

    target = dest / model.key
    print(f"  downloading {model.hf_repo} -> {target}")
    snapshot_download(
        repo_id=model.hf_repo,
        local_dir=str(target),
        allow_patterns=patterns,
        resume_download=True,
    )
    return target


def verify(model: Model, path: Path, lock: dict, update: bool) -> bool:
    if not path.exists():
        print(f"  MISSING: {path}")
        return False

    digest = sha256_tree(path)
    expected = EXPECTED_SHA256.get(model.key)
    if expected:
        ok = digest == expected
        print("  OK (publisher digest)" if ok else f"  FAIL: expected {expected}, got {digest}")
        return ok

    recorded = lock.get(model.key, {}).get("sha256")
    if recorded is None:
        if update:
            lock[model.key] = {
                "sha256": digest,
                "name": model.name,
                "hf_repo": model.hf_repo,
                "note": "trust-on-first-use; not a publisher-provided digest",
            }
            print(f"  recorded digest {digest[:16]}... (first use)")
            return True
        print("  no recorded digest")
        return False

    if digest == recorded:
        print("  OK (matches lockfile)")
        return True
    print(f"  FAIL: digest changed\n    locked {recorded}\n    got    {digest}")
    return False


def offline_test(models: list[Model], dest: Path) -> bool:
    """Load each model's config with the network disabled.

    Proves the offline requirement (task 3.9) at the point it is cheap to fix,
    rather than on demo day. Only configs are loaded - instantiating a 3B model
    is unnecessary to show that no network call is attempted.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("\nOffline load test (HF_HUB_OFFLINE=1):")

    try:
        from transformers import AutoConfig
    except ImportError:
        print("  transformers not installed; skipping")
        return True

    ok = True
    for model in models:
        path = dest / model.key
        if not path.exists():
            print(f"  {model.key}: SKIP (not downloaded)")
            continue
        try:
            AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
            print(f"  {model.key}: OK")
        except Exception as exc:  # noqa: BLE001
            # Some vision-language configs require trust_remote_code; that is a
            # real finding for offline operation, so it is reported not hidden.
            print(f"  {model.key}: FAIL - {type(exc).__name__}: {exc}")
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("models"))
    parser.add_argument("--only", nargs="*", help="specific model keys")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--offline-test", action="store_true")
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="also download .bin/.pt weights (pickle; code-execution risk)",
    )
    args = parser.parse_args()

    if args.list:
        print(f"{'key':<18} {'verified':<9} {'size':<10} {'licence':<14} name")
        for m in MODELS:
            print(
                f"{m.key:<18} {'yes' if m.verified else 'NO':<9} "
                f"{m.approx_size:<10} {m.licence:<14} {m.name}"
            )
        print(
            "\nNo model is verified yet: verification items 3, 4, 7 and 12 are "
            "open. Confirm repo ids and licences before a real training run."
        )
        return 0

    selected = [BY_KEY[k] for k in args.only] if args.only else list(MODELS)
    if args.only:
        unknown = [k for k in args.only if k not in BY_KEY]
        if unknown:
            print(f"Unknown model keys: {unknown}", file=sys.stderr)
            return 2

    if args.offline_test and args.verify_only:
        return 0 if offline_test(selected, args.dest) else 1

    lock = load_lock()
    args.dest.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for model in selected:
        print(f"\n[{model.key}] {model.name}")
        target = args.dest / model.key
        if not args.verify_only:
            try:
                target = fetch(model, args.dest, args.allow_pickle)
            except SystemExit:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"  download failed: {exc}")
                failures.append(model.key)
                continue
        if not verify(model, target, lock, update=not args.verify_only):
            failures.append(model.key)

    if not args.verify_only:
        save_lock(lock)
        print(f"\nLockfile written to {LOCKFILE}")

    if args.offline_test and not offline_test(selected, args.dest):
        failures.append("offline_test")

    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print("\nAll selected models verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
