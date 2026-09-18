"""Launch a SatQuery retrain as a Hugging Face Job: no browser, no session limit.

Reuses training/kaggle/retrain.py unchanged - it only ever assumed Kaggle's
/kaggle/tmp and /kaggle/working as *defaults*, both overridable, so the same
driver runs here with generic paths. What HF Jobs needs that Kaggle's UI gave
for free: a way to get the result back out of an ephemeral container. This
pushes the packaged output to a private Hub model repo at the end of the job,
which scripts/install_retrained.py --hf-repo then downloads.

Whoever's account authenticates the machine that RUNS this script is who
gets billed - the job is not "sent" anywhere else. See docs/hf-jobs-retrain.md
for how to use a friend's credits with a scoped token instead of full account
access.

Usage (from a machine logged in via `hf auth login`, or with HF_TOKEN set):
    python training/hf_jobs/launch.py --model vqa
    python training/hf_jobs/launch.py --model change_mask --flavor a10g-small
    python training/hf_jobs/launch.py --model vqa --dry-run   # print, don't submit

Then, once the job's status is COMPLETED:
    python scripts/install_retrained.py --hf-repo <printed repo id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
REPO_URL = "https://github.com/hs-zz27/sih2.git"

# Timeout = the most a job can bill, with headroom over the estimates in
# docs/kaggle-retrain.md. vqa includes the official scoring (~1-1.5 h) after
# training, in the same job. You pay for actual seconds, not the timeout.
DEFAULT_TIMEOUT = {"vqa": "14h", "change_mask": "9h", "caption": "4h"}
MODELS = sorted(DEFAULT_TIMEOUT)
# t4-small ($0.40/h) matches what the recipes were tuned for; every model
# here fits in its 16 GB. A friend's $11 covers ~27h on it.
DEFAULT_FLAVOR = "t4-small"


def build_command(model: str, smoke: bool, fp32_compute: bool, push_to: str) -> str:
    """One shell script: clone, retrain, (vqa: score), and ALWAYS push.

    The push runs whatever happened before it. A job that trained for eight
    hours and then failed its scoring step must still hand back the adapter,
    and a job that failed outright must still hand back its manifest and logs
    - they are the only way to see why from outside an ephemeral container.
    The script exits with the training step's code, so a failed training run
    still shows as ERROR on the Hub.
    """
    # retrain.py is stdlib-only at import time and installs its own training
    # packages as its first step (same PIP_PACKAGES as Kaggle) - nothing needs
    # installing here beyond git to clone the repo. The base image already
    # carries torch + torchvision matched to its CUDA build; retrain.py never
    # touches those.
    smoke_flag = "--smoke " if smoke else ""
    fp32_flag = "--fp32-compute " if fp32_compute else ""
    # Scoring reuses the base model training downloaded (same --work), and
    # the adapter training packaged (same --out). A fresh HF job would have
    # neither, which is why it is not a separate job.
    score = ""
    if model == "vqa":
        score = (
            "if [ \"$TRAIN\" = 0 ]; then\n"
            f"  python training/kaggle/retrain.py --model eval_vqa {smoke_flag}--skip-install"
            " --adapter /tmp/output/checkpoints/v2/track_b_vqa/adapter_final"
            " --work /tmp/satquery --out /tmp/output"
            " || echo '!! official scoring failed - the trained adapter is still pushed'\n"
            "fi\n"
        )
    return (
        "set -uo pipefail\n"
        "apt-get update -qq && apt-get install -y -qq git > /dev/null || exit 1\n"
        f"git clone --depth 1 {REPO_URL} /workspace/sih2 || exit 1\n"
        "cd /workspace/sih2\n"
        f"python training/kaggle/retrain.py --model {model} {smoke_flag}{fp32_flag}"
        " --work /tmp/satquery --out /tmp/output\n"
        "TRAIN=$?\n"
        + score +
        "mkdir -p /tmp/output\n"
        "python -c \""
        "from huggingface_hub import HfApi; "
        "HfApi().create_repo(repo_id='" + push_to + "', repo_type='model', "
        "private=True, exist_ok=True); "
        "HfApi().upload_folder(repo_id='" + push_to + "', repo_type='model', "
        "folder_path='/tmp/output')\"\n"
        f"echo 'PUSHED TO: {push_to}  (training exit code '$TRAIN')'\n"
        "exit $TRAIN"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, choices=MODELS,
                   help="vqa also scores itself on the official RSVQA-LR split")
    p.add_argument("--flavor", default=DEFAULT_FLAVOR,
                   help=f"HF Jobs hardware flavor (default {DEFAULT_FLAVOR}, $0.40/h). "
                        "'hf jobs hardware' lists all options and current prices.")
    p.add_argument("--timeout", help="e.g. '10h'; default varies by model, see DEFAULT_TIMEOUT")
    p.add_argument("--push-to", help="Hub repo id for the result (default: "
                                     "<your-username>/satquery-retrain-<model>)")
    p.add_argument("--smoke", action="store_true", help="tiny limits: prove the job works first")
    p.add_argument("--fp32-compute", action="store_true", help="vqa only: fp16 NaN fallback")
    p.add_argument("--dry-run", action="store_true", help="print the plan, submit nothing")
    args = p.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception as exc:  # noqa: BLE001
        if not args.dry_run:
            print(f"!! Not logged in to Hugging Face ({exc}).\n"
                  "   Run `hf auth login` and paste a token first - see "
                  "docs/hf-jobs-retrain.md for how to use a friend's token safely.")
            return 1
        who = "<not logged in>"

    push_to = args.push_to or f"{who}/satquery-retrain-{args.model}{'-smoke' if args.smoke else ''}"
    timeout = args.timeout or DEFAULT_TIMEOUT[args.model]
    command = build_command(args.model, args.smoke, args.fp32_compute, push_to)

    print(f"Account that will be billed : {who}")
    print(f"Flavor                      : {args.flavor}")
    print(f"Timeout                     : {timeout}")
    print(f"Result will be pushed to    : https://huggingface.co/{push_to} (private)")
    print(f"\n--- command ---\n{command}\n---------------")

    if args.dry_run:
        print("\n(dry run: nothing submitted)")
        return 0

    from huggingface_hub import get_token, run_job

    # The container has no credentials of its own. Without this the job would
    # train for hours and then fail at the final upload_folder. Sent as a
    # secret, so HF encrypts it server-side and it never appears in the logs.
    token = get_token()
    if not token:
        print("!! No token found to pass to the job for the final upload. Run `hf auth login`.")
        return 1
    job = run_job(image=IMAGE, command=["bash", "-c", command],
                  flavor=args.flavor, timeout=timeout, secrets={"HF_TOKEN": token})
    print(f"\nSubmitted: {job.url}")
    print(f"Job id: {job.id}")
    print("\nWatch it with:")
    print(f"  hf jobs logs {job.id}")
    print("When it says COMPLETED, pull the result down with:")
    print(f"  python scripts/install_retrained.py --hf-repo {push_to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
