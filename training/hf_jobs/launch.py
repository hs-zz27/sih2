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

# Every number in this table exists to bound what a FAILURE can cost, because
# a job that hangs bills its timeout and hands back nothing.
#
#   budget    hours from job start after which retrain.py stops TRAINING and
#             packages whatever it has. Passed together with --session-start,
#             so the clock starts when the job does, not when the step does.
#   deadline  hours from job start by which vqa's official scoring must also be
#             finished. Training and scoring therefore share ONE clock; before
#             this, scoring began a second full-length budget of its own and
#             could run the container past its timeout - losing the adapter
#             that had just been trained, because the push never ran.
#   hard      SIGTERM backstop, in hours, around the whole driver call. It
#             covers the steps retrain.py does not budget: the dataset and
#             base-model downloads. A stalled download is the one failure the
#             driver cannot stop by itself.
#   timeout   the job's own limit, and the most HF can bill for it. Always
#             comfortably above `hard`, so the final push still runs.
#
# change_mask used to set budget=10.5h (retrain.py's default) under a 9h
# timeout, so its own budget could never fire: the container was killed first
# and the result was never pushed. Every budget here is now under its timeout.
PROFILE = {
    ("vqa", False):         {"budget": 7.5, "deadline": 10.5, "hard": 8.5, "timeout": "12h"},
    ("vqa", True):          {"budget": 0.5, "deadline": 1.5, "hard": 1.6, "timeout": "3h"},
    ("change_mask", False): {"budget": 6.0, "deadline": 6.0, "hard": 7.0, "timeout": "8h"},
    ("change_mask", True):  {"budget": 0.4, "deadline": 0.4, "hard": 1.2, "timeout": "2h"},
    ("caption", False):     {"budget": 3.0, "deadline": 3.0, "hard": 3.5, "timeout": "4h"},
    ("caption", True):      {"budget": 0.4, "deadline": 0.4, "hard": 1.2, "timeout": "2h"},
}
DEFAULT_TIMEOUT = {m: PROFILE[(m, False)]["timeout"]
                   for m in ("vqa", "change_mask", "caption")}
MODELS = sorted(DEFAULT_TIMEOUT)

# Only the two flavors docs/hf-jobs-retrain.md actually quotes. Anything else
# is priced "unknown" rather than guessed, because a wrong number here would
# silently disarm the credit guard below. `hf jobs hardware` lists the rest.
FLAVOR_USD_PER_HOUR = {"cpu-basic": 0.0, "t4-small": 0.40, "t4-medium": 0.60}


def hours(timeout: str) -> float | None:
    """'12h' -> 12.0, '90m' -> 1.5. None when the unit is not one we price."""
    try:
        if timeout.endswith("h"):
            return float(timeout[:-1])
        if timeout.endswith("m"):
            return float(timeout[:-1]) / 60
        if timeout.endswith("s"):
            return float(timeout[:-1]) / 3600
    except ValueError:
        return None
    return None
# t4-small ($0.40/h) matches what the recipes were tuned for; every model
# here fits in its 16 GB. A friend's $11 covers ~27h on it.
DEFAULT_FLAVOR = "t4-small"


def build_command(model: str, smoke: bool, fp32_compute: bool, push_to: str) -> str:
    """One shell script: prove the push works, retrain, (vqa: score), and push.

    The push runs whatever happened before it. A job that trained for eight
    hours and then failed its scoring step must still hand back the adapter,
    and a job that failed outright must still hand back its manifest and logs
    - they are the only way to see why from outside an ephemeral container.
    The script exits with the training step's code, so a failed training run
    still shows as ERROR on the Hub.

    Three things here exist only to stop a broken run from spending money:

    1. A push PREFLIGHT before any GPU work. Repo-write used to be exercised
       for the first time at the very end, so a token without write access, or
       a name collision, turned eight paid hours into nothing. It now fails in
       the first seconds instead.
    2. One shared deadline, via --session-start, across training and scoring.
    3. A `timeout` around each driver call, covering the download steps that
       retrain.py does not budget and therefore cannot interrupt.
    """
    prof = PROFILE[(model, smoke)]
    smoke_flag = "--smoke " if smoke else ""
    fp32_flag = "--fp32-compute " if fp32_compute else ""
    deadline_s = int(prof["deadline"] * 3600)
    hard_s = int(prof["hard"] * 3600)

    # Scoring reuses the base model training downloaded (same --work), and
    # the adapter training packaged (same --out). A fresh HF job would have
    # neither, which is why it is not a separate job. It gets whatever is left
    # of the shared deadline, and is skipped outright when that is under ten
    # minutes - an adapter pushed unscored beats a container killed mid-upload.
    score = ""
    if model == "vqa":
        score = (
            'if [ "$TRAIN" = 0 ]; then\n'
            f"  LEFT=$(( {deadline_s} - ($(date +%s) - START) ))\n"
            '  if [ "$LEFT" -gt 600 ]; then\n'
            "    timeout --signal=TERM --kill-after=300 ${LEFT}s"
            f" python training/kaggle/retrain.py --model eval_vqa {smoke_flag}--skip-install"
            " --adapter /tmp/output/checkpoints/v2/track_b_vqa/adapter_final"
            " --work /tmp/satquery --out /tmp/output"
            f" --session-start $START --budget-hours {prof['deadline']}"
            " || echo '!! official scoring failed - the trained adapter is still pushed'\n"
            "  else\n"
            "    echo '!! no time left in the budget for official scoring -"
            " skipping it; the trained adapter is still pushed'\n"
            "  fi\n"
            "fi\n"
        )

    # retrain.py is stdlib-only at import time and installs its own training
    # packages as its first step (same PIP_PACKAGES as Kaggle) - nothing needs
    # installing here beyond git to clone the repo. The base image already
    # carries torch + torchvision matched to its CUDA build; retrain.py never
    # touches those.
    return (
        "set -uo pipefail\n"
        "START=$(date +%s)\n"
        f"export PUSH_TO='{push_to}'\n"
        f"export MODEL='{model}'\n"
        "apt-get update -qq && apt-get install -y -qq git > /dev/null || exit 1\n"
        f"git clone --depth 1 {REPO_URL} /workspace/sih2 || exit 1\n"
        "cd /workspace/sih2\n"
        # The base image does not carry huggingface_hub, and retrain.py
        # installs it only as its OWN first step - too late for the
        # preflight below, and too late for the final push if retrain.py
        # dies before reaching that step, which would lose the logs saying
        # why. It pulls no torch, so the image's CUDA build is untouched.
        "pip install -q \"huggingface_hub>=0.26\" || { echo '!! could not"
        " install huggingface_hub - the job would have no way to hand back"
        " its result, so stopping before any GPU time is billed'; exit 1; }\n"
        "cat > /tmp/preflight.py <<'PYEOF'\n"
        "import json, os, time\n"
        "from huggingface_hub import HfApi\n"
        "repo = os.environ['PUSH_TO']\n"
        "api = HfApi()\n"
        "api.create_repo(repo_id=repo, repo_type='model', private=True, exist_ok=True)\n"
        "api.upload_file(path_in_repo='job_started.json', repo_id=repo, repo_type='model',\n"
        "                path_or_fileobj=json.dumps({'started': time.time(),\n"
        "                                            'model': os.environ['MODEL']}).encode())\n"
        "print('preflight ok: the result repo exists and is writable')\n"
        "PYEOF\n"
        "python /tmp/preflight.py || { echo '!! cannot write to the result repo -"
        " aborting now, before any GPU time is billed'; exit 1; }\n"
        f"timeout --signal=TERM --kill-after=300 {hard_s}s"
        f" python training/kaggle/retrain.py --model {model} {smoke_flag}{fp32_flag}"
        " --work /tmp/satquery --out /tmp/output"
        f" --session-start $START --budget-hours {prof['budget']}\n"
        "TRAIN=$?\n"
        '[ "$TRAIN" = 124 ] && echo "!! the driver hit its hard time limit'
        ' - pushing whatever it produced"\n'
        + score +
        "mkdir -p /tmp/output\n"
        "cat > /tmp/push.py <<'PYEOF'\n"
        "import os\n"
        "from huggingface_hub import HfApi\n"
        "repo = os.environ['PUSH_TO']\n"
        "api = HfApi()\n"
        "api.create_repo(repo_id=repo, repo_type='model', private=True, exist_ok=True)\n"
        "api.upload_folder(repo_id=repo, repo_type='model', folder_path='/tmp/output')\n"
        "PYEOF\n"
        # Retried: losing hours of finished training to one flaky upload would
        # be the silliest way to spend the budget.
        "for attempt in 1 2 3; do\n"
        "  python /tmp/push.py && break\n"
        '  echo "!! push attempt $attempt failed"\n'
        "  sleep 30\n"
        "done\n"
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
    p.add_argument("--credits", type=float,
                   help="USD left on the billed account. Given, the launcher refuses "
                        "to submit a job whose worst case (flavor price x timeout) "
                        "exceeds it.")
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
    prof = PROFILE[(args.model, args.smoke)]
    timeout = args.timeout or prof["timeout"]
    command = build_command(args.model, args.smoke, args.fp32_compute, push_to)

    price = FLAVOR_USD_PER_HOUR.get(args.flavor)
    billable = hours(timeout)
    worst = price * billable if (price is not None and billable is not None) else None

    print(f"Account that will be billed : {who}")
    print(f"Flavor                      : {args.flavor}"
          + (f" (${price:.2f}/h)" if price is not None else " (price unknown here)"))
    print(f"Timeout                     : {timeout}  <- the most this job can bill")
    print(f"Stops training after        : {prof['budget']} h, then packages what it has")
    if args.model == "vqa":
        print(f"Scoring must finish by      : {prof['deadline']} h after the job starts")
    print(f"Worst case                  : "
          + (f"${worst:.2f}" if worst is not None else "unknown (unpriced flavor)"))
    print(f"Result will be pushed to    : https://huggingface.co/{push_to} (private)")

    # The guard is deliberately on the WORST case, not the estimate. An
    # estimate that fits the balance is no comfort if the failure mode bills
    # the timeout.
    if args.credits is not None and worst is not None and worst > args.credits:
        print(f"\n!! REFUSING TO SUBMIT: worst case ${worst:.2f} exceeds the "
              f"${args.credits:.2f} you said is left.\n"
              f"   Lower --timeout (e.g. --timeout {max(1, int(args.credits / max(price, 0.01)))}h) "
              "or pick a cheaper --flavor.")
        return 1

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
