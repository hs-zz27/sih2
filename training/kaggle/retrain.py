"""Retrain a demo model on a free Kaggle GPU, unattended, and package it.

The trained checkpoints were lost (2026-09-17) and the team has no GPU. This
rebuilds the three models the demo video needs, one Kaggle session each:

    vqa          rs_vqa_v1 adapter   dmarsili/RSVQA-LR-2k + designed refusals
    change_mask  change_mask_v1      ericyu/LEVIRCD_Cropped256 (7,120/1,024/2,048)
    caption      caption_v1          arampacha/rsicd (test n=1,093)

Every dataset id was checked against the HuggingFace API; LEVIR-CD and RSICD
match the split sizes the original runs recorded. The recipes are the Phase 5
ones from configs/campaign.yaml, adjusted for a T4 and a 12-hour session - each
adjustment is listed in DEVIATIONS and written into the manifest.

Run inside a Kaggle notebook (GPU on, internet on):

    python training/kaggle/retrain.py --model vqa

Heavy files (base model, datasets, working checkpoints) go to --work, which
Kaggle does not save. Only the packaged result goes to --out, which Kaggle
keeps as the notebook's output:

    <out>/checkpoints/<path the demo reads>     the weights
    <out>/retrain_manifest_<model>.json          datasets, args, GPU, timing,
                                                 metrics, deviations, status
    <out>/logs/<model>.log                       everything the steps printed

Training is stopped at --budget-hours so the session never hits Kaggle's hard
limit with nothing saved; a stopped run is evaluated from its last checkpoint.
`--dry-run` prints the plan without downloading or running anything.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Packages the trainers import that a stock Kaggle GPU image may lack or carry
# too old. The repository's own pins are NOT installed: they target the API
# runtime (Python >= 3.12, exact rasterio/numpy) and would fight Kaggle's image.
PIP_PACKAGES = [
    "transformers>=4.49",
    "peft>=0.14",
    "bitsandbytes>=0.45",
    "accelerate>=1.2",
    "huggingface_hub>=0.26",
    "pyarrow>=18.0",
]

# Kaggle stops a session at 12 h. Leave room for download, eval and packaging.
DEFAULT_BUDGET_HOURS = 10.5


@dataclass
class Step:
    name: str
    cmd: list[str]
    # Only the training step is bound by the wall-clock budget.
    budgeted: bool = False


@dataclass
class Plan:
    model: str
    datasets: dict[str, str]
    deploy_path: str            # relative to <out>/checkpoints, = demo_assets path
    steps: list[Step]
    ckpt_dir: Path
    package_from: list[str]     # candidates inside ckpt_dir, first existing wins
    eval_cmd: list[str] | None  # re-evaluate a checkpoint if training was cut short
    deviations: list[str] = field(default_factory=list)


def py(*args: str) -> list[str]:
    return [sys.executable, *args]


def download(repo: str, dest: Path, repo_type: str = "dataset") -> list[str]:
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={repo!r}, repo_type={repo_type!r}, local_dir={str(dest)!r})"
    )
    return py("-c", code)


def build_plan(model: str, work: Path) -> Plan:
    data = work / "data"
    ckpt = work / "ckpt"

    if model == "vqa":
        rsvqa = data / "rsvqa_lr_2k"
        mix = data / "instruct_mix"
        base = work / "models" / "qwen25_vl_3b"
        run = ckpt / "track_b_vqa"
        return Plan(
            model="vqa",
            datasets={"rsvqa_lr_2k": "dmarsili/RSVQA-LR-2k",
                      "base_model": "Qwen/Qwen2.5-VL-3B-Instruct"},
            deploy_path="v2/track_b_vqa/adapter_final",
            steps=[
                Step("download RSVQA-LR-2k", download("dmarsili/RSVQA-LR-2k", rsvqa)),
                Step("prepare RSVQA instruct.jsonl",
                     py("training/prepare/rsvqa.py", "--src", str(rsvqa),
                        "--out", str(rsvqa / "instruct.jsonl"))),
                Step("build instruction mix",
                     py("training/prepare/instruction_mix.py", "--data-root", str(data),
                        "--out", str(mix))),
                Step("download base model",
                     download("Qwen/Qwen2.5-VL-3B-Instruct", base, repo_type="model")),
                Step("dry-run the trainer (data + config, no GPU)",
                     py("training/track_b_vlm_qlora.py", "--model", str(base), "--data", str(mix),
                        "--ckpt-dir", str(run), "--dry-run")),
                Step("train QLoRA adapter",
                     py("training/track_b_vlm_qlora.py", "--model", str(base), "--data", str(mix),
                        "--ckpt-dir", str(run), "--max-steps", "1500",
                        "--batch-size", "2", "--grad-accum", "8",
                        "--lr", "0.0001", "--warmup-steps", "45",
                        "--val-every", "250", "--save-every", "250", "--patience", "3",
                        "--resume"),
                     budgeted=True),
            ],
            ckpt_dir=run,
            # adapter_best is the lowest held-out loss; the Phase 5 early-stopped
            # run recorded that as the one to deploy.
            package_from=["adapter_best", "adapter_final"],
            eval_cmd=None,  # validation runs inside training and writes adapter_best
            deviations=[
                "WHU-OPT-SAR (~10 GB, not on HuggingFace) is not downloaded, so the mix has "
                "no SAR examples and no not_in_image refusals; instruction_mix.py skips it "
                "when absent.",
                "Early-stopping recipe (Phase 5 track_b_vqa_es: best at step 1500), not the "
                "6,000-step run; --max-steps 1500, --patience 3.",
                "Effective batch 16 as micro-batch 2 x grad-accum 8 for a 16 GB T4; "
                "warmup scaled to 3% of steps (45).",
                "fp16 compute on a T4 (no bf16), chosen by the trainer.",
                "Official RSVQA-LR accuracy is NOT measured here; do not quote 0.8947 for "
                "this adapter until it is re-measured.",
            ],
        )

    if model == "change_mask":
        src = data / "levircd_parquet"
        out = data / "levircd"
        run = ckpt / "change_mask"
        train = py("training/train_change_mask.py", "--index", str(out / "index.json"),
                   "--ckpt-dir", str(run), "--arch", "v2", "--epochs", "30", "--dim", "48")
        return Plan(
            model="change_mask",
            datasets={"levir_cd": "ericyu/LEVIRCD_Cropped256"},
            deploy_path="v2/change_mask",
            steps=[
                Step("download LEVIR-CD", download("ericyu/LEVIRCD_Cropped256", src)),
                Step("extract tiles + index",
                     py("training/prepare/levir.py", "--src", str(src),
                        "--out", str(out / "index.json"), "--dest", str(out / "tiles"))),
                Step("train change mask", train + ["--resume"], budgeted=True),
            ],
            ckpt_dir=run,
            package_from=["."],
            eval_cmd=train + ["--eval-only", "--out", str(run / "metrics_after_budget.json")],
            deviations=["30 epochs instead of Phase 5's 60, to fit one T4 session."],
        )

    if model == "caption":
        src = data / "rsicd"
        run = ckpt / "caption_pre"
        train = py("training/train_caption.py", "--data", str(src), "--ckpt-dir", str(run),
                   "--arch", "v2", "--pretrained", "--epochs", "60", "--dim", "192")
        return Plan(
            model="caption",
            datasets={"rsicd": "arampacha/rsicd"},
            deploy_path="v2/caption_pre",
            steps=[
                Step("download RSICD", download("arampacha/rsicd", src)),
                Step("train caption model", train + ["--resume"], budgeted=True),
            ],
            ckpt_dir=run,
            package_from=["."],
            eval_cmd=train + ["--eval-only", "--out", str(run / "metrics_after_budget.json")],
            deviations=["ImageNet ResNet-50 weights are downloaded at train start (internet on)."],
        )

    raise ValueError(f"unknown model {model!r}; choose vqa, change_mask or caption")


def run_step(step: Step, log, deadline: float | None) -> tuple[int, bool]:
    """Run one step, teeing output to the log. Returns (exit code, stopped_by_budget)."""
    log.write(f"\n=== {step.name}\n$ {' '.join(step.cmd)}\n")
    log.flush()
    proc = subprocess.Popen(step.cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env={**os.environ, "PYTHONPATH": str(ROOT)})
    stopped = False
    assert proc.stdout is not None
    os.set_blocking(proc.stdout.fileno(), False)
    while True:
        line = proc.stdout.readline()
        if line:
            sys.stdout.write(line)
            log.write(line)
            continue
        if proc.poll() is not None:
            rest = proc.stdout.read() or ""
            sys.stdout.write(rest)
            log.write(rest)
            break
        if step.budgeted and deadline is not None and time.time() > deadline:
            msg = f"\n!! wall-clock budget reached - stopping '{step.name}' (checkpoints are kept)\n"
            sys.stdout.write(msg)
            log.write(msg)
            proc.terminate()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
            stopped = True
            break
        time.sleep(0.2)
    log.flush()
    return proc.returncode if proc.returncode is not None else -1, stopped


def package(plan: Plan, out: Path) -> str | None:
    """Copy the deployable weights to <out>/checkpoints/<deploy_path>."""
    target = out / "checkpoints" / plan.deploy_path
    for candidate in plan.package_from:
        source = plan.ckpt_dir / candidate if candidate != "." else plan.ckpt_dir
        if source.is_dir() and any(source.iterdir()):
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            return candidate
    return None


def gpu_name() -> str | None:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, choices=["vqa", "change_mask", "caption"])
    p.add_argument("--work", type=Path, default=Path("/kaggle/tmp/satquery"))
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/retrained"))
    p.add_argument("--budget-hours", type=float, default=DEFAULT_BUDGET_HOURS)
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = p.parse_args()

    plan = build_plan(args.model, args.work)
    if args.dry_run:
        print(json.dumps({**asdict(plan), "ckpt_dir": str(plan.ckpt_dir)}, indent=2, default=str))
        return 0

    args.work.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + args.budget_hours * 3600
    manifest: dict = {
        "model": plan.model, "datasets": plan.datasets, "deploy_path": plan.deploy_path,
        "deviations": plan.deviations, "started_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_name(), "python": platform.python_version(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "budget_hours": args.budget_hours, "steps": [],
    }
    manifest_path = args.out / f"retrain_manifest_{plan.model}.json"

    def save(status: str) -> None:
        manifest["status"] = status
        manifest["elapsed_hours"] = round((time.time() - started) / 3600, 3)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with open(args.out / "logs" / f"{plan.model}.log", "a", encoding="utf-8") as log:
        steps = list(plan.steps)
        if not args.skip_install:
            steps.insert(0, Step("install training packages",
                                 py("-m", "pip", "install", "-q", *PIP_PACKAGES)))
        stopped_by_budget = False
        for step in steps:
            code, stopped = run_step(step, log, deadline)
            manifest["steps"].append({"name": step.name, "exit_code": code, "stopped_by_budget": stopped})
            save("running")
            if stopped:
                stopped_by_budget = True
                break
            if code != 0:
                save(f"failed at: {step.name}")
                print(f"\n!! FAILED at '{step.name}' (exit {code}). See the log in {args.out / 'logs'}.")
                return 1

        if stopped_by_budget and plan.eval_cmd:
            code, _ = run_step(Step("evaluate last checkpoint", plan.eval_cmd), log, None)
            manifest["steps"].append({"name": "evaluate last checkpoint", "exit_code": code})

    for name in ("metrics.json", "metrics_after_budget.json"):
        path = plan.ckpt_dir / name
        if path.exists():
            manifest[name] = json.loads(path.read_text(encoding="utf-8"))
    packaged = package(plan, args.out)
    manifest["packaged_from"] = packaged
    manifest["stopped_by_budget"] = stopped_by_budget
    if packaged is None:
        save("no deployable weights produced")
        print("\n!! Training produced nothing to package. See the log.")
        return 1
    save("stopped by budget - packaged last good weights" if stopped_by_budget else "complete")
    print(f"\nDONE: {plan.model} -> {args.out / 'checkpoints' / plan.deploy_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
