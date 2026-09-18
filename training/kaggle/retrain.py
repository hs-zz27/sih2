"""Retrain a demo model on a free Kaggle GPU, unattended, and package it.

The trained checkpoints were lost (2026-09-17) and the team has no GPU. This
rebuilds the three models the demo video needs, one Kaggle session each:

    vqa          rs_vqa_v1 adapter   dmarsili/RSVQA-LR-2k + designed refusals
    change_mask  change_mask_v1      ericyu/LEVIRCD_Cropped256 (7,120/1,024/2,048)
    caption      caption_v1          arampacha/rsicd (test n=1,093)

and one evaluation, which is how a retrained adapter earns a quotable number:

    eval_vqa     rs_vqa_v1 adapter   official RSVQA-LR test split (Zenodo
                                     6344334): 10,004 questions, 100 images

Every dataset id was checked against the HuggingFace API, and RSVQA-LR-2k was
run through the prepare scripts end to end. The recipes are the Phase 5 ones
from configs/campaign.yaml, adjusted for a T4 and a 12-hour session; each
adjustment is listed in DEVIATIONS and written into the manifest.

Run inside a Kaggle notebook (GPU on, internet on). The notebooks run a smoke
pass first and the real run second, in the same session:

    python training/kaggle/retrain.py --model vqa --smoke
    python training/kaggle/retrain.py --model vqa --session-start <epoch>

What a run does, in order:
  1. pip install the pinned training stack (torch/torchvision are NOT touched)
  2. envcheck.py: CUDA, imports, free disk, internet - fail in minute one
  3. download, prepare, train (training is the only budgeted step)
  4. if training was stopped by the budget, evaluate the last checkpoint
  5. refuse to package weights whose losses or metrics are not finite
  6. package to <out>/checkpoints/<the path the demo reads>

A smoke pass uses tiny limits (minutes of training), its own checkpoint
directory and its own output directory, so the real run never resumes from it
and scripts/install_retrained.py refuses to install it.
"""

from __future__ import annotations

import argparse
import json
import math
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

# The versions the team's trainer last ran with (docs/phase1-status.md:
# bitsandbytes 0.50.2, peft 0.20.0, accelerate 1.14.0; training/cluster/
# bootstrap.sh: transformers 5.x). torch and torchvision are deliberately NOT
# installed: pip would replace Kaggle's CUDA build. envcheck.py fails the run
# if they are missing or broken instead.
PIP_PACKAGES = [
    "transformers>=5.0,<6",
    "peft==0.20.0",
    "bitsandbytes==0.50.2",
    "accelerate==1.14.0",
    "huggingface_hub>=0.26",
    "pyarrow>=18.0",
]

# Scoring an adapter imports evaluation.track_b_eval for its SYSTEM_PROMPT,
# which executes satquery/tools/__init__.py, which imports the index engine and
# therefore rasterio and scikit-image. None of that is needed to TRAIN, so it
# is installed only for the evaluation plan. Deliberately unpinned, and
# deliberately not `-r requirements.txt`: that file pins numpy and pillow, and
# replacing the ones a CUDA torch build was compiled against is how a working
# GPU environment gets broken minutes before it is needed.
EVAL_PIP_PACKAGES = [
    "rasterio", "scikit-image", "scipy", "scikit-learn", "pydantic", "pyyaml",
]

# Kaggle stops a session at 12 h, counting from notebook start.
DEFAULT_BUDGET_HOURS = 10.5
SMOKE_BUDGET_HOURS = 1.0


@dataclass
class Step:
    name: str
    cmd: list[str]
    # Only the training step is bound by the wall-clock budget.
    budgeted: bool = False
    env: dict[str, str] = field(default_factory=dict)


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
    smoke: bool = False
    # Installed before this plan's steps even under --skip-install: that flag
    # means "the training stack is already in this container", which says
    # nothing about the evaluation stack.
    pip_extra: list[str] = field(default_factory=list)
    # Evaluation runs produce a JSON result instead of weights.
    result_file: Path | None = None


def py(*args: str) -> list[str]:
    return [sys.executable, *args]


def download(repo: str, dest: Path, repo_type: str = "dataset") -> list[str]:
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={repo!r}, repo_type={repo_type!r}, local_dir={str(dest)!r})"
    )
    return py("-c", code)


ZENODO_RSVQA_LR = "https://zenodo.org/records/6344334/files"
RSVQA_LR_FILES = [
    "LR_split_test_questions.json", "LR_split_test_answers.json",
    "LR_split_train_questions.json", "LR_split_train_answers.json",
    "Images_LR.zip",
]


def fetch_and_unzip(base_url: str, names: list[str], dest: Path) -> list[str]:
    """Download named files, unzipping any .zip, skipping what is already there."""
    code = f"""
import urllib.request, zipfile, pathlib
dest = pathlib.Path({str(dest)!r}); dest.mkdir(parents=True, exist_ok=True)
for name in {names!r}:
    target = dest / name
    if not target.exists():
        print('downloading', name, flush=True)
        urllib.request.urlretrieve(f'{base_url}/' + name + '?download=1', target)
    if name.endswith('.zip'):
        marker = dest / name[:-4]
        if not marker.exists():
            print('unzipping', name, flush=True)
            with zipfile.ZipFile(target) as zf:
                zf.extractall(dest)
print('ready:', sorted(p.name for p in dest.iterdir())[:8], flush=True)
"""
    return py("-c", code)


def build_plan(model: str, work: Path, smoke: bool = False, fp32_compute: bool = False,
               adapter: Path | None = None) -> Plan:
    data = work / "data"
    # Smoke checkpoints live apart, so the real run's --resume cannot pick
    # up a 10-step smoke checkpoint and call it progress.
    ckpt = work / ("ckpt_smoke" if smoke else "ckpt")

    if model == "vqa":
        rsvqa = data / "rsvqa_lr_2k"
        mix = data / "instruct_mix"
        base = work / "models" / "qwen25_vl_3b"
        run = ckpt / "track_b_vqa"
        # The trainer processes `--grad-accum` examples per optimiser step,
        # one at a time; `--batch-size` only feeds its printed "effective
        # batch". So 1 x 8 is honestly 8 examples per step - what Phase 5 used.
        train_args = ["--batch-size", "1", "--grad-accum", "8", "--lr", "0.0001"]
        if smoke:
            train_args += ["--limit", "64", "--max-steps", "10", "--warmup-steps", "2",
                           "--val-every", "5", "--val-limit", "8", "--save-every", "5",
                           "--patience", "0"]
        else:
            train_args += ["--max-steps", "1500", "--warmup-steps", "45",
                           "--val-every", "250", "--save-every", "250", "--patience", "3"]
        train_env = {"SATQUERY_BNB_COMPUTE_DTYPE": "float32"} if fp32_compute else {}
        deviations = [
            "WHU-OPT-SAR (~10 GB, not on HuggingFace) is not downloaded, so the mix has "
            "no SAR examples and no not_in_image refusals; instruction_mix.py skips it "
            "when absent.",
            "Early-stopping recipe (Phase 5 track_b_vqa_es: best at step 1500), not the "
            "6,000-step run; --max-steps 1500, --patience 3.",
            "8 examples per optimiser step (--batch-size 1 --grad-accum 8). The trainer "
            "ignores --batch-size in its loop, so this matches Phase 5's actual 8.",
            "fp32 4-bit compute (SATQUERY_BNB_COMPUTE_DTYPE=float32) - fallback after an "
            "fp16 NaN." if fp32_compute else "fp16 4-bit compute on a T4 (no bf16).",
            "Official RSVQA-LR accuracy is not measured by this run: score the adapter "
            "with --model eval_vqa (the notebook's stage 3). Until then the adapter has "
            "no quotable accuracy - 0.8947 belongs to the lost Phase 5 weights.",
        ]
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
                        "--ckpt-dir", str(run), *train_args, "--resume"),
                     budgeted=True, env=train_env),
            ],
            ckpt_dir=run,
            # adapter_best is the lowest held-out loss; the Phase 5 early-stopped
            # run recorded that as the one to deploy.
            package_from=["adapter_best", "adapter_final"],
            eval_cmd=None,  # validation runs inside training and writes adapter_best
            deviations=deviations,
            smoke=smoke,
        )

    if model == "change_mask":
        src = data / "levircd_parquet"
        out = data / "levircd"
        run = ckpt / "change_mask"
        limits = (["--epochs", "1", "--limit-train", "64", "--limit-eval", "32"] if smoke
                  else ["--epochs", "30"])
        train = py("training/train_change_mask.py", "--index", str(out / "index.json"),
                   "--ckpt-dir", str(run), "--arch", "v2", "--dim", "48", *limits)
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
            smoke=smoke,
        )

    if model == "caption":
        src = data / "rsicd"
        run = ckpt / "caption_pre"
        # train_caption.py has no row limit, so a smoke pass is one real epoch.
        epochs = "1" if smoke else "60"
        train = py("training/train_caption.py", "--data", str(src), "--ckpt-dir", str(run),
                   "--arch", "v2", "--pretrained", "--epochs", epochs, "--dim", "192")
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
            smoke=smoke,
        )

    if model == "eval_vqa":
        official = data / "rsvqa_lr_official"
        base = work / "models" / "qwen25_vl_3b"
        adapter_path = adapter or Path(
            "/kaggle/working/retrained/checkpoints/v2/track_b_vqa/adapter_final")
        result = work / "results" / "rsvqa_lr_official_test.json"
        limit = ["--limit", "40"] if smoke else []
        return Plan(
            model="eval_vqa",
            datasets={"rsvqa_lr_official": "Zenodo 10.5281/zenodo.6344334 (CC-BY-4.0)"},
            deploy_path="",
            steps=[
                Step("download official RSVQA-LR",
                     fetch_and_unzip(ZENODO_RSVQA_LR, RSVQA_LR_FILES, official)),
                Step("resolve the official test split",
                     py("training/prepare/rsvqa_official.py", "--src", str(official),
                        "--out", str(official))),
                Step("download base model",
                     download("Qwen/Qwen2.5-VL-3B-Instruct", base, repo_type="model")),
                Step("score the adapter on the official split",
                     py("evaluation/rsvqa_official_eval.py", "--base", str(base),
                        "--data", str(official), "--arms", f"retrained={adapter_path}",
                        "--out", str(result), *limit),
                     budgeted=True),
            ],
            ckpt_dir=result.parent,
            package_from=[],
            eval_cmd=None,
            result_file=result,
            pip_extra=EVAL_PIP_PACKAGES,
            deviations=[
                "Scores whichever adapter --adapter names; by default the one this "
                "session's vqa run packaged.",
                "Published convention excludes count questions, as the literature does; "
                "the report carries both conventions.",
            ] + (["SMOKE: --limit 40 questions, not the 10,004-question split."] if smoke else []),
            smoke=smoke,
        )

    raise ValueError(
        f"unknown model {model!r}; choose vqa, change_mask, caption or eval_vqa")


def _read(stream, method: str) -> str:
    """Read from a NON-BLOCKING text stream, tolerating "no data right now".

    `os.set_blocking(fd, False)` below makes the underlying raw read return
    None whenever the pipe happens to be empty. TextIOWrapper does not pass
    that through: it raises from inside its decoder ("can't concat NoneType to
    bytes"), which is why the `or ""` that used to guard the drain could never
    catch it.

    Measured on a Hugging Face Job (2026-09-18): this killed a vqa run at the
    end of its environment-check step, fourteen seconds in, while the
    identical change_mask run survived all five of its steps. It is a race, so
    it fails intermittently - and on a paid GPU it would fail just as happily
    after seven hours of training as after fourteen seconds, losing the run.
    """
    try:
        return getattr(stream, method)() or ""
    except (TypeError, ValueError):
        return ""


def run_step(step: Step, log, deadline: float | None) -> tuple[int, bool]:
    """Run one step, teeing output to the log. Returns (exit code, stopped_by_budget)."""
    log.write(f"\n=== {step.name}\n$ {' '.join(step.cmd)}\n")
    log.flush()
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        # One GPU. Kaggle offers T4 x2, and the VQA trainer loads with
        # device_map="auto", which would split the model across both cards -
        # a known source of cross-device errors mid-training.
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        "PYTHONUNBUFFERED": "1",
        **step.env,
    }
    proc = subprocess.Popen(step.cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    stopped = False
    assert proc.stdout is not None
    os.set_blocking(proc.stdout.fileno(), False)
    while True:
        line = _read(proc.stdout, "readline")
        if line:
            sys.stdout.write(line)
            log.write(line)
            continue
        if proc.poll() is not None:
            # The child has exited; drain the tail. Retried, because a
            # non-blocking pipe can report "nothing yet" a moment before the
            # last bytes land - and that tail is usually the error message
            # explaining why the step failed.
            rest, empty = "", 0
            while empty < 3:
                chunk = _read(proc.stdout, "read")
                if chunk:
                    rest, empty = rest + chunk, 0
                else:
                    empty += 1
                    time.sleep(0.05)
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


def _all_finite(value) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_all_finite(v) for v in value)
    return True


def check_trained(plan: Plan) -> tuple[bool, str]:
    """Is there something real to package? (ok, reason).

    Guards the failure that would otherwise ship silently: a NaN loss. The
    VQA trainer picks its best adapter with min(), and a NaN first validation
    compares False against everything after it, so a NaN adapter can be saved
    as "adapter_best" and early-stop the run - which then looks complete.
    """
    if plan.result_file is not None:
        if not plan.result_file.exists():
            return False, "the evaluator wrote no result file"
        report = json.loads(plan.result_file.read_text(encoding="utf-8"))
        arms = report.get("arms") or {}
        if not arms:
            return False, "the result file has no scored arm"
        if not _all_finite(arms):
            return False, "the result file holds a non-finite score"
        first = next(iter(arms.values()))
        accuracy = (first.get("published_convention") or {}).get("micro_accuracy")
        return True, f"scored: published-convention accuracy {accuracy}"

    if plan.model == "vqa":
        history_path = plan.ckpt_dir / "val_history.json"
        if not history_path.exists():
            return False, "no val_history.json - training never reached a validation"
        history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
        if not history:
            return False, "val_history.json has no validations"
        bad = [r["step"] for r in history
               if not (math.isfinite(r.get("val_loss", math.nan))
                       and math.isfinite(r.get("train_loss", math.nan)))]
        if bad:
            return False, (f"non-finite loss at step(s) {bad} - fp16 overflow is likely; "
                           "re-run with --fp32-compute")
        if not (plan.ckpt_dir / "adapter_best").is_dir():
            return False, "no adapter_best saved"
        return True, f"{len(history)} finite validations, best val_loss " \
                     f"{min(r['val_loss'] for r in history):.4f}"

    for name in ("metrics.json", "metrics_after_budget.json"):
        path = plan.ckpt_dir / name
        if path.exists():
            metrics = json.loads(path.read_text(encoding="utf-8"))
            if not _all_finite(metrics):
                return False, f"{name} contains a non-finite value: {json.dumps(metrics)[:200]}"
            return True, f"{name} finite"
    return False, "no metrics.json - training or its evaluation did not finish"


def package(plan: Plan, out: Path) -> str | None:
    """Copy the deployable weights - or an evaluation's result - into <out>."""
    if plan.result_file is not None:
        target = out / "results" / plan.result_file.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan.result_file, target)
        return plan.result_file.name

    target = out / "checkpoints" / plan.deploy_path
    for candidate in plan.package_from:
        source = plan.ckpt_dir / candidate if candidate != "." else plan.ckpt_dir
        if source.is_dir() and any(source.iterdir()):
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            return candidate
    return None


def envcheck_report(output: str) -> dict | None:
    for line in output.splitlines():
        if line.startswith("ENVCHECK "):
            try:
                return json.loads(line[len("ENVCHECK "):])
            except json.JSONDecodeError:
                return None
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True,
                   choices=["vqa", "change_mask", "caption", "eval_vqa"])
    p.add_argument("--adapter", type=Path,
                   help="eval_vqa: adapter to score (default: this session's vqa output)")
    p.add_argument("--work", type=Path, default=Path("/kaggle/tmp/satquery"))
    p.add_argument("--out", type=Path, help="default /kaggle/working/retrained[_smoke]")
    p.add_argument("--smoke", action="store_true",
                   help="tiny limits, separate checkpoints and output: prove the path works")
    p.add_argument("--fp32-compute", action="store_true",
                   help="vqa only: 4-bit compute in float32, the fallback for fp16 NaN on a T4")
    p.add_argument("--budget-hours", type=float,
                   help=f"default {DEFAULT_BUDGET_HOURS} (smoke {SMOKE_BUDGET_HOURS})")
    p.add_argument("--session-start", type=float,
                   help="epoch seconds the Kaggle session started; the budget counts from "
                        "here, so a smoke pass earlier in the session is included")
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = p.parse_args()

    plan = build_plan(args.model, args.work, smoke=args.smoke,
                      fp32_compute=args.fp32_compute, adapter=args.adapter)
    out = args.out or Path("/kaggle/working/retrained_smoke" if args.smoke
                           else "/kaggle/working/retrained")
    budget = args.budget_hours or (SMOKE_BUDGET_HOURS if args.smoke else DEFAULT_BUDGET_HOURS)
    if args.dry_run:
        print(json.dumps({**asdict(plan), "ckpt_dir": str(plan.ckpt_dir), "out": str(out),
                          "budget_hours": budget}, indent=2, default=str))
        return 0

    args.work.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = (args.session_start or started) + budget * 3600
    manifest: dict = {
        "model": plan.model, "smoke": plan.smoke, "datasets": plan.datasets,
        "deploy_path": plan.deploy_path, "deviations": plan.deviations,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "budget_hours": budget, "session_start": args.session_start, "steps": [],
    }
    manifest_path = out / f"retrain_manifest_{plan.model}.json"

    def save(status: str) -> None:
        manifest["status"] = status
        manifest["elapsed_hours"] = round((time.time() - started) / 3600, 3)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    save("running")
    log_path = out / "logs" / f"{plan.model}.log"
    with open(log_path, "a", encoding="utf-8") as log:
        pre: list[Step] = []
        if not args.skip_install:
            pre.append(Step("install training packages",
                            py("-m", "pip", "install", "-q", *PIP_PACKAGES)))
        # Not gated on --skip-install: see Plan.pip_extra.
        if plan.pip_extra:
            pre.append(Step("install evaluation packages",
                            py("-m", "pip", "install", "-q", *plan.pip_extra)))
        pre.append(Step("environment check",
                        py("training/kaggle/envcheck.py", "--model", plan.model,
                           "--work", str(args.work))))

        stopped_by_budget = False
        for step in pre + plan.steps:
            mark = log.tell()
            code, stopped = run_step(step, log, deadline)
            manifest["steps"].append({"name": step.name, "exit_code": code,
                                      "stopped_by_budget": stopped})
            if step.name == "environment check":
                log.flush()
                with open(log_path, encoding="utf-8") as reread:
                    reread.seek(mark)
                    manifest["environment"] = envcheck_report(reread.read())
            save("running")
            if stopped:
                stopped_by_budget = True
                break
            if code != 0:
                save(f"failed at: {step.name}")
                print(f"\n!! FAILED at '{step.name}' (exit {code}). Log: {log_path}")
                return 1

        if stopped_by_budget and plan.eval_cmd:
            code, _ = run_step(Step("evaluate last checkpoint", plan.eval_cmd), log, None)
            manifest["steps"].append({"name": "evaluate last checkpoint", "exit_code": code})

    for name in ("metrics.json", "metrics_after_budget.json", "val_history.json"):
        path = plan.ckpt_dir / name
        if path.exists():
            manifest[name] = json.loads(path.read_text(encoding="utf-8"))
    manifest["stopped_by_budget"] = stopped_by_budget

    ok, reason = check_trained(plan)
    manifest["training_check"] = reason
    if not ok:
        save(f"failed check: {reason}")
        print(f"\n!! NOT PACKAGED: {reason}")
        return 1

    packaged = package(plan, out)
    manifest["packaged_from"] = packaged
    if packaged is None:
        save("no deployable weights produced")
        print("\n!! Training produced nothing to package. See the log.")
        return 1
    status = "stopped by budget - packaged last good weights" if stopped_by_budget else "complete"
    save(f"smoke {status}" if plan.smoke else status)
    landed = (out / "results" / plan.result_file.name) if plan.result_file is not None \
        else (out / "checkpoints" / plan.deploy_path)
    print(f"\nDONE: {plan.model}{' (smoke)' if plan.smoke else ''} -> {landed}"
          f"\nmanifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
