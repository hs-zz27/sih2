"""Track B v0: QLoRA instruction tuning of the base VLM (plan task 1.7).

Goal for v0, per docs/04 section 4: *"Adapter trains, loads, and answers
through the real pipeline."* Quality is explicitly not the objective. The point
is to prove the whole path works - data prep, 4-bit load, LoRA attach, train,
checkpoint, kill, resume, save adapter, load adapter, answer - before any GPU
hours are spent chasing metrics.

Designed for a free-tier T4 (16 GB), which drives every default here:
4-bit NF4 quantisation, gradient checkpointing, batch size 1 with accumulation,
and frequent checkpoints because the session can be killed at any moment.

Originally written for Kaggle/Colab because the development machine had no
GPU. That is no longer true - it runs locally on an RTX 4050 with CUDA,
bitsandbytes, peft and accelerate installed - but every constraint above is
kept, because 6 GB of VRAM is tighter than a T4's 16 GB, not looser.

Everything that does not need a GPU - argument handling, dataset construction,
prompt formatting, checkpoint/resume - stays importable and unit tested; see
tests/test_training.py.

Usage on a GPU box:
    pip install peft bitsandbytes accelerate datasets
    python training/track_b_vlm_qlora.py \
        --model models/qwen25_vl_3b \
        --data data/vrsbench \
        --ckpt-dir checkpoints/track_b_v0 \
        --max-steps 200

    # after a kill:
    python training/track_b_vlm_qlora.py ... --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Repo root on the path so this runs as a plain script inside a notebook.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common.checkpointing import (  # noqa: E402
    TrainingState,
    maybe_resume,
    save_checkpoint,
    set_seed,
    write_run_metadata,
)

# LoRA targets for the language tower. The vision tower is left frozen in v0:
# it is the expensive half, and the adaptation the plan actually wants at this
# stage is instruction-following, not new visual features.
DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

SYSTEM_PROMPT = (
    "You are a remote-sensing image analyst. Answer only from what is visible "
    "in the imagery. If the image does not support an answer, say so."
)


@dataclass
class Example:
    """One instruction-tuning example."""

    image_path: str
    question: str
    answer: str
    source: str = "unknown"


def load_examples(
    data_dir: Path, limit: int | None = None, filename: str = "instruct.jsonl"
) -> list[Example]:
    """Load examples from a prepared JSONL file.

    Expected format, one object per line:
        {"image": "rel/path.jpg", "question": "...", "answer": "..."}

    A prepared JSONL is required rather than reading each benchmark's native
    format here: keeping dataset-specific parsing in training/prepare/ means
    this script stays the same regardless of which corpus is being used.
    """
    jsonl = data_dir / filename
    if not jsonl.exists():
        raise FileNotFoundError(
            f"{jsonl} not found. Build it first with a prepare script, e.g.\n"
            f"    python training/prepare/vrsbench.py --src {data_dir} --out {jsonl}"
        )

    examples: list[Example] = []
    with jsonl.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl}:{lineno} is not valid JSON: {exc}") from exc
            missing = {"image", "question", "answer"} - set(row)
            if missing:
                raise ValueError(f"{jsonl}:{lineno} missing fields: {sorted(missing)}")
            examples.append(
                Example(
                    image_path=str(data_dir / row["image"]),
                    question=row["question"],
                    answer=row["answer"],
                    source=row.get("source", "unknown"),
                )
            )
            if limit is not None and len(examples) >= limit:
                break

    if not examples:
        raise ValueError(f"{jsonl} contained no examples")
    return examples


def build_chat(example: Example) -> list[dict[str, Any]]:
    """Chat-format one example for a Qwen-VL style processor."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example.image_path},
                {"type": "text", "text": example.question},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": example.answer}]},
    ]


def supervised_start(processor, prompt_text: str, image) -> int:
    """Index of the first assistant token in the PROCESSED sequence.

    This must be measured with the image, not without it. A Qwen2.5-VL prompt
    renders one `<|image|>` marker that the processor then expands into one
    `<|image_pad|>` per visual patch - 323 of them for a 512x512 input. Taking
    the length of the text-only tokenisation instead returns the length before
    that expansion, so the mask cuts hundreds of tokens too early and the model
    is trained to predict image placeholders.

    Measured on data/instruct_mix before this was fixed: the boundary came back
    as 52 where the processed sequence needed 375, leaving 341 supervised
    tokens of which 312 were `<|image_pad|>`. Across six examples 89.1% of the
    supervised tokens were placeholders and 10.9% were the answer, and the run
    that produced checkpoints/track_b_v3 sat at loss ~6.8 for 1,950 steps
    because of it - a 0.3% change over 1,800 steps, inside the noise.
    """
    return processor(
        text=[prompt_text], images=[image], return_tensors="pt"
    )["input_ids"].shape[1]


def mask_prompt_labels(input_ids, assistant_start: int, pad_token_id: int):
    """Train only on the assistant's tokens.

    Without this the model is also trained to reproduce the question and the
    system prompt, which wastes capacity and measurably degrades answer
    quality. -100 is the ignore index for the HF loss.
    """
    import torch

    labels = input_ids.clone()
    labels[:assistant_start] = -100
    labels[labels == pad_token_id] = -100
    return labels


def encode_supervised(processor, example: Example, device=None):
    """One example -> a batch whose labels cover the answer and nothing else.

    Extracted so training and validation cannot drift apart. The image-token
    bug was possible because the boundary was computed inline; computing it in
    two places would let it reappear on the validation side alone, which is
    worse than the original bug - the number you trust to stop training would
    be the wrong one.
    """
    from PIL import Image

    chat = build_chat(example)
    text = processor.apply_chat_template(chat, tokenize=False)
    image = Image.open(example.image_path).convert("RGB")
    batch = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    if device is not None:
        batch = batch.to(device)

    prompt_text = processor.apply_chat_template(chat[:-1], tokenize=False)
    prompt_len = supervised_start(processor, prompt_text, image)
    pad_id = processor.tokenizer.pad_token_id or 0
    batch["labels"] = mask_prompt_labels(
        batch["input_ids"][0], prompt_len, pad_id
    ).unsqueeze(0)
    return batch


def evaluate(model, processor, examples, torch) -> float:
    """Mean loss over held-out examples, using the training encode path.

    `model.eval()` and `no_grad` for the duration, then training mode is
    restored by the caller. Examples whose labels are entirely masked - an
    empty answer would do it - are skipped rather than counted as zero, since
    a zero would drag the mean down and read as improvement.
    """
    was_training = model.training
    model.eval()
    total, counted = 0.0, 0
    try:
        with torch.no_grad():
            for example in examples:
                batch = encode_supervised(processor, example, model.device)
                if int((batch["labels"] != -100).sum()) == 0:
                    continue
                total += float(model(**batch).loss)
                counted += 1
    finally:
        if was_training:
            model.train()
    return total / counted if counted else float("nan")


def require_gpu_stack() -> tuple[Any, Any, Any]:
    """Import the GPU-only training stack with an actionable error."""
    missing: list[str] = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
        torch = None  # type: ignore[assignment]
    try:
        import peft
    except ImportError:
        missing.append("peft")
        peft = None  # type: ignore[assignment]
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        missing.append("bitsandbytes")

    if missing:
        raise SystemExit(
            "Missing training dependencies: "
            + ", ".join(missing)
            + "\nInstall with: pip install peft bitsandbytes accelerate datasets\n"
            "Note: bitsandbytes requires CUDA. This script cannot run on CPU."
        )

    if not torch.cuda.is_available():  # type: ignore[union-attr]
        raise SystemExit(
            "No CUDA device visible. QLoRA 4-bit training requires a GPU.\n"
            "On Kaggle: Settings -> Accelerator -> GPU T4 x2."
        )
    import transformers

    return torch, peft, transformers


def build_model(args, torch, peft, transformers):
    """Load the base model in 4-bit and attach a LoRA adapter."""
    from transformers import AutoProcessor, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # Double quantisation saves ~0.4 GB, which matters on a 16 GB T4.
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16,
    )

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)

    # transformers v5 renamed the vision-language auto class:
    # AutoModelForVision2Seq -> AutoModelForImageTextToText. Try the new name
    # first and fall back, so this works on both v4 and v5 installs.
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # transformers < 5
        from transformers import AutoModelForVision2Seq as AutoVLM

    model = AutoVLM.from_pretrained(
        args.model,
        quantization_config=quant,
        device_map="auto",
        local_files_only=True,
        # Left False deliberately. Qwen2.5-VL is natively supported, so no
        # remote code is needed; enabling it would execute arbitrary Python
        # downloaded from the model repo. Models that require it (InternVL3,
        # Florence-2) need a conscious decision, not a silent default.
        trust_remote_code=False,
    )

    model = peft.prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    lora = peft.LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_targets,
    )
    model = peft.get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, processor


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="local base model dir")
    p.add_argument("--data", type=Path, required=True, help="dir containing instruct.jsonl")
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/track_b_v0"))
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, help="cap the number of examples")
    # Frequent by design: a free-tier session can vanish without warning, and
    # a 40 MB adapter save is cheap.
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--keep-last", type=int, default=3)
    p.add_argument(
        "--patience", type=int, default=0,
        help="stop after this many validations with no improvement "
             "(0 = never stop early). The v2 run overfit for 2,000 steps "
             "past its optimum; --patience 3 would have stopped it.",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", nargs="*", default=DEFAULT_LORA_TARGETS)
    p.add_argument("--val-file", default="val.jsonl",
                   help="held-out split inside --data; empty string disables")
    p.add_argument("--val-every", type=int, default=100,
                   help="run validation every N optimiser steps")
    p.add_argument("--val-limit", type=int, default=160,
                   help="validate on the first N held-out examples. A fixed "
                        "prefix, not a sample, so the number is comparable "
                        "across steps; the full 534 costs ~4x more per check.")
    p.add_argument(
        "--optim",
        choices=["adamw8bit", "adamw"],
        default="adamw8bit",
        help="8-bit Adam moments (default) or fp32. Must MATCH on --resume: "
             "the checkpoint stores an optimiser state_dict of one shape.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="validate data and config without importing the GPU stack",
    )
    return p


def dry_run(args) -> int:
    """Validate everything that does not need a GPU.

    Runs on any machine, so a broken JSONL or a missing model directory is
    caught before a GPU session is booked rather than during it.
    """
    print("DRY RUN - no GPU stack imported\n")
    problems: list[str] = []

    if not args.model.exists():
        problems.append(f"model dir missing: {args.model} (run scripts/fetch_models.py)")
    else:
        print(f"model dir      : {args.model}")

    try:
        examples = load_examples(args.data, limit=args.limit)
        print(f"examples       : {len(examples)}")
        missing_images = [e for e in examples[:200] if not Path(e.image_path).exists()]
        if missing_images:
            problems.append(
                f"{len(missing_images)} of the first 200 images do not exist, "
                f"e.g. {missing_images[0].image_path}"
            )
        else:
            print("images         : first 200 all present")
        print(f"sample prompt  : {examples[0].question[:60]!r}")
        print(f"chat turns     : {len(build_chat(examples[0]))}")
    except (FileNotFoundError, ValueError) as exc:
        problems.append(str(exc))

    effective = args.batch_size * args.grad_accum
    print(f"effective batch: {effective} ({args.batch_size} x {args.grad_accum})")
    print(f"optimiser steps: {args.max_steps}")
    print(f"checkpoint every {args.save_every} steps -> {args.ckpt_dir}")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nReady to train on a GPU box.")
    return 0


def main() -> int:
    args = build_parser().parse_args()

    if args.dry_run:
        return dry_run(args)

    torch, peft, transformers = require_gpu_stack()
    set_seed(args.seed)

    examples = load_examples(args.data, limit=args.limit)
    print(f"Loaded {len(examples)} examples")

    val_examples: list[Example] = []
    if args.val_file:
        try:
            val_examples = load_examples(
                args.data, limit=args.val_limit, filename=args.val_file
            )
            print(f"Loaded {len(val_examples)} validation examples "
                  f"from {args.val_file}")
        except (FileNotFoundError, ValueError) as exc:
            # A missing held-out split must not kill a training run that was
            # going to succeed; it costs the stopping signal, not the run.
            print(f"[warn] validation disabled: {exc}")

    model, processor = build_model(args, torch, peft, transformers)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "adamw8bit":
        # Adam keeps two moments per trainable parameter. In fp32 that is
        # 8 bytes each on top of the 8 the parameter and its gradient already
        # cost; 8-bit quantises the moments to 2 bytes, saving 6 bytes per
        # parameter. Measured on the RTX 4050 (6,141 MiB) with the full target
        # set at r=32 and every micro-batch at 512x512 - the worst case this
        # dataset contains:
        #
        #     fp32 AdamW    6,036 MiB reserved   (105 MiB headroom)
        #     8-bit AdamW   5,668 MiB reserved   (473 MiB headroom)
        #
        # The 105 MiB margin is not one to stake an 11-hour run on: on Windows
        # an over-allocation does not raise, it spills into shared system RAM
        # and the run silently slows by an order of magnitude.
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(trainable, lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, args.warmup_steps, args.max_steps
    )

    state: TrainingState
    state, _ = maybe_resume(
        args.ckpt_dir, model, optimizer, scheduler, enabled=args.resume
    )

    write_run_metadata(
        args.ckpt_dir,
        {
            "task": "track_b_vlm_qlora_v0",
            "base_model": str(args.model),
            "data": str(args.data),
            "n_examples": len(examples),
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "targets": args.lora_targets,
            },
            "lr": args.lr,
            "optim": args.optim,
            "val_file": args.val_file,
            "val_every": args.val_every,
            "n_val": len(val_examples),
            "effective_batch": args.batch_size * args.grad_accum,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
    )

    val_history: list[dict] = []
    stop_early = False

    model.train()
    step = state.step
    index = (step * args.batch_size * args.grad_accum) % len(examples)

    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(args.grad_accum):
            example = examples[index % len(examples)]
            index += 1

            # The same encode path validation uses, so the two cannot drift.
            batch = encode_supervised(processor, example, model.device)

            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()
        step += 1

        lr_now = scheduler.get_last_lr()[0]
        if step % 5 == 0:
            print(
                f"step {step}/{args.max_steps}  loss {total_loss:.4f}  "
                f"lr {lr_now:.3e}",
                flush=True,
            )

        if val_examples and (
            step % args.val_every == 0 or step == args.max_steps
        ):
            val_loss = evaluate(model, processor, val_examples, torch)
            val_history.append(
                {"step": step, "train_loss": round(total_loss, 6),
                 "val_loss": round(val_loss, 6), "lr": lr_now,
                 "n_val": len(val_examples)}
            )
            best = min(val_history, key=lambda r: r["val_loss"])
            marker = "  <-- best so far" if best["step"] == step else (
                f"  (best {best['val_loss']:.4f} @ step {best['step']})"
            )
            print(
                f"VAL step {step}  train {total_loss:.4f}  val {val_loss:.4f}"
                f"  lr {lr_now:.3e}{marker}",
                flush=True,
            )
            # Save the best adapter THE MOMENT it is best, rather than
            # trusting it to survive pruning. It did not: val bottomed at
            # step 4000 and `keep_last=3` deleted that checkpoint while
            # `val_history.json` still named it, so the adapter the pipeline
            # loaded was the overfit one from step 6000. An adapter is ~40 MB
            # and this costs one copy per improvement.
            if best["step"] == step:
                best_dir = args.ckpt_dir / "adapter_best"
                model.save_pretrained(str(best_dir))
                print(f"  best adapter -> {best_dir}", flush=True)

            # Validations since the best one. Counted on the history rather
            # than a running tally so a resumed run inherits it correctly.
            since_best = sum(1 for r in val_history if r["step"] > best["step"])
            if args.patience and since_best >= args.patience:
                print(
                    f"\nEarly stop: {since_best} validations without "
                    f"improvement on {best['val_loss']:.4f} @ step "
                    f"{best['step']}. Training past this point is what made "
                    "the previous adapter worse than one already saved.",
                    flush=True,
                )
                stop_early = True

            # Written every time, so an interrupted run still says which
            # checkpoint was best rather than losing the record.
            args.ckpt_dir.mkdir(parents=True, exist_ok=True)
            (args.ckpt_dir / "val_history.json").write_text(
                json.dumps(
                    {"history": val_history, "best": best,
                     "best_checkpoint": f"ckpt_step_{best['step']}.pt",
                     "best_adapter": "adapter_best"},
                    indent=2,
                ),
                encoding="utf-8",
            )

        if step % args.save_every == 0 or step == args.max_steps or stop_early:
            state.step = step
            state.metrics_history.append({"step": step, "loss": total_loss})
            best_step = (min(val_history, key=lambda r: r["val_loss"])["step"]
                         if val_history else None)
            path = save_checkpoint(
                args.ckpt_dir, step, model, optimizer, scheduler,
                state=state, keep_last=args.keep_last, is_peft=True,
                protect=best_step,
            )
            print(f"  checkpoint -> {path}", flush=True)

        if stop_early:
            break

    final = args.ckpt_dir / "adapter_final"
    model.save_pretrained(str(final))

    # `rs_vqa_v1` finished its v2 run with no metrics.json at all, so the one
    # tool whose v1 weights were destroyed had no number of any kind. Held-out
    # loss is not a benchmark score, but it is a measurement, it is written
    # where every other run writes one, and it names which adapter it belongs
    # to - which is exactly what went wrong before.
    if val_history:
        best = min(val_history, key=lambda r: r["val_loss"])
        (args.ckpt_dir / "metrics.json").write_text(
            json.dumps({
                "val_loss_best": best["val_loss"],
                "val_loss_final": val_history[-1]["val_loss"],
                "best_step": best["step"],
                "steps_trained": val_history[-1]["step"],
                "n_val": best["n_val"],
                "stopped_early": stop_early,
                "note": (
                    "val_loss is held-out cross-entropy on the instruction "
                    "mix. It is NOT an RSVQA score and, measured on the "
                    "official RSVQA-LR test split, it did not predict one: "
                    "an adapter at val 0.1301 and one at val 0.2529 scored "
                    "0.6958 all-types EITHER WAY, and the higher-loss adapter "
                    "was marginally ahead on the published convention "
                    "(0.8947 vs 0.8851, overlapping CIs). Use this number to "
                    "spot divergence, not to choose which adapter to deploy - "
                    "run evaluation/rsvqa_official_eval.py for that."
                ),
            }, indent=2),
            encoding="utf-8",
        )
        print(f"  metrics -> {args.ckpt_dir / 'metrics.json'}")
        print(f"  BEST adapter: {args.ckpt_dir / 'adapter_best'} "
              f"(val {best['val_loss']:.4f} @ step {best['step']})")

    print(f"\nTraining complete. Adapter saved to {final}")
    print("Load it in the pipeline with rs_vqa_v1's adapter_path parameter.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
