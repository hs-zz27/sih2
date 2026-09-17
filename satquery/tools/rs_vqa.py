"""`rs_vqa_v1` backed by the QLoRA adapter from Track B (plan task 1.7).

Closes the last part of 1.7's criterion - *"adapter trains, loads, and answers
through the real pipeline"*. Training and loading were proven separately; this
is the component that makes a real fine-tuned model answer inside the trace.

Activation is explicit and opt-in via environment variables:

    SATQUERY_VQA_BASE       path to the base model directory
    SATQUERY_VQA_ADAPTER    path to the trained LoRA adapter
    SATQUERY_VQA_CPU_DTYPE  bfloat16 (default) or float32, CPU only

With a CUDA device the base loads in 4-bit NF4, the configuration every
published number was measured on. Without one it loads unquantised on the CPU
(bitsandbytes' 4-bit kernels are CUDA-only), which needs about 7.5 GB of RAM in
bfloat16 and answers in tens of seconds rather than one. The answers come from
the same adapter; the precision differs, and every CPU answer says so in its
warnings.

If either path is unset, or torch/peft/transformers are missing, the registry
keeps the stub.
That is deliberate: CI and every developer without a GPU must still get a
green test suite, and a silently-degraded model answering real questions would
be worse than an obvious stub.

Confidence comes from the model's own token log-probabilities
(`confidence_method="logprob"`), not a hardcoded constant - it is the `model`
component of the three-part confidence score, and it must move with the
model's actual certainty for the combined score to mean anything.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from satquery.contracts.input_manifest import InputManifest
from satquery.contracts.tool_result import ToolPayload, ToolResult
from satquery.tools.base import ToolProtocol
from satquery.tools.provenance import record
from satquery.tools.imaging import to_rgb_preview
from satquery.tools.sidecars import readable_safetensors

TOOL_NAME = "rs_vqa"
TOOL_VERSION = "1.0.0-qlora"

ENV_BASE = "SATQUERY_VQA_BASE"
ENV_ADAPTER = "SATQUERY_VQA_ADAPTER"
ENV_CPU_DTYPE = "SATQUERY_VQA_CPU_DTYPE"

# Answers are short in this domain ("rural", "5", "yes"). A tight cap keeps
# latency and VRAM predictable and discourages the model from rambling.
DEFAULT_MAX_NEW_TOKENS = 48

SYSTEM_PROMPT = (
    "You are a remote-sensing image analyst. Answer only from what is visible "
    "in the imagery. If the image does not support an answer, say so."
)


class VQAPayload(ToolPayload):
    data: dict[str, Any]


class _ModelHandle:
    """Lazily loaded, process-wide base model plus adapter.

    Loading costs ~20 s and several GiB, so it happens once on first use and
    is shared. Guarded by a lock because FastAPI serves from a thread pool.
    """

    _instance: "_ModelHandle | None" = None
    _lock = threading.Lock()

    def __init__(self, base: Path, adapter: Path):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:  # transformers < 5
            from transformers import AutoModelForVision2Seq as AutoVLM

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(base), local_files_only=True)
        self.on_cpu = not torch.cuda.is_available()
        self.precision = load_precision(on_cpu=self.on_cpu)
        model = AutoVLM.from_pretrained(
            str(base),
            local_files_only=True,
            trust_remote_code=False,
            **model_load_kwargs(torch, on_cpu=self.on_cpu),
        )
        # The adapter is ours, produced by training/track_b_vlm_qlora.py, so
        # loading it is not the third-party-code risk that trust_remote_code is.
        self.model = PeftModel.from_pretrained(model, str(adapter))
        self.model.eval()
        self.adapter_path = str(adapter)
        self.base_path = str(base)
        # The adapter only, not the base. The adapter is what this project
        # trained and what changes between runs; the base is gigabytes of
        # third-party weights whose digest is already recorded in
        # configs/model_lock.json by scripts/fetch_models.py. Hashing it on
        # every cold start would restate a number that is already on disk and
        # cost the demo its warm-up budget.
        record("rs_vqa_v1", adapter)

    @classmethod
    def get(cls, base: Path, adapter: Path) -> "_ModelHandle":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(base, adapter)
        return cls._instance


def is_available() -> tuple[bool, str]:
    """Whether the real VQA tool can run, and why not if it cannot."""
    base = os.getenv(ENV_BASE)
    adapter = os.getenv(ENV_ADAPTER)
    if not base or not adapter:
        return False, f"{ENV_BASE} and {ENV_ADAPTER} are not both set"
    if not Path(base).exists():
        return False, f"base model not found: {base}"
    if not Path(adapter).exists():
        return False, f"adapter not found: {adapter}"
    # Readable, not merely present. The recovered adapter was 99.99% NUL and
    # this function answered "ready", so the tool was selected and then died
    # inside the loader. See satquery/tools/sidecars.py.
    ok, reason = readable_safetensors(Path(adapter))
    if not ok:
        return False, reason
    for module in ("torch", "peft", "transformers"):
        try:
            __import__(module)
        except ImportError:
            return False, f"{module} is not installed"
    return True, "ready"


CPU_PRECISION_WARNING = (
    "rs_vqa_v1 is running unquantised on the CPU ({precision}); its published "
    "accuracy was measured on the 4-bit GPU build, so answers can differ slightly"
)


def load_precision(on_cpu: bool) -> str:
    """The precision the base model is loaded in, as recorded in warnings."""
    if not on_cpu:
        return "nf4-4bit"
    choice = os.getenv(ENV_CPU_DTYPE, "bfloat16").lower()
    return choice if choice in ("bfloat16", "float32") else "bfloat16"


def model_load_kwargs(torch, on_cpu: bool) -> dict:
    """`from_pretrained` keyword arguments for the device actually present.

    Separate from the handle so both branches are testable without weights.
    """
    if not on_cpu:
        from transformers import BitsAndBytesConfig

        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                ),
            ),
            "device_map": {"": 0},
        }
    # bitsandbytes' 4-bit path needs CUDA, so the CPU loads full weights.
    # bfloat16 halves the memory of float32 (~7.5 GB instead of ~15 GB for
    # the 3B model), which is what makes a 16 GB machine viable at all.
    # low_cpu_mem_usage streams the checkpoint in rather than materialising a
    # random-init copy first, which would double the peak.
    return {
        "torch_dtype": getattr(torch, load_precision(on_cpu=True)),
        "low_cpu_mem_usage": True,
    }


class RSVQATool(ToolProtocol):
    """Answers a question about one image using the fine-tuned adapter."""

    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        warnings: list[str] = []

        # The executor injects the user's question at run time under a
        # reserved key. It is deliberately NOT a plan parameter: the
        # capability matrix governs tunable parameters, and the query is
        # input data, already recorded verbatim elsewhere in the trace.
        question = str(params.get("_query") or "").strip()
        if not question:
            question = "Describe what is visible in this image."
            warnings.append("no question supplied; fell back to a description prompt")

        base = Path(os.environ[ENV_BASE])
        adapter = Path(os.environ[ENV_ADAPTER])
        handle = _ModelHandle.get(base, adapter)
        torch = handle.torch

        image_meta = manifest.images[0]
        image, provenance = to_rgb_preview(image_meta)

        chat = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            },
        ]
        text = handle.processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        batch = handle.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(handle.model.device)

        with torch.no_grad():
            generated = handle.model.generate(
                **batch,
                max_new_tokens=int(params.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
                do_sample=False,               # deterministic: same input, same answer
                output_scores=True,
                return_dict_in_generate=True,
            )

        prompt_len = batch["input_ids"].shape[1]
        new_tokens = generated.sequences[0][prompt_len:]
        answer = handle.processor.decode(new_tokens, skip_special_tokens=True).strip()

        confidence = _mean_token_probability(torch, generated.scores)

        if not answer:
            answer = "The image does not support an answer to that question."
            warnings.append("model produced an empty answer; abstained in wording")
        if handle.on_cpu:
            warnings.append(CPU_PRECISION_WARNING.format(precision=handle.precision))

        payload = VQAPayload(
            data={
                "answer": answer,
                "question": question,
                "image_provenance": provenance,
                "n_generated_tokens": int(new_tokens.shape[0]),
            }
        )

        return ToolResult(
            tool=TOOL_NAME,
            version=TOOL_VERSION,
            payload=payload,
            artifacts=[],
            confidence=confidence,
            confidence_method="logprob",
            model_card=f"{Path(handle.base_path).name} ({handle.precision}) + "
                       f"QLoRA adapter {Path(handle.adapter_path).name}",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=warnings,
        )

    def run_batch(
        self, manifests: list[InputManifest], params: dict[str, Any]
    ) -> list[ToolResult]:
        # Sequential by design: batching a VLM on a 6 GB GPU is the fastest
        # route to an OOM, and throughput optimisation is task 2.13.
        return [self.run(m, params) for m in manifests]


def _mean_token_probability(torch, scores) -> float:
    """Mean probability of the tokens the model actually chose.

    A greedy decode picks the argmax at each step; averaging those
    probabilities gives a calibrated-ish scalar in (0, 1] that falls when the
    model is unsure. It is uncalibrated - temperature scaling is task 3.3 -
    so it feeds the confidence combiner rather than being reported as a
    probability of correctness.
    """
    if not scores:
        return 0.0
    probs = []
    for step_scores in scores:
        step_probs = torch.softmax(step_scores[0].float(), dim=-1)
        probs.append(float(step_probs.max()))
    if not probs:
        return 0.0
    return round(sum(probs) / len(probs), 6)
