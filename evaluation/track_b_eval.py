"""Score a Track B adapter, and compare v0 against v1 (plan task 3.1).

The acceptance criterion is "improved metrics across VQA/caption; model
declines appropriately". Both halves need care to be worth anything.

## Improved compared to what, on what

v0 trained on RSVQA-LR alone; v1 trains on the full mix. Scoring v1 on the mix
val split and calling it an improvement over v0's old number would compare two
models on two different sets, which is the error this project has already
corrected twice. **Both adapters are scored on the identical split here**, and
results are broken down by source, because the two questions being asked are
different:

* on the **whu_opt_sar** and **refusal** rows, v1 is expected to win - it saw
  that distribution and v0 did not. That measures whether the new data taught
  anything, not whether v1 is a better model.
* on the **rsvqa_lr** rows, both models trained on that distribution, so this
  is the fair comparison and the one that can regress. Adding SAR and refusals
  to the mix could easily cost general VQA accuracy, and if it does, that is
  the finding.

## Generation is mirrored from the tool, not reimplemented

Same 4-bit config, same chat template, same greedy decode as
`satquery/tools/rs_vqa.py`. A benchmark that decodes differently from the
deployed path measures a model nobody ships.

`_ModelHandle` is a process-wide singleton and cannot hold two adapters, so
this module loads its own and frees it between arms.

Usage:
    python evaluation/track_b_eval.py \
        --adapters v0=checkpoints/killtest/adapter_final \
                   v1=checkpoints/track_b_v1/adapter_final
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.refusal import is_refusal, matched_pairs  # noqa: E402
from satquery.tools.rs_vqa import SYSTEM_PROMPT  # noqa: E402

REPORT = Path("docs/assets/refusal/track_b.json")
DEFAULT_MAX_NEW_TOKENS = 48


def normalise(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", str(text).lower())


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1.

    Exact match alone is too brittle for generated text - "yes" against "Yes,
    forest covers about 78% of the scene" scores 0 while being right - and too
    lenient nowhere. F1 is reported alongside exact match, not instead of it.
    """
    from collections import Counter

    pred, ref = normalise(prediction), normalise(reference)
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


class Adapter:
    """One base model + adapter, loaded and freed explicitly."""

    def __init__(self, base: Path, adapter: Path):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, BitsAndBytesConfig

        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ),
        )
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            str(base), local_files_only=True
        )
        model = AutoVLM.from_pretrained(
            str(base),
            quantization_config=quant,
            device_map={"": 0} if torch.cuda.is_available() else "cpu",
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = PeftModel.from_pretrained(model, str(adapter)).eval()

    def answer(self, image_path: Path, question: str) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        chat = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            },
        ]
        text = self.processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        batch = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.model.device)

        with self.torch.no_grad():
            generated = self.model.generate(
                **batch,
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                do_sample=False,   # deterministic, as the deployed tool is
            )
        prompt_len = batch["input_ids"].shape[1]
        return self.processor.decode(
            generated[0][prompt_len:], skip_special_tokens=True
        ).strip()

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def score(adapter: Adapter, examples: list[dict], root: Path) -> dict:
    predictions = {}
    for i, example in enumerate(examples):
        predictions[i] = adapter.answer(root / example["image"], example["question"])

    by_group: dict[str, list[tuple[dict, str]]] = defaultdict(list)
    for i, example in enumerate(examples):
        by_group[example.get("source", "unknown")].append(
            (example, predictions[i])
        )
        by_group[f"kind:{example.get('kind', 'vqa')}"].append(
            (example, predictions[i])
        )
        # Per-reason, because aggregate refusal recall hides the only
        # distinction that matters. On the v1 run it was 0.4118 overall,
        # which decomposes into 5/5 on the lexical categories and 2/12 on
        # the image-conditional one - two completely different verdicts
        # averaged into one uninformative number.
        if example.get("refusal_reason"):
            by_group[f"reason:{example['refusal_reason']}"].append(
                (example, predictions[i])
            )

    groups = {}
    for name, rows in sorted(by_group.items()):
        answerable = [(e, p) for e, p in rows if e.get("kind") != "refusal"]
        groups[name] = {
            "n": len(rows),
            "exact_match": (
                sum(
                    normalise(p) == normalise(e["answer"]) for e, p in answerable
                ) / len(answerable)
                if answerable else None
            ),
            "token_f1": (
                sum(token_f1(p, e["answer"]) for e, p in answerable)
                / len(answerable)
                if answerable else None
            ),
            "refusal_rate": sum(is_refusal(p) for _, p in rows) / len(rows),
        }

    refusals = [(e, predictions[i]) for i, e in enumerate(examples)
                if e.get("kind") == "refusal"]
    answerable = [(e, predictions[i]) for i, e in enumerate(examples)
                  if e.get("kind") != "refusal"]
    index = {id(e): predictions[i] for i, e in enumerate(examples)}
    pairs = matched_pairs(examples)
    pair_correct = sum(
        not is_refusal(index[id(a)]) and is_refusal(index[id(r)])
        for a, r in pairs
        if id(a) in index and id(r) in index
    )

    return {
        "n": len(examples),
        "overall": {
            "exact_match": (
                sum(normalise(p) == normalise(e["answer"]) for e, p in answerable)
                / len(answerable) if answerable else None
            ),
            "token_f1": (
                sum(token_f1(p, e["answer"]) for e, p in answerable)
                / len(answerable) if answerable else None
            ),
        },
        "refusal": {
            "refusal_recall": (
                sum(is_refusal(p) for _, p in refusals) / len(refusals)
                if refusals else None
            ),
            "false_refusal_rate": (
                sum(is_refusal(p) for _, p in answerable) / len(answerable)
                if answerable else None
            ),
            "n_matched_pairs": len(pairs),
            "lexical_shortcut_probe": (
                pair_correct / len(pairs) if pairs else None
            ),
        },
        "groups": groups,
        "samples": [
            {
                "question": e["question"],
                "expected": e["answer"][:120],
                "predicted": predictions[i][:120],
                "kind": e.get("kind"),
            }
            for i, e in list(enumerate(examples))[:8]
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("models/qwen25_vl_3b"))
    p.add_argument("--adapters", nargs="+", required=True,
                   help="name=path pairs, e.g. v0=checkpoints/killtest/adapter_final")
    p.add_argument("--data", type=Path, default=Path("data/instruct_mix"))
    p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--out", type=Path, default=REPORT)
    args = p.parse_args()

    examples = [
        json.loads(line)
        for line in (args.data / f"{args.split}.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    print(f"scoring {len(examples)} examples from {args.split}")

    results = {}
    for spec in args.adapters:
        name, _, path = spec.partition("=")
        adapter_path = Path(path)
        if not adapter_path.exists():
            print(f"  {name}: {adapter_path} not found, skipping", file=sys.stderr)
            continue
        print(f"\nloading {name} from {adapter_path}")
        adapter = Adapter(args.base, adapter_path)
        try:
            results[name] = score(adapter, examples, args.data)
        finally:
            adapter.close()

        row = results[name]
        print(f"  exact match {row['overall']['exact_match']:.4f}  "
              f"token F1 {row['overall']['token_f1']:.4f}")
        for key, value in row["refusal"].items():
            print(f"  {key:24s} "
                  f"{'n/a' if value is None else (f'{value:.4f}' if isinstance(value, float) else value)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "split": args.split,
                "n_examples": len(examples),
                "results": results,
                "note": (
                    "Both adapters are scored on the IDENTICAL split. The "
                    "whu_opt_sar and refusal rows favour v1 by construction - "
                    "it saw that distribution and v0 did not - so they measure "
                    "whether the new data taught anything. The rsvqa_lr rows "
                    "are the fair comparison, since both models trained on "
                    "that distribution, and are the ones that can regress."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
