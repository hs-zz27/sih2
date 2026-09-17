"""Score Track B adapters on the OFFICIAL RSVQA-LR test split.

Why this exists
---------------
Every Track B VQA number this project has published was measured on
`data/rsvqa_lr_2k` - a 2,000-question HuggingFace redistribution of the RSVQA-LR
**validation** split, cut 90/10 by `training/prepare/instruction_mix.py` into
1,793 train / 207 val. That slice has three properties that make it
incomparable to the published literature:

* it is the **validation** split, not the official **test** split;
* **n = 207**, so its 95% interval is about +/-6.5 points;
* it **includes count questions** (27.5% of it), which the RSVQA-LR literature
  conventionally excludes because their answers are open numerals.

Phase 2 established the sharpest consequence: a train-fitted per-type constant
scores 0.6473 on that slice, **identical** to the deployed `track_b_v2`
headline. The benchmark cannot distinguish the deployed model from a constant.

This module scores the official test split instead: **10,004 questions over
100 images**, from Zenodo record 6344334 (CC-BY-4.0), all four question types
present and typed by the dataset itself rather than inferred from wording.

What it reports
---------------
Both conventions, always, side by side:

* **all types** - every question, the metric this project has been quoting;
* **published convention** - presence + comparison + rural-urban, count
  excluded, which is what the 89-93% figures in the literature measure.

And, per the discipline Phase 2 section 7.4 established, a **train-fitted
per-type constant baseline** alongside every model number. A model score
without that baseline is uninterpretable on a skewed benchmark.

Read-only
---------
This module writes to `--out` and nowhere else. It loads checkpoints, never
saves them, and touches no `metrics.json`, no `configs/`, and nothing under
`docs/`. Generation mirrors `satquery/tools/rs_vqa.py` exactly - same 4-bit
config, same system prompt, same chat template, same greedy decode - so it
measures the model that is actually deployed.

Usage
-----
    python evaluation/rsvqa_official_eval.py \\
        --base models/qwen25_vl_3b \\
        --data data/rsvqa_lr_official \\
        --arms v2=checkpoints/track_b_v2/adapter_final \\
               v3=checkpoints/track_b_v3/adapter_final \\
        --out artifacts/phase4_rsvqa/official_test.json
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics.vqa import normalise_answer  # noqa: E402
from evaluation.track_b_eval import Adapter  # noqa: E402

PUBLISHED_TYPES = ("presence", "comp", "rural_urban")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - honest at the small n some types have."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


class BaseOnly(Adapter):
    """The identical load path, minus PeftModel - to score the un-adapted base."""

    def __init__(self, base: Path):
        import torch
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
        self.processor = AutoProcessor.from_pretrained(str(base), local_files_only=True)
        self.model = AutoVLM.from_pretrained(
            str(base),
            quantization_config=quant,
            device_map={"": 0} if torch.cuda.is_available() else "cpu",
            local_files_only=True,
            trust_remote_code=False,
        ).eval()


def load_split(data: Path) -> tuple[list[dict], dict[str, str]]:
    """The resolved official test rows, and the train-fitted per-type constant."""
    rows = json.loads((data / "test_resolved.json").read_text(encoding="utf-8"))
    constant = json.loads((data / "train_majority.json").read_text(encoding="utf-8"))
    return rows, constant


def score(hits: list[bool], rows: list[dict], constant_hits: list[bool]) -> dict:
    """Both conventions, per-type detail, and the contingency against the constant."""
    types = sorted({r["type"] for r in rows})
    by_type = {}
    for t in types:
        idx = [i for i, r in enumerate(rows) if r["type"] == t]
        k = sum(hits[i] for i in idx)
        lo, hi = wilson(k, len(idx))
        ck = sum(constant_hits[i] for i in idx)
        by_type[t] = {
            "n": len(idx),
            "correct": k,
            "accuracy": k / len(idx),
            "ci95": [round(lo, 4), round(hi, 4)],
            "constant_accuracy": ck / len(idx),
        }

    pub_idx = [i for i, r in enumerate(rows) if r["type"] in PUBLISHED_TYPES]
    pub_k = sum(hits[i] for i in pub_idx)
    pub_lo, pub_hi = wilson(pub_k, len(pub_idx))
    all_lo, all_hi = wilson(sum(hits), len(rows))

    both = sum(1 for h, c in zip(hits, constant_hits) if h and c)
    model_only = sum(1 for h, c in zip(hits, constant_hits) if h and not c)
    const_only = sum(1 for h, c in zip(hits, constant_hits) if c and not h)
    disc = model_only + const_only
    chi2 = ((abs(model_only - const_only) - 1) ** 2 / disc) if disc else 0.0

    return {
        "all_types": {
            "n": len(rows),
            "accuracy": sum(hits) / len(rows),
            "ci95": [round(all_lo, 4), round(all_hi, 4)],
        },
        "published_convention": {
            "types": list(PUBLISHED_TYPES),
            "n": len(pub_idx),
            "micro_accuracy": pub_k / len(pub_idx),
            "ci95": [round(pub_lo, 4), round(pub_hi, 4)],
            "macro_average_accuracy": sum(
                by_type[t]["accuracy"] for t in PUBLISHED_TYPES if t in by_type
            ) / len([t for t in PUBLISHED_TYPES if t in by_type]),
        },
        "by_type": by_type,
        "vs_constant": {
            "both_correct": both,
            "model_only_correct": model_only,
            "constant_only_correct": const_only,
            "mcnemar_chi2_cc": round(chi2, 3),
            "significant_at_0.05": chi2 > 3.841,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--arms", nargs="+", required=True, help="name=path, or name=BASE")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, help="first N rows only (diagnostic)")
    args = p.parse_args()

    rows, constant = load_split(args.data)
    if args.limit:
        rows = rows[: args.limit]
    images = args.data / "Images_LR"

    gold = [normalise_answer(r["answer"]) for r in rows]
    constant_hits = [
        normalise_answer(constant[r["type"]]) == g for r, g in zip(rows, gold)
    ]
    print(f"[official] {len(rows)} questions, {len({r['img'] for r in rows})} images")
    print(f"[official] types: {dict(collections.Counter(r['type'] for r in rows))}")
    print(f"[official] train-fitted per-type constant: "
          f"{sum(constant_hits) / len(rows):.4f} all types", flush=True)

    out = {
        "benchmark": "RSVQA-LR official test split",
        "source": "Zenodo 10.5281/zenodo.6344334, CC-BY-4.0",
        "n_questions": len(rows),
        "n_images": len({r["img"] for r in rows}),
        "type_counts": dict(collections.Counter(r["type"] for r in rows)),
        "constant_baseline": {
            "answers": constant,
            "all_types_accuracy": sum(constant_hits) / len(rows),
            "published_convention_accuracy": (
                sum(c for c, r in zip(constant_hits, rows)
                    if r["type"] in PUBLISHED_TYPES)
                / sum(1 for r in rows if r["type"] in PUBLISHED_TYPES)
            ),
        },
        "arms": {},
        "timing": {},
    }
    if args.out.exists():
        try:
            prev = json.loads(args.out.read_text(encoding="utf-8"))
            if prev.get("n_questions") == len(rows):
                out = prev
                print(f"[official] resuming; have {list(out['arms'])}", flush=True)
        except Exception:
            pass

    for spec in args.arms:
        name, _, path = spec.partition("=")
        if name in out["arms"]:
            print(f"[official] {name}: already scored, skipping", flush=True)
            continue
        t0 = time.time()
        handle = BaseOnly(args.base) if path == "BASE" else Adapter(args.base, Path(path))
        print(f"[official] loaded {name} ({time.time() - t0:.1f}s)", flush=True)

        hits, t_gen = [], time.time()
        try:
            for i, r in enumerate(rows):
                pred = handle.answer(images / f"{r['img']}.tif", r["question"])
                hits.append(normalise_answer(pred) == gold[i])
                if (i + 1) % 500 == 0:
                    print(f"[official]   {name} {i+1}/{len(rows)}  "
                          f"running acc {sum(hits)/len(hits):.4f}  "
                          f"{(time.time()-t_gen)/(i+1):.3f}s/q", flush=True)
        finally:
            handle.close()
            gc.collect()

        row = score(hits, rows, constant_hits)
        out["arms"][name] = row
        out["timing"][name] = {
            "generate_s": round(time.time() - t_gen, 1),
            "s_per_question": round((time.time() - t_gen) / len(rows), 4),
            "adapter": path,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

        print(f"[official] {name}: all-types {row['all_types']['accuracy']:.4f}  "
              f"published-convention {row['published_convention']['micro_accuracy']:.4f}  "
              f"(model-only {row['vs_constant']['model_only_correct']} / "
              f"constant-only {row['vs_constant']['constant_only_correct']})", flush=True)

    print(f"[official] done -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
