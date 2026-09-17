"""Score Track B adapters on the VRSBench VQA evaluation set.

Why this exists
---------------
VRSBench is prescribed by PS-26167 and was the last prescribed benchmark with
no result. `docs/00` section 3.6 L11 recorded it as unevaluable because "its
142k rows reference images that live in the separate DOTA and DIOR datasets".
**That premise is wrong**: the VRSBench HuggingFace repository hosts the
imagery directly as `Images_val.zip` (3.977 GB) and `Images_train.zip`
(8.359 GB). Only the val images are needed to evaluate, and only those were
downloaded.

Evaluation set: **37,409 questions over 9,349 images**, twelve question types,
matching the published 37,408 / 9,350 to within one row.

This is a ZERO-SHOT measurement
-------------------------------
SatQuery has never trained on VRSBench. The honest comparison is therefore
against other models' **zero-shot** VRSBench numbers, not their fine-tuned
ones. For GeoChat the published pair is **40.8 zero-shot / 60.6 fine-tuned**;
only the first is comparable to what this script produces.

Baselines
---------
Two, because neither alone is honest on a skewed benchmark:

* **train-fitted global constant** - the most common answer in
  `VRSBench_train.json`, applied to every test question. No peeking. This is
  the floor a model must clear to have demonstrated anything.
* **test-fitted per-type constant** - the most common answer *within each
  question type of the evaluation set itself*. This **peeks at the test set**
  and is therefore an **optimistic upper bound** on what a constant could
  achieve, not a fair baseline. It is reported because a model that fails to
  beat it has not learned the task, and labelled so it cannot be misread.

Read-only
---------
Writes to `--out` and nowhere else. Loads checkpoints, never saves them.
Generation mirrors `satquery/tools/rs_vqa.py` exactly.

Usage
-----
    python evaluation/vrsbench_eval.py \\
        --base models/qwen25_vl_3b --data data/vrsbench \\
        --arms v3=checkpoints/track_b_v3/adapter_final \\
        --out artifacts/phase4_vrsbench/vqa.json
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import math
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics.vqa import normalise_answer  # noqa: E402
from evaluation.rsvqa_official_eval import BaseOnly, wilson  # noqa: E402
from evaluation.track_b_eval import Adapter  # noqa: E402


def train_global_constant(data: Path) -> str | None:
    """Most common VQA answer in VRSBench_train.json. No test peeking."""
    path = data / "VRSBench_train.json"
    if not path.exists():
        return None
    counts: collections.Counter = collections.Counter()
    for row in json.loads(path.read_text(encoding="utf-8")):
        conv = row.get("conversations") or []
        for i, turn in enumerate(conv):
            if turn.get("from") != "human":
                continue
            if "[vqa]" not in (turn.get("value") or "").lower():
                continue
            if i + 1 < len(conv) and conv[i + 1].get("from") == "gpt":
                counts[normalise_answer(conv[i + 1].get("value", ""))] += 1
    return counts.most_common(1)[0][0] if counts else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--sample", type=int,
                   help="stratified subsample: N questions, proportional by type. "
                        "Reduces precision; never changes the task. Recorded in the output.")
    p.add_argument("--seed", type=int, default=20260904)
    args = p.parse_args()

    rows = json.loads((args.data / "VRSBench_EVAL_vqa.json").read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]

    sampling = None
    if args.sample and args.sample < len(rows):
        # Proportional stratified draw: every question type keeps its share of the
        # eval set, so per-type accuracy stays estimable and the overall figure is
        # not shifted by dropping a type. Seeded, so the draw is reproducible.
        by_t: dict[str, list[int]] = collections.defaultdict(list)
        for i, r in enumerate(rows):
            by_t[r["type"]].append(i)
        rng = random.Random(args.seed)
        keep: list[int] = []
        for t, idx in sorted(by_t.items()):
            k = max(1, round(len(idx) * args.sample / len(rows)))
            keep.extend(rng.sample(idx, min(k, len(idx))))
        keep.sort()
        sampling = {
            "stratified_subsample": True,
            "n_drawn": len(keep),
            "n_full_eval_set": len(rows),
            "fraction": round(len(keep) / len(rows), 4),
            "seed": args.seed,
            "method": "proportional by question type, seeded",
            "note": "REDUCED PRECISION relative to a full-set arm. Per-type CIs widen "
                    "accordingly and are reported. The task, prompt, decode and metric "
                    "are unchanged; only the number of questions differs.",
        }
        rows = [rows[i] for i in keep]

    images = args.data / "Images_val"
    gold = [normalise_answer(r["ground_truth"]) for r in rows]
    types = sorted({r["type"] for r in rows})

    # honest floor
    tgc = train_global_constant(args.data)
    tgc_hits = [tgc == g for g in gold] if tgc else None

    # optimistic ceiling on constants — fitted on the eval set itself
    per_type_majority = {}
    for t in types:
        c = collections.Counter(gold[i] for i, r in enumerate(rows) if r["type"] == t)
        per_type_majority[t] = c.most_common(1)[0][0]
    ptc_hits = [per_type_majority[r["type"]] == g for r, g in zip(rows, gold)]

    print(f"[vrsbench] {len(rows)} questions over {len({r['image_id'] for r in rows})} images")
    print(f"[vrsbench] types: {dict(collections.Counter(r['type'] for r in rows).most_common())}")
    if tgc:
        print(f"[vrsbench] train-fitted GLOBAL constant '{tgc}': "
              f"{sum(tgc_hits)/len(rows):.4f}  (honest floor)")
    print(f"[vrsbench] test-fitted PER-TYPE constant: {sum(ptc_hits)/len(rows):.4f}  "
          f"(OPTIMISTIC — peeks at the eval set)", flush=True)

    out = {
        "benchmark": "VRSBench VQA (val/eval split)",
        "source": "HuggingFace xiang709/VRSBench, Images_val.zip sha256 67f99c1d…",
        "setting": "ZERO-SHOT — SatQuery has never trained on VRSBench",
        "n_questions": len(rows),
        "n_images": len({r["image_id"] for r in rows}),
        "type_counts": dict(collections.Counter(r["type"] for r in rows)),
        "baselines": {
            "train_fitted_global_constant": {
                "answer": tgc,
                "accuracy": (sum(tgc_hits) / len(rows)) if tgc else None,
                "note": "honest floor; fitted on VRSBench_train.json, no test peeking",
            },
            "test_fitted_per_type_constant": {
                "answers": per_type_majority,
                "accuracy": sum(ptc_hits) / len(rows),
                "note": "OPTIMISTIC UPPER BOUND — fitted on the evaluation set itself. "
                        "Not a fair baseline; a model failing to beat it has not learned the task.",
            },
        },
        "arms": {},
        "timing": {},
    }
    if sampling:
        out["sampling"] = sampling
    if args.out.exists():
        try:
            prev = json.loads(args.out.read_text(encoding="utf-8"))
            if prev.get("n_questions") == len(rows):
                out = prev
                print(f"[vrsbench] resuming; have {list(out['arms'])}", flush=True)
        except Exception:
            pass

    for spec in args.arms:
        name, _, path = spec.partition("=")
        if name in out["arms"]:
            print(f"[vrsbench] {name}: already scored, skipping", flush=True)
            continue
        # Mid-arm checkpoint. An arm is ~3-12 h; writing only on completion means a
        # kill at 88% loses everything, which is exactly what happened on 2026-09-04.
        part = args.out.parent / f"{args.out.stem}.{name}.partial.json"
        strict: list[bool] = []
        lenient: list[bool] = []
        if part.exists():
            try:
                pj = json.loads(part.read_text(encoding="utf-8"))
                if pj.get("n_rows") == len(rows) and pj.get("arm") == name:
                    strict = list(pj["strict"])
                    lenient = list(pj["lenient"])
                    print(f"[vrsbench] {name}: resuming mid-arm at {len(strict)}/{len(rows)}",
                          flush=True)
            except Exception:
                strict, lenient = [], []
        if len(strict) >= len(rows):
            print(f"[vrsbench] {name}: partial already complete", flush=True)

        def _save_partial() -> None:
            part.parent.mkdir(parents=True, exist_ok=True)
            tmp = part.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "arm": name, "n_rows": len(rows), "done": len(strict),
                "strict": strict, "lenient": lenient,
            }), encoding="utf-8")
            tmp.replace(part)  # atomic: never leave a half-written checkpoint

        t0 = time.time()
        done0 = len(strict)
        if done0 < len(rows):
            handle = BaseOnly(args.base) if path == "BASE" else Adapter(args.base, Path(path))
            print(f"[vrsbench] loaded {name} ({time.time()-t0:.1f}s)", flush=True)
        else:
            handle = None

        t_gen = time.time()
        try:
            for i in range(done0, len(rows)):
                r = rows[i]
                pred = normalise_answer(handle.answer(images / r["image_id"], r["question"]))
                strict.append(pred == gold[i])
                lenient.append(gold[i] in re.findall(r"[a-z0-9]+", pred) or pred == gold[i])
                if (i + 1) % 250 == 0:
                    _save_partial()
                if (i + 1) % 1000 == 0:
                    n_new = i + 1 - done0
                    print(f"[vrsbench]   {name} {i+1}/{len(rows)}  "
                          f"acc {sum(strict)/len(strict):.4f}  "
                          f"{(time.time()-t_gen)/max(n_new,1):.3f}s/q", flush=True)
            _save_partial()
        finally:
            if handle is not None:
                handle.close()
            gc.collect()

        by_type = {}
        for t in types:
            idx = [i for i, r in enumerate(rows) if r["type"] == t]
            k = sum(strict[i] for i in idx)
            lo, hi = wilson(k, len(idx))
            by_type[t] = {
                "n": len(idx), "accuracy": k / len(idx), "ci95": [round(lo, 4), round(hi, 4)],
                "test_fitted_constant": sum(ptc_hits[i] for i in idx) / len(idx),
            }
        lo, hi = wilson(sum(strict), len(rows))
        both = sum(1 for a, b in zip(strict, ptc_hits) if a and b)
        m_only = sum(1 for a, b in zip(strict, ptc_hits) if a and not b)
        c_only = sum(1 for a, b in zip(strict, ptc_hits) if b and not a)
        disc = m_only + c_only

        out["arms"][name] = {
            "overall_accuracy": sum(strict) / len(rows),
            "ci95": [round(lo, 4), round(hi, 4)],
            "lenient_contains_accuracy": sum(lenient) / len(rows),
            "by_type": by_type,
            "vs_test_fitted_constant": {
                "model_only_correct": m_only, "constant_only_correct": c_only,
                "both_correct": both,
                "mcnemar_chi2_cc": round(((abs(m_only - c_only) - 1) ** 2 / disc) if disc else 0.0, 3),
            },
        }
        out["timing"][name] = {
            "generate_s": round(time.time() - t_gen, 1),
            "s_per_question": round((time.time() - t_gen) / len(rows), 4),
            "adapter": path,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[vrsbench] {name}: accuracy {sum(strict)/len(rows):.4f} "
              f"(lenient {sum(lenient)/len(rows):.4f})", flush=True)

    print(f"[vrsbench] done -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
