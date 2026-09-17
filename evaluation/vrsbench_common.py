"""Pure helpers shared by the VRSBench evaluators. No torch, no model loading.

Split out of `vrsbench_eval.py`, where the baselines and the stratified draw
lived inline in `main()` and so could not be tested without a GPU, a 3.9 GB
image archive and a QLoRA adapter. Everything here runs on a few rows of JSON.

File shapes, read from the published files (HuggingFace xiang709/VRSBench,
checked 2026-09-17):

    VRSBench_EVAL_vqa.json        {image_id, question, ground_truth, question_id, type}
    VRSBench_EVAL_Cap.json        {image_id, question, ground_truth, question_id, type="caption"}
    VRSBench_EVAL_referring.json  {image_id, question, ground_truth="{<x1><y1><x2><y2>}",
                                   question_id, type="ref", unique, obj_cls, ...}

Referring boxes are corner coordinates normalised to 0-100.
"""

from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path

from evaluation.metrics.vqa import normalise_answer

EVAL_FILES = {
    "vqa": "VRSBench_EVAL_vqa.json",
    "caption": "VRSBench_EVAL_Cap.json",
    "referring": "VRSBench_EVAL_referring.json",
}

_BOX = re.compile(r"\{\s*<\s*(-?[\d.]+)\s*>\s*<\s*(-?[\d.]+)\s*>\s*<\s*(-?[\d.]+)\s*>\s*<\s*(-?[\d.]+)\s*>\s*\}")


def load_rows(data: Path, task: str) -> list[dict]:
    return json.loads((data / EVAL_FILES[task]).read_text(encoding="utf-8"))


def stratified_sample(
    rows: list[dict], n: int, seed: int, key: str = "type"
) -> tuple[list[dict], dict | None]:
    """Proportional, seeded draw of about `n` rows, keeping each stratum's share.

    Returns the rows in their original order and a record of the draw, or the
    input unchanged and None when no reduction was asked for.
    """
    if not n or n >= len(rows):
        return rows, None
    by_key: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_key[str(r.get(key))].append(i)
    rng = random.Random(seed)
    keep: list[int] = []
    for _, idx in sorted(by_key.items()):
        k = max(1, round(len(idx) * n / len(rows)))
        keep.extend(rng.sample(idx, min(k, len(idx))))
    keep.sort()
    record = {
        "stratified_subsample": True,
        "n_drawn": len(keep),
        "n_full_eval_set": len(rows),
        "fraction": round(len(keep) / len(rows), 4),
        "seed": seed,
        "stratified_by": key,
        "method": f"proportional by {key}, seeded",
        "note": "REDUCED PRECISION relative to a full-set run. The task, prompt, "
                "decode and metric are unchanged; only the number of items differs.",
    }
    return [rows[i] for i in keep], record


def per_type_majority(rows: list[dict], gold: list[str]) -> dict[str, str]:
    """Most common normalised answer within each question type of THESE rows.

    Fitted on the evaluation rows themselves, so it is an optimistic upper
    bound for a constant answerer, never a fair baseline.
    """
    out = {}
    for t in sorted({r["type"] for r in rows}):
        c = collections.Counter(gold[i] for i, r in enumerate(rows) if r["type"] == t)
        out[t] = c.most_common(1)[0][0]
    return out


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


def parse_box(text: str) -> dict:
    """`"{<25><40><33><60>}"` -> corner box on the 0-100 scale."""
    m = _BOX.search(str(text))
    if not m:
        raise ValueError(f"not a VRSBench box: {text!r}")
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    return {"xmin": min(x0, x1), "ymin": min(y0, y1), "xmax": max(x0, x1), "ymax": max(y0, y1)}


def pixel_box_to_percent(box: dict, width: int, height: int) -> dict:
    """A pixel-corner box (x0, y0, x1, y1) on the VRSBench 0-100 scale."""
    return {
        "xmin": 100.0 * box["x0"] / width,
        "ymin": 100.0 * box["y0"] / height,
        "xmax": 100.0 * box["x1"] / width,
        "ymax": 100.0 * box["y1"] / height,
    }
