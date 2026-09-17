"""Score caption_v1 and grounding_v1 on VRSBench captioning and referring.

Why this exists
---------------
PS-26167 assigns VRSBench three roles - "single-image captioning, grounding,
and visual question answering". Only VQA was ever measured
(`vrsbench_eval.py`, zero-shot 0.2968 on a 7,999-question stratified draw).
The other two had no evaluator, so the benchmark was a third covered however
the traceability row was worded. This closes the gap in code; it produces a
number only when run on a machine with the val imagery and the checkpoints.

It is a ZERO-SHOT measurement
-----------------------------
caption_v1 trained on RSICD and grounding_v1 on DIOR-RSVG. Neither has seen
VRSBench, whose captions are long multi-sentence GPT-assisted descriptions and
whose referring boxes are often small vehicles in 512 px DOTA crops. Expect
low numbers, and compare them only against other zero-shot results.

Metric caveats, stated before anyone quotes a figure
----------------------------------------------------
* Caption BLEU-4 here is the project's sentence-level BLEU
  (`evaluation.metrics.all_tasks.bleu`), the same one behind every other caption
  number in this repository. It is **not** the pycocoevalcap corpus BLEU the
  VRSBench paper reports, so it is not comparable to that table.
* Referring accuracy is Acc@0.5 / Acc@0.7 on the single predicted box, split by
  the dataset's own `unique` flag exactly as the paper splits it. The 0-100
  normalised boxes are compared directly: IoU is scale-invariant only when
  both axes are normalised the same way, which they are.

Usage (on the GPU box, with the val imagery extracted)
------------------------------------------------------
    python evaluation/vrsbench_tasks.py --task caption \\
        --data data/vrsbench --out artifacts/vrsbench/caption.json --sample 2000
    python evaluation/vrsbench_tasks.py --task referring \\
        --data data/vrsbench --out artifacts/vrsbench/referring.json

Predictions are appended to `<out>.predictions.jsonl` after every item and
reused on restart, so a killed run resumes rather than starting over.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics.all_tasks import bleu, iou  # noqa: E402
from evaluation.rsvqa_official_eval import wilson  # noqa: E402
from evaluation.vrsbench_common import (  # noqa: E402
    load_rows,
    parse_box,
    pixel_box_to_percent,
    stratified_sample,
)

CaptionPredictor = Callable[[Path, str], str]
ReferringPredictor = Callable[[Path, str], "dict | None"]


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"n": n, "accuracy": round(k / n, 6) if n else 0.0,
            "ci95": [round(lo, 4), round(hi, 4)]}


def score_captions(rows: list[dict], predictions: dict[int, str]) -> dict:
    """Sentence BLEU-4 against the single VRSBench reference caption."""
    scores = []
    missing = 0
    for r in rows:
        pred = predictions.get(r["question_id"])
        if not pred:
            missing += 1
            scores.append(0.0)
            continue
        scores.append(bleu(pred, [r["ground_truth"]]))
    n = len(rows)
    return {
        "n_items": n,
        "n_missing_or_empty": missing,
        "bleu4_sentence_mean": round(sum(scores) / n, 6) if n else 0.0,
        "metric_note": "project sentence-level BLEU-4; not pycocoevalcap corpus BLEU",
    }


def score_referring(rows: list[dict], predictions: dict[int, dict | None]) -> dict:
    """Acc@0.5 / Acc@0.7 / mIoU, overall and split by the `unique` flag."""
    groups: dict[str, list[float]] = {"all": [], "unique": [], "non_unique": []}
    no_box = 0
    for r in rows:
        target = parse_box(r["ground_truth"])
        pred = predictions.get(r["question_id"])
        value = iou(pred, target) if pred else 0.0
        no_box += pred is None
        groups["all"].append(value)
        groups["unique" if r.get("unique") else "non_unique"].append(value)
    out: dict = {"n_items": len(rows), "n_no_box": no_box}
    for name, values in groups.items():
        n = len(values)
        out[name] = {
            "miou": round(sum(values) / n, 6) if n else 0.0,
            "acc@0.5": _rate(sum(v >= 0.5 for v in values), n),
            "acc@0.7": _rate(sum(v >= 0.7 for v in values), n),
        }
    return out


def load_predictions(path: Path) -> dict[int, object]:
    if not path.exists():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            done[rec["question_id"]] = rec["prediction"]
    return done


def run_predictions(
    rows: list[dict], images: Path, predict: Callable, cache: Path
) -> dict[int, object]:
    """Predict every row not already in `cache`, appending as it goes."""
    done = load_predictions(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with cache.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            if r["question_id"] in done:
                continue
            pred = predict(images / r["image_id"], r["question"])
            done[r["question_id"]] = pred
            fh.write(json.dumps({"question_id": r["question_id"], "prediction": pred}) + "\n")
            fh.flush()
            if (i + 1) % 250 == 0:
                rate = (time.time() - started) / (i + 1)
                print(f"[vrsbench] {i + 1}/{len(rows)}  {rate:.3f}s/item", flush=True)
    return done


def tool_predictor(task: str) -> Callable:
    """Adapters from the production tools to (image, question) -> prediction.

    Goes through `ingest` in BENCHMARK mode, the same path a benchmark PNG
    takes through the API, so preprocessing is the served preprocessing.
    Raises if the registry resolved to a stub: scoring a placeholder would
    publish a number about nothing.
    """
    from satquery.contracts.input_manifest import IngestMode
    from satquery.ingest import ingest
    from satquery.tools.stubs import REGISTRY

    name = {"caption": "caption_v1", "referring": "grounding_v1"}[task]
    tool = REGISTRY[name]

    def run(image: Path, question: str):
        manifest = ingest([image], mode=IngestMode.BENCHMARK, benchmark="vrsbench")
        result = tool.run(manifest, {"_query": question})
        if result.confidence_method == "stub":
            raise RuntimeError(
                f"{name} resolved to its stub - set its checkpoint before scoring"
            )
        data = result.payload.data
        if task == "caption":
            return str(data.get("caption", ""))
        boxes = data.get("bounding_boxes") or []
        if not boxes:
            return None
        width, height = data["image_size"]
        return pixel_box_to_percent(boxes[0], width, height)

    return run


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", choices=["caption", "referring"], required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample", type=int, help="stratified subsample size")
    p.add_argument("--seed", type=int, default=20260904)
    args = p.parse_args()

    rows = load_rows(args.data, args.task)
    # Referring queries are stratified by the unique/non-unique split the
    # paper reports. Captions all share type "caption" and there is one per
    # image, so the same call reduces to a seeded simple random draw.
    key = "unique" if args.task == "referring" else "type"
    rows, sampling = stratified_sample(rows, args.sample, args.seed, key=key)
    cache = args.out.with_suffix(".predictions.jsonl")
    preds = run_predictions(rows, args.data / "Images_val", tool_predictor(args.task), cache)

    scored = score_captions(rows, preds) if args.task == "caption" else score_referring(rows, preds)
    report = {
        "benchmark": f"VRSBench {args.task} (val/eval split)",
        "setting": "ZERO-SHOT - the tool has never trained on VRSBench",
        "tool": {"caption": "caption_v1", "referring": "grounding_v1"}[args.task],
        "sampling": sampling,
        "result": scored,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(scored, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
