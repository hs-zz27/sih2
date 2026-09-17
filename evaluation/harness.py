"""Evaluation harness v1 (plan tasks 1.8, 1.9).

Runs a benchmark manifest through the real controller, writes a schema-valid
predictions file, and scores it into a single JSON report.

A benchmark manifest is a JSON list of items:
    [{"item_id": "...", "images": ["a.tif"], "question": "...",
      "answer": "...", "answer_type": "count"}, ...]

`--dry-run` validates the manifest and reports what would run without loading
a single tool, which is the cheap way to catch a broken split file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from satquery.contracts.input_manifest import IngestMode
from satquery.controller.executor import CODE_VERSION
from satquery.controller.pipeline import Controller

from .metrics.all_tasks import score_caption, score_grounding, score_landcover
from .metrics.vqa import score_vqa
from .schemas import PredictionsFile

REQUIRED_FIELDS = {"item_id", "images"}


def load_benchmark(path: str | Path) -> list[dict]:
    """Load and structurally validate a benchmark manifest."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("benchmark manifest must be a JSON list")

    seen: set[str] = set()
    for i, item in enumerate(items):
        missing = REQUIRED_FIELDS - set(item)
        if missing:
            raise ValueError(f"item {i} is missing required fields: {sorted(missing)}")
        if not item["images"]:
            raise ValueError(f"item {item['item_id']} has no images")
        if item["item_id"] in seen:
            raise ValueError(f"duplicate item_id: {item['item_id']}")
        seen.add(item["item_id"])
    return items


def dry_run(items: list[dict], root: Path) -> dict:
    """Report what would run, and which inputs are missing, without running."""
    missing_files = []
    for item in items:
        for rel in item["images"]:
            if not (root / rel).exists():
                missing_files.append(f"{item['item_id']}: {rel}")

    return {
        "dry_run": True,
        "n_items": len(items),
        "n_missing_files": len(missing_files),
        "missing_files": missing_files[:50],
        "ready": not missing_files,
    }


def run_benchmark(
    items: list[dict],
    root: Path,
    benchmark: str,
    annotation_type: str = "vqa",
    controller: Controller | None = None,
    limit: int | None = None,
) -> PredictionsFile:
    """Execute every item through the real controller and collect predictions."""
    controller = controller or Controller()
    if limit is not None:
        items = items[:limit]

    predictions: list[dict] = []
    model_cards: dict[str, str] = {}

    for item in items:
        paths = [root / rel for rel in item["images"]]
        query = item.get("question") or item.get("query") or "Describe this image."
        try:
            trace = controller.run(
                paths,
                query,
                mode=IngestMode.BENCHMARK,
                benchmark=benchmark,
                run_id=f"bench_{item['item_id']}",
            )
        except Exception as exc:  # noqa: BLE001 - a failed item is a wrong item
            predictions.append(
                {
                    "item_id": item["item_id"],
                    "question": query,
                    "answer": "",
                    "confidence": 0.0,
                    "abstained": True,
                    "answer_type": item.get("answer_type"),
                    "error": str(exc),
                }
            )
            continue

        for step in trace.execution:
            model_cards.setdefault(step.tool, step.version)

        predictions.append(
            _to_prediction(annotation_type, item, query, trace)
        )

    return PredictionsFile(
        benchmark=benchmark,
        annotation_type=annotation_type,  # type: ignore[arg-type]
        code_version=CODE_VERSION,
        matrix_version=controller.matrix.version,
        model_cards=model_cards,
        n_items=len(predictions),
        predictions=predictions,
    )


def _to_prediction(annotation_type: str, item: dict, query: str, trace) -> dict:
    base = {
        "item_id": item["item_id"],
        "confidence": trace.confidence.final,
        "abstained": trace.abstained,
    }
    if annotation_type == "vqa":
        return {
            **base,
            "question": query,
            "answer": trace.answer,
            "answer_type": item.get("answer_type"),
        }
    if annotation_type == "caption":
        return {**base, "caption": trace.answer}
    if annotation_type == "grounding":
        boxes: list[dict] = []
        for step in trace.execution:
            for box in step.outputs.get("bounding_boxes", []) or []:
                boxes.append(box)
        return {**base, "referring_expression": query, "boxes": boxes}
    if annotation_type == "landcover":
        labels: list[str] = []
        for step in trace.execution:
            labels.extend(step.outputs.get("labels", []) or [])
        return {**base, "labels": labels, "scores": {}}
    raise ValueError(f"unknown annotation type: {annotation_type}")


def score(predictions: PredictionsFile, items: list[dict]) -> dict:
    """Score a predictions file against the benchmark's ground truth.

    Every annotation type is now covered (task 2.14), so a run produces a
    filled row rather than "not_implemented" for three of four types.
    """
    kind = predictions.annotation_type

    if kind == "vqa":
        truth = {
            i["item_id"]: {
                "answer": i.get("answer", ""),
                "answer_type": i.get("answer_type", "unknown"),
            }
            for i in items if "answer" in i
        }
        scorer = score_vqa
    elif kind == "caption":
        truth = {
            i["item_id"]: {
                "caption": i.get("caption", ""),
                "captions": i.get("captions"),
            }
            for i in items if ("caption" in i or "captions" in i)
        }
        scorer = score_caption
    elif kind == "grounding":
        truth = {i["item_id"]: {"box": i.get("box")} for i in items if i.get("box")}
        scorer = score_grounding
    elif kind == "landcover":
        truth = {
            i["item_id"]: {"labels": i.get("labels", [])}
            for i in items if "labels" in i
        }
        scorer = score_landcover
    else:
        return {"metric_status": "unknown_annotation_type", "type": kind}

    if not truth:
        return {"metric_status": "no_ground_truth", "type": kind}
    return {"metric_status": "ok", "type": kind, **scorer(predictions.predictions, truth)}


def evaluate(
    benchmark_path: str | Path,
    root: str | Path,
    benchmark: str,
    annotation_type: str = "vqa",
    limit: int | None = None,
    controller: Controller | None = None,
) -> dict[str, Any]:
    """One command, one JSON report."""
    started = time.perf_counter()
    root = Path(root)
    items = load_benchmark(benchmark_path)
    preds = run_benchmark(
        items, root, benchmark, annotation_type, controller=controller, limit=limit
    )
    scored = score(preds, items)

    return {
        "benchmark": benchmark,
        "annotation_type": annotation_type,
        "code_version": preds.code_version,
        "matrix_version": preds.matrix_version,
        "model_cards": preds.model_cards,
        "runtime_s": round(time.perf_counter() - started, 3),
        "metrics": scored,
        "predictions": json.loads(preds.model_dump_json())["predictions"],
    }
