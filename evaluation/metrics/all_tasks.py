"""Metrics for every annotation type (plan task 2.14).

Phase 1 scored VQA only and declared the rest "not_implemented", which was
honest but left the harness unable to produce a row for most of the nine
tasks. This adds caption, grounding and land-cover metrics so a single run
yields one report with every row filled.

Where a metric is a simplification of the standard one, it says so rather
than borrowing the standard name and implying equivalence.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# --- Captioning -------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu(prediction: str, references: list[str], max_n: int = 4) -> float:
    """Sentence-level BLEU-4 with add-one smoothing on higher orders.

    This is a genuine BLEU implementation, but sentence-level and
    single-prediction; corpus BLEU as reported in captioning papers aggregates
    counts across the corpus first and will not match this exactly. Reported
    as `bleu4_sentence_mean` for that reason.

    Known inflation: add-one smoothing above unigrams means two *unrelated*
    short captions sharing only a stopword still score ~0.25 rather than 0.
    The smoothing is deliberate - an unsmoothed 4-gram zero collapses the
    score for short but reasonable captions - but it makes absolute values
    optimistic. Compare models against each other, not against a paper's
    corpus BLEU.
    """
    pred = tokenize(prediction)
    if not pred or not references:
        return 0.0
    refs = [tokenize(r) for r in references if str(r).strip()]
    if not refs:
        return 0.0

    log_precision = 0.0
    for n in range(1, max_n + 1):
        pred_ngrams = _ngrams(pred, n)
        if not pred_ngrams:
            return 0.0
        best = Counter()
        for ref in refs:
            ref_ngrams = _ngrams(ref, n)
            for gram, count in ref_ngrams.items():
                best[gram] = max(best[gram], count)
        overlap = sum(min(c, best[g]) for g, c in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        # Add-one smoothing above unigrams: an unsmoothed zero at n=4 would
        # collapse the whole score for a short but reasonable caption.
        if n == 1:
            if overlap == 0:
                return 0.0
            precision = overlap / total
        else:
            precision = (overlap + 1) / (total + 1)
        log_precision += math.log(precision) / max_n

    closest = min(refs, key=lambda r: (abs(len(r) - len(pred)), len(r)))
    brevity = 1.0 if len(pred) > len(closest) else math.exp(
        1 - len(closest) / max(len(pred), 1)
    )
    return brevity * math.exp(log_precision)


def score_caption(predictions: list[dict], ground_truth: dict[str, dict]) -> dict:
    scores, answered, abstained, missing = [], 0, 0, 0
    by_id = {p["item_id"]: p for p in predictions}

    for item_id, truth in ground_truth.items():
        pred = by_id.get(item_id)
        if pred is None:
            missing += 1
            scores.append(0.0)
            continue
        if pred.get("abstained"):
            abstained += 1
            scores.append(0.0)
            continue
        answered += 1
        refs = truth.get("captions") or [truth.get("caption", "")]
        scores.append(bleu(pred.get("caption", ""), refs))

    total = len(ground_truth)
    return {
        "n_items": total,
        "n_answered": answered,
        "n_abstained": abstained,
        "n_missing": missing,
        "bleu4_sentence_mean": round(sum(scores) / total, 6) if total else 0.0,
        "coverage": round(answered / total, 6) if total else 0.0,
    }


# --- Grounding --------------------------------------------------------------


def iou(a: dict, b: dict) -> float:
    """Intersection over union of two axis-aligned boxes."""
    ax0, ay0, ax1, ay1 = a["xmin"], a["ymin"], a["xmax"], a["ymax"]
    bx0, by0, bx1, by1 = b["xmin"], b["ymin"], b["xmax"], b["ymax"]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def score_grounding(
    predictions: list[dict], ground_truth: dict[str, dict],
    thresholds: tuple[float, ...] = (0.5, 0.7),
) -> dict:
    """Acc@IoU for referring expressions.

    Referring expressions have one target, so the highest-scoring predicted
    box is compared against it. Taking the best-matching box instead would
    reward a model for spraying boxes across the image.
    """
    best_ious, answered, abstained, missing = [], 0, 0, 0
    by_id = {p["item_id"]: p for p in predictions}

    for item_id, truth in ground_truth.items():
        target = truth.get("box")
        if not target:
            continue
        pred = by_id.get(item_id)
        if pred is None:
            missing += 1
            best_ious.append(0.0)
            continue
        if pred.get("abstained") or not pred.get("boxes"):
            abstained += 1
            best_ious.append(0.0)
            continue
        answered += 1
        boxes = pred["boxes"]
        top = max(boxes, key=lambda b: b.get("score") or 0.0)
        best_ious.append(iou(top, target))

    n = len(best_ious)
    result = {
        "n_items": n,
        "n_answered": answered,
        "n_abstained": abstained,
        "n_missing": missing,
        "miou": round(sum(best_ious) / n, 6) if n else 0.0,
    }
    for t in thresholds:
        hits = sum(1 for v in best_ious if v >= t)
        result[f"acc@{t}"] = round(hits / n, 6) if n else 0.0
    return result


# --- Land cover (multi-label) ----------------------------------------------


def score_landcover(predictions: list[dict], ground_truth: dict[str, dict]) -> dict:
    """Micro/macro F1 over multi-label class sets.

    Land cover is multi-label, so exact-set accuracy would be needlessly harsh
    - predicting 3 of 4 correct classes is not the same as predicting none.
    """
    by_id = {p["item_id"]: p for p in predictions}
    tp = fp = fn = 0
    per_class: dict[str, list[int]] = {}
    answered = abstained = missing = 0

    for item_id, truth in ground_truth.items():
        expected = set(truth.get("labels", []))
        pred = by_id.get(item_id)
        if pred is None:
            missing += 1
            predicted: set = set()
        elif pred.get("abstained"):
            abstained += 1
            predicted = set()
        else:
            answered += 1
            predicted = set(pred.get("labels", []))

        for label in expected | predicted:
            counts = per_class.setdefault(label, [0, 0, 0])
            if label in expected and label in predicted:
                counts[0] += 1
                tp += 1
            elif label in predicted:
                counts[1] += 1
                fp += 1
            else:
                counts[2] += 1
                fn += 1

    def f1(t: int, p: int, n: int) -> float:
        denom = 2 * t + p + n
        return (2 * t / denom) if denom else 0.0

    macro = [f1(*c) for c in per_class.values()]
    total = len(ground_truth)
    return {
        "n_items": total,
        "n_answered": answered,
        "n_abstained": abstained,
        "n_missing": missing,
        "micro_f1": round(f1(tp, fp, fn), 6),
        "macro_f1": round(sum(macro) / len(macro), 6) if macro else 0.0,
        "n_classes_seen": len(per_class),
    }


SCORERS = {
    "vqa": None,  # provided by evaluation.metrics.vqa.score_vqa
    "caption": score_caption,
    "grounding": score_grounding,
    "landcover": score_landcover,
}
