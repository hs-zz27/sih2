"""VQA metrics (plan task 1.9).

Standard RSVQA-style scoring: normalised exact match overall and broken down
by answer type, plus abstention-aware coverage. Coverage matters because a
system that abstains on everything would otherwise score perfectly on the
items it did answer.
"""

from __future__ import annotations

import re
from collections import defaultdict

# Articles and filler that must not affect an exact-match comparison.
_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")

# Yes/no synonyms, normalised so "yep" scores as "yes".
_YES = {"yes", "yeah", "yep", "true", "correct"}
_NO = {"no", "nope", "false", "incorrect"}

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}


def normalise_answer(answer: str) -> str:
    """Lowercase, strip punctuation/articles, canonicalise yes-no and numbers."""
    text = _PUNCT.sub(" ", str(answer).lower())
    tokens = [t for t in _WS.split(text.strip()) if t and t not in _ARTICLES]
    tokens = [_NUMBER_WORDS.get(t, t) for t in tokens]
    joined = " ".join(tokens)
    if joined in _YES:
        return "yes"
    if joined in _NO:
        return "no"
    return joined


def exact_match(prediction: str, truth: str) -> bool:
    return normalise_answer(prediction) == normalise_answer(truth)


def score_vqa(
    predictions: list[dict], ground_truth: dict[str, dict]
) -> dict:
    """Score VQA predictions against ground truth keyed by item_id.

    Ground-truth entries look like {"answer": ..., "answer_type": ...}.
    Items present in the truth but missing from predictions count as wrong,
    so a truncated predictions file cannot inflate the score.
    """
    by_type_correct: dict[str, int] = defaultdict(int)
    by_type_total: dict[str, int] = defaultdict(int)

    correct = 0
    answered = 0
    abstained = 0
    missing = 0

    predicted_by_id = {p["item_id"]: p for p in predictions}

    for item_id, truth in ground_truth.items():
        answer_type = truth.get("answer_type", "unknown")
        by_type_total[answer_type] += 1

        pred = predicted_by_id.get(item_id)
        if pred is None:
            missing += 1
            continue

        if pred.get("abstained"):
            abstained += 1
            continue

        answered += 1
        if exact_match(pred.get("answer", ""), truth["answer"]):
            correct += 1
            by_type_correct[answer_type] += 1

    total = len(ground_truth)
    return {
        "n_items": total,
        "n_answered": answered,
        "n_abstained": abstained,
        "n_missing": missing,
        # Accuracy over ALL items: abstentions and omissions count against it.
        "accuracy": round(correct / total, 6) if total else 0.0,
        # Accuracy over answered items only, paired with coverage.
        "accuracy_when_answered": round(correct / answered, 6) if answered else 0.0,
        "coverage": round(answered / total, 6) if total else 0.0,
        "by_answer_type": {
            t: {
                "n": by_type_total[t],
                "accuracy": round(by_type_correct[t] / by_type_total[t], 6)
                if by_type_total[t]
                else 0.0,
            }
            for t in sorted(by_type_total)
        },
    }
