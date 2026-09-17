"""Deterministic change answers from a pair of semantic change maps.

The CDVQA measurement (docs/phase1-status.md) found the system scoring 0.0000
on the PS's prescribed change-VQA benchmark, for a structural reason: seven of
its eight question types ask which *class* changed, and nothing in the system
produced per-class semantic change.

This module is the arithmetic half of the fix. Given two class maps - one per
date, over the six SECOND change classes plus "unchanged" - every CDVQA
question type is answerable in closed form. **Verified against CDVQA's own
train annotations: all eight types derive at 1.0000 exact match from
ground-truth maps** (`evaluation/cdvqa_oracle.py`).

That matters for more than CDVQA. It means the learned component is *only*
semantic change segmentation, and the answer is exact arithmetic over its
output rather than generated text - which is the system's own axiom that
quantitative answers come from computation, not from a language model
estimating a number the pixels already determine.

Nothing here is learned and nothing here raises: pure array functions over two
integer class maps, testable against hand-computed answers.

The rules, each confirmed rather than assumed:

* **Date qualifiers select a date.** "in the first image" / "pre-event" /
  "pre-change" means the t1 map alone; "second" / "post-event" / "post-change"
  means t2 alone; an unqualified question sums both. Assuming the sum
  everywhere scored 0.935 on `change_or_not`; reading the qualifier scores
  1.0000, and every disagreement was a class absent on the named date.
* **Ratios are fractions of the whole scene**, not of the changed area. Of the
  two candidate denominators, the scene gives 1.0000 and the changed area
  0.6199.
* **"Changed to what" is the majority destination class** over pixels holding
  the subject class at the source date.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# SECOND's label palette. Index 0 is "unchanged"; the rest are the six change
# classes, named as CDVQA's answer vocabulary spells them so that a derived
# answer is directly comparable to the ground truth with no aliasing step.
PALETTE: tuple[tuple[tuple[int, int, int], str], ...] = (
    ((255, 255, 255), "unchanged"),
    ((0, 0, 255), "water"),
    ((128, 128, 128), "NVG_surface"),
    ((0, 128, 0), "low_vegetation"),
    ((0, 255, 0), "trees"),
    ((128, 0, 0), "buildings"),
    ((255, 0, 0), "playgrounds"),
)

CLASSES: tuple[str, ...] = tuple(name for _, name in PALETTE)
CHANGE_CLASSES: tuple[str, ...] = CLASSES[1:]
UNCHANGED = 0

# How CDVQA words each class in a question, longest first so that
# "low vegetation" is matched before "vegetation" could be.
SUBJECT_PHRASES: tuple[tuple[str, str], ...] = (
    ("non-vegetated ground surface", "NVG_surface"),
    ("low vegetation", "low_vegetation"),
    ("playgrounds", "playgrounds"),
    ("buildings", "buildings"),
    ("water", "water"),
    ("trees", "trees"),
)

_T1 = re.compile(r"first image|pre-event|pre-change|before")
_T2 = re.compile(r"second image|post-event|post-change|after")

# Question shapes, in the order CDVQA names them.
_RATIO_TYPES = re.compile(
    r"(change (?:ratio|percentage|proportion)|how much area) of (?:the )?\w"
)
_RATIO = re.compile(
    # "What is the [non-]change ratio of the imagery?"
    r"(?:non-)?change ratio of the imagery"
    # "What is the percentage of changed/unchanged/non-change areas|regions?"
    r"|percentage of (?:changed|unchanged|non-change) (?:areas|regions)"
    # "How much of the area has [not] changed?"
    r"|how much of the (?:area|imagery) has"
)
_TO_WHAT = re.compile(r"changed to|change[d]? into")
_LARGEST = re.compile(r"largest")
_SMALLEST = re.compile(r"smallest")
_INCREASE = re.compile(r"increase")
_DECREASE = re.compile(r"decrease")
# "how much has NOT changed" inverts the measured fraction.
_INVERTED = re.compile(r"\bnot\b|unchanged|non-change")


@dataclass(frozen=True)
class Question:
    """What a CDVQA question is asking, once parsed."""

    kind: str
    subject: str | None
    scope: str  # "t1", "t2" or "both"


def decode_label(rgb: np.ndarray) -> np.ndarray:
    """Map an RGB label image to class indices.

    Colours outside the palette become `UNCHANGED` rather than raising: a
    resampled or JPEG-compressed label is a data problem to notice in the
    caller, not a crash here.
    """
    out = np.zeros(rgb.shape[:2], dtype=np.int8)
    for index, (colour, _) in enumerate(PALETTE):
        if index == UNCHANGED:
            continue
        out[(rgb == np.array(colour, dtype=rgb.dtype)).all(-1)] = index
    return out


def class_areas(labels: np.ndarray) -> dict[str, int]:
    """Pixel count per class, including `unchanged`."""
    return {name: int((labels == i).sum()) for i, name in enumerate(CLASSES)}


def parse_question(question: str) -> Question:
    """Read the class, the date scope and the question shape out of the text."""
    text = question.lower()

    subject = None
    for phrase, name in SUBJECT_PHRASES:
        if phrase in text:
            subject = name
            break

    scope = "t1" if _T1.search(text) else "t2" if _T2.search(text) else "both"

    if _TO_WHAT.search(text):
        kind = "change_to_what"
    elif _LARGEST.search(text):
        kind = "largest_change"
    elif _SMALLEST.search(text):
        kind = "smallest_change"
    elif _INCREASE.search(text):
        kind = "increase_or_not"
    elif _DECREASE.search(text):
        kind = "decrease_or_not"
    elif subject is not None and _RATIO_TYPES.search(text):
        kind = "change_ratio_types"
    elif _RATIO.search(text):
        kind = "change_ratio"
    elif subject is not None:
        kind = "change_or_not"
    else:
        kind = "unknown"

    return Question(kind=kind, subject=subject, scope=scope)


def decile_bin(percent: float) -> str:
    """CDVQA's answer vocabulary for a percentage.

    Exactly zero has its own answer token, distinct from the `0_to_10` bin.
    """
    if percent <= 0:
        return "0"
    low = min(int(percent // 10) * 10, 90)
    return f"{low}_to_{low + 10}"


def answer(question: str, t1: np.ndarray, t2: np.ndarray) -> str | None:
    """Answer one CDVQA question from two class maps.

    Returns None when the question shape is not one this module derives, so
    the caller defers rather than guessing - the same contract the
    deterministic index path already follows.
    """
    parsed = parse_question(question)
    a1, a2 = class_areas(t1), class_areas(t2)
    both = {name: a1[name] + a2[name] for name in CLASSES}
    scoped = {"t1": a1, "t2": a2, "both": both}[parsed.scope]

    if parsed.kind == "change_ratio":
        changed = float((t1 != UNCHANGED).sum()) / max(t1.size, 1) * 100.0
        if _INVERTED.search(question.lower()):
            changed = 100.0 - changed
        return decile_bin(changed)

    if parsed.kind == "change_ratio_types" and parsed.subject:
        # Fraction of the whole scene, not of the changed area - measured, see
        # the module docstring.
        return decile_bin(scoped[parsed.subject] / max(t1.size, 1) * 100.0)

    if parsed.kind == "change_or_not" and parsed.subject:
        return "yes" if scoped[parsed.subject] > 0 else "no"

    if parsed.kind == "increase_or_not" and parsed.subject:
        return "yes" if a2[parsed.subject] > a1[parsed.subject] else "no"

    if parsed.kind == "decrease_or_not" and parsed.subject:
        return "yes" if a2[parsed.subject] < a1[parsed.subject] else "no"

    if parsed.kind in ("largest_change", "smallest_change"):
        areas = {name: n for name, n in scoped.items() if name in CHANGE_CLASSES and n > 0}
        if not areas:
            # Nothing changed, so no class changed most or least. CDVQA still
            # carries an answer here - its generator takes an argmax over a
            # row of zeros and returns whichever class its own ordering puts
            # first - but that is a property of the generator, not of the
            # image, and reproducing it would be scoring by imitation.
            return None
        extreme = (max if parsed.kind == "largest_change" else min)(areas.values())
        tied = [name for name, n in areas.items() if n == extreme]
        if len(tied) > 1:
            # Two classes changed by exactly the same area. The measurement
            # does not discriminate, so neither do we.
            return None
        return tied[0]

    if parsed.kind == "change_to_what" and parsed.subject:
        source, destination = (t2, t1) if parsed.scope == "t2" else (t1, t2)
        selected = (source == CLASSES.index(parsed.subject)) & (t1 != UNCHANGED)
        if not selected.any():
            return None
        values, counts = np.unique(destination[selected], return_counts=True)
        return CLASSES[int(values[int(np.argmax(counts))])]

    return None
