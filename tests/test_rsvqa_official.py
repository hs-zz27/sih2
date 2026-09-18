"""Resolving the official RSVQA-LR split - the step that was never committed.

`evaluation/rsvqa_official_eval.py` reads test_resolved.json and
train_majority.json; nothing in the repository produced them, so the headline
0.8947 could not be reproduced. Fixtures here mirror the real release's shape
(verified against Zenodo 6344334 on 2026-09-18): inactive rows carry only
`id` and `active`, answers are joined by `question_id`, and the types are
presence / comp / count / rural_urban.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.prepare.rsvqa_official import load_split, majority_by_type  # noqa: E402


def write_split(src: Path, split: str, rows: list[dict], inactive: int = 2) -> None:
    """rows: [{id, img, type, question, answer, active}]"""
    questions = [{"id": i, "active": False} for i in range(inactive)]
    answers = [{"id": i, "active": False} for i in range(inactive)]
    for row in rows:
        questions.append({
            "id": row["id"], "img_id": row["img"], "type": row["type"],
            "question": row["question"], "answers_ids": [row["id"]],
            "active": row.get("active", True),
        })
        answers.append({
            "id": row["id"], "question_id": row["id"], "answer": row["answer"],
            "active": row.get("active", True),
        })
    src.mkdir(parents=True, exist_ok=True)
    (src / f"LR_split_{split}_questions.json").write_text(json.dumps({"questions": questions}))
    (src / f"LR_split_{split}_answers.json").write_text(json.dumps({"answers": answers}))


ROWS = [
    {"id": 10, "img": 1, "type": "presence", "question": "Is there water", "answer": "yes"},
    {"id": 11, "img": 1, "type": "count", "question": "How many roads", "answer": "3"},
    {"id": 12, "img": 2, "type": "comp", "question": "More trees than roads", "answer": "no"},
    {"id": 13, "img": 2, "type": "rural_urban", "question": "Rural or urban", "answer": "urban"},
]


def test_inactive_rows_are_dropped(tmp_path):
    write_split(tmp_path, "test", ROWS + [
        {"id": 99, "img": 3, "type": "presence", "question": "dropped", "answer": "yes",
         "active": False},
    ])
    rows = load_split(tmp_path, "test")
    assert len(rows) == len(ROWS)
    assert "dropped" not in {r["question"] for r in rows}


def test_rows_carry_the_fields_the_evaluator_reads(tmp_path):
    write_split(tmp_path, "test", ROWS)
    row = load_split(tmp_path, "test")[0]
    assert set(row) == {"img", "question", "answer", "type"}
    # The evaluator opens Images_LR/<img>.tif, so img must stay the raw id.
    assert row["img"] == 1


def test_answers_are_joined_by_question_id_not_order(tmp_path):
    write_split(tmp_path, "test", ROWS)
    answers = json.loads((tmp_path / "LR_split_test_answers.json").read_text())
    answers["answers"].reverse()
    (tmp_path / "LR_split_test_answers.json").write_text(json.dumps(answers))
    by_question = {r["question"]: r["answer"] for r in load_split(tmp_path, "test")}
    assert by_question["Rural or urban"] == "urban"
    assert by_question["How many roads"] == "3"


def test_a_question_with_no_active_answer_is_skipped(tmp_path):
    write_split(tmp_path, "test", ROWS)
    answers = json.loads((tmp_path / "LR_split_test_answers.json").read_text())
    for answer in answers["answers"]:
        if answer.get("question_id") == 10:
            answer["active"] = False
    (tmp_path / "LR_split_test_answers.json").write_text(json.dumps(answers))
    assert len(load_split(tmp_path, "test")) == len(ROWS) - 1


def test_majority_is_per_type_and_normalised(tmp_path):
    rows = [
        {"type": "presence", "answer": "Yes"}, {"type": "presence", "answer": "yes"},
        {"type": "presence", "answer": "no"},
        {"type": "count", "answer": "0"}, {"type": "count", "answer": "0"},
        {"type": "count", "answer": "7"},
    ]
    assert majority_by_type(rows) == {"presence": "yes", "count": "0"}


def test_missing_release_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path, "test")
