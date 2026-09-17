"""VRSBench shared helpers, on synthetic rows shaped like the published files.

No imagery, checkpoints or torch.
"""

from __future__ import annotations

import collections
import json
import random

import pytest

from evaluation.metrics.vqa import normalise_answer
from evaluation.vrsbench_common import (
    parse_box,
    per_type_majority,
    pixel_box_to_percent,
    stratified_sample,
    train_global_constant,
)


def vqa_rows(n_per_type=(40, 25, 10)):
    rows, qid = [], 0
    for t, n in zip(("object quantity", "object existence", "object color"), n_per_type):
        for i in range(n):
            rows.append({"image_id": f"P{qid:04d}.png", "question": "q", "type": t,
                         "ground_truth": "Yes" if i % 3 else "2", "question_id": qid})
            qid += 1
    return rows


def original_draw(rows, sample, seed):
    """The inline algorithm vrsbench_eval.main() used when 0.2968 was measured."""
    by_t = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_t[r["type"]].append(i)
    rng = random.Random(seed)
    keep = []
    for _, idx in sorted(by_t.items()):
        k = max(1, round(len(idx) * sample / len(rows)))
        keep.extend(rng.sample(idx, min(k, len(idx))))
    keep.sort()
    return [rows[i] for i in keep]


class TestSampling:
    def test_reproduces_the_draw_behind_the_published_number(self):
        rows = vqa_rows()
        drawn, record = stratified_sample(rows, 30, seed=20260904)
        assert drawn == original_draw(rows, 30, 20260904)
        assert record["n_drawn"] == len(drawn)

    def test_every_type_keeps_its_share(self):
        drawn, _ = stratified_sample(vqa_rows(), 30, seed=1)
        counts = collections.Counter(r["type"] for r in drawn)
        assert counts == {"object quantity": 16, "object existence": 10, "object color": 4}

    def test_no_reduction_returns_input_and_no_record(self):
        rows = vqa_rows()
        assert stratified_sample(rows, None, seed=1) == (rows, None)
        assert stratified_sample(rows, 10_000, seed=1) == (rows, None)


class TestBaselines:
    def test_per_type_majority(self):
        rows = vqa_rows()
        gold = [normalise_answer(r["ground_truth"]) for r in rows]
        assert per_type_majority(rows, gold) == {
            "object color": "yes", "object existence": "yes", "object quantity": "yes",
        }

    def test_train_constant_counts_only_vqa_turns(self, tmp_path):
        train = [
            {"conversations": [{"from": "human", "value": "[vqa] how many?"},
                               {"from": "gpt", "value": "3"}]},
            {"conversations": [{"from": "human", "value": "[vqa] any ships?"},
                               {"from": "gpt", "value": "No"}]},
            {"conversations": [{"from": "human", "value": "[vqa] any roads?"},
                               {"from": "gpt", "value": "no"}]},
            {"conversations": [{"from": "human", "value": "[caption] describe"},
                               {"from": "gpt", "value": "3"}]},
            {"conversations": [{"from": "human", "value": "[refer] the ship"},
                               {"from": "gpt", "value": "3"}]},
        ]
        (tmp_path / "VRSBench_train.json").write_text(json.dumps(train))
        assert train_global_constant(tmp_path) == "no"

    def test_train_constant_absent_file(self, tmp_path):
        assert train_global_constant(tmp_path) is None


class TestBoxes:
    def test_parses_the_published_format(self):
        assert parse_box("{<25><40><33><60>}") == {"xmin": 25, "ymin": 40, "xmax": 33, "ymax": 60}

    def test_reversed_corners_are_ordered(self):
        assert parse_box("{<33><60><25><40>}") == {"xmin": 25, "ymin": 40, "xmax": 33, "ymax": 60}

    def test_rejects_other_text(self):
        with pytest.raises(ValueError):
            parse_box("[25, 40, 33, 60]")

    def test_pixel_to_percent_on_a_non_square_image(self):
        box = pixel_box_to_percent({"x0": 100, "y0": 50, "x1": 300, "y1": 150}, 400, 200)
        assert box == {"xmin": 25.0, "ymin": 25.0, "xmax": 75.0, "ymax": 75.0}
