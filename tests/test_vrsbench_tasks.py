"""VRSBench caption and referring evaluator: scoring, resume, stub refusal.

No imagery, checkpoints or torch: the scoring and resume logic is what can be
wrong silently, and it is testable on a few dicts.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from evaluation.vrsbench_tasks import (
    load_predictions,
    run_predictions,
    score_captions,
    score_referring,
    tool_predictor,
)


class TestScoring:
    def test_captions(self):
        rows = [{"question_id": 0, "ground_truth": "a red car parked near a road"},
                {"question_id": 1, "ground_truth": "two ships in a harbour"}]
        exact = score_captions(rows, {0: "a red car parked near a road", 1: "two ships in a harbour"})
        assert exact["bleu4_sentence_mean"] == pytest.approx(1.0)
        partial = score_captions(rows, {0: "a red car parked near a road"})
        assert partial["n_missing_or_empty"] == 1
        assert partial["bleu4_sentence_mean"] == pytest.approx(0.5)

    def test_referring_split_by_unique(self):
        rows = [
            {"question_id": 0, "ground_truth": "{<0><0><50><50>}", "unique": True},
            {"question_id": 1, "ground_truth": "{<50><50><100><100>}", "unique": False},
            {"question_id": 2, "ground_truth": "{<0><0><10><10>}", "unique": False},
        ]
        preds = {
            0: {"xmin": 0, "ymin": 0, "xmax": 50, "ymax": 50},      # IoU 1.0
            1: {"xmin": 50, "ymin": 50, "xmax": 90, "ymax": 100},   # IoU 0.8
            2: None,                                                # no box
        }
        r = score_referring(rows, preds)
        assert r["n_no_box"] == 1
        assert r["all"]["acc@0.5"]["accuracy"] == pytest.approx(2 / 3, abs=1e-6)
        assert r["all"]["acc@0.7"]["accuracy"] == pytest.approx(2 / 3, abs=1e-6)
        assert r["unique"]["acc@0.7"]["accuracy"] == 1.0
        assert r["non_unique"]["acc@0.5"]["accuracy"] == 0.5
        assert r["non_unique"]["miou"] == pytest.approx(0.4)


class TestResume:
    def test_a_killed_run_does_not_predict_twice(self, tmp_path):
        rows = [{"question_id": i, "image_id": f"{i}.png", "question": "q"} for i in range(5)]
        cache = tmp_path / "out.predictions.jsonl"
        calls = []

        def flaky(image, question):
            calls.append(image.name)
            if len(calls) == 3:
                raise KeyboardInterrupt
            return f"caption for {image.name}"

        with pytest.raises(KeyboardInterrupt):
            run_predictions(rows, tmp_path, flaky, cache)
        assert set(load_predictions(cache)) == {0, 1}

        calls.clear()
        done = run_predictions(rows, tmp_path, lambda image, q: calls.append(image.name) or "x", cache)
        assert calls == ["2.png", "3.png", "4.png"]
        assert set(done) == {0, 1, 2, 3, 4}


def test_refuses_to_score_a_stub(tmp_path):
    """Without checkpoints the registry serves stubs; scoring one must fail loudly."""
    image = tmp_path / "P0001_0001.png"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(image)
    predict = tool_predictor("caption")
    try:
        predict(image, "Describe the image in detail")
    except RuntimeError as exc:
        assert "stub" in str(exc)
    else:
        pytest.skip("caption_v1 has a real checkpoint here; the stub guard is not exercised")
