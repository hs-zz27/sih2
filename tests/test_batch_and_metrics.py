"""Batch execution and all-task metrics (plan tasks 2.13, 2.14)."""

from __future__ import annotations

import pytest

from evaluation.metrics.all_tasks import (
    bleu, iou, score_caption, score_grounding, score_landcover, tokenize,
)
from satquery.ingest import ingest
from satquery.tools.batch import benchmark, run_batch


class TestBatchExecution:
    def test_results_align_with_inputs(self, msi_6band):
        m = ingest([msi_6band])
        results, report = run_batch("index_engine_v1", [m, m, m],
                                    {"write_artifacts": False})
        assert len(results) == 3 == report.n_items
        assert report.n_ok == 3

    def test_one_failure_does_not_lose_the_batch(self, msi_6band, monkeypatch):
        """A single bad item must not discard 199 good ones."""
        from satquery.tools.stubs import REGISTRY

        good = ingest([msi_6band])
        calls = {"n": 0}
        original = REGISTRY["index_engine_v1"].run

        def flaky(manifest, params):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return original(manifest, params)

        monkeypatch.setattr(REGISTRY["index_engine_v1"], "run", flaky)
        results, report = run_batch("index_engine_v1", [good, good, good],
                                    {"write_artifacts": False})
        assert report.n_ok == 2 and report.n_failed == 1
        assert results[1] is None
        assert results[0] is not None and results[2] is not None

    def test_failed_slot_is_none_preserving_index_alignment(self, msi_6band, monkeypatch):
        """Dropping failures would mis-attribute every later answer."""
        from satquery.tools.stubs import REGISTRY

        m = ingest([msi_6band])
        monkeypatch.setattr(
            REGISTRY["index_engine_v1"], "run",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("x")),
        )
        results, report = run_batch("index_engine_v1", [m, m])
        assert results == [None, None]
        assert report.failures[0]["index"] == 0

    def test_throughput_reported(self, msi_6band):
        m = ingest([msi_6band])
        _, report = run_batch("index_engine_v1", [m], {"write_artifacts": False})
        d = report.to_dict()
        assert d["items_per_second"] > 0
        assert d["seconds_per_item"] > 0

    def test_empty_batch_is_safe(self):
        results, report = run_batch("index_engine_v1", [])
        assert results == [] and report.n_items == 0
        assert report.items_per_second == 0.0

    def test_benchmark_covers_several_tools(self, msi_6band):
        m = ingest([msi_6band])
        out = benchmark(["caption_v1", "landcover_v1"], [m, m])
        assert set(out["tools"]) == {"caption_v1", "landcover_v1"}
        assert out["n_items"] == 2


class TestCaptionMetric:
    def test_identical_caption_scores_high(self):
        assert bleu("a river runs through farmland",
                    ["a river runs through farmland"]) > 0.9

    def test_unrelated_caption_scores_far_below_a_matching_one(self):
        """Not zero: a shared stopword ("a") gives real unigram precision, and
        add-one smoothing lifts the higher orders. That inflation is a known
        property of smoothed sentence-BLEU on short captions, so the test
        asserts the ordering rather than an absolute floor."""
        ref = ["a river runs through farmland"]
        unrelated = bleu("aircraft on a runway", ref)
        matching = bleu("a river runs through farmland", ref)
        assert unrelated < 0.35
        assert matching > 2.5 * unrelated

    def test_zero_when_no_tokens_shared_at_all(self):
        assert bleu("aircraft runway", ["river farmland"]) == 0.0

    def test_empty_prediction_scores_zero(self):
        assert bleu("", ["something"]) == 0.0

    def test_brevity_penalty_applies(self):
        long_ref = ["a river runs through farmland near a small village"]
        assert bleu("a river", long_ref) < bleu("a river runs through farmland", long_ref)

    def test_abstention_counts_against_score(self):
        truth = {"a": {"caption": "a river"}}
        result = score_caption([{"item_id": "a", "abstained": True}], truth)
        assert result["bleu4_sentence_mean"] == 0.0
        assert result["n_abstained"] == 1

    def test_missing_prediction_counts_as_zero(self):
        result = score_caption([], {"a": {"caption": "a river"}})
        assert result["n_missing"] == 1


class TestGroundingMetric:
    def test_identical_boxes_iou_one(self):
        b = {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
        assert iou(b, b) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_zero(self):
        a = {"xmin": 0, "ymin": 0, "xmax": 5, "ymax": 5}
        b = {"xmin": 10, "ymin": 10, "xmax": 15, "ymax": 15}
        assert iou(a, b) == 0.0

    def test_half_overlap(self):
        a = {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
        b = {"xmin": 5, "ymin": 0, "xmax": 15, "ymax": 10}
        assert iou(a, b) == pytest.approx(50 / 150)

    def test_highest_scoring_box_is_used_not_best_matching(self):
        """Using the best-matching box would reward spraying boxes."""
        target = {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
        pred = [{"item_id": "a", "boxes": [
            {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10, "score": 0.1},
            {"xmin": 90, "ymin": 90, "xmax": 99, "ymax": 99, "score": 0.9},
        ]}]
        result = score_grounding(pred, {"a": {"box": target}})
        assert result["acc@0.5"] == 0.0

    def test_accuracy_thresholds_present(self):
        target = {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}
        pred = [{"item_id": "a", "boxes": [{**target, "score": 1.0}]}]
        result = score_grounding(pred, {"a": {"box": target}})
        assert result["acc@0.5"] == 1.0 and result["acc@0.7"] == 1.0


class TestLandcoverMetric:
    def test_perfect_match(self):
        result = score_landcover(
            [{"item_id": "a", "labels": ["water", "forest"]}],
            {"a": {"labels": ["water", "forest"]}},
        )
        assert result["micro_f1"] == pytest.approx(1.0)

    def test_partial_credit_for_multi_label(self):
        """3 of 4 correct must not score the same as none."""
        result = score_landcover(
            [{"item_id": "a", "labels": ["water", "forest", "urban"]}],
            {"a": {"labels": ["water", "forest", "urban", "roads"]}},
        )
        assert 0.0 < result["micro_f1"] < 1.0

    def test_no_overlap_scores_zero(self):
        result = score_landcover(
            [{"item_id": "a", "labels": ["urban"]}], {"a": {"labels": ["water"]}}
        )
        assert result["micro_f1"] == 0.0

    def test_abstention_counted(self):
        result = score_landcover(
            [{"item_id": "a", "abstained": True}], {"a": {"labels": ["water"]}}
        )
        assert result["n_abstained"] == 1
        assert result["micro_f1"] == 0.0
