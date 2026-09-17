"""Evaluation harness, metrics and CLI tests (plan tasks 1.8, 1.9)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from evaluation.harness import dry_run, evaluate, load_benchmark, run_benchmark, score
from evaluation.metrics.vqa import exact_match, normalise_answer, score_vqa
from evaluation.schemas import PredictionsFile


@pytest.fixture
def benchmark(tmp_path, msi_6band):
    """A small benchmark manifest pointing at a real synthetic raster."""
    items = [
        {
            "item_id": f"item_{i}",
            "images": [msi_6band.name],
            "question": q,
            "answer": a,
            "answer_type": t,
        }
        for i, (q, a, t) in enumerate(
            [
                ("How many buildings are visible?", "4", "count"),
                ("Is there water in this scene?", "yes", "presence"),
                ("How many roads are there?", "2", "count"),
            ]
        )
    ]
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return path, msi_6band.parent, items


class TestNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("The Building", "building"),
            ("YES!", "yes"),
            ("yep", "yes"),
            ("Nope.", "no"),
            ("  two  ", "2"),
            ("a river", "river"),
        ],
    )
    def test_normalise(self, raw, expected):
        assert normalise_answer(raw) == expected

    def test_exact_match_is_normalisation_insensitive(self):
        assert exact_match("Yes", "yeah")
        assert exact_match("two", "2")
        assert not exact_match("yes", "no")


class TestVQAScoring:
    TRUTH = {
        "a": {"answer": "yes", "answer_type": "presence"},
        "b": {"answer": "3", "answer_type": "count"},
        "c": {"answer": "no", "answer_type": "presence"},
    }

    def test_all_correct(self):
        preds = [
            {"item_id": "a", "answer": "yes", "abstained": False},
            {"item_id": "b", "answer": "three", "abstained": False},
            {"item_id": "c", "answer": "NO", "abstained": False},
        ]
        result = score_vqa(preds, self.TRUTH)
        assert result["accuracy"] == 1.0
        assert result["coverage"] == 1.0

    def test_abstention_counts_against_accuracy_but_lowers_coverage(self):
        preds = [
            {"item_id": "a", "answer": "yes", "abstained": False},
            {"item_id": "b", "answer": "", "abstained": True},
            {"item_id": "c", "answer": "no", "abstained": False},
        ]
        result = score_vqa(preds, self.TRUTH)
        assert result["accuracy"] == pytest.approx(2 / 3)
        assert result["accuracy_when_answered"] == 1.0
        assert result["coverage"] == pytest.approx(2 / 3)
        assert result["n_abstained"] == 1

    def test_missing_predictions_count_as_wrong(self):
        """A truncated predictions file must not inflate the score."""
        result = score_vqa([{"item_id": "a", "answer": "yes"}], self.TRUTH)
        assert result["n_missing"] == 2
        assert result["accuracy"] == pytest.approx(1 / 3)

    def test_per_answer_type_breakdown(self):
        preds = [
            {"item_id": "a", "answer": "yes"},
            {"item_id": "b", "answer": "999"},
            {"item_id": "c", "answer": "no"},
        ]
        result = score_vqa(preds, self.TRUTH)
        assert result["by_answer_type"]["presence"]["accuracy"] == 1.0
        assert result["by_answer_type"]["count"]["accuracy"] == 0.0


class TestBenchmarkLoading:
    def test_loads_valid_manifest(self, benchmark):
        path, _, items = benchmark
        assert len(load_benchmark(path)) == len(items)

    def test_rejects_non_list(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text('{"not": "a list"}', encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON list"):
            load_benchmark(p)

    def test_rejects_missing_fields(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text('[{"item_id": "x"}]', encoding="utf-8")
        with pytest.raises(ValueError, match="missing required fields"):
            load_benchmark(p)

    def test_rejects_duplicate_ids(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(
            '[{"item_id":"x","images":["a.tif"]},{"item_id":"x","images":["b.tif"]}]',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate item_id"):
            load_benchmark(p)


class TestDryRun:
    def test_ready_when_files_present(self, benchmark):
        path, root, items = benchmark
        report = dry_run(load_benchmark(path), root)
        assert report["ready"] is True
        assert report["n_items"] == len(items)

    def test_reports_missing_files(self, tmp_path):
        p = tmp_path / "b.json"
        p.write_text('[{"item_id":"x","images":["nope.tif"]}]', encoding="utf-8")
        report = dry_run(load_benchmark(p), tmp_path)
        assert report["ready"] is False
        assert report["n_missing_files"] == 1


class TestHarnessRun:
    def test_produces_schema_valid_predictions(self, benchmark):
        path, root, items = benchmark
        preds = run_benchmark(load_benchmark(path), root, "test_bench")
        assert isinstance(preds, PredictionsFile)
        assert preds.n_items == len(items)
        # Raises if any prediction violates the schema.
        assert len(preds.validated_predictions()) == len(items)

    def test_report_has_metrics_and_provenance(self, benchmark):
        path, root, _ = benchmark
        report = evaluate(path, root, "test_bench")
        assert report["metrics"]["metric_status"] == "ok"
        assert report["matrix_version"]
        assert report["code_version"]
        assert "accuracy" in report["metrics"]

    def test_limit_respected(self, benchmark):
        path, root, _ = benchmark
        preds = run_benchmark(load_benchmark(path), root, "b", limit=2)
        assert preds.n_items == 2

    def test_missing_image_becomes_abstention_not_crash(self, tmp_path):
        p = tmp_path / "b.json"
        p.write_text(
            '[{"item_id":"x","images":["missing.tif"],"question":"hi","answer":"y"}]',
            encoding="utf-8",
        )
        preds = run_benchmark(load_benchmark(p), tmp_path, "b")
        assert preds.predictions[0]["abstained"] is True

    def test_caption_metrics_now_implemented(self, benchmark):
        """Task 2.14 replaced the Phase 1 "not_implemented" stub for the three
        non-VQA annotation types."""
        path, root, items = benchmark
        preds = run_benchmark(load_benchmark(path), root, "b", annotation_type="caption")
        for item in items:
            item["caption"] = "a reference caption"
        result = score(preds, items)
        assert result["metric_status"] == "ok"
        assert "bleu4_sentence_mean" in result

    def test_no_ground_truth_is_reported_not_scored(self, benchmark):
        """Absent references must not silently produce a zero score."""
        path, root, items = benchmark
        preds = run_benchmark(load_benchmark(path), root, "b", annotation_type="grounding")
        assert score(preds, items)["metric_status"] == "no_ground_truth"


class TestCLI:
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "satquery", *args],
            capture_output=True, text=True,
        )

    def test_dry_run_exits_zero_when_ready(self, benchmark):
        path, root, _ = benchmark
        r = self._run(
            "eval", "--benchmark", "b", "--manifest", str(path),
            "--root", str(root), "--dry-run",
        )
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout)["ready"] is True

    def test_dry_run_exits_nonzero_on_missing_files(self, tmp_path):
        p = tmp_path / "b.json"
        p.write_text('[{"item_id":"x","images":["nope.tif"]}]', encoding="utf-8")
        r = self._run(
            "eval", "--benchmark", "b", "--manifest", str(p),
            "--root", str(tmp_path), "--dry-run",
        )
        assert r.returncode == 1

    def test_invalid_manifest_exits_two(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{}", encoding="utf-8")
        r = self._run("eval", "--benchmark", "b", "--manifest", str(p), "--dry-run")
        assert r.returncode == 2

    def test_eval_writes_report(self, benchmark, tmp_path):
        path, root, _ = benchmark
        out = tmp_path / "report.json"
        r = self._run(
            "eval", "--benchmark", "b", "--manifest", str(path),
            "--root", str(root), "--out", str(out),
        )
        assert r.returncode == 0, r.stderr
        report = json.loads(out.read_text(encoding="utf-8"))
        assert "metrics" in report and "predictions" in report

    def test_matrix_validate_still_works(self):
        r = self._run("matrix", "--validate")
        assert r.returncode == 0, r.stderr

    def test_ask_command(self, msi_6band):
        r = self._run("ask", str(msi_6band), "--query", "Describe this image.")
        assert r.returncode == 0, r.stderr
        assert "Answer" in r.stdout
        assert "Confidence" in r.stdout
