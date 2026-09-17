"""The v1/v2 comparison table: fixed metrics, quoted baselines, honest verdicts.

The failure this guards against is not a crash. It is a table that quietly
reports a worse model as an improvement - by picking the metric after seeing
the result, by mistyping a baseline, or by dropping a row that regressed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.compare_v2 import (
    COMPARISONS, POST_CAMPAIGN_RUNS, build_rows, load_metrics, render, verdict,
)

REPO = Path(__file__).resolve().parent.parent


def test_every_comparison_names_a_real_campaign_run():
    import yaml

    config = yaml.safe_load((REPO / "configs/campaign.yaml").read_text(encoding="utf-8"))
    run_ids = {run["id"] for run in config["runs"]} | set(POST_CAMPAIGN_RUNS)
    for entry in COMPARISONS:
        # Post-campaign arms are allowed, but only by being declared. An
        # undeclared id is a typo, and a typo here hides a result.
        assert entry.run_id in run_ids, f"unknown run '{entry.run_id}'"


def test_baselines_match_the_v1_metrics_files_when_present():
    """The transcribed baselines must equal what the v1 runs actually wrote.

    A mistyped baseline is the quietest way to manufacture an improvement.
    `checkpoints/` is gitignored, so this only runs where v1 weights exist -
    which is exactly where the transcription can be checked.
    """
    checked = 0
    for entry in COMPARISONS:
        # `path:key` says the v1 number lives under a DIFFERENT key from the
        # one v2 writes. change_caption is the case: v1 recorded
        # `bleu4_aggregate`, the current evaluator writes
        # `bleu4_sentence_mean`, and the two are the same quantity over the
        # same 1,929 pairs. Spelling that out keeps the baseline checkable
        # instead of quietly unverified.
        source_spec, _, key = entry.v1_source.partition(":")
        source = REPO / source_spec
        if not source.is_file():
            continue
        recorded = json.loads(source.read_text(encoding="utf-8"))
        v1_key = key or entry.metric
        assert v1_key in recorded, f"{source_spec} has no key '{v1_key}'"
        # v1 cards quote some figures to 4 dp, so compare at that precision.
        assert recorded[v1_key] == pytest.approx(entry.v1_value, abs=1e-4), \
            f"{entry.run_id}.{entry.metric}: table says {entry.v1_value}, " \
            f"{source_spec}:{v1_key} says {recorded[v1_key]}"
        checked += 1
    if checked == 0:
        pytest.skip("no v1 checkpoints on this machine")


def test_verdict_calls_a_regression_a_regression():
    assert verdict(0.50, 0.40, higher_is_better=True) == "WORSE"
    assert verdict(0.50, 0.60, higher_is_better=True) == "BETTER"
    assert verdict(0.50, 0.50, higher_is_better=True) == "unchanged"
    assert verdict(0.50, None, higher_is_better=True) == "pending"


def test_lower_is_better_metrics_are_not_inverted():
    assert verdict(0.50, 0.40, higher_is_better=False) == "BETTER"
    assert verdict(0.50, 0.60, higher_is_better=False) == "WORSE"


def test_missing_results_are_pending_not_zero(tmp_path):
    """An unrun run must not read as a catastrophic regression to 0.0.

    `v2` stays None and the verdict stays `pending`; nothing is reported as a
    regression, and no delta is computed against a value that does not exist.
    """
    rows = build_rows(tmp_path)
    assert rows
    assert all(row["v2"] is None for row in rows)
    assert all(row["delta"] is None for row in rows)
    assert {row["verdict"] for row in rows} == {"pending"}

    text = render(rows)
    assert "REGRESSIONS" not in text
    assert f"0/{len(COMPARISONS)} metrics measured" in text


def test_regressions_are_printed_in_their_own_section(tmp_path):
    """A row that got worse must be impossible to miss in the output."""
    run = COMPARISONS[4]          # change_mask F1, v1 = 0.5597
    assert run.run_id == "change_mask"
    (tmp_path / run.run_id).mkdir(parents=True)
    (tmp_path / run.run_id / "metrics.json").write_text(
        json.dumps({run.metric: run.v1_value - 0.2}), encoding="utf-8"
    )
    text = render(build_rows(tmp_path))
    assert "REGRESSIONS" in text
    assert "change_mask" in text.split("REGRESSIONS")[1]


def test_unreadable_metrics_file_does_not_crash(tmp_path):
    (tmp_path / "change_mask").mkdir(parents=True)
    (tmp_path / "change_mask" / "metrics.json").write_text("{not json",
                                                           encoding="utf-8")
    assert load_metrics(tmp_path, "change_mask") is None
    rows = build_rows(tmp_path)          # must not raise
    assert any(r["run"] == "change_mask" for r in rows)


def test_optsar_baseline_is_recorded_as_negative():
    """v1's fused head was WORSE than optical alone, and the table must say so.

    If this baseline were ever silently corrected upward, a v2 gain would be
    measured against a fiction.
    """
    entry = next(c for c in COMPARISONS
                 if c.metric == "complementarity_gain")
    assert entry.v1_value < 0
