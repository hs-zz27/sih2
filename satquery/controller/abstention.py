"""Abstention policy: when to decline, and what would fix it (task 3.6).

Phase 1 abstained on exactly one condition - the router selected
`CLARIFY_OR_ABSTAIN` - and produced one of two fixed sentences. That is a
routing outcome, not a policy: a confidently-routed plan whose answer the
physics contradicts still came back as an answer.

This adds the rest, under one rule that shapes every message here:

    **Every abstention names the input that would resolve it.**

"Confidence too low" tells a user nothing. "The limiting component is input
quality (0.50), because the CRS check failed - supply a georeferenced raster"
tells them what to do next. The policy therefore identifies the *limiting*
component rather than reporting the combined score, because the combined
score is a geometric mean and says nothing about which of the three collapsed.

Thresholds live in `configs/thresholds.yaml` so they can be changed without a
code edit, and the defaults below apply when that file is absent or empty -
which it was for all of Phase 1 and 2.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_THRESHOLDS_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "thresholds.yaml"
)
ENV_THRESHOLDS = "SATQUERY_THRESHOLDS"

Trigger = Literal[
    "input_validation",
    "routing",
    "tool_failure",
    "profile_degraded",
    "low_confidence",
    "no_supported_content",
]


@dataclass(frozen=True)
class AbstentionPolicy:
    """Thresholds governing when the system declines to answer.

    Defaults are deliberately permissive. An abstention policy tuned to look
    good on a demo set is worse than none: it converts silent errors into
    silent refusals, and a system that abstains on everything has a perfect
    risk-coverage curve and zero utility. `evaluation/abstention.py` is how
    these should be set - pick the coverage that holds risk under target on a
    real labelled set, not by eye.
    """

    min_final_confidence: float = 0.25
    min_input_quality: float = 0.50
    abstain_when_all_sentences_flagged: bool = True

    @classmethod
    def load(cls, path: str | Path | None = None) -> AbstentionPolicy:
        """Load thresholds, falling back to the defaults on any problem.

        A malformed policy file must not take the system down: the failure
        mode of "cannot read thresholds" should be the documented default
        behaviour, not a crash on the first query of a demo.
        """
        target = Path(
            path or os.environ.get(ENV_THRESHOLDS) or DEFAULT_THRESHOLDS_PATH
        )
        if not target.exists():
            return cls()
        try:
            import yaml

            blob = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
            section = blob.get("abstention") or {}
            return cls(
                min_final_confidence=float(
                    section.get("min_final_confidence", cls.min_final_confidence)
                ),
                min_input_quality=float(
                    section.get("min_input_quality", cls.min_input_quality)
                ),
                abstain_when_all_sentences_flagged=bool(
                    section.get(
                        "abstain_when_all_sentences_flagged",
                        cls.abstain_when_all_sentences_flagged,
                    )
                ),
            )
        except Exception:  # noqa: BLE001 - degradation, not a crash
            return cls()


@dataclass(frozen=True)
class AbstentionDecision:
    abstained: bool
    trigger: Trigger | None = None
    reason: str | None = None
    resolving_input: str | None = None
    limiting_component: str | None = None


# What a user can actually do about each component being the limiting one.
_RESOLUTION = {
    "input_quality": (
        "supply inputs that pass the failing ingest checks - the check names "
        "are listed in the trace under ingest.checks"
    ),
    "agreement": (
        "the measured indices disagree with the answer; supply imagery where "
        "the relevant index has a bimodal, data-derived threshold rather than "
        "a fixed prior, or ask about a class this sensor's bands can measure"
    ),
    "model": (
        "the model itself was uncertain; a higher-resolution or less cloudy "
        "scene, or a more specific question, is what moves this component"
    ),
}


def decide(
    *,
    policy: AbstentionPolicy,
    routed_to_abstain: bool,
    blocking_failures: list[str],
    final_confidence: float,
    components: dict[str, float],
    failing_checks: list[str],
    conflicts: list[str],
    gate_sentences: int,
    gate_flagged: int,
    tool_failure: str | None = None,
    profile_degraded: str | None = None,
) -> AbstentionDecision:
    """Decide whether to answer, and if not, say what would change it.

    Order matters and is not arbitrary. An input that failed validation is
    reported as such even if the confidence would also have been too low,
    because "your file has no CRS" is actionable and "confidence was 0.2" is
    not. The most specific cause wins.
    """
    if blocking_failures:
        failed = ", ".join(blocking_failures)
        return AbstentionDecision(
            True, "input_validation",
            f"input validation failed ({failed}) - no reliable answer is "
            "possible from these inputs",
            # The failed check names are already in the reason, so repeating
            # them here would add length without adding information. What is
            # missing is where to read the specific complaint.
            "supply inputs that pass those checks; each check's own message "
            "is in the trace under ingest.checks",
            "input_quality",
        )

    if profile_degraded:
        # The tool did not fail - it was never loaded, because the active
        # profile's resource budget excluded it. Telling the user to retry
        # would be wrong: the same profile will do the same thing.
        return AbstentionDecision(
            True, "profile_degraded",
            f"the active profile cannot answer this task: {profile_degraded}",
            "run the full profile on a machine with a GPU, or ask a question "
            "the deterministic index engine can answer - land cover, "
            "vegetation, water and built-up extent are all available here",
            "model",
        )

    if tool_failure:
        # A tool that could not run is a system fault, not a user error, and
        # the message says so: telling someone to rephrase when the model
        # failed to load would send them chasing the wrong problem.
        return AbstentionDecision(
            True, "tool_failure",
            f"a required tool did not complete: {tool_failure}",
            "this is a system-side failure, not a problem with the query or "
            "the inputs; the trace's warnings list carries the error, and "
            "retrying with the same inputs is reasonable",
            "model",
        )

    if routed_to_abstain:
        return AbstentionDecision(
            True, "routing",
            "the query could not be mapped to a supported task with "
            "sufficient confidence",
            "rephrase the question naming the task explicitly, for example "
            "'describe the change between these two images' or 'classify the "
            "land cover'",
            None,
        )

    # Every sentence of the draft contradicted the measured indices. Showing
    # a corrected-to-nothing answer would be worse than declining.
    if (
        policy.abstain_when_all_sentences_flagged
        and gate_sentences > 0
        and gate_flagged == gate_sentences
    ):
        detail = conflicts[0] if conflicts else "see verification.conflicts"
        return AbstentionDecision(
            True, "no_supported_content",
            "every sentence of the draft answer contradicted the measured "
            f"indices ({detail})",
            "the model's output disagrees with the physics on this scene; a "
            "cleaner or higher-resolution input is what changes this, not a "
            "rephrasing",
            "agreement",
        )

    if final_confidence < policy.min_final_confidence:
        limiting = min(components, key=components.get) if components else None
        value = components.get(limiting, 0.0) if limiting else 0.0
        resolution = _RESOLUTION.get(limiting or "", "no specific resolving input")
        if limiting == "input_quality" and failing_checks:
            resolution = (
                "the failing or warning ingest checks are: "
                + ", ".join(failing_checks)
            )
        return AbstentionDecision(
            True, "low_confidence",
            f"combined confidence {final_confidence:.2f} is below the "
            f"{policy.min_final_confidence:.2f} threshold; the limiting "
            f"component is {limiting} at {value:.2f}",
            resolution,
            limiting,
        )

    # Input quality can be the sole reason even when the geometric mean
    # survives, because a bad input makes every downstream number suspect
    # regardless of how confident the model was about it.
    quality = components.get("input_quality", 1.0)
    if quality < policy.min_input_quality:
        resolution = (
            "the failing or warning ingest checks are: " + ", ".join(failing_checks)
            if failing_checks
            else _RESOLUTION["input_quality"]
        )
        return AbstentionDecision(
            True, "low_confidence",
            f"input quality {quality:.2f} is below the "
            f"{policy.min_input_quality:.2f} threshold, so no answer from "
            "these inputs is trustworthy",
            resolution,
            "input_quality",
        )

    return AbstentionDecision(False)
