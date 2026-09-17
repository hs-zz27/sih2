"""Executor: runs a validated plan and builds a real trace.

Phase 0 filled the trace with placeholders. This version derives every field
from something that actually happened: ingest from the manifest, routing from
the classifier's own probabilities, verification from the deterministic index
engine, and confidence from the three-component combiner.

Where a component genuinely does not exist yet the trace says so explicitly
rather than reporting a fabricated number.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any

from satquery.contracts.input_manifest import InputManifest
from satquery.jsonsafe import json_safe as _json_safe
from satquery.contracts.plan import Plan
from satquery.contracts.trace import (
    ClassifierTrace,
    ConfidenceTrace,
    EntailmentGateTrace,
    FlaggedSentenceTrace,
    IngestTrace,
    RoutingTrace,
    StepExecutionTrace,
    Trace,
    VerificationTrace,
)
from satquery.controller.abstention import AbstentionPolicy, decide
from satquery.controller.calibration import CALIBRATABLE_CONFIDENCE_METHODS
from satquery.controller.confidence import compute_confidence
from satquery.controller.intent import CLASSIFIER_NAME, IntentPrediction
from satquery.geo import lookup as lookup_place
from satquery.synth.narrative import compose_answer
from satquery.verify.entailment import run_gate
from satquery.verify.verifier import verify as verify_claims
from satquery.tools.provenance import hashes_for as weights_hashes_for
from satquery.tools.stubs import REGISTRY

CODE_VERSION = "0.2.0-phase1"

# Answer-bearing payload keys, in priority order.
_ANSWER_KEYS = ("answer", "caption", "description", "summary")


def physics_agreement_from_indices(payload: dict) -> tuple[dict[str, float], list[str]]:
    """Turn index-engine output into per-claim agreement scores and conflicts.

    Phase 1 scope: agreement is high when an index produced a confident,
    genuinely bimodal split (the threshold is trustworthy) and low when the
    engine had to fall back to a fixed prior. Task 2.9 extends this to
    per-claim entailment against the generated text.
    """
    agreements: dict[str, float] = {}
    conflicts: list[str] = []

    for report in payload.get("thresholds", []):
        name = report["index"]
        if report["method"] == "fixed_prior":
            agreements[name] = 0.4
            conflicts.append(
                f"{name}: no bimodal split found, threshold is a fixed prior "
                "rather than data-derived"
            )
        elif report.get("bimodal"):
            agreements[name] = 1.0
        else:
            agreements[name] = 0.7

    return agreements, conflicts


def built_up_path(payload: dict) -> str:
    """Which built-up derivation was used - NDBI or the SWIR-free proxy."""
    indices = payload.get("indices", {})
    if "ndbi" in indices:
        return "ndbi"
    if "builtup_proxy" in indices:
        return "swir_free_proxy"
    return "not_computed"


# Phrases the VQA adapter emits when it declines a question it reads as
# out-of-scope. Matched, not parsed: the point is only to notice that the tool
# declined, so that a decline about LOCATION on a georeferenced input can be
# corrected with evidence the manifest already holds.
_REFUSAL_MARKERS = ("i cannot answer that", "i can't answer that")

# The specific claim that is FALSE on a georeferenced input. Only this clause
# is dropped, and only when the manifest contradicts it - "I cannot answer that
# from this image" is left standing, because it is true: the pixels do not
# carry the location, the header does.
_FALSE_ON_GEOREFERENCED = (
    "does not carry that information",
    "doesn't carry that information",
    "do not carry that information",
)

_LOCATION_TERMS = (
    "where", "place", "location", "located", "city", "town", "country",
    "region", "coordinates", "latitude", "longitude", "which part of the world",
)


def _drop_false_location_claim(answer: str) -> str:
    """Remove the one sentence the manifest proves wrong, keep the rest.

    Appending a correction after "Satellite imagery does not carry that
    information." leaves the answer arguing with itself, and the false half
    is the half a reader sees first. Only sentences making that specific
    claim are dropped; the accompanying "I cannot answer that from this
    image" stays, because it is true - the pixels do not carry the location,
    the header does.
    """
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
        if any(claim in sentence.lower() for claim in _FALSE_ON_GEOREFERENCED):
            continue
        if sentence:
            kept.append(sentence)
    return " ".join(kept)


def _location_disclosure(query: str, answer: str, manifest: InputManifest) -> str | None:
    """Coordinates for a location question the tool wrongly called unanswerable.

    `rs_vqa_v1` declines location questions with "Satellite imagery does not
    carry that information." On a GeoTIFF that sentence is simply false - the
    file carries a CRS and an affine transform, which is precisely that
    information - and the adapter cannot know the difference because it never
    sees the georeferencing.

    So the correction is made here, from the manifest, and only when all three
    hold: the question was about location, the tool declined, and the input is
    actually georeferenced. An ungeoreferenced PNG keeps the decline, because
    for a PNG the decline is true.

    What is deliberately NOT done is name the place. Coordinates are measured;
    a place name would need a gazetteer, and this system has none and runs
    offline. Saying so is the honest half of the answer.
    """
    lowered = query.lower()
    if not any(term in lowered for term in _LOCATION_TERMS):
        return None
    if not any(marker in answer.lower() for marker in _REFUSAL_MARKERS):
        return None

    located = [img for img in manifest.images if img.lonlat_bounds]
    if not located:
        return None

    west, south, east, north = located[0].lonlat_bounds  # type: ignore[misc]
    lat, lon = (south + north) / 2.0, (west + east) / 2.0
    return (
        f"The input is georeferenced, so the location IS known: this scene is "
        f"centred at {abs(lat):.4f}°{'N' if lat >= 0 else 'S'}, "
        f"{abs(lon):.4f}°{'E' if lon >= 0 else 'W'} "
        f"({located[0].crs}), spanning {west:.4f} to {east:.4f} longitude and "
        f"{south:.4f} to {north:.4f} latitude. Naming the place would need a "
        f"gazetteer, which this system does not have and cannot reach offline."
    )


class Executor:
    """Runs plan steps and assembles the trace."""

    def __init__(
        self,
        verifier_enabled: bool = True,
        abstention_policy: AbstentionPolicy | None = None,
    ):
        # The off arm of the verifier ablation (task 3.7). Disabling it skips
        # the entailment gate entirely rather than running it and ignoring the
        # result, so the ablation measures the gate's real cost as well as its
        # effect.
        self.verifier_enabled = verifier_enabled
        # Loaded once per executor rather than per query: the thresholds are
        # configuration, and re-reading the file mid-run would let a demo
        # change behaviour between two queries in the same session.
        self.abstention_policy = abstention_policy or AbstentionPolicy.load()

    def execute(
        self,
        plan: Plan,
        manifest: InputManifest,
        query: str,
        prediction: IntentPrediction | None = None,
        on_event: Callable[[str, dict], None] | None = None,
        config_excluded: str | None = None,
    ) -> Trace:
        """Run the plan. `on_event(name, data)` fires as each stage completes,
        which is what lets the API stream the trace live rather than posting it
        all at the end."""

        def emit(name: str, data: dict) -> None:
            if on_event is not None:
                on_event(name, _json_safe(data))

        ingest_trace = IngestTrace(
            mode=manifest.ingest_mode.value,
            config=manifest.config,
            images=[
                {
                    "role": img.role,
                    "path": str(img.path),
                    "modality": img.modality,
                    "modality_reason": img.modality_evidence.get("reason"),
                    "crs": img.crs,
                    "container_format": img.container_format,
                    "georeferenced": img.georeferenced,
                    # None, not the identity-transform 1.0, when the container
                    # carried no georeferencing: reporting a GSD nobody
                    # measured is the kind of quiet fabrication the trace
                    # exists to prevent.
                    "gsd_m": img.gsd_m if img.georeferenced else None,
                    # Where the scene actually is. The CRS and transform were
                    # always in the manifest and nothing ever reported them,
                    # so a run's own trace could not answer "where is this?".
                    "lonlat_bounds": list(img.lonlat_bounds)
                    if img.lonlat_bounds else None,
                    "bands": img.bands,
                    "nodata_pct": img.nodata_pct,
                    "sensor_guess": img.sensor_guess,
                    "polarisations": img.polarisations,
                    # Present only on a run that was given a crop, so an
                    # ordinary run's trace - and every golden - is unchanged.
                    **(
                        {
                            "aoi_applied": list(img.aoi_applied),
                            "source_lonlat_bounds": list(img.source_lonlat_bounds)
                            if img.source_lonlat_bounds
                            else None,
                        }
                        if img.aoi_applied
                        else {}
                    ),
                }
                for img in manifest.images
            ],
            index_availability=manifest.index_availability,
            checks=[
                {"name": c.name, "status": c.status, "message": c.message}
                for c in manifest.checks
            ],
            tiling=manifest.tiling.model_dump() if manifest.tiling else {"applied": False},
        )

        emit("ingest", ingest_trace.model_dump())

        execution_traces: list[StepExecutionTrace] = []
        artifacts: list[str] = []
        # Artifact KEYS go in `artifacts` (stable, golden-comparable); the
        # filesystem paths go here. The PDF report (task 3.12) needs the
        # paths, and they were previously written to disk without ever
        # reaching the trace, so nothing downstream could find them.
        artifact_paths: dict[str, str] = {}
        # Prose emitted by a tool, if any. Kept separate from `final_answer`
        # so the composition step below can put the measured description
        # alongside it instead of one silently overwriting the other.
        tool_answer = ""
        final_answer = ""
        model_confidence = 1.0
        # The `confidence_method` of the tool that set the running minimum.
        # Calibration is only defined on a probability of correctness, so the
        # method - not merely "a learned tool ran" - decides whether the
        # fitted transform may be applied. See CALIBRATABLE_CONFIDENCE_METHODS.
        confidence_method: str | None = None
        # Set when any executed tool was a placeholder rather than a model.
        stubbed = False
        # Learned tools that ran and asserted nothing. They do not enter the
        # model-confidence minimum - see the `no_assertion` branch below.
        silent_tools: list[str] = []
        # Set when a learned tool actually contributed a score.
        model_scored = False
        index_payload: dict = {}
        # optsar_fusion_v1 computes a per-query triad (optical / SAR / fused)
        # and puts the score in its payload. The trace hardcoded `{}`, so the
        # number task 2.3 exists to produce was computed and then dropped on
        # the floor.
        complementarity: dict = {}
        warnings: list[str] = []
        # Set when a step with on_failure="abort" raised. The run stops and
        # becomes a named abstention rather than a traceback (task 3.13).
        tool_failure: str | None = None
        # Set when no tool could answer because the profile shed them all
        # (task 3.10). Distinct from tool_failure: nothing broke.
        profile_degraded: str | None = None

        if prediction is not None:
            classifier = ClassifierTrace(
                name=CLASSIFIER_NAME,
                top1=prediction.top1,
                margin=prediction.margin,
            )
        else:
            # Abstention path: routing was forced by input checks, not by the
            # classifier, so report that rather than inventing a score.
            classifier = ClassifierTrace(name="not_invoked", top1=0.0, margin=0.0)

        routing = RoutingTrace(
            legal_tasks=plan.legal_tasks,
            selected_task=plan.tasks[0],
            classifier=classifier,
            # Tier-2 tiebreak is unbuilt, and this flag is honest about it
            # rather than absent. It is not one of the plan's tasks and the
            # PS does not ask for one: docs/ps-26167.md says the controller
            # MAY plan internally and that only the observable trace is
            # evaluated. Recorded as limitation L9. (The comment here used
            # to say "Phase 3", which passed without it being built.)
            llm_tiebreak_invoked=False,
            config_excluded_task=config_excluded,
            capability_matrix_version=plan.matrix_version,
        )

        emit("routing", routing.model_dump())

        for step in plan.steps:
            tool = REGISTRY[step.tool]
            # The user's question is injected at execution time under a
            # reserved key rather than being placed in the plan. The
            # capability matrix governs *tunable parameters*; the query is
            # input data. Keeping it out of step.params means the plan that
            # gets validated for legality stays exactly what the matrix
            # permits, and the query is already recorded verbatim in the
            # trace's own `query` field.
            runtime_params = {**step.params, "_query": query}
            try:
                result = tool.run(manifest, runtime_params)
            except Exception as exc:  # noqa: BLE001 - degradation, not a crash
                if step.on_failure == "abort":
                    # Task 3.13: an aborting tool used to re-raise, which put
                    # a Python traceback in front of whoever called the API.
                    # A crash is not a graceful degradation, and "zero stack
                    # traces surfaced to the user" is the requirement. The
                    # run stops here and becomes an abstention that names the
                    # tool and the error; the traceback belongs in the logs,
                    # not the answer.
                    tool_failure = (
                        f"{step.tool} failed and the plan cannot continue "
                        f"without it ({type(exc).__name__}: {exc})"
                    )
                    warnings.append(tool_failure)
                    break
                warnings.append(f"{step.tool} failed ({exc}); continuing degraded")
                continue

            data = result.payload.data
            if isinstance(data.get("complementarity"), dict):
                complementarity = data["complementarity"]
            if step.tool == "index_engine_v1":
                index_payload = data
            else:
                # Only learned tools contribute to the model confidence
                # component; the index engine is deterministic by construction.
                #
                # A stub contributes nothing either. It reports 0.0 under the
                # `stub` method, and feeding that into the geometric mean
                # would collapse the score below the abstention threshold and
                # turn every CI run into a refusal. Instead the run is flagged
                # and the FINAL score is capped - see STUB_CONFIDENCE_CAP.
                if result.confidence_method == "stub":
                    stubbed = True
                elif result.confidence_method == "no_assertion":
                    # A selective head that stayed inside its abstention band
                    # on every class. Its 0.0 means "no claim", and letting it
                    # set the minimum vetoed answers the OTHER tools gave.
                    #
                    # Measured 2026-09-12 on the demo bundle with the learned
                    # tools enabled: landcover asserted nothing on the Cartosat
                    # scenes (correctly - a 4-band 1.6 m image is far outside
                    # its BigEarthNet training), reported 0.0, and four of nine
                    # beats abstained with "the model itself was unconfident"
                    # while caption_v1 had answered at 0.56. The matrix
                    # declares index_engine_v1 as landcover's fallback and it
                    # had already run; the plan was answerable and said not.
                    #
                    # v1 passed the same beats only because it asserted classes
                    # at 0.98 on imagery it had never seen - overconfidence
                    # masking the defect, not the absence of it.
                    silent_tools.append(step.tool)
                elif result.confidence <= model_confidence:
                    model_confidence = result.confidence
                    confidence_method = result.confidence_method
                    model_scored = True

            for key in _ANSWER_KEYS:
                if key in data:
                    tool_answer = str(data[key])
                    break

            execution_traces.append(
                StepExecutionTrace(
                    step=step.step_id,
                    tool=step.tool,
                    version=result.version,
                    params=step.params,  # plan params only; _query is not one
                    rationale_tag=step.rationale_tag,
                    outputs=_json_safe(data),
                    confidence=result.confidence,
                    confidence_method=result.confidence_method,
                    runtime_ms=result.runtime_ms,
                )
            )
            warnings.extend(result.warnings)
            artifacts.extend(a.key for a in result.artifacts)
            artifact_paths.update({a.key: str(a.path) for a in result.artifacts})
            emit("step", execution_traces[-1].model_dump())

        # Tools that return structure rather than prose (land cover, grounding)
        # still owe the user a sentence, and tools that DO return prose return
        # one short one - the captioner's whole output is a single clause. Both
        # are handled here: the synthesised description is built deterministically
        # from the numbers already computed, so it cannot assert anything the
        # index engine did not measure, and for descriptive tasks it is appended
        # to the tool's answer rather than only standing in for a missing one.
        first = manifest.images[0] if manifest.images else None
        # Degrees to words. Returns an empty Place when no gazetteer is
        # installed, which is the normal case and not a degraded one - the
        # answer then reports the coordinate and says nothing about the
        # region, rather than guessing at a country.
        centre = first.centroid_latlon if first else None
        place = lookup_place(*centre) if centre else None
        final_answer = compose_answer(
            plan.tasks[0],
            tool_answer,
            [t.outputs for t in execution_traces],
            index_payload,
            artifacts=artifacts,
            georeferenced=first.georeferenced if first else True,
            container_format=first.container_format if first else None,
            centroid=centre,
            extent_m=first.ground_extent_m if first else None,
            place=place,
        )

        # `model_confidence == 0.0` means a learned tool ran and explicitly
        # reported that it could not measure anything - change_vqa_v1's
        # documented "cannot measure" path. That is a deliberate decline, not
        # a fault, so it is left to the low-confidence rule, which produces
        # the right message. Overriding it with "the tool failed, retrying is
        # reasonable" would send the user to repeat something that already
        # worked as designed.
        tool_declined = model_confidence <= 0.0

        # CLARIFY_OR_ABSTAIN declares no tools in the matrix, so it reaches
        # here with no steps and no answer *by design*. That is a routing
        # outcome, and `decide()` has a `routing` trigger that says so. Without
        # this guard it fell into the branch below, which found that no learned
        # tool had run and blamed the profile - telling a user who typed "hmm"
        # to "run the full profile on a machine with a GPU". The condition the
        # branch is for is a task that HAS tools whose tools were all shed.
        routed_to_abstain = plan.tasks[0] == "CLARIFY_OR_ABSTAIN"

        if (
            not final_answer.strip()
            and not tool_failure
            and not tool_declined
            and not routed_to_abstain
        ):
            # Nothing produced a sentence and nothing raised. This is the
            # degraded-profile case (task 3.10): the learned tool that would
            # have answered was shed by the VRAM budget. An empty answer is a
            # silent failure, so it becomes a named one.
            learned_ran = any(
                t.tool != "index_engine_v1" for t in execution_traces
            )
            if learned_ran:
                # A learned tool ran and returned nothing usable. That is a
                # tool problem, not a profile problem, and saying "run the
                # full profile" would send the user somewhere that will do
                # exactly the same thing.
                tool_failure = (
                    f"the tool for {plan.tasks[0]} ran but produced no "
                    f"answer for these inputs"
                )
            else:
                profile_degraded = (
                    f"{plan.tasks[0]} needs a learned tool, and none was "
                    f"available under this profile's resource budget; the "
                    f"task has no deterministic fallback"
                )

        agreements, conflicts = physics_agreement_from_indices(index_payload)
        for sub in index_payload.get("substitutions", []):
            conflicts.append(f"substitution: {sub}")

        # Task 2.9: check what the answer actually *claims* against measured
        # indices. Threshold quality says whether the instrument is
        # trustworthy; this says whether the statement is true.
        claim_report = verify_claims(final_answer, index_payload)
        agreements.update(claim_report["agreements"])
        conflicts.extend(claim_report["conflicts"])
        if claim_report["built_up_path"]:
            built_up = claim_report["built_up_path"]
        else:
            built_up = built_up_path(index_payload)

        # Task 3.5: gate every sentence of the answer against the payload.
        # This runs AFTER verify_claims because it reuses those verdicts, and
        # it can rewrite `final_answer` - a sentence that contradicts the
        # measured indices is removed rather than shown. The original text is
        # kept verbatim in the trace, so nothing is hidden.
        gate = run_gate(final_answer, index_payload, enabled=self.verifier_enabled)
        final_answer = gate.answer
        conflicts.extend(
            f"entailment gate removed: {v.reason}"
            for v in gate.verdicts
            if v.status == "flagged"
        )

        verification = VerificationTrace(
            physics_agreement=agreements,
            built_up_path=built_up,
            # Empty only when no fusion step ran, which is every
            # single-image and bi-temporal configuration.
            complementarity=complementarity,
            conflicts=conflicts,
            entailment_gate=EntailmentGateTrace(
                sentences=gate.sentences,
                retained=gate.retained,
                flagged=gate.flagged,
                unverifiable=gate.unverifiable,
                backend=gate.backend,
                action=gate.action,
                flagged_detail=[
                    FlaggedSentenceTrace(
                        sentence=v.sentence, reason=v.reason,
                        backend=v.backend, score=v.score,
                    )
                    for v in gate.verdicts
                    if v.status == "flagged"
                ],
            ),
        )

        emit("verification", verification.model_dump())

        # A plan whose only learned tools stayed silent has no model claim
        # at all. Left at the 1.0 it started from, the combined score would
        # describe input quality and physics agreement while reading as a
        # model result - the exact situation STUB_CONFIDENCE_CAP exists for,
        # so the same cap applies, with a warning that says which tools
        # declined rather than which were placeholders.
        no_model_claim = bool(silent_tools) and not model_scored and not stubbed
        if silent_tools:
            warnings.append(
                f"{', '.join(silent_tools)} asserted nothing at the measured "
                "threshold and did not contribute to the model confidence"
                + ("; no learned tool made a claim, so the score is capped"
                   if no_model_claim else "")
            )

        confidence: ConfidenceTrace = compute_confidence(
            model_confidence=model_confidence,
            manifest=manifest,
            agreements=agreements,
            head=(
                plan.tasks[0]
                if confidence_method in CALIBRATABLE_CONFIDENCE_METHODS
                else None
            ),
            stubbed=stubbed or no_model_claim,
        )

        emit("confidence", confidence.model_dump())

        # Task 3.6: abstention is a policy over the whole run, not just a
        # routing outcome. A confidently-routed plan whose answer the physics
        # contradicts must also be able to decline.
        decision = decide(
            policy=self.abstention_policy,
            routed_to_abstain=routed_to_abstain,
            blocking_failures=list(manifest.blocking_failures),
            final_confidence=confidence.final,
            components=confidence.components.model_dump(),
            failing_checks=[
                c.name for c in manifest.checks if c.status in ("FAIL", "WARN")
            ],
            conflicts=conflicts,
            gate_sentences=gate.sentences,
            gate_flagged=gate.flagged,
            tool_failure=tool_failure,
            profile_degraded=profile_degraded,
        )
        abstained = decision.abstained
        abstain_reason = decision.reason
        if abstained and decision.resolving_input:
            # The resolving input is part of the message the user sees, not
            # only trace metadata: an abstention nobody can act on is just a
            # refusal.
            abstain_reason = f"{decision.reason} - {decision.resolving_input}"
        location = _location_disclosure(query, final_answer, manifest)
        if location:
            # The tool was right that it could not answer from the pixels, so
            # that half stays; the claim the manifest disproves is dropped
            # rather than left to be argued with.
            final_answer = f"{_drop_false_location_claim(final_answer)} {location}"

        if abstained:
            final_answer = abstain_reason or final_answer
        elif config_excluded:
            # The user asked for something these inputs cannot support. The
            # plan is legal and the answer is real, but it is not the answer
            # that was asked for, and saying so is the difference between a
            # helpful fallback and a silent substitution. This is a prefix
            # rather than an abstention because the fallback answer is often
            # still useful - abstaining on every mismatch would trade a large
            # amount of coverage for a small amount of precision.
            final_answer = (
                f"Note: this input configuration ({manifest.config}) cannot "
                f"support {config_excluded}, which is what the question asks "
                f"for. Answering with {plan.tasks[0]} instead. "
            ) + final_answer

        return Trace(
            run_id=plan.run_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            code_version=CODE_VERSION,
            query=query,
            ingest=ingest_trace,
            routing=routing,
            execution=execution_traces,
            verification=verification,
            confidence=confidence,
            answer=final_answer,
            artifacts=artifacts,
            artifact_paths=artifact_paths,
            abstained=abstained,
            abstain_reason=abstain_reason,
            abstain_trigger=decision.trigger,
            abstain_resolving_input=decision.resolving_input,
            abstain_limiting_component=decision.limiting_component,
            # SHA-256 of the weights the tools in THIS plan actually loaded.
            # Empty when every step ran from a stub or from deterministic
            # arithmetic, which is the CI and no-checkpoint case: a stub loads
            # no bytes, so it gets no digest. See satquery/tools/provenance.py.
            data_sources=list(place.sources) if place else [],
            weights_hashes=weights_hashes_for(t.tool for t in execution_traces),
        )
