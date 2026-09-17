"""Adversarial routing suite (plan task 3.8).

200 deliberately hostile queries x 3 input configurations = 600 plans. The
requirement is an illegal-plan rate of 0 and a named reason on every
rejection.

What this DOES establish: no query text can cause a plan that violates the
capability matrix. That guarantee is structural - the legal task set is
computed from the images, never from the words - so the suite confirms the
structure rather than sampling for luck.

What it does NOT establish is in `test_what_the_suite_does_not_prove`, which
is a test rather than a comment because the distinction is exactly the kind
that gets dropped when a headline number is quoted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from satquery.controller.matrix_loader import load_matrix
from satquery.controller.pipeline import Controller
from satquery.controller.router import Router
from satquery.controller.validator import validate_plan
from satquery.ingest import ingest
from satquery.synth.adversarial import (
    ADVERSARIAL_QUERIES,
    CATEGORIES,
    CATEGORY_OF,
    summary,
)

MATRIX_PATH = Path("configs/capability_matrix.yaml")

# Tools that must never appear in a plan for a single-image input, whatever
# the query asks for.
BITEMPORAL_ONLY = {"change_mask_v1", "change_caption_v1", "change_vqa_v1"}
CROSSMODAL_ONLY = {"optsar_fusion_v1"}


@pytest.fixture(scope="module")
def matrix():
    return load_matrix(MATRIX_PATH)


@pytest.fixture(scope="module")
def router(matrix):
    return Router(matrix)


@pytest.fixture(scope="module")
def manifests(tmp_path_factory):
    from evaluation.scenes import build_configurations

    directory = tmp_path_factory.mktemp("adversarial")
    return {
        name: ingest(paths)
        for name, paths in build_configurations(directory).items()
    }


class TestSuiteShape:
    def test_suite_is_exactly_200_unique_queries(self):
        assert len(ADVERSARIAL_QUERIES) == 200
        assert len(set(ADVERSARIAL_QUERIES)) == 200

    def test_every_query_belongs_to_a_category(self):
        assert set(CATEGORY_OF) == set(ADVERSARIAL_QUERIES)
        assert sum(summary().values()) == 200

    def test_categories_attack_distinct_gates(self):
        """Nine mechanisms, not nine rewordings of one."""
        assert len(CATEGORIES) == 9
        assert {"config_impossible", "tool_coercion", "parameter_injection"} <= set(
            CATEGORIES
        )


class TestIllegalPlanRate:
    def test_zero_illegal_plans_over_600(self, router, matrix, manifests):
        """The headline requirement."""
        illegal = []
        total = 0
        for config, manifest in manifests.items():
            for query in ADVERSARIAL_QUERIES:
                total += 1
                plan = router.route(query, manifest)
                violations = validate_plan(plan, matrix)
                if violations:
                    illegal.append((config, query, violations))

        assert total == 600
        assert not illegal, f"{len(illegal)}/600 plans violated the matrix: {illegal[:3]}"

    def test_single_image_never_plans_a_bitemporal_tool(self, router, manifests):
        for query in ADVERSARIAL_QUERIES:
            plan = router.route(query, manifests["SINGLE"])
            tools = {step.tool for step in plan.steps}
            assert not (tools & BITEMPORAL_ONLY), query
            assert not (tools & CROSSMODAL_ONLY), query

    def test_crossmodal_pair_never_plans_a_change_tool(self, router, manifests):
        for query in ADVERSARIAL_QUERIES:
            plan = router.route(query, manifests["CROSSMODAL"])
            assert not ({s.tool for s in plan.steps} & BITEMPORAL_ONLY), query

    def test_query_text_never_reaches_plan_params(self, router, manifests):
        """Parameter injection has nowhere to land.

        The query is injected at execution time under a reserved key; the plan
        that gets validated carries only matrix-declared parameters.
        """
        for query in CATEGORIES["parameter_injection"]:
            for manifest in manifests.values():
                plan = router.route(query, manifest)
                for step in plan.steps:
                    serialised = repr(step.params)
                    assert query[:20] not in serialised
                    assert "hack" not in serialised
                    assert "99999" not in serialised


class TestConfigExclusionIsNamed:
    def test_change_query_on_single_image_names_the_excluded_task(
        self, router, manifests
    ):
        router.route("Produce a change mask for these images.", manifests["SINGLE"])
        assert router.last_config_excluded == "TEMPORAL_CHANGE_MAP"

    def test_a_satisfiable_query_excludes_nothing(self, router, manifests):
        router.route("Describe this image.", manifests["SINGLE"])
        assert router.last_config_excluded is None

    def test_the_answer_says_what_was_asked_for_and_what_was_answered(
        self, matrix, manifests
    ):
        """A silent substitution is the failure this prevents."""
        controller = Controller(matrix=matrix)
        trace = controller.run_on_manifest(
            manifests["SINGLE"], "Produce a change mask for these images."
        )
        assert trace.routing.config_excluded_task == "TEMPORAL_CHANGE_MAP"
        assert "TEMPORAL_CHANGE_MAP" in trace.answer
        assert trace.routing.selected_task in trace.answer


class TestNamedReasons:
    """Every abstention that actually occurs must name a reason.

    Parametrising one query per category made every case skip, because only
    ~13% of adversarial queries abstain - a vacuous green. This runs a
    stratified sample end to end and asserts on the abstentions it finds,
    plus asserts it found some, so the test cannot pass by abstaining never.
    """

    @pytest.fixture(scope="class")
    def traces(self, manifests):
        controller = Controller()
        sample = [q for queries in CATEGORIES.values() for q in queries[:3]]
        return [
            controller.run_on_manifest(manifests["SINGLE"], query)
            for query in sample
        ]

    def test_the_sample_contains_abstentions(self, traces):
        """Guards against the assertion below becoming vacuous."""
        assert sum(t.abstained for t in traces) >= 1

    def test_every_abstention_names_a_reason_and_a_resolving_input(self, traces):
        unnamed = [
            t.query
            for t in traces
            if t.abstained
            and not (
                t.abstain_trigger and t.abstain_reason and t.abstain_resolving_input
            )
        ]
        assert not unnamed, f"abstentions without a named reason: {unnamed}"

    def test_no_answer_is_returned_empty(self, traces):
        """An empty answer is a silent failure whether or not it abstained."""
        assert all(t.answer.strip() for t in traces)

    def test_blocking_input_failure_abstains_with_the_check_named(
        self, no_crs_raster
    ):
        controller = Controller()
        trace = controller.run([no_crs_raster], "Describe this image.")
        assert trace.abstained
        assert trace.abstain_trigger == "input_validation"
        assert "crs" in trace.abstain_reason.lower()
        assert trace.abstain_resolving_input


def test_what_the_suite_does_not_prove(router, manifests):
    """The 0/600 headline is narrower than it sounds, and this pins that down.

    An illegal-plan rate of 0 means no plan violated the capability matrix. It
    does NOT mean the system declines questions it cannot answer. Out-of-scope
    queries - the weather, land ownership, population - route to a legal task
    and get answered, because the matrix constrains which TOOLS may run, not
    whether the QUESTION is answerable. Out-of-scope detection is a separate
    capability and is not built.

    This test asserts the gap exists so that if it is ever closed, someone has
    to come here and update the claim rather than leaving a stale caveat in
    the docs.
    """
    answered = 0
    for query in CATEGORIES["out_of_scope"]:
        plan = router.route(query, manifests["SINGLE"])
        if plan.tasks[0] != "CLARIFY_OR_ABSTAIN":
            answered += 1

    assert answered > len(CATEGORIES["out_of_scope"]) / 2, (
        "Out-of-scope queries now mostly abstain. That is an improvement - "
        "update docs/phase1-status.md task 3.8, which currently records that "
        "they do not."
    )
