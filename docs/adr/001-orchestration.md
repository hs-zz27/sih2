# ADR 001: LangGraph vs. hand-rolled executor

- **Status:** Accepted
- **Date:** 2026-08-29
- **Deciders:** Controller/orchestration owner
- **Related:** `docs/04-Implementation-Plan.md` item 0.9, `docs/02-Agentic-Workflow-and-Orchestration.md`

## Context

The problem statement makes orchestration a first-class, separately-evaluated
requirement: the system must interpret a query, pick a task, select a tool
from a predefined registry, and execute it with permitted parameters — and
the evaluators grade the *observable decision*, not the reasoning trace
(`docs/02` §1). The design in `docs/02` centers this on a version-controlled
`capability_matrix.yaml` as "the auditable artifact" (§4), validated at
startup and referenced in every trace, with a stated goal of a provable
zero illegal-plan rate.

The plan (`docs/04-Implementation-Plan.md`, item 0.9) called for deciding
once, permanently, between adopting LangGraph (or a similar graph-based
agent-orchestration framework) versus a hand-rolled router + executor.

## Decision

**Hand-rolled router + executor**, as already implemented in
`satquery/controller/router.py` and `satquery/controller/executor.py`.

## Rationale

- **The task graph is small and closed, not open-ended.** The capability
  matrix defines nine fixed tasks with explicit allowed/forbidden tools and
  permitted parameters (`configs/capability_matrix.yaml`). This is a
  classification-then-dispatch problem, not a multi-step agentic loop with
  branching, cycles, or dynamic replanning — the class of problem LangGraph
  is built for. Adopting a graph-execution engine here would add a
  dependency and an abstraction layer without buying capability we need.
- **Auditability requires the matrix to be the source of truth, not a graph
  runtime's internal state.** The design explicitly wants illegal plans to
  be structurally unrepresentable (validated against the matrix before
  execution), and every trace to cite the matrix version. A thin, fully
  understood executor makes it straightforward to guarantee "route → validate
  against matrix → execute → trace" with no hidden control flow to audit.
- **Fewer dependencies, easier to reason about under contest deadlines.** No
  new framework version to pin, upgrade, or debug; `router.py`/`executor.py`
  are plain Python that any contributor can read start to finish in a few
  minutes. This matters for a small team on a fixed SIH timeline.
- **Testability without a GPU.** `docs/02` §9 emphasizes that orchestration
  correctness should be fully testable in isolation. A hand-rolled executor
  with a plain function-call interface is trivial to unit test (see
  `tests/test_controller_e2e.py`); a graph-framework executor adds a layer
  of framework-specific test harnessing for no corresponding benefit at this
  scale.

## Consequences

- If a future phase needs genuine multi-step replanning, cycles, or
  human-in-the-loop interrupts (none are in scope for Phases 1–4 per
  `docs/04`), this decision should be revisited — the capability matrix and
  contracts (`satquery/contracts/`) are framework-agnostic, so migrating the
  executor internals later would not require changing the matrix schema or
  the trace format.
- The router currently hardcodes routing to a single task
  (`satquery/controller/router.py`, Phase 0 stub) — this ADR governs the
  *execution* model, not the intent-classification model, which is separate
  Phase 1 work (real intent classifier, per the vertical-slice plan).

## Alternatives considered

- **LangGraph.** Rejected: general-purpose graph orchestration is unneeded
  complexity for a fixed, auditable, nine-task dispatch problem, and
  conflicts with the goal of a minimal, fully-inspectable illegal-plan gate.
- **A different agent framework (e.g. AutoGen, CrewAI).** Not seriously
  evaluated — same objection as LangGraph: these target open-ended
  multi-agent conversation, not closed-set validated dispatch.
