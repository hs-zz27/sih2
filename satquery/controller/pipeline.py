"""Controller: the single entry point from files + query to a Trace.

Wires the vertical slice together: ingest -> route -> execute -> trace.
Everything the API and the eval CLI need goes through here, so there is one
code path and one place where the ordering guarantees hold.
"""

from __future__ import annotations

from pathlib import Path

from satquery.contracts.input_manifest import IngestMode, InputManifest
from satquery.contracts.plan import Plan
from satquery.contracts.trace import Trace
from satquery.controller.executor import Executor
from satquery.controller.matrix_loader import CapabilityMatrix, load_matrix
from satquery.controller.profiles import Profile, load_profile
from satquery.controller.router import Router
from satquery.ingest import ingest

DEFAULT_MATRIX_PATH = Path("configs/capability_matrix.yaml")


class Controller:
    def __init__(
        self,
        matrix: CapabilityMatrix | None = None,
        matrix_path: str | Path = DEFAULT_MATRIX_PATH,
        vram_budget_mb: int | None = None,
        verifier_enabled: bool | None = None,
        profile: str | Profile | None = None,
    ):
        # A profile supplies defaults; explicit arguments still win, so a
        # caller can run the lite profile with the verifier forced on (or
        # off, for the 3.7 ablation) without editing a YAML file.
        self.profile = (
            profile if isinstance(profile, Profile) else load_profile(profile)
        )
        budget = (
            vram_budget_mb
            if vram_budget_mb is not None
            else self.profile.vram_budget_mb
        )
        self.matrix = matrix or load_matrix(matrix_path)
        self.router = Router(self.matrix, vram_budget_mb=budget)
        # `verifier_enabled=False` is the off arm of the verifier ablation
        # (task 3.7), plumbed from here so the ablation runs the real
        # controller rather than a reimplementation of it.
        self.executor = Executor(
            verifier_enabled=(
                self.profile.verifier_enabled
                if verifier_enabled is None
                else verifier_enabled
            )
        )

    def run(
        self,
        paths: list[str | Path],
        query: str,
        mode: IngestMode = IngestMode.OPERATIONAL,
        benchmark: str | None = None,
        run_id: str | None = None,
        tool_params: dict | None = None,
    ) -> Trace:
        """Full pipeline from raster paths and a query to a validated Trace."""
        manifest = ingest(paths, mode=mode, benchmark=benchmark, run_id=run_id)
        return self.run_on_manifest(manifest, query, tool_params=tool_params)

    def run_on_manifest(
        self,
        manifest: InputManifest,
        query: str,
        tool_params: dict | None = None,
        on_event=None,
    ) -> Trace:
        """Route and execute against an already-built manifest.

        `on_event` is forwarded to the executor so the SSE endpoint can stream
        stages through THIS method rather than reaching past it into the
        executor. It used to do the latter, and the paths drifted: the
        streamed path never passed `config_excluded`, so the task 3.8 notice
        ("this input configuration cannot support TEMPORAL_CHANGE_MAP") was
        missing from the streamed answer - which is the path the frontend
        actually uses. One method, one behaviour.
        """
        # `decide`, not `route`: the prediction and the config-excluded task
        # come back with the plan instead of being read off the router
        # afterwards. One Router is shared by every request the API serves,
        # and /runs/stream runs each request on its own thread, so reading
        # `router.last_*` here could pick up another run's values - which
        # would put another user's "this input cannot support X" notice into
        # this answer with nothing in the trace to show for it.
        decision = self.router.decide(query, manifest)
        plan = decision.plan

        if tool_params:
            plan = self._apply_tool_params(plan, tool_params)

        return self.executor.execute(
            plan, manifest, query, prediction=decision.prediction,
            on_event=on_event,
            config_excluded=decision.config_excluded,
        )

    def _apply_tool_params(self, plan: Plan, tool_params: dict) -> Plan:
        """Merge caller-supplied params, then re-validate.

        Caller parameters are not trusted: the plan is validated against the
        matrix again after merging, so a caller cannot inject a parameter the
        matrix does not permit.
        """
        from satquery.controller.validator import assert_legal

        steps = []
        for step in plan.steps:
            extra = tool_params.get(step.tool, {})
            steps.append(step.model_copy(update={"params": {**step.params, **extra}}))
        updated = plan.model_copy(update={"steps": steps})
        assert_legal(updated, self.matrix)
        return updated
