"""Plan validation - the mechanism behind the zero-illegal-plan claim.

The PS requires that the system select tools from a predefined registry and
execute them with permitted parameters. Rather than trusting the planner to
behave, every plan is validated against the capability matrix before it runs.
A plan that violates the matrix is rejected outright, so an illegal plan can
never reach the executor regardless of what the intent classifier predicted.

This is the file to point at when asked how illegal-plan rate can be zero: it
is enforced structurally, not learned.
"""

from __future__ import annotations

from dataclasses import dataclass

from satquery.contracts.plan import Plan
from satquery.controller.matrix_loader import CapabilityMatrix, ParameterSchema
from satquery.tools.stubs import REGISTRY


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


class IllegalPlanError(Exception):
    """Raised when a plan violates the capability matrix."""

    def __init__(self, violations: list[Violation]):
        self.violations = violations
        super().__init__("; ".join(str(v) for v in violations))


def _validate_param(
    name: str, value, schema: ParameterSchema
) -> list[Violation]:
    out: list[Violation] = []

    if schema.enum is not None and value not in schema.enum:
        out.append(
            Violation("PARAM_NOT_IN_ENUM", f"{name}={value!r} not in {schema.enum}")
        )

    if schema.enum_subset is not None:
        if not isinstance(value, (list, tuple, set)):
            out.append(
                Violation(
                    "PARAM_NOT_A_SUBSET", f"{name}={value!r} must be a list"
                )
            )
        else:
            extra = [v for v in value if v not in schema.enum_subset]
            if extra:
                out.append(
                    Violation(
                        "PARAM_NOT_IN_ENUM_SUBSET",
                        f"{name} contains {extra} not in {schema.enum_subset}",
                    )
                )

    if schema.type in ("number", "integer"):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            out.append(
                Violation("PARAM_WRONG_TYPE", f"{name}={value!r} is not numeric")
            )
        else:
            if schema.min is not None and value < schema.min:
                out.append(
                    Violation(
                        "PARAM_BELOW_MIN", f"{name}={value} < min {schema.min}"
                    )
                )
            if schema.max is not None and value > schema.max:
                out.append(
                    Violation(
                        "PARAM_ABOVE_MAX", f"{name}={value} > max {schema.max}"
                    )
                )

    if schema.type == "integer" and isinstance(value, float) and value != int(value):
        out.append(Violation("PARAM_NOT_INTEGER", f"{name}={value} is not an integer"))

    if schema.type == "string" and not isinstance(value, str):
        out.append(Violation("PARAM_WRONG_TYPE", f"{name}={value!r} is not a string"))

    if schema.type == "boolean" and not isinstance(value, bool):
        out.append(Violation("PARAM_WRONG_TYPE", f"{name}={value!r} is not a boolean"))

    return out


def validate_plan(plan: Plan, matrix: CapabilityMatrix) -> list[Violation]:
    """Return every way `plan` violates `matrix`. Empty list means legal."""
    violations: list[Violation] = []

    if plan.matrix_version != matrix.version:
        violations.append(
            Violation(
                "MATRIX_VERSION_MISMATCH",
                f"plan built against {plan.matrix_version}, running {matrix.version}",
            )
        )

    for task in plan.tasks:
        if task not in plan.legal_tasks:
            violations.append(
                Violation(
                    "TASK_NOT_LEGAL_FOR_CONFIG",
                    f"{task} is not legal for this input configuration",
                )
            )
        if task not in matrix.tasks:
            violations.append(
                Violation("TASK_NOT_IN_MATRIX", f"{task} is not defined in the matrix")
            )

    known_tasks = [t for t in plan.tasks if t in matrix.tasks]
    allowed_tools: set[str] = set()
    forbidden_tools: set[str] = set()
    permitted_params: dict[str, ParameterSchema] = {}
    for task in known_tasks:
        cfg = matrix.tasks[task]
        allowed_tools |= set(cfg.tools) | set(cfg.optional_tools)
        forbidden_tools |= set(cfg.forbidden_tools)
        permitted_params.update(cfg.permitted_params)

    seen_ids: set[str] = set()
    for step in plan.steps:
        if step.step_id in seen_ids:
            violations.append(
                Violation("DUPLICATE_STEP_ID", f"step_id {step.step_id} reused")
            )
        seen_ids.add(step.step_id)

        if step.tool not in REGISTRY:
            violations.append(
                Violation("TOOL_NOT_IN_REGISTRY", f"{step.tool} is not a known tool")
            )
        if step.tool in forbidden_tools:
            violations.append(
                Violation(
                    "FORBIDDEN_TOOL",
                    f"{step.tool} is explicitly forbidden for {known_tasks}",
                )
            )
        elif known_tasks and step.tool not in allowed_tools:
            violations.append(
                Violation(
                    "TOOL_NOT_PERMITTED",
                    f"{step.tool} is not among the permitted tools for {known_tasks}",
                )
            )

        for pname, pvalue in step.params.items():
            schema = permitted_params.get(pname)
            if schema is None:
                violations.append(
                    Violation(
                        "PARAM_NOT_PERMITTED",
                        f"{pname} is not a permitted parameter for {known_tasks}",
                    )
                )
                continue
            violations.extend(_validate_param(pname, pvalue, schema))

    return violations


def assert_legal(plan: Plan, matrix: CapabilityMatrix) -> None:
    """Raise `IllegalPlanError` if the plan violates the matrix."""
    violations = validate_plan(plan, matrix)
    if violations:
        raise IllegalPlanError(violations)
