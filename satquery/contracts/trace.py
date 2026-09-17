from typing import Literal
from pydantic import BaseModel
from .plan import TaskID, RationaleTag

class IngestTrace(BaseModel):
    mode: str
    config: str
    images: list[dict]
    index_availability: dict[str, bool]
    checks: list[dict]
    tiling: dict

class ClassifierTrace(BaseModel):
    name: str
    top1: float
    margin: float

class RoutingTrace(BaseModel):
    legal_tasks: list[TaskID]
    selected_task: TaskID
    classifier: ClassifierTrace
    llm_tiebreak_invoked: bool
    # Set when the query's most likely task was removed by config gating -
    # the user asked for something these inputs cannot support. The plan is
    # still legal; this records that the answer is not what was asked for.
    config_excluded_task: str | None = None
    capability_matrix_version: str

class StepExecutionTrace(BaseModel):
    step: str
    tool: str
    version: str
    params: dict
    rationale_tag: RationaleTag
    outputs: dict
    confidence: float
    confidence_method: str
    runtime_ms: int

class FlaggedSentenceTrace(BaseModel):
    sentence: str
    reason: str
    backend: str
    score: float | None = None

class EntailmentGateTrace(BaseModel):
    sentences: int
    retained: int
    flagged: int
    # `retained` alone would be read as "verified". It is not: a sentence
    # nothing in the payload can speak to is neither supported nor
    # contradicted, and lumping it in with the supported ones turns "we did
    # not check this" into "this passed". The three counts sum to
    # `sentences` so the split is always visible.
    unverifiable: int = 0
    backend: str = "not_run"
    action: str = "none"
    flagged_detail: list[FlaggedSentenceTrace] = []

class VerificationTrace(BaseModel):
    physics_agreement: dict[str, float]
    built_up_path: str
    complementarity: dict
    conflicts: list[str]
    entailment_gate: EntailmentGateTrace

class ConfidenceComponentsTrace(BaseModel):
    model: float
    agreement: float
    input_quality: float

class ConfidenceCalibrationTrace(BaseModel):
    method: str
    T: float
    ece_after: float

class ConfidenceTrace(BaseModel):
    final: float
    band: Literal["HIGH", "MEDIUM", "LOW"]
    components: ConfidenceComponentsTrace
    calibration: ConfidenceCalibrationTrace

class Trace(BaseModel):
    run_id: str
    timestamp_utc: str
    code_version: str
    query: str
    ingest: IngestTrace
    routing: RoutingTrace
    execution: list[StepExecutionTrace]
    verification: VerificationTrace
    confidence: ConfidenceTrace
    answer: str
    artifacts: list[str]
    # Artifact key -> filesystem path. Separate from `artifacts` because the
    # keys are stable and comparable while the paths are per-run temporary
    # directories; the golden traces normalise this field away.
    artifact_paths: dict[str, str] = {}
    abstained: bool
    abstain_reason: str | None = None
    # Task 3.6. `abstain_reason` says what went wrong; these say which rule
    # fired and what the user can change. An abstention that does not name a
    # resolving input is a dead end for whoever receives it.
    abstain_trigger: str | None = None
    abstain_resolving_input: str | None = None
    abstain_limiting_component: str | None = None
    weights_hashes: dict[str, str]
    # Attribution for any third-party reference data an answer drew on - the
    # gazetteer rasters, today. The sibling of `weights_hashes`: that one
    # names the weights that produced an answer, this one names the data.
    # CC BY layers require the credit to travel with the output, and an
    # answer that names a country without saying whose boundaries it used is
    # not reproducible either. Empty when no such data was consulted.
    data_sources: list[str] = []
