from typing import Literal
from enum import Enum
from pydantic import BaseModel

TaskID = Literal[
    "SINGLE_VQA",
    "SINGLE_CAPTION",
    "SINGLE_GROUND",
    "SINGLE_LANDCOVER",
    "XMODAL_JOINT_EXTRACT",
    "TEMPORAL_CHANGE_DESC",
    "TEMPORAL_CHANGE_VQA",
    "TEMPORAL_CHANGE_MAP",
    "CLARIFY_OR_ABSTAIN"
]

class RationaleTag(str, Enum):
    NDBI_UNAVAILABLE_SWIR_FREE_FALLBACK = "NDBI_UNAVAILABLE_SWIR_FREE_FALLBACK"
    DETECTED_THEN_COUNTED = "DETECTED_THEN_COUNTED"
    MASK_CONDITIONED_CAPTION = "MASK_CONDITIONED_CAPTION"
    EXPLICIT_CHANGE_LANGUAGE = "EXPLICIT_CHANGE_LANGUAGE"
    QUANTITATIVE_REQUEST = "QUANTITATIVE_REQUEST"
    AMBIGUOUS_DEFAULTED_TO_VQA = "AMBIGUOUS_DEFAULTED_TO_VQA"
    MISSING_SECOND_IMAGE = "MISSING_SECOND_IMAGE"
    VQA_INFERENCE = "VQA_INFERENCE"

class PlanStep(BaseModel):
    step_id: str
    tool: str
    tool_version: str
    inputs: list[str]
    params: dict
    rationale_tag: RationaleTag
    on_failure: Literal["abort", "fallback", "continue_degraded"]

class Plan(BaseModel):
    run_id: str
    legal_tasks: list[TaskID]
    tasks: list[TaskID]
    steps: list[PlanStep]
    fallbacks: dict[str, str]
    matrix_version: str
    estimated_vram_mb: int
    estimated_runtime_ms: int

