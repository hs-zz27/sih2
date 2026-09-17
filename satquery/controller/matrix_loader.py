import yaml
from pathlib import Path
from typing import Literal, Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator
from satquery.contracts.plan import TaskID

class ParameterSchema(BaseModel):
    type: Optional[Literal["number", "integer", "string", "boolean"]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    enum: Optional[List[str]] = None
    enum_subset: Optional[List[str]] = None
    default: Any = None

class DegradationRule(BaseModel):
    check: str
    equals: Optional[Union[bool, str, int]] = None
    gt: Optional[float] = None
    effect: str
    confidence_penalty: float

class RequiresSchema(BaseModel):
    model_config = {"extra": "allow"}
    config: Union[str, List[str]]

class TaskConfig(BaseModel):
    description: str
    requires: RequiresSchema
    tools: List[str]
    optional_tools: List[str]
    forbidden_tools: List[str]
    permitted_params: Dict[str, ParameterSchema]
    fallbacks: Dict[str, str]
    degraded_if: List[DegradationRule] = Field(default_factory=list)

class CapabilityMatrix(BaseModel):
    version: str
    tasks: Dict[TaskID, TaskConfig]

def load_matrix(path: str | Path) -> CapabilityMatrix:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    version = data.pop("version", "unknown")
    
    # Rest of data are tasks
    tasks = {}
    for k, v in data.items():
        tasks[k] = TaskConfig(**v)
        
    return CapabilityMatrix(version=version, tasks=tasks)
