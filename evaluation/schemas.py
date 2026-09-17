"""Prediction schemas for all four annotation types (plan task 1.8).

One schema per annotation type the benchmarks use, so a predictions file can
be validated before metrics are computed. Catching a malformed predictions
file at write time is far cheaper than discovering it mid-scoring.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AnnotationType = Literal["vqa", "caption", "grounding", "landcover"]


class VQAPrediction(BaseModel):
    item_id: str
    question: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool = False
    answer_type: str | None = None  # e.g. count / presence / comparison


class CaptionPrediction(BaseModel):
    item_id: str
    caption: str
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool = False


class BoundingBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)


class GroundingPrediction(BaseModel):
    item_id: str
    referring_expression: str
    boxes: list[BoundingBox]
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool = False


class LandcoverPrediction(BaseModel):
    item_id: str
    labels: list[str]
    scores: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    abstained: bool = False


PREDICTION_MODELS: dict[str, type[BaseModel]] = {
    "vqa": VQAPrediction,
    "caption": CaptionPrediction,
    "grounding": GroundingPrediction,
    "landcover": LandcoverPrediction,
}


class PredictionsFile(BaseModel):
    """The container written by `satquery eval`."""

    benchmark: str
    annotation_type: AnnotationType
    code_version: str
    matrix_version: str
    model_cards: dict[str, str] = Field(default_factory=dict)
    n_items: int
    predictions: list[dict]

    def validated_predictions(self) -> list[BaseModel]:
        """Parse each prediction against the schema for this annotation type."""
        model = PREDICTION_MODELS[self.annotation_type]
        return [model.model_validate(p) for p in self.predictions]
