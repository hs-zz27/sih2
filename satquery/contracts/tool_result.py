from typing import Literal
from pathlib import Path
from pydantic import BaseModel

class Artifact(BaseModel):
    key: str
    kind: Literal["geojson", "geotiff", "cog", "png", "json"]
    path: Path
    crs: str | None = None
    description: str | None = None

class ToolPayload(BaseModel):
    pass

class ToolResult(BaseModel):
    tool: str
    version: str
    payload: ToolPayload
    artifacts: list[Artifact]
    confidence: float
    # What the `confidence` number actually IS, so a consumer can tell whether
    # it is a probability of correctness or something else entirely. Only
    # `logprob` and `mean_asserted_probability` are probabilities at all, and
    # neither is P(correct) - see CALIBRATABLE_CONFIDENCE_METHODS in
    # satquery/controller/calibration.py for what each one measures.
    #
    # `softmax_temp_scaled` was removed: it was claimed by two tools that
    # never temperature-scaled anything, and keeping a name in the contract
    # that nothing computes invites the same mistake again.
    #
    # `segmentation_derived` was added for `change_vqa_v1`'s semantic path.
    # The arithmetic over the predicted change maps is exact; every bit of
    # the uncertainty lives in the segmentation that produced them, and no
    # per-answer probability is available from an argmax over class logits.
    # It is a fixed conservative constant, named so that nobody mistakes it
    # for a measured one.
    #
    # `stub` was added 2026-09-01 for the placeholder tools the registry falls
    # back to when a learned tool is unavailable. They previously reported
    # `threshold_rule` with values of 0.80-0.95, which is what allowed a
    # stubbed answer to reach the user as "0.9473 HIGH". A stub measures
    # nothing, so it now reports 0.0 under a name that says so - and because
    # the combiner takes a *geometric* mean, a zero component collapses the
    # final score to 0.0 and the band to LOW without any special case.
    confidence_method: Literal[
        "logprob",
        "sharpness",
        "mean_asserted_probability",
        "threshold_rule",
        "deterministic",
        "segmentation_derived",
        "stub",
        # The tool ran, was not a stub, and made NO claim - a selective head
        # whose every class fell inside its abstention band. Its 0.0 is "I
        # am not asserting anything", not "I am certain of nothing", and the
        # executor treats it accordingly: excluded from the model-confidence
        # minimum, so it cannot veto an answer another tool did give.
        "no_assertion",
    ]
    model_card: str
    runtime_ms: int
    warnings: list[str]
