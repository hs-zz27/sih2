from datetime import datetime
from pathlib import Path
from satquery.contracts import (
    InputManifest, IngestMode, ImageMeta, CheckResult,
    Plan, PlanStep, RationaleTag, TaskID,
    ToolResult, Artifact, ToolPayload,
    Trace, IngestTrace, RoutingTrace, ClassifierTrace,
    StepExecutionTrace, VerificationTrace, EntailmentGateTrace,
    ConfidenceTrace, ConfidenceComponentsTrace, ConfidenceCalibrationTrace
)
import pytest
from pydantic import ValidationError

def test_input_manifest_instantiation():
    manifest = InputManifest(
        run_id="run_123",
        ingest_mode=IngestMode.OPERATIONAL,
        images=[
            ImageMeta(
                role="optical",
                path=Path("dummy.tif"),
                modality="MSI",
                modality_evidence={"band_count": 4, "local_cov": 0.07},
                crs="EPSG:32643",
                gsd_m=1.6,
                width=1024,
                height=1024,
                bands=["Blue", "Green", "Red", "NIR"],
                band_presence=[True, True, True, True],
                dtype="uint16",
                effective_bits=11,
                acquisition_dt=datetime.now(),
                nodata_pct=0.0,
                cloud_pct=3.1,
                sensor_guess="Cartosat-2S MX",
                polarisations=None,
                look_count_est=None
            )
        ],
        config="SINGLE",
        checks=[
            CheckResult(name="format_gate", status="PASS", value="GeoTIFF", message="OK")
        ],
        coreg=None,
        tiling=None,
        artifacts={"rgb": Path("rgb.tif")},
        blocking_failures=[],
        index_availability={"NDVI": True, "NDWI": True}
    )
    assert manifest.run_id == "run_123"

def test_plan_instantiation():
    plan = Plan(
        run_id="run_123",
        legal_tasks=["SINGLE_VQA", "SINGLE_CAPTION", "SINGLE_GROUND", "SINGLE_LANDCOVER", "CLARIFY_OR_ABSTAIN"],
        tasks=["SINGLE_VQA"],
        steps=[
            PlanStep(
                step_id="s1",
                tool="rs_vqa_v1",
                tool_version="0.8.0",
                inputs=["rgb.tif"],
                params={"max_new_tokens": 128},
                rationale_tag=RationaleTag.VQA_INFERENCE,
                on_failure="abort"
            )
        ],
        fallbacks={},
        matrix_version="cm-2026.11.02",
        estimated_vram_mb=4200,
        estimated_runtime_ms=1890
    )
    assert plan.run_id == "run_123"

def test_tool_result_instantiation():
    res = ToolResult(
        tool="rs_vqa_v1",
        version="0.8.0",
        payload=ToolPayload(),
        artifacts=[
            Artifact(key="mask", kind="geotiff", path=Path("mask.tif"))
        ],
        confidence=0.9,
        confidence_method="logprob",
        model_card="Qwen-VL base",
        runtime_ms=1890,
        warnings=[]
    )
    assert res.tool == "rs_vqa_v1"

def test_trace_instantiation():
    trace = Trace(
        run_id="8f2c1a",
        timestamp_utc="2026-11-14T09:21:04Z",
        code_version="git:4b91e2c",
        query="Use the optical and SAR images together...",
        ingest=IngestTrace(
            mode="operational",
            config="CROSSMODAL_PAIR",
            images=[],
            index_availability={"NDVI": True},
            checks=[],
            tiling={"applied": True}
        ),
        routing=RoutingTrace(
            legal_tasks=["XMODAL_JOINT_EXTRACT"],
            selected_task="XMODAL_JOINT_EXTRACT",
            classifier=ClassifierTrace(name="intent-tfidf-lr-v3", top1=0.91, margin=0.44),
            llm_tiebreak_invoked=False,
            capability_matrix_version="cm-2026.11.02"
        ),
        execution=[
            StepExecutionTrace(
                step="s1",
                tool="index_engine_v1",
                version="1.0.2",
                params={"indices": ["NDVI"]},
                rationale_tag=RationaleTag.VQA_INFERENCE,
                outputs={"ndwi": "art/ndwi.tif"},
                confidence=1.0,
                confidence_method="deterministic",
                runtime_ms=410
            )
        ],
        verification=VerificationTrace(
            physics_agreement={"water": 0.93},
            built_up_path="sar_primary_texture_secondary",
            complementarity={"gain_iou": {"water": 0.09}},
            conflicts=[],
            entailment_gate=EntailmentGateTrace(sentences=6, retained=6, flagged=0)
        ),
        confidence=ConfidenceTrace(
            final=0.79,
            band="HIGH",
            components=ConfidenceComponentsTrace(model=0.75, agreement=0.86, input_quality=0.81),
            calibration=ConfidenceCalibrationTrace(method="temperature_scaling", T=1.34, ece_after=0.041)
        ),
        answer="...",
        artifacts=["art/lc_fused.tif"],
        abstained=False,
        weights_hashes={"optsar_fusion_v1": "sha256:..."}
    )
    assert trace.run_id == "8f2c1a"

def test_negative_plan_invalid_task():
    with pytest.raises(ValidationError):
        Plan(
            run_id="run_123",
            legal_tasks=["SINGLE_VQA"],
            tasks=["INVALID_TASK_NAME"],
            steps=[],
            fallbacks={},
            matrix_version="cm-1",
            estimated_vram_mb=1000,
            estimated_runtime_ms=1000
        )

def test_negative_trace_missing_field():
    with pytest.raises(ValidationError):
        Trace(
            run_id="run_123",
            # missing timestamp_utc, query, etc.
            code_version="git:hash"
        )
