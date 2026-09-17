from pathlib import Path
from satquery.contracts.input_manifest import InputManifest, IngestMode, ImageMeta
from satquery.controller.matrix_loader import load_matrix
from satquery.controller.router import Router
from satquery.controller.executor import Executor

def test_controller_e2e(tmp_path):
    # Setup capability matrix path (assuming run from root dir)
    matrix_path = Path("configs/capability_matrix.yaml")
    
    # Create a fake InputManifest
    manifest = InputManifest(
        run_id="test-run-123",
        ingest_mode=IngestMode.OPERATIONAL,
        images=[
            ImageMeta(
                role="single",
                path=Path("/fake/image.tif"),
                modality="OPTICAL",
                modality_evidence={},
                crs="EPSG:4326",
                gsd_m=1.0,
                width=1024,
                height=1024,
                bands=["R", "G", "B"],
                band_presence=[True, True, True],
                dtype="uint8",
                effective_bits=8,
                acquisition_dt=None,
                nodata_pct=0.0,
                cloud_pct=0.0,
                sensor_guess="Sentinel-2",
                polarisations=None,
                look_count_est=None
            )
        ],
        config="SINGLE",
        checks=[],
        coreg=None,
        tiling=None,
        artifacts={},
        blocking_failures=[],
        index_availability={}
    )
    
    query = "How many planes are in this image?"
    
    # Initialize components
    matrix = load_matrix(matrix_path)
    router = Router(matrix)
    executor = Executor()
    
    # Run pipeline
    plan = router.route(query, manifest)
    trace = executor.execute(plan, manifest, query)
    
    # Assertions
    assert trace.run_id == "test-run-123"
    assert trace.query == query
    assert not trace.abstained
    
    # Check execution step
    assert len(trace.execution) == 1
    assert trace.execution[0].tool == "rs_vqa_v1"
    
    # Check answer
    assert trace.answer != ""
    # The stub still reaches the answer - what changed is that it now says
    # so. A placeholder that reads like a result is the defect this asserts
    # against (satquery/tools/stubs.py STUB_NOTICE).
    assert "[STUB - no model loaded]" in trace.answer
    assert "the VQA model is not loaded" in trace.answer

    print(trace.model_dump_json(indent=2))
