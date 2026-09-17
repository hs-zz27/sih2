import pytest
from datetime import datetime
from pathlib import Path

from satquery.contracts.input_manifest import InputManifest, IngestMode, ImageMeta
from satquery.contracts.tool_result import ToolResult
from satquery.tools.stubs import (
    RSVQAStub,
    CaptionStub,
    GroundingStub,
    LandcoverStub,
    OptSARFusionStub,
    ChangeMaskStub,
    ChangeCaptionStub,
    ChangeVQAStub,
    IndexEngineStub
)

@pytest.fixture
def fake_manifest():
    image = ImageMeta(
        role="single",
        path=Path("/fake/image.tif"),
        modality="OPTICAL",
        modality_evidence={"source": "fake"},
        crs="EPSG:4326",
        gsd_m=10.0,
        width=100,
        height=100,
        bands=["B4", "B3", "B2"],
        band_presence=[True, True, True],
        dtype="uint16",
        effective_bits=12,
        acquisition_dt=datetime(2026, 1, 1),
        nodata_pct=0.0,
        cloud_pct=0.0,
        sensor_guess="Sentinel-2",
        polarisations=None,
        look_count_est=None
    )
    return InputManifest(
        run_id="test_run",
        ingest_mode=IngestMode.OPERATIONAL,
        images=[image],
        config="SINGLE",
        checks=[],
        artifacts={},
        blocking_failures=[],
        index_availability={}
    )


@pytest.mark.parametrize("stub_class", [
    RSVQAStub,
    CaptionStub,
    GroundingStub,
    LandcoverStub,
    OptSARFusionStub,
    ChangeMaskStub,
    ChangeCaptionStub,
    ChangeVQAStub,
    IndexEngineStub
])
def test_stub_returns_valid_tool_result(stub_class, fake_manifest):
    stub = stub_class()
    result = stub.run(fake_manifest, params={"some": "param"})
    
    assert isinstance(result, ToolResult)
    assert result.tool is not None


@pytest.mark.parametrize("stub_class", [
    RSVQAStub,
    CaptionStub,
    GroundingStub,
    LandcoverStub,
    OptSARFusionStub,
    ChangeMaskStub,
    ChangeCaptionStub,
    ChangeVQAStub,
    IndexEngineStub
])
def test_stub_run_batch(stub_class, fake_manifest):
    stub = stub_class()
    manifests = [fake_manifest, fake_manifest, fake_manifest]
    results = stub.run_batch(manifests, params={"some": "param"})
    
    assert len(results) == 3
    for res in results:
        assert isinstance(res, ToolResult)

def test_stub_payload_differences(fake_manifest):
    grounding_stub = GroundingStub()
    index_stub = IndexEngineStub()
    
    g_res = grounding_stub.run(fake_manifest, params={})
    i_res = index_stub.run(fake_manifest, params={})
    
    assert "bounding_boxes" in g_res.payload.data
    assert "NDVI" in i_res.payload.data
    assert "NDVI" not in g_res.payload.data
    assert "bounding_boxes" not in i_res.payload.data
