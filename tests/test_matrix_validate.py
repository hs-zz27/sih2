import pytest
import yaml
from pathlib import Path
from satquery.cli.matrix_validate import validate_matrix

@pytest.fixture
def valid_matrix_path():
    return Path("configs/capability_matrix.yaml")

@pytest.fixture
def broken_matrix_path(tmp_path):
    broken_yaml = """
version: cm-2026.11.02
SINGLE_VQA:
  description: "Question answering on one image"
  requires:
    config: SINGLE
  tools: [nonexistent_tool_v1]
  optional_tools: []
  forbidden_tools: []
  permitted_params: {}
  fallbacks: {}
  degraded_if: []
"""
    p = tmp_path / "broken_matrix.yaml"
    p.write_text(broken_yaml)
    return p

@pytest.fixture
def broken_config_matrix_path(tmp_path):
    broken_yaml = """
version: cm-2026.11.02
XMODAL_JOINT_EXTRACT:
  description: "Extract complementary information"
  requires:
    config: SINGLE
  tools: [index_engine_v1, optsar_fusion_v1, rs_vqa_v1]
  optional_tools: []
  forbidden_tools: []
  permitted_params: {}
  fallbacks: {}
  degraded_if: []
"""
    p = tmp_path / "broken_config_matrix.yaml"
    p.write_text(broken_yaml)
    return p

def test_valid_matrix(valid_matrix_path):
    assert validate_matrix(valid_matrix_path) is True

def test_broken_tool_matrix(broken_matrix_path):
    assert validate_matrix(broken_matrix_path) is False

def test_broken_config_matrix(broken_config_matrix_path):
    assert validate_matrix(broken_config_matrix_path) is False
