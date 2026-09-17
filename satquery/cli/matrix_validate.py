import sys
import argparse
from pathlib import Path
from pydantic import ValidationError
from satquery.controller.matrix_loader import load_matrix
from satquery.tools.stubs import REGISTRY

EXPECTED_CONFIGS = {
    "SINGLE_VQA": ["SINGLE", "CROSSMODAL_PAIR", "BITEMPORAL_PAIR"],
    "SINGLE_CAPTION": ["SINGLE", "CROSSMODAL_PAIR", "BITEMPORAL_PAIR"],
    "SINGLE_GROUND": ["SINGLE", "CROSSMODAL_PAIR", "BITEMPORAL_PAIR"],
    "SINGLE_LANDCOVER": ["SINGLE", "CROSSMODAL_PAIR", "BITEMPORAL_PAIR"],
    "XMODAL_JOINT_EXTRACT": ["CROSSMODAL_PAIR"],
    "TEMPORAL_CHANGE_DESC": ["BITEMPORAL_PAIR"],
    "TEMPORAL_CHANGE_VQA": ["BITEMPORAL_PAIR"],
    "TEMPORAL_CHANGE_MAP": ["BITEMPORAL_PAIR"],
    "CLARIFY_OR_ABSTAIN": ["any"],
}

def validate_matrix(path: Path) -> bool:
    try:
        matrix = load_matrix(path)
    except ValidationError as e:
        print(f"Validation Error: {e}")
        return False
    except Exception as e:
        print(f"Error loading matrix: {e}")
        return False

    success = True
    for task_name, task_config in matrix.tasks.items():
        # Task legality check
        if task_name not in EXPECTED_CONFIGS:
            print(f"Task {task_name} is not a valid TaskID.")
            success = False

        # Config match check
        config = task_config.requires.config
        expected = EXPECTED_CONFIGS.get(task_name, [])

        if isinstance(config, list):
            if not all(c in expected for c in config):
                print(f"Task {task_name} has invalid config requirement: {config}. Expected subset of {expected}")
                success = False
        else:
            if config not in expected and config != "any":
                print(f"Task {task_name} has invalid config requirement: {config}. Expected one of {expected}")
                success = False

        # Tool registry check
        all_tools = task_config.tools + task_config.optional_tools + task_config.forbidden_tools
        for tool in all_tools:
            if tool not in REGISTRY:
                print(f"Task {task_name} references nonexistent tool: {tool}")
                success = False

        # Fallback references check
        for src, target in task_config.fallbacks.items():
            if target not in REGISTRY:
                print(f"Task {task_name} fallback points to nonexistent tool: {target}")
                success = False
            if src not in REGISTRY:
                print(f"Task {task_name} fallback source is nonexistent tool: {src}")
                success = False

    return success

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=Path("configs/capability_matrix.yaml"))
    args = parser.parse_args()

    if validate_matrix(args.matrix):
        print("Matrix is valid.")
        sys.exit(0)
    else:
        print("Matrix validation failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
