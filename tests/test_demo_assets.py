"""The native demo's asset list stays in step with docker-compose.yml."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from demo_assets import ASSETS, FIXED_ENV, env_lines, status  # noqa: E402


def compose_paths() -> dict[str, str]:
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    return {k: v for k, v in re.findall(r"-\s*(SATQUERY_[A-Z_]+)=/app/(\S+)", text)}


def test_every_compose_weight_path_is_listed_with_the_same_path():
    compose = compose_paths()
    listed = {a.env: a.path for a in ASSETS}
    assert compose, "no SATQUERY_* paths found in docker-compose.yml"
    assert listed == compose


def test_the_native_run_uses_the_cpu_profile_offline():
    assert FIXED_ENV["SATQUERY_PROFILE"] == "cpu"
    assert FIXED_ENV["HF_HUB_OFFLINE"] == "1"


def test_env_lines_export_every_asset_even_when_missing(tmp_path):
    lines = env_lines(tmp_path)
    for a in ASSETS:
        assert any(line.startswith(f"export {a.env}=") for line in lines)


def test_status_reports_missing_checkpoints(tmp_path):
    rows = status(tmp_path)
    assert all(not r["present"] for r in rows)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "calibration.v2.json").write_text("{}")
    rows = {r["env"]: r for r in status(tmp_path)}
    assert rows["SATQUERY_CALIBRATION"]["present"] is True
