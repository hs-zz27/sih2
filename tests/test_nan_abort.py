"""A diverged run must stop in its first minutes, not at the end of its budget.

The HF Job bills by the second, so the expensive failure is not a crash - it is
a run that trains for hours on NaN weights and only fails `check_trained()` at
packaging time, having spent the whole GPU budget to produce nothing. Both
trainers now abort on a streak of non-finite losses.

This drives the real trainer end to end on four synthetic tiles with an
absurd learning rate, which makes divergence certain rather than likely.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")


def _tiles(root: Path, n: int = 8, size: int = 32) -> list[dict]:
    """n change pairs with some positive pixels, so pos_weight stays finite."""
    rows = []
    for split in ("train",):
        for i in range(n):
            d = root / "tiles" / split
            (d / "a").mkdir(parents=True, exist_ok=True)
            (d / "b").mkdir(parents=True, exist_ok=True)
            (d / "label").mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(i)
            a = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            b = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[: size // 2, : size // 2] = 255      # a quarter changed
            Image.fromarray(a).save(d / "a" / f"{i:03d}.png")
            Image.fromarray(b).save(d / "b" / f"{i:03d}.png")
            Image.fromarray(mask).save(d / "label" / f"{i:03d}.png")
            rows.append({
                "a": f"tiles/{split}/a/{i:03d}.png",
                "b": f"tiles/{split}/b/{i:03d}.png",
                "label": f"tiles/{split}/label/{i:03d}.png",
            })
    return rows


def test_change_mask_aborts_on_a_streak_of_non_finite_losses(tmp_path):
    rows = _tiles(tmp_path)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"splits": {"train": rows, "test": []}}), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(ROOT / "training" / "train_change_mask.py"),
         "--index", str(index), "--ckpt-dir", str(tmp_path / "ckpt"),
         "--epochs", "3", "--batch-size", "1", "--dim", "4",
         # An absurd learning rate: AdamW's first step moves the weights far
         # enough that the forward pass overflows, so divergence is certain.
         "--lr", "1e30", "--nan-patience", "2"],
        cwd=tmp_path, capture_output=True, text=True, timeout=600,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )

    assert result.returncode == 2, (result.returncode, result.stdout[-3000:], result.stderr[-2000:])
    assert "Training aborted" in result.stdout
    assert "non-finite loss" in result.stdout


def test_change_mask_trains_normally_when_the_loss_stays_finite(tmp_path):
    """The guard must not fire on a healthy run - an abort that triggers on
    ordinary training would cost more than the failure it prevents."""
    rows = _tiles(tmp_path)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"splits": {"train": rows, "test": []}}), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(ROOT / "training" / "train_change_mask.py"),
         "--index", str(index), "--ckpt-dir", str(tmp_path / "ckpt"),
         "--epochs", "2", "--batch-size", "2", "--dim", "4", "--lr", "1e-3"],
        cwd=tmp_path, capture_output=True, text=True, timeout=600,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )

    assert result.returncode == 0, (result.stdout[-3000:], result.stderr[-2000:])
    assert "Training aborted" not in result.stdout
    assert "non-finite" not in result.stdout
