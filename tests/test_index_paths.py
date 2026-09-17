"""Dataset index paths must resolve on the OS that reads them, not the one
that wrote them.

The four dataset indexes and the demo manifest were generated on Windows and
store backslash separators. On Linux a backslash is an ordinary filename
character, so every open failed and `make_demo_bundle.py --verify` could not
run inside a container at all - the demo's 9/9 guarantee was Windows-only.
Measured 2026-09-07 in the API container:

    FileNotFoundError: 'data\\levircd\\tiles\\test\\label\\000000.png'

`training.common.paths.index_path` normalises on read. These tests pin both
halves: that the helper handles either convention, and that the call sites
actually use it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PureWindowsPath

import pytest

from training.common.paths import index_path

BACKSLASH = chr(92)

# Every module that turns a string stored in an index or manifest into a file
# it opens. If a new one appears, it belongs here.
CALL_SITES = [
    "training/track_a_encoder.py",
    "training/stage_a2_transfer.py",
    "training/train_change_mask.py",
    "training/train_change_caption.py",
    "training/train_optsar_fusion.py",
    "training/prepare/instruction_mix.py",
    "scripts/make_demo_bundle.py",
    "scripts/rehearse.py",
]

INDEXES = [
    "data/levircd/index.json",
    "data/bigearthnet_14k/index.json",
    "data/levir_mci/index.json",
    "data/whu_opt_sar/index.json",
]


class TestSeparatorHandling:
    def test_windows_relative_path_is_split(self):
        stored = BACKSLASH.join(["data", "levircd", "tiles", "a.png"])
        assert index_path(stored) == Path("data/levircd/tiles/a.png")

    def test_posix_relative_path_is_unchanged_in_meaning(self):
        assert index_path("data/levircd/tiles/a.png") == Path(
            "data/levircd/tiles/a.png"
        )

    def test_both_conventions_agree(self):
        win = BACKSLASH.join(["data", "x", "y.png"])
        assert index_path(win) == index_path("data/x/y.png")

    def test_parts_are_real_components_not_one_filename(self):
        # The actual bug: the whole string was read as a single filename.
        stored = BACKSLASH.join(["data", "levircd", "tiles", "a.png"])
        assert len(index_path(stored).parts) == 4
        assert index_path(stored).name == "a.png"

    def test_a_path_object_survives_a_round_trip(self):
        assert index_path(Path("data/x/y.png")) == Path("data/x/y.png")


class TestAbsolutePathsArePassedThrough:
    """An absolute path from the other OS is unusable; do not mangle it."""

    def test_posix_absolute_is_not_resplit(self):
        assert index_path("/app/data/x.png") == Path("/app/data/x.png")

    def test_windows_absolute_is_not_resplit(self):
        stored = "C:" + BACKSLASH + "data" + BACKSLASH + "x.png"
        # Rebuilt as itself rather than silently demoted to a relative path.
        assert str(index_path(stored)).startswith("C:")

    def test_empty_string_does_not_raise(self):
        assert index_path("") == Path("")


class TestAgainstTheRealIndexes:
    @pytest.mark.parametrize("rel", INDEXES)
    def test_first_stored_path_resolves(self, rel):
        idx = Path(rel)
        if not idx.exists():
            pytest.skip(f"{rel} not on disk")
        blob = json.loads(idx.read_text(encoding="utf-8"))

        stored = None
        for value in _walk_strings(blob):
            if PureWindowsPath(value).suffix in {".png", ".tif", ".tiff", ".jpg"}:
                stored = value
                break
        if stored is None:
            pytest.skip(f"no image path found in {rel}")

        resolved = index_path(stored)
        assert resolved.exists(), (
            f"{rel}: {stored!r} did not resolve to a real file "
            f"(got {resolved}). This is the container failure."
        )


def _walk_strings(blob, budget=2000):
    """Yield string leaves, breadth-limited so a huge index stays cheap."""
    stack = [blob]
    seen = 0
    while stack and seen < budget:
        item = stack.pop()
        if isinstance(item, str):
            seen += 1
            yield item
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item[:50])


class TestCallSitesUseTheHelper:
    """A raw open() on a stored path is the bug; catch it generically."""

    @pytest.mark.parametrize("rel", CALL_SITES)
    def test_no_raw_open_of_a_stored_path(self, rel):
        text = Path(rel).read_text(encoding="utf-8")
        # `Image.open(row["a"])` / `rasterio.open(row["label"])` and friends,
        # where the argument is a bare index lookup rather than index_path(...).
        raw = re.findall(
            r"(?:Image|rasterio)\.open\(\s*(?:row|r|best|item|rows\[[^\]]*\])\[",
            text,
        )
        assert not raw, (
            f"{rel} opens a path straight out of an index: {raw}. "
            "Wrap it in training.common.paths.index_path so it resolves on "
            "Linux as well as Windows."
        )

    @pytest.mark.parametrize("rel", CALL_SITES)
    def test_helper_is_imported(self, rel):
        text = Path(rel).read_text(encoding="utf-8")
        assert "from training.common.paths import index_path" in text
