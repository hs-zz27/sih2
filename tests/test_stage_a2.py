"""Stage A2 resolution-bridge tests (plan task 2.2).

The dataset is large and gitignored, so these test the logic that decides
*what gets trained* - band slot mapping, encoder transfer, split honesty -
rather than requiring the download.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

# torch is a TRAINING dependency, not a runtime one: the ingest pipeline,
# index engine, controller, API and evidence pack all work without it. CI
# therefore does not install it, and these tests skip there rather than
# adding an ~800 MB download to every run. They do execute locally, where
# the training environment exists.
pytest.importorskip("torch")
import torch

from training.prepare.whu_opt_sar import CLASSES, build_index, discover
from training.stage_a2_transfer import (
    N_WHU_CLASSES,
    WHU_BAND_ORDER,
    WHU_TO_SLOT,
    load_pretrained_encoder,
    mean_average_precision_n,
    replace_head,
)
from training.track_a_encoder import BAND_NAMES, build_model


class TestBandSlotMapping:
    def test_maps_onto_canonical_slots_not_positions(self):
        """WHU optical is R,G,B,NIR - loading it positionally would tell the
        encoder that RED is blue and NIR is a red-edge band."""
        assert WHU_BAND_ORDER == ["RED", "GREEN", "BLUE", "NIR"]
        assert WHU_TO_SLOT["RED"] == BAND_NAMES.index("B04")
        assert WHU_TO_SLOT["BLUE"] == BAND_NAMES.index("B02")
        assert WHU_TO_SLOT["NIR"] == BAND_NAMES.index("B08")

    def test_slots_are_distinct_and_in_range(self):
        slots = list(WHU_TO_SLOT.values())
        assert len(set(slots)) == len(slots)
        assert all(0 <= s < len(BAND_NAMES) for s in slots)

    def test_whu_bands_are_a_subset_of_cartosat_bands(self):
        """Both sensors are 4-band VNIR, which is why A2 bridges to Cartosat."""
        from training.track_a_encoder import CARTOSAT_INDICES

        assert set(WHU_TO_SLOT.values()) == set(CARTOSAT_INDICES)


class TestEncoderTransfer:
    def test_head_replaced_with_new_class_count(self):
        model = build_model(n_bands=len(BAND_NAMES), dim=8)
        model = replace_head(model, torch, dim=8, n_classes=N_WHU_CLASSES)
        out = model(torch.randn(2, len(BAND_NAMES), 32, 32), torch.ones(2, len(BAND_NAMES)))
        assert out.shape == (2, N_WHU_CLASSES)

    def test_encoder_weights_carry_over(self, tmp_path):
        """The point of A2 is adapting learned features, not retraining."""
        from training.common.checkpointing import save_checkpoint

        source = build_model(n_bands=len(BAND_NAMES), dim=8)
        with torch.no_grad():
            source.band_embed.fill_(0.1234)
        save_checkpoint(tmp_path, 1, source)

        loaded, path = load_pretrained_encoder(tmp_path, torch, "cpu", dim=8)
        assert path is not None
        assert torch.allclose(loaded.band_embed, torch.full_like(loaded.band_embed, 0.1234))

    def test_missing_checkpoint_warns_and_starts_fresh(self, tmp_path, capsys):
        """A silent partial load is how a random encoder gets reported as
        fine-tuned."""
        model, path = load_pretrained_encoder(tmp_path, torch, "cpu", dim=8)
        assert path is None
        assert "training from scratch" in capsys.readouterr().out

    def test_old_head_is_not_transferred(self, tmp_path, capsys):
        from training.common.checkpointing import save_checkpoint

        save_checkpoint(tmp_path, 1, build_model(n_bands=len(BAND_NAMES), dim=8))
        load_pretrained_encoder(tmp_path, torch, "cpu", dim=8)
        assert "reinitialised (head)" in capsys.readouterr().out


class TestIndexBuilding:
    def _tree(self, tmp_path, n=6, with_sar=True):
        for kind in ("optical", "sar", "lbl"):
            (tmp_path / kind).mkdir()
        for i in range(n):
            (tmp_path / "optical" / f"tile_{i}.tif").write_bytes(b"x")
            (tmp_path / "lbl" / f"tile_{i}.tif").write_bytes(b"x")
            if with_sar:
                (tmp_path / "sar" / f"tile_{i}.tif").write_bytes(b"x")
        return tmp_path

    def test_discovers_all_three_kinds(self, tmp_path):
        found = discover(self._tree(tmp_path))
        assert len(found) == 6
        assert set(found["tile_0"]) == {"optical", "sar", "label"}

    def test_label_hint_wins_over_sar_hint(self, tmp_path):
        """A directory called 'sar_label' is a label directory."""
        (tmp_path / "sar_label").mkdir()
        (tmp_path / "sar_label" / "a.tif").write_bytes(b"x")
        assert discover(tmp_path)["a"] == {"label": tmp_path / "sar_label" / "a.tif"}

    def test_index_splits_are_disjoint(self, tmp_path):
        index = build_index(self._tree(tmp_path, n=10), 0.2, seed=1)
        train = {r["id"] for r in index["splits"]["train"]}
        val = {r["id"] for r in index["splits"]["validation"]}
        assert train & val == set()
        assert len(train) + len(val) == 10

    def test_split_is_deterministic(self, tmp_path):
        tree = self._tree(tmp_path, n=10)
        a = build_index(tree, 0.2, seed=7)["splits"]["validation"]
        b = build_index(tree, 0.2, seed=7)["splits"]["validation"]
        assert [r["id"] for r in a] == [r["id"] for r in b]

    def test_split_method_declares_it_is_not_geographic(self, tmp_path):
        """Adjacent WHU tiles come from the same scenes, so the split is
        optimistic. That must be stated, not hidden."""
        index = build_index(self._tree(tmp_path), 0.2, seed=1)
        assert "NOT geographic" in index["split_method"]
        assert "optimistic" in index["split_method"]

    def test_tiles_without_labels_are_skipped(self, tmp_path):
        tree = self._tree(tmp_path, n=4)
        (tree / "optical" / "orphan.tif").write_bytes(b"x")
        index = build_index(tree, 0.0, seed=1)
        ids = {r["id"] for r in index["splits"]["train"]}
        assert "orphan" not in ids
        assert index["n_incomplete"] == 1

    def test_sar_recorded_when_present(self, tmp_path):
        index = build_index(self._tree(tmp_path), 0.0, seed=1)
        assert index["n_with_sar"] == 6
        assert index["splits"]["train"][0]["sar"] is not None

    def test_works_without_sar(self, tmp_path):
        index = build_index(self._tree(tmp_path, with_sar=False), 0.0, seed=1)
        assert index["n_with_sar"] == 0
        assert all(r["sar"] is None for r in index["splits"]["train"])

    def test_class_vocabulary(self):
        assert len(CLASSES) == N_WHU_CLASSES
        assert CLASSES[0] == "background"


class TestMetrics:
    def test_map_handles_arbitrary_class_count(self):
        scores = np.zeros((4, N_WHU_CLASSES))
        targets = np.zeros((4, N_WHU_CLASSES))
        scores[:, 2] = [0.9, 0.8, 0.1, 0.2]
        targets[:, 2] = [1, 1, 0, 0]
        value, per_class = mean_average_precision_n(scores, targets)
        assert value == pytest.approx(1.0)
        assert len(per_class) == N_WHU_CLASSES
