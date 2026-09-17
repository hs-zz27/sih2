"""Track A band-agnostic encoder tests (plan task 1.10).

The property under test is the one the project's adaptation claim depends on:
the model must accept an arbitrary subset of bands, and a missing band must
contribute nothing rather than contributing zeros.
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

from training.prepare.bigearthnet import CLASS_INDEX, CLASSES
from training.track_a_encoder import (
    BAND_NAMES,
    CARTOSAT_INDICES,
    MIN_BANDS_KEPT,
    N_CLASSES,
    average_precision,
    band_dropout_mask,
    build_model,
    load_index,
    mean_average_precision,
)


class TestClassVocabulary:
    def test_nineteen_classes(self):
        assert len(CLASSES) == N_CLASSES == 19

    def test_class_index_is_bijective(self):
        assert len(CLASS_INDEX) == len(CLASSES)
        assert all(CLASSES[i] == name for name, i in CLASS_INDEX.items())

    def test_order_is_fixed(self):
        """A trained head maps index -> class; reordering would silently
        invalidate every saved checkpoint."""
        assert CLASSES[0] == "Agro-forestry areas"
        assert CLASSES[-1] == "Urban fabric"


class TestBandDropout:
    def test_always_keeps_minimum_bands(self):
        rng = np.random.default_rng(0)
        # p=0.99 would otherwise drop nearly everything.
        mask = band_dropout_mask(200, 10, 0.99, rng)
        assert mask.sum(axis=1).min() >= MIN_BANDS_KEPT

    def test_never_exceeds_band_count(self):
        rng = np.random.default_rng(1)
        mask = band_dropout_mask(50, 10, 0.0, rng)
        assert mask.sum(axis=1).max() <= 10

    def test_zero_probability_keeps_all(self):
        rng = np.random.default_rng(2)
        mask = band_dropout_mask(32, 10, 0.0, rng)
        assert (mask == 1.0).all()

    def test_dropout_actually_varies_across_samples(self):
        rng = np.random.default_rng(3)
        mask = band_dropout_mask(64, 10, 0.3, rng)
        assert len({tuple(row) for row in mask}) > 1

    def test_mask_is_float_for_multiplication(self):
        rng = np.random.default_rng(4)
        assert band_dropout_mask(4, 10, 0.3, rng).dtype == np.float32


class TestBandAgnosticModel:
    @pytest.fixture(scope="class")
    @classmethod
    def model(cls):
        torch.manual_seed(0)
        return build_model(n_bands=10, dim=8).eval()

    @staticmethod
    def _batch(n=2, c=10, s=32):
        g = torch.Generator().manual_seed(7)
        return torch.randn(n, c, s, s, generator=g)

    def test_output_shape_is_class_logits(self, model):
        out = model(self._batch(), torch.ones(2, 10))
        assert out.shape == (2, N_CLASSES)

    def test_runs_with_only_four_bands(self, model):
        """Cartosat-2E MX has 4 VNIR bands; the model must accept that."""
        mask = torch.zeros(2, 10)
        mask[:, CARTOSAT_INDICES] = 1.0
        out = model(self._batch(), mask)
        assert out.shape == (2, N_CLASSES)
        assert torch.isfinite(out).all()

    def test_runs_with_a_single_band(self, model):
        mask = torch.zeros(2, 10)
        mask[:, 0] = 1.0
        assert torch.isfinite(model(self._batch(), mask)).all()

    def test_absent_bands_do_not_change_the_result(self, model):
        """The masking guarantee.

        Garbage in a masked-out band must not reach the output. If it does,
        the mask is decorative and 4-band inference would be corrupted by
        whatever happened to sit in the missing channels.
        """
        x = self._batch()
        mask = torch.zeros(2, 10)
        mask[:, CARTOSAT_INDICES] = 1.0

        baseline = model(x, mask)

        corrupted = x.clone()
        absent = [i for i in range(10) if i not in CARTOSAT_INDICES]
        corrupted[:, absent] = 1e4  # extreme values in the absent bands

        assert torch.allclose(baseline, model(corrupted, mask), atol=1e-4)

    def test_present_bands_do_change_the_result(self, model):
        """Sanity check on the previous test: the mask must not ignore
        everything."""
        x = self._batch()
        mask = torch.ones(2, 10)
        changed = x.clone()
        changed[:, 0] += 5.0
        assert not torch.allclose(model(x, mask), model(changed, mask), atol=1e-4)

    def test_masked_mean_preserves_scale(self, model):
        """Dropping bands must not shrink activations toward zero.

        A plain sum would; the divisor is the number of present bands
        precisely so that a 4-band input is not four-tenths the magnitude of
        a 10-band one.
        """
        x = self._batch()
        full = model(x, torch.ones(2, 10))
        mask = torch.zeros(2, 10)
        mask[:, CARTOSAT_INDICES] = 1.0
        partial = model(x, mask)
        ratio = partial.abs().mean() / full.abs().mean()
        assert 0.2 < ratio < 5.0, f"scale collapsed or exploded: ratio {ratio:.3f}"

    def test_band_embedding_is_learnable(self, model):
        assert model.band_embed.requires_grad
        assert model.band_embed.shape == (10, 8)

    def test_gradients_flow_to_present_bands(self, model):
        x = self._batch().requires_grad_(True)
        mask = torch.zeros(2, 10)
        mask[:, CARTOSAT_INDICES] = 1.0
        model(x, mask).sum().backward()
        grad = x.grad.abs().sum(dim=(0, 2, 3))
        assert grad[CARTOSAT_INDICES].sum() > 0
        absent = [i for i in range(10) if i not in CARTOSAT_INDICES]
        assert grad[absent].sum() == pytest.approx(0.0, abs=1e-6)


class TestBandDefinitions:
    def test_ten_bands_expected(self):
        assert len(BAND_NAMES) == 10

    def test_cartosat_subset_is_vnir_without_swir(self):
        from training.track_a_encoder import CANONICAL, CARTOSAT_BANDS

        names = [CANONICAL[b] for b in CARTOSAT_BANDS]
        assert names == ["BLUE", "GREEN", "RED", "NIR"]
        assert "SWIR1" not in names and "SWIR2" not in names

    def test_cartosat_indices_within_range(self):
        assert all(0 <= i < len(BAND_NAMES) for i in CARTOSAT_INDICES)
        assert len(set(CARTOSAT_INDICES)) == 4


class TestMetrics:
    def test_perfect_ranking_scores_one(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        targets = np.array([1, 1, 0, 0])
        assert average_precision(scores, targets) == pytest.approx(1.0)

    def test_worst_ranking_scores_low(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        targets = np.array([0, 0, 1, 1])
        assert average_precision(scores, targets) < 0.6

    def test_absent_class_is_nan_not_zero(self):
        """A class with no positives is undefined, not a score of zero -
        averaging zeros would understate mAP."""
        assert np.isnan(average_precision(np.array([0.5, 0.2]), np.array([0, 0])))

    def test_map_ignores_undefined_classes(self):
        scores = np.zeros((4, N_CLASSES))
        targets = np.zeros((4, N_CLASSES))
        scores[:, 0] = [0.9, 0.8, 0.1, 0.2]
        targets[:, 0] = [1, 1, 0, 0]
        value, per_class = mean_average_precision(scores, targets)
        assert value == pytest.approx(1.0)
        assert np.isnan(per_class[1])


class TestIndex:
    def test_index_roundtrip(self, tmp_path):
        payload = {
            "classes": CLASSES,
            "splits": {"train": [{"patch_id": "p", "s2": "a.tif", "s1": None,
                                  "labels": [0, 5]}]},
        }
        path = tmp_path / "index.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = load_index(path)
        assert loaded["splits"]["train"][0]["labels"] == [0, 5]
        assert loaded["classes"] == CLASSES
