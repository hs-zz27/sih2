"""Track A full: 12-band encoder with GSD conditioning (plan task 2.1)."""

from __future__ import annotations

import numpy as np
import pytest

# torch is a TRAINING dependency, not a runtime one: the ingest pipeline,
# index engine, controller, API and evidence pack all work without it. CI
# therefore does not install it, and these tests skip there rather than
# adding an ~800 MB download to every run. They do execute locally, where
# the training environment exists.
pytest.importorskip("torch")
import torch

from training.track_a_full import (
    BAND_NAMES_12, CANONICAL_12, CARTOSAT_IDX_12, N_CLASSES, build_model,
)


class TestBandDefinitions:
    def test_twelve_bands(self):
        assert len(BAND_NAMES_12) == 12

    def test_cartosat_subset_is_vnir_without_swir(self):
        names = [CANONICAL_12[b] for b in ("B02", "B03", "B04", "B08")]
        assert names == ["BLUE", "GREEN", "RED", "NIR"]
        assert len(CARTOSAT_IDX_12) == 4

    def test_indices_point_at_the_right_bands(self):
        """An off-by-one here would silently feed the model red-edge bands
        while the trace claimed VNIR."""
        assert [BAND_NAMES_12[i] for i in CARTOSAT_IDX_12] == [
            "B02", "B03", "B04", "B08"
        ]


class TestGSDConditioning:
    @staticmethod
    def _inputs(n=2, c=12, s=32):
        g = torch.Generator().manual_seed(3)
        return torch.randn(n, c, s, s, generator=g), torch.ones(n, c)

    def test_output_shape(self):
        model = build_model(dim=8).eval()
        x, m = self._inputs()
        assert model(x, m, torch.full((2,), 10.0)).shape == (2, N_CLASSES)

    def test_gsd_changes_the_prediction(self):
        """If GSD did not alter the output, the conditioning would be inert
        and the resolution gap could not be compensated for."""
        model = build_model(dim=8).eval()
        x, m = self._inputs()
        at_10m = model(x, m, torch.full((2,), 10.0))
        at_1m6 = model(x, m, torch.full((2,), 1.6))
        assert not torch.allclose(at_10m, at_1m6, atol=1e-5)

    def test_disabling_gsd_ignores_it(self):
        model = build_model(dim=8, gsd_conditioning=False).eval()
        x, m = self._inputs()
        a = model(x, m, torch.full((2,), 10.0))
        b = model(x, m, torch.full((2,), 1.6))
        assert torch.allclose(a, b)

    def test_missing_gsd_is_tolerated(self):
        model = build_model(dim=8).eval()
        x, m = self._inputs()
        assert torch.isfinite(model(x, m, None)).all()

    def test_zero_gsd_does_not_produce_nan(self):
        """log(0) would be -inf; the clamp must hold."""
        model = build_model(dim=8).eval()
        x, m = self._inputs()
        assert torch.isfinite(model(x, m, torch.zeros(2))).all()


class TestBandMasking:
    @staticmethod
    def _inputs(n=2, c=12, s=32):
        g = torch.Generator().manual_seed(5)
        return torch.randn(n, c, s, s, generator=g)

    def test_runs_with_only_cartosat_bands(self):
        model = build_model(dim=8).eval()
        mask = torch.zeros(2, 12)
        mask[:, CARTOSAT_IDX_12] = 1.0
        out = model(self._inputs(), mask, torch.full((2,), 1.6))
        assert out.shape == (2, N_CLASSES) and torch.isfinite(out).all()

    def test_absent_bands_cannot_influence_the_output(self):
        model = build_model(dim=8).eval()
        x = self._inputs()
        mask = torch.zeros(2, 12)
        mask[:, CARTOSAT_IDX_12] = 1.0
        baseline = model(x, mask, torch.full((2,), 1.6))

        corrupted = x.clone()
        absent = [i for i in range(12) if i not in CARTOSAT_IDX_12]
        corrupted[:, absent] = 1e4
        assert torch.allclose(baseline, model(corrupted, mask, torch.full((2,), 1.6)),
                              atol=1e-4)

    def test_scale_preserved_across_band_counts(self):
        model = build_model(dim=8).eval()
        x = self._inputs()
        full = model(x, torch.ones(2, 12), torch.full((2,), 10.0))
        mask = torch.zeros(2, 12)
        mask[:, CARTOSAT_IDX_12] = 1.0
        partial = model(x, mask, torch.full((2,), 10.0))
        ratio = partial.abs().mean() / full.abs().mean()
        assert 0.2 < ratio < 5.0


class TestMultiResolutionAugmentation:
    """BigEarthNet is uniformly 10 m, so without augmentation the GSD input is
    constant and the conditioning has nothing to learn from."""

    def test_factor_one_is_identity(self):
        from training.track_a_full import degrade_resolution

        x = np.random.default_rng(1).random((2, 3, 12, 12)).astype("float32")
        assert np.array_equal(degrade_resolution(x, 1), x)

    def test_shape_preserved(self):
        from training.track_a_full import degrade_resolution

        x = np.random.default_rng(2).random((2, 3, 120, 120)).astype("float32")
        for f in (2, 3, 4):
            assert degrade_resolution(x, f).shape == x.shape

    def test_coarser_factor_removes_more_detail(self):
        """Monotonic smoothing is the property; if it were not monotonic the
        'effective GSD' label would not match the actual degradation."""
        from training.track_a_full import degrade_resolution

        x = np.random.default_rng(3).random((2, 3, 120, 120)).astype("float32")
        stds = [degrade_resolution(x, f).std() for f in (1, 2, 3, 4)]
        assert stds == sorted(stds, reverse=True)

    def test_mean_roughly_preserved(self):
        """Block averaging must not shift the radiometry, only the detail."""
        from training.track_a_full import degrade_resolution

        x = np.random.default_rng(4).random((2, 3, 120, 120)).astype("float32")
        assert degrade_resolution(x, 4).mean() == pytest.approx(x.mean(), rel=0.05)
