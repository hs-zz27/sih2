"""Tests for the deterministic index engine (plan task 1.2).

The point of these tests is exact arithmetic against hand-computed answers.
The index engine is what the neural outputs get checked against, so it is the
one component that must be provably right rather than approximately right.
"""

from __future__ import annotations

import numpy as np
import pytest

from satquery.verify import (
    adaptive_threshold,
    apply_threshold,
    coefficient_of_variation,
    glcm_features,
    gmm_threshold,
    index_stats,
    mndwi,
    ndbi,
    ndvi,
    ndwi,
    normalised_difference,
    otsu_threshold,
    polarisation_ratio_db,
    sigma0_db,
    swir_free_builtup_proxy,
)


class TestNormalisedDifference:
    def test_known_answer(self):
        """(8-2)/(8+2) = 0.6, computed by hand."""
        result = normalised_difference(np.array([8.0]), np.array([2.0]))
        assert result[0] == pytest.approx(0.6)

    def test_identical_inputs_give_zero(self):
        result = normalised_difference(np.array([5.0]), np.array([5.0]))
        assert result[0] == pytest.approx(0.0)

    def test_zero_denominator_is_nan_not_crash(self):
        result = normalised_difference(np.array([0.0]), np.array([0.0]))
        assert np.isnan(result[0])

    def test_opposite_signs_cancelling_denominator(self):
        result = normalised_difference(np.array([3.0]), np.array([-3.0]))
        assert np.isnan(result[0])

    def test_nan_input_propagates(self):
        result = normalised_difference(np.array([np.nan]), np.array([2.0]))
        assert np.isnan(result[0])

    def test_output_within_physical_range(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(1, 1000, 500)
        b = rng.uniform(1, 1000, 500)
        out = normalised_difference(a, b)
        assert np.all(out >= -1.0) and np.all(out <= 1.0)


class TestIndexFormulae:
    def test_ndvi_argument_order(self):
        """NDVI = (NIR-RED)/(NIR+RED). Vegetation (high NIR) must be positive."""
        red = np.array([1000.0])
        nir = np.array([3000.0])
        assert ndvi(red, nir)[0] == pytest.approx((3000 - 1000) / (3000 + 1000))
        assert ndvi(red, nir)[0] > 0

    def test_ndwi_argument_order(self):
        """NDWI = (GREEN-NIR)/(GREEN+NIR). Water (low NIR) must be positive."""
        green = np.array([2000.0])
        nir = np.array([200.0])
        assert ndwi(green, nir)[0] == pytest.approx((2000 - 200) / (2000 + 200))
        assert ndwi(green, nir)[0] > 0

    def test_mndwi_argument_order(self):
        """MNDWI = (GREEN-SWIR1)/(GREEN+SWIR1)."""
        green = np.array([2000.0])
        swir1 = np.array([500.0])
        assert mndwi(green, swir1)[0] == pytest.approx((2000 - 500) / (2000 + 500))

    def test_ndbi_argument_order(self):
        """NDBI = (SWIR1-NIR)/(SWIR1+NIR). Built-up (high SWIR) positive."""
        swir1 = np.array([2500.0])
        nir = np.array([1500.0])
        assert ndbi(swir1, nir)[0] == pytest.approx((2500 - 1500) / (2500 + 1500))

    def test_ndvi_and_ndbi_disagree_on_vegetation(self):
        """A vegetated pixel: high NDVI, negative NDBI. Sanity of sign conventions."""
        red, nir, swir1 = np.array([500.0]), np.array([4000.0]), np.array([1000.0])
        assert ndvi(red, nir)[0] > 0.5
        assert ndbi(swir1, nir)[0] < 0


class TestSAR:
    def test_sigma0_amplitude_known_answer(self):
        """Amplitude 10 -> power 100 -> 10*log10(100) = 20 dB."""
        assert sigma0_db(np.array([10.0]))[0] == pytest.approx(20.0)

    def test_sigma0_intensity_known_answer(self):
        """Intensity 100 -> 10*log10(100) = 20 dB."""
        assert sigma0_db(np.array([100.0]), is_intensity=True)[0] == pytest.approx(20.0)

    def test_sigma0_calibration_constant_is_additive_in_db(self):
        base = sigma0_db(np.array([10.0]))[0]
        shifted = sigma0_db(np.array([10.0]), calibration_constant=-83.0)[0]
        assert shifted == pytest.approx(base - 83.0)

    def test_sigma0_zero_is_nan_not_negative_infinity(self):
        assert np.isnan(sigma0_db(np.array([0.0]))[0])

    def test_polarisation_ratio_known_answer(self):
        """VH/VV = 1/10 -> 10*log10(0.1) = -10 dB."""
        result = polarisation_ratio_db(np.array([1.0]), np.array([10.0]))
        assert result[0] == pytest.approx(-10.0)

    def test_equal_polarisations_give_zero_db(self):
        result = polarisation_ratio_db(np.array([5.0]), np.array([5.0]))
        assert result[0] == pytest.approx(0.0)

    def test_zero_denominator_is_nan(self):
        assert np.isnan(polarisation_ratio_db(np.array([1.0]), np.array([0.0]))[0])


class TestTexture:
    def test_cov_of_constant_field_is_zero(self):
        """No variation -> coefficient of variation 0."""
        cov = coefficient_of_variation(np.full((32, 32), 7.0), window=5)
        centre = cov[8:24, 8:24]  # avoid edge effects
        assert np.allclose(centre, 0.0, atol=1e-9)

    def test_cov_higher_for_noisier_field(self):
        rng = np.random.default_rng(1)
        smooth = np.full((64, 64), 10.0) + rng.normal(0, 0.1, (64, 64))
        rough = np.full((64, 64), 10.0) + rng.normal(0, 3.0, (64, 64))
        assert np.nanmean(coefficient_of_variation(rough)) > np.nanmean(
            coefficient_of_variation(smooth)
        )

    def test_glcm_returns_all_properties(self):
        rng = np.random.default_rng(2)
        feats = glcm_features(rng.random((64, 64)))
        assert set(feats) == {"contrast", "homogeneity", "energy", "correlation"}
        assert all(np.isfinite(v) for v in feats.values())

    def test_glcm_contrast_higher_for_noisy_image(self):
        rng = np.random.default_rng(3)
        smooth = np.tile(np.linspace(0, 1, 64), (64, 1))
        noisy = rng.random((64, 64))
        assert glcm_features(noisy)["contrast"] > glcm_features(smooth)["contrast"]


class TestThresholding:
    def _bimodal(self, n=2000, seed=4):
        rng = np.random.default_rng(seed)
        return np.concatenate(
            [rng.normal(-0.6, 0.08, n), rng.normal(0.6, 0.08, n)]
        )

    def test_otsu_finds_split_between_two_clusters(self):
        data = self._bimodal()
        result = otsu_threshold(data)
        assert result is not None
        assert result.method == "otsu"
        assert result.bimodal is True
        assert -0.4 < result.value < 0.4  # between the two modes

    def test_gmm_finds_split_between_two_clusters(self):
        result = gmm_threshold(self._bimodal())
        assert result is not None
        assert result.method == "gmm"
        assert -0.4 < result.value < 0.4

    def test_adaptive_prefers_otsu_when_bimodal(self):
        result = adaptive_threshold(self._bimodal(), fixed_prior=0.0)
        assert result.method == "otsu"
        assert result.fallback_reason is None

    def test_unimodal_data_falls_back_to_fixed_prior(self):
        """A single-class scene has no meaningful threshold; say so explicitly."""
        rng = np.random.default_rng(5)
        result = adaptive_threshold(rng.normal(0.5, 0.01, 2000), fixed_prior=0.3)
        assert result.method == "fixed_prior"
        assert result.value == 0.3
        assert result.bimodal is False
        assert "single class" in result.fallback_reason

    def test_too_few_pixels_falls_back_with_named_reason(self):
        result = adaptive_threshold(np.array([1.0, 2.0, 3.0]), fixed_prior=0.25)
        assert result.method == "fixed_prior"
        assert "too_few_valid_pixels" in result.fallback_reason

    def test_all_nan_input_does_not_crash(self):
        result = adaptive_threshold(np.full(100, np.nan), fixed_prior=0.1)
        assert result.method == "fixed_prior"
        assert result.n_pixels == 0

    def test_apply_threshold_masks_correctly(self):
        arr = np.array([0.0, 0.5, 1.0])
        result = adaptive_threshold(arr, fixed_prior=0.4)
        mask = apply_threshold(arr, result)
        assert mask.tolist() == [False, True, True]

    def test_apply_threshold_treats_nan_as_false(self):
        arr = np.array([np.nan, 1.0])
        result = adaptive_threshold(arr, fixed_prior=0.5)
        assert apply_threshold(arr, result).tolist() == [False, True]


class TestSwirFreeFallback:
    def test_builtup_proxy_in_unit_range(self):
        rng = np.random.default_rng(6)
        red = rng.uniform(100, 2000, (32, 32))
        nir = rng.uniform(100, 2000, (32, 32))
        proxy = swir_free_builtup_proxy(red, nir)
        finite = proxy[np.isfinite(proxy)]
        assert finite.min() >= 0.0 and finite.max() <= 1.0

    def test_vegetated_pixels_score_low(self):
        """High NIR relative to RED means vegetation, so low built-up score."""
        red = np.full((16, 16), 500.0)
        nir = np.full((16, 16), 4000.0)  # NDVI ~ 0.78
        assert float(np.nanmean(swir_free_builtup_proxy(red, nir))) < 0.2

    def test_bare_pixels_score_higher_than_vegetated(self):
        veg = swir_free_builtup_proxy(
            np.full((16, 16), 500.0), np.full((16, 16), 4000.0)
        )
        bare = swir_free_builtup_proxy(
            np.full((16, 16), 2000.0), np.full((16, 16), 2100.0)
        )
        assert np.nanmean(bare) > np.nanmean(veg)

    def test_sar_term_included_when_available(self):
        """Adding a SAR term must change the result, proving it is used."""
        red = np.full((16, 16), 1500.0)
        nir = np.full((16, 16), 1600.0)
        rng = np.random.default_rng(9)
        without = swir_free_builtup_proxy(red, nir)
        with_sar = swir_free_builtup_proxy(
            red, nir, sigma0_vv=rng.uniform(-20, 0, (16, 16))
        )
        assert not np.allclose(np.nanmean(without), np.nanmean(with_sar))


class TestIndexStats:
    def test_known_values(self):
        stats = index_stats(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["min"] == pytest.approx(0.0)
        assert stats["max"] == pytest.approx(4.0)
        assert stats["p50"] == pytest.approx(2.0)
        assert stats["valid_fraction"] == pytest.approx(1.0)

    def test_valid_fraction_accounts_for_nan(self):
        stats = index_stats(np.array([1.0, np.nan, 3.0, np.nan]))
        assert stats["valid_fraction"] == pytest.approx(0.5)
        assert stats["mean"] == pytest.approx(2.0)

    def test_all_nan_returns_nan_not_crash(self):
        stats = index_stats(np.full(10, np.nan))
        assert np.isnan(stats["mean"])
        assert stats["valid_fraction"] == 0.0
