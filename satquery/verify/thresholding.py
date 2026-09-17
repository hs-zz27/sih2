"""Adaptive thresholding.

Fixed index thresholds (e.g. "water is NDWI > 0.3") do not survive a change of
sensor, season or calibration - and since the evaluation sensor is not yet
confirmed (docs/verification.md items 5 and 6), every threshold in this system
is derived from the data itself. Otsu is the default; a two-component GMM is
used when the histogram is not cleanly bimodal, and the fallback is an
explicit fixed prior that is always reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture

# Below this many valid pixels, adaptive estimation is not meaningful.
MIN_PIXELS_FOR_ADAPTIVE = 64

# Separation below this means the histogram is not usefully bimodal, so the
# adaptive estimate should not be trusted on its own. See
# `bimodality_separation` for the scale: splitting a single Gaussian at its
# own mean yields ~1.32, so anything at or below that is indistinguishable
# from a unimodal distribution. Two clearly separated populations score well
# above 2.
MIN_BIMODAL_SEPARATION = 2.0


@dataclass
class ThresholdResult:
    """A threshold plus everything needed to audit how it was chosen."""

    value: float
    method: Literal["otsu", "gmm", "fixed_prior"]
    bimodal: bool
    separation: float
    n_pixels: int
    fallback_reason: str | None = None
    diagnostics: dict = field(default_factory=dict)


def _finite(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype="float64").ravel()
    return a[np.isfinite(a)]


def bimodality_separation(values: np.ndarray, threshold: float) -> float:
    """How well `threshold` splits `values` into two distinct populations.

    Fisher-style: the distance between the two class means divided by the sum
    of their standard deviations. Normalising by *within-class spread* rather
    than by the overall data range is what makes this discriminate genuine
    bimodality - splitting any single Gaussian at its own mean scores a
    constant ~1.32 regardless of its variance, whereas two separated
    populations score far higher. Normalising by the data range instead would
    return a similar value in both cases and would be useless as a test.
    """
    below = values[values <= threshold]
    above = values[values > threshold]
    if below.size < 2 or above.size < 2:
        return 0.0
    within = float(below.std() + above.std())
    gap = float(abs(above.mean() - below.mean()))
    if within == 0:
        # Two perfectly tight, distinct clusters: maximally separated.
        return float("inf") if gap > 0 else 0.0
    return gap / within


def otsu_threshold(arr: np.ndarray) -> ThresholdResult | None:
    """Otsu's between-class variance maximisation. None if not applicable."""
    values = _finite(arr)
    if values.size < MIN_PIXELS_FOR_ADAPTIVE or values.min() == values.max():
        return None
    try:
        t = float(threshold_otsu(values))
    except (ValueError, RuntimeError):
        return None
    sep = bimodality_separation(values, t)
    return ThresholdResult(
        value=t,
        method="otsu",
        bimodal=sep >= MIN_BIMODAL_SEPARATION,
        separation=sep,
        n_pixels=int(values.size),
    )


def gmm_threshold(arr: np.ndarray, *, seed: int = 0) -> ThresholdResult | None:
    """Two-component Gaussian mixture; threshold at the crossing point.

    More tolerant than Otsu when the two populations have very different
    variances - common for water (tight) against land (broad).
    """
    values = _finite(arr)
    if values.size < MIN_PIXELS_FOR_ADAPTIVE or values.min() == values.max():
        return None

    # Subsample: GMM on millions of pixels is slow and no more accurate.
    if values.size > 50_000:
        rng = np.random.default_rng(seed)
        values_fit = rng.choice(values, size=50_000, replace=False)
    else:
        values_fit = values

    try:
        gmm = GaussianMixture(n_components=2, random_state=seed, n_init=2)
        gmm.fit(values_fit.reshape(-1, 1))
    except (ValueError, RuntimeError):
        return None

    means = gmm.means_.ravel()
    order = np.argsort(means)
    lo_mean, hi_mean = means[order]
    lo_sd, hi_sd = np.sqrt(gmm.covariances_.ravel()[order])

    # Threshold at the variance-weighted midpoint between the components.
    denom = lo_sd + hi_sd
    t = float((lo_mean * hi_sd + hi_mean * lo_sd) / denom) if denom > 0 else float(
        (lo_mean + hi_mean) / 2
    )

    sep = bimodality_separation(values, t)
    return ThresholdResult(
        value=t,
        method="gmm",
        bimodal=sep >= MIN_BIMODAL_SEPARATION,
        separation=sep,
        n_pixels=int(values.size),
        diagnostics={
            "component_means": [float(lo_mean), float(hi_mean)],
            "component_sds": [float(lo_sd), float(hi_sd)],
            "weights": [float(w) for w in gmm.weights_[order]],
        },
    )


def adaptive_threshold(
    arr: np.ndarray, *, fixed_prior: float, seed: int = 0
) -> ThresholdResult:
    """Otsu, falling back to GMM, falling back to a stated fixed prior.

    `fixed_prior` is the literature default for the index in question. It is
    only ever used when the data cannot support an adaptive estimate, and the
    result records that this happened so the trace and the confidence penalty
    can reflect it.
    """
    otsu = otsu_threshold(arr)
    if otsu is not None and otsu.bimodal:
        return otsu

    gmm = gmm_threshold(arr, seed=seed)
    if gmm is not None and gmm.bimodal:
        gmm.fallback_reason = (
            "otsu_not_bimodal" if otsu is not None else "otsu_not_applicable"
        )
        return gmm

    # Neither adaptive method found two populations. That is a real finding -
    # a single-class scene (all water, all land) has no meaningful threshold.
    values = _finite(arr)
    best = otsu or gmm
    return ThresholdResult(
        value=float(fixed_prior),
        method="fixed_prior",
        bimodal=False,
        separation=best.separation if best else 0.0,
        n_pixels=int(values.size),
        fallback_reason=(
            "no_bimodal_split_found - scene appears to contain a single class"
            if values.size >= MIN_PIXELS_FOR_ADAPTIVE
            else "too_few_valid_pixels_for_adaptive_estimate"
        ),
    )


def apply_threshold(arr: np.ndarray, result: ThresholdResult) -> np.ndarray:
    """Boolean mask of pixels above the threshold; NaN pixels are False."""
    a = np.asarray(arr, dtype="float64")
    return np.where(np.isfinite(a), a > result.value, False)
