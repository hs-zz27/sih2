"""Texture measures: GLCM statistics and speckle coefficient of variation.

Texture is the main SWIR-free discriminator for built-up surfaces, and CoV is
the standard SAR speckle diagnostic. Both are windowed, so they return a
raster of the same shape as the input rather than a single number.

MEMORY AND NUMERICS. Real Cartosat scenes are ~59 megapixels (7687x7640), so
each full-resolution float64 intermediate costs ~450 MiB and the naive
implementation exhausted memory on a 16 GB machine. Two changes fix that:

* work in **float32**, halving every intermediate, and
* compute the windowed variance about a **shifted origin** (the global mean)
  rather than as ``E[x^2] - E[x]^2``.

The second point is not only about memory. For SAR intensity, where values are
large and the variance is comparatively small, ``E[x^2] - E[x]^2`` suffers
catastrophic cancellation - in float32 it can return small negative numbers for
genuinely positive variances. Centring first removes the large common term
before squaring, which keeps the subtraction well conditioned.

Full-resolution processing still has a ceiling; the tile pyramid (task 2.10)
is the structural fix for scenes larger than memory.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops

GLCM_PROPERTIES = ("contrast", "homogeneity", "energy", "correlation")

# Windowed moments are computed in this dtype. float32 halves peak memory
# versus float64 on the large scenes this runs against.
_WORK_DTYPE = "float32"


def _windowed_moments(
    arr: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (valid mask, windowed mean, windowed variance).

    Variance is computed about the global mean to avoid the cancellation that
    makes ``E[x^2] - E[x]^2`` unreliable in float32.
    """
    a = np.asarray(arr, dtype=_WORK_DTYPE)
    valid = np.isfinite(a)

    finite = a[valid]
    if finite.size == 0:
        nan = np.full(a.shape, np.nan, dtype=_WORK_DTYPE)
        return valid, nan, nan.copy()

    # Shift so the windowed sums are of small numbers, not large ones.
    origin = np.float32(finite.mean())
    centred = np.where(valid, a - origin, np.float32(0.0))

    ones = valid.astype(_WORK_DTYPE)
    # uniform_filter returns the mean over the window; multiplying by the
    # window area recovers the sum without a second full-size temporary.
    area = float(window * window)
    count = ndimage.uniform_filter(ones, size=window) * area
    total = ndimage.uniform_filter(centred, size=window) * area
    np.square(centred, out=centred)  # in place: no extra full-size array
    total_sq = ndimage.uniform_filter(centred, size=window) * area

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_centred = np.where(count > 0, total / count, np.nan)
        mean_sq = np.where(count > 0, total_sq / count, np.nan)
        variance = np.clip(mean_sq - mean_centred * mean_centred, 0.0, None)

    mean = mean_centred + origin  # undo the shift for the true mean
    return valid, mean, variance


def coefficient_of_variation(arr: np.ndarray, window: int = 7) -> np.ndarray:
    """Windowed std/mean - the classic SAR speckle measure.

    For fully developed speckle CoV approaches a known constant set by the
    number of looks, so departures from it indicate real texture rather than
    noise. Smooth surfaces (water) give low CoV; urban areas give high CoV.
    """
    valid, mean, variance = _windowed_moments(arr, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = np.where(mean > 0, np.sqrt(variance) / mean, np.nan)
    return np.where(valid, cov, np.nan)


def local_variance(arr: np.ndarray, window: int = 7) -> np.ndarray:
    """Windowed variance - a cheap texture-roughness raster.

    Used by the SWIR-free built-up proxy, where a full GLCM per pixel would be
    far too slow.
    """
    valid, _, variance = _windowed_moments(arr, window)
    return np.where(valid, variance, np.nan)


def _quantise(arr: np.ndarray, levels: int) -> np.ndarray:
    """Rescale finite values to 0..levels-1 integers for GLCM."""
    a = np.asarray(arr, dtype=_WORK_DTYPE)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, dtype="uint8")
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype="uint8")
    scaled = (a - lo) / (hi - lo) * (levels - 1)
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return np.clip(np.rint(scaled), 0, levels - 1).astype("uint8")


# GLCM cost grows with pixel count; above this many pixels the image is
# decimated first. Texture statistics are distributional, so a regular
# subsample of a large scene preserves them while keeping runtime bounded.
_GLCM_MAX_PIXELS = 4_000_000


def glcm_features(
    arr: np.ndarray,
    *,
    levels: int = 32,
    distances: tuple[int, ...] = (1,),
    angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> dict[str, float]:
    """Scene-level GLCM statistics, averaged over angles (rotation invariant).

    Returns one number per property. Windowed GLCM over a whole scene is
    prohibitively slow, and the verifier compares distributions rather than
    per-pixel values, so scene-level statistics are the right granularity.
    """
    a = np.asarray(arr)
    if a.size > _GLCM_MAX_PIXELS:
        step = int(np.ceil(np.sqrt(a.size / _GLCM_MAX_PIXELS)))
        a = a[::step, ::step]

    q = _quantise(a, levels)
    glcm = graycomatrix(
        q, distances=list(distances), angles=list(angles), levels=levels,
        symmetric=True, normed=True,
    )
    return {prop: float(np.mean(graycoprops(glcm, prop))) for prop in GLCM_PROPERTIES}
