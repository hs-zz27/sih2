"""Co-registration.

Same-modality pairs co-register well with plain phase correlation on
intensity. Optical-SAR pairs do not: backscatter and reflectance are not
linearly related, so intensity correlation is unreliable. For those we
correlate *gradient magnitude* instead - edges (field boundaries, coastlines,
built-up outlines) survive across both modalities even though brightness does
not. This is the `gradient_phase_correlation` method named in the contract.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from satquery.contracts.input_manifest import CoregReport, ImageMeta

# Work at a bounded size: co-registration shift is a global rigid estimate and
# does not need full resolution, and full reads on large scenes are wasteful.
_COREG_SIZE = 512

# Above this residual the correction is not trusted and is not applied.
RESIDUAL_REJECT_PX = 12.0

OPTICAL_MODALITIES = {"OPTICAL", "MSI", "PAN"}


def _read_for_coreg(path, size: int = _COREG_SIZE) -> np.ndarray:
    """Read band 1 at a bounded size as float, NaN-filled."""
    with rasterio.open(path) as src:
        arr = src.read(
            1,
            out_shape=(min(src.height, size), min(src.width, size)),
            resampling=Resampling.bilinear,
            masked=True,
        )
    return np.ma.filled(arr.astype("float64"), np.nan)


def _clean(a: np.ndarray) -> np.ndarray:
    """Replace non-finite values with the finite mean, then zero-centre."""
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a)
    filled = np.where(finite, a, a[finite].mean())
    std = filled.std()
    if std == 0:
        return filled - filled.mean()
    return (filled - filled.mean()) / std


def gradient_magnitude(a: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude - the modality-invariant structural signal."""
    cleaned = _clean(a)
    # Mild smoothing first: SAR speckle otherwise dominates the gradient.
    smoothed = ndimage.gaussian_filter(cleaned, sigma=1.5)
    gx = ndimage.sobel(smoothed, axis=1)
    gy = ndimage.sobel(smoothed, axis=0)
    return np.hypot(gx, gy)


def _match_shapes(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop both arrays to their common shape."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def coregister(reference: ImageMeta, moving: ImageMeta) -> CoregReport:
    """Estimate the rigid shift from `moving` onto `reference`.

    Returns a report describing the shift, its residual, and whether it was
    trusted enough to apply. The shift is reported in both pixels and metres
    so the trace can be read without knowing the GSD.
    """
    cross_modal = (
        reference.modality in OPTICAL_MODALITIES and moving.modality == "SAR"
    ) or (moving.modality in OPTICAL_MODALITIES and reference.modality == "SAR")

    ref_arr = _read_for_coreg(reference.path)
    mov_arr = _read_for_coreg(moving.path)
    ref_arr, mov_arr = _match_shapes(ref_arr, mov_arr)

    if cross_modal:
        method = "gradient_phase_correlation"
        ref_signal = gradient_magnitude(ref_arr)
        mov_signal = gradient_magnitude(mov_arr)
    else:
        method = "phase_correlation"
        ref_signal = _clean(ref_arr)
        mov_signal = _clean(mov_arr)

    shift, error, _ = phase_cross_correlation(
        ref_signal, mov_signal, upsample_factor=10, normalization=None
    )
    dy, dx = float(shift[0]), float(shift[1])

    # `error` from phase_cross_correlation is a normalised translation-
    # invariant error, not a pixel residual. Use the shift magnitude scaled by
    # that error as an interpretable residual estimate.
    residual_px = float(np.hypot(dy, dx) * float(error)) if np.isfinite(error) else float("inf")

    # Scale pixel shift back to the source resolution: we correlated a
    # decimated view, so a shift of 1 decimated pixel is several source pixels.
    with rasterio.open(reference.path) as src:
        scale = max(src.height, src.width) / max(ref_signal.shape)
    dy_src, dx_src = dy * scale, dx * scale

    applied = bool(np.isfinite(residual_px) and residual_px <= RESIDUAL_REJECT_PX)

    return CoregReport(
        method=method,
        shift_px=(round(dx_src, 4), round(dy_src, 4)),
        shift_m=(
            round(dx_src * reference.gsd_m, 4),
            round(dy_src * reference.gsd_m, 4),
        ),
        residual_px=round(residual_px, 4) if np.isfinite(residual_px) else 999.0,
        applied_correction=applied,
    )
