"""Turn a geospatial raster into an RGB image a vision model can consume.

VLMs expect 8-bit RGB. Remote-sensing rasters are typically 11-16 bit, have
4-12 bands in no fixed order, and can be 59 megapixels. Bridging that gap
correctly matters more than it looks:

* **Band selection** must go through the canonical names, because band 1 is
  blue on Cartosat MX and HH on EOS-04. Assuming positional RGB would render
  a false-colour image and silently change what the model is asked about.
* **Stretching** must use percentiles, not the dtype range. Cartosat MX is
  11-bit data inside a uint16 container, so dividing by 65535 produces a
  near-black image - the model would be answering questions about darkness.
* **Downsampling** is required: a 7687x7640 scene would expand into an
  enormous number of visual tokens and exhaust VRAM.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling

from satquery.contracts.input_manifest import ImageMeta

# Longest edge of the RGB preview handed to the model. Qwen2.5-VL uses dynamic
# resolution, so this directly bounds the visual token count - and therefore
# the activation memory, which is the binding constraint on a 6 GB GPU.
DEFAULT_MAX_EDGE = 512

# Percentile clip for contrast stretching. 2-98 keeps sensor hot pixels and
# deep shadow from consuming the whole dynamic range.
CLIP_PERCENTILES = (2.0, 98.0)

RGB_BANDS = ("RED", "GREEN", "BLUE")


def _select_band_indices(meta: ImageMeta) -> tuple[list[int], str]:
    """Return 1-based band indices for RGB, and how they were chosen."""
    if all(b in meta.bands for b in RGB_BANDS):
        return [meta.bands.index(b) + 1 for b in RGB_BANDS], "canonical_rgb"

    # SAR and other non-optical products have no RGB. Showing one band as
    # greyscale is honest; inventing colour would not be.
    if len(meta.bands) >= 3:
        return [1, 2, 3], "first_three_bands"
    return [1, 1, 1], "single_band_greyscale"


def _stretch(band: np.ndarray) -> np.ndarray:
    """Percentile stretch a single band to 0-255 uint8."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros(band.shape, dtype="uint8")

    lo, hi = np.percentile(finite, CLIP_PERCENTILES)
    if hi <= lo:
        # Flat band: mid-grey is more honest than an arbitrary ramp.
        return np.full(band.shape, 128, dtype="uint8")

    scaled = (band - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return (scaled * 255).astype("uint8")


def to_rgb_preview(meta: ImageMeta, max_edge: int = DEFAULT_MAX_EDGE):
    """Render an `ImageMeta` as a PIL RGB image, downsampled and stretched.

    Returns (PIL.Image, provenance dict). The provenance goes into the trace
    so it is always visible which bands the model actually saw.
    """
    from PIL import Image

    indices, how = _select_band_indices(meta)

    scale = max(meta.width, meta.height) / max_edge
    out_h = max(1, int(meta.height / scale)) if scale > 1 else meta.height
    out_w = max(1, int(meta.width / scale)) if scale > 1 else meta.width

    channels = []
    with rasterio.open(meta.path) as src:
        for idx in indices:
            arr = src.read(
                idx,
                out_shape=(out_h, out_w),
                resampling=Resampling.average,
                masked=True,
            )
            channels.append(_stretch(np.ma.filled(arr.astype("float64"), np.nan)))

    rgb = np.dstack(channels)
    return Image.fromarray(rgb, mode="RGB"), {
        "band_selection": how,
        "bands_shown": [meta.bands[i - 1] if i <= len(meta.bands) else f"band_{i}"
                        for i in indices],
        "preview_size": [out_w, out_h],
        "source_size": [meta.width, meta.height],
        "downsample_factor": round(scale, 3) if scale > 1 else 1.0,
        "stretch": f"percentile_{CLIP_PERCENTILES[0]}_{CLIP_PERCENTILES[1]}",
    }
