"""Adaptive modality inference and band harmonisation.

The band-agnostic design in `docs/03` requires that we never assume a fixed
band layout. This module maps whatever bands a product actually carries onto
a canonical set, and reports which spectral indices are therefore computable.
Nothing here hardcodes a sensor: everything is inferred from metadata and,
where metadata is silent, from the data itself.
"""

from __future__ import annotations

import re

import numpy as np

# The canonical bands we harmonise onto. Deliberately the minimal set needed
# by the index engine (docs/01) - not a full Sentinel-2 band list, because
# most target products carry a strict subset.
CANONICAL_BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]

# Substrings that identify a canonical band from a band description string.
# Order matters: longer/more specific patterns are checked first so that
# "SWIR1" is not swallowed by a bare "SWIR" rule.
_BAND_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("SWIR1", ("swir1", "swir_1", "b11", "swir 1")),
    ("SWIR2", ("swir2", "swir_2", "b12", "swir 2")),
    ("NIR", ("nir", "b08", "b8", "near infrared", "infrared")),
    ("RED", ("red", "b04", "b4")),
    ("GREEN", ("green", "b03", "b3")),
    ("BLUE", ("blue", "b02", "b2")),
]

# Sensor name fragments that imply SAR. Used only as evidence, never alone.
_SAR_SENSOR_HINTS = (
    "sar",
    "risat",
    "eos-04",
    "eos04",
    "sentinel-1",
    "sentinel1",
    "s1",
    "novasar",
    "alos",
    "terrasar",
    "capella",
    "umbra",
    "iceye",
)

_POL_PATTERN = re.compile(r"\b(HH|HV|VH|VV|RH|RV)\b", re.IGNORECASE)


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def match_canonical_band(description: str | None) -> str | None:
    """Map a raw band description onto a canonical band name, or None."""
    d = _norm(description)
    if not d:
        return None
    for canonical, patterns in _BAND_PATTERNS:
        for p in patterns:
            if p in d:
                return canonical
    return None


def harmonise_bands(
    descriptions: list[str | None], modality: str
) -> tuple[list[str], list[bool]]:
    """Return (band names, canonical presence flags).

    `band names` is one entry per actual raster band, using the canonical name
    where it could be identified and a positional fallback otherwise.
    `presence` is one flag per entry of CANONICAL_BANDS.
    """
    names: list[str] = []
    for i, desc in enumerate(descriptions):
        canonical = match_canonical_band(desc)
        if canonical is not None:
            names.append(canonical)
        elif desc:
            names.append(desc.strip())
        else:
            names.append(f"BAND_{i + 1}")

    # If descriptions were absent or unhelpful, fall back to the conventional
    # band ordering for the band count. This is an assumption, and it is
    # recorded as such in modality_evidence by the caller.
    if not any(n in CANONICAL_BANDS for n in names) and modality in ("MSI", "OPTICAL"):
        assumed = {
            3: ["BLUE", "GREEN", "RED"],
            4: ["BLUE", "GREEN", "RED", "NIR"],
            6: CANONICAL_BANDS,
        }.get(len(descriptions))
        if assumed:
            names = list(assumed)

    presence = [b in names for b in CANONICAL_BANDS]
    return names, presence


def detect_polarisations(tags: dict, band_descriptions: list[str | None]) -> list[str]:
    """Extract SAR polarisations from metadata tags or band descriptions."""
    found: list[str] = []
    haystack = " ".join(
        [f"{k} {v}" for k, v in tags.items()] + [d or "" for d in band_descriptions]
    )
    for m in _POL_PATTERN.finditer(haystack):
        pol = m.group(1).upper()
        if pol not in found:
            found.append(pol)
    return found


def infer_modality(
    band_count: int,
    dtype: str,
    tags: dict,
    band_descriptions: list[str | None],
    sample: np.ndarray | None = None,
) -> tuple[str, dict]:
    """Infer OPTICAL / MSI / PAN / SAR, returning (modality, evidence).

    Evidence is recorded so that every downstream decision can be audited -
    the PS requires the observable decision to be explainable.
    """
    evidence: dict = {
        "band_count": band_count,
        "dtype": dtype,
        "signals": [],
    }

    tag_blob = _norm(" ".join(f"{k}={v}" for k, v in tags.items()))
    desc_blob = _norm(" ".join(d or "" for d in band_descriptions))

    sensor_hit = next((h for h in _SAR_SENSOR_HINTS if h in tag_blob), None)
    if sensor_hit:
        evidence["signals"].append(f"sensor_tag_matches_sar:{sensor_hit}")

    pols = detect_polarisations(tags, band_descriptions)
    if pols:
        evidence["signals"].append(f"polarisations_found:{','.join(pols)}")
    evidence["polarisations"] = pols

    canonical_hits = [
        b for b in (match_canonical_band(d) for d in band_descriptions) if b
    ]
    if canonical_hits:
        evidence["signals"].append(f"optical_band_names:{','.join(canonical_hits)}")

    # Decision. SAR evidence is strong and specific; optical band naming is
    # equally specific. Band count alone only breaks ties.
    sar_score = int(bool(sensor_hit)) + int(bool(pols))
    optical_score = len(canonical_hits)

    if sar_score > 0 and optical_score == 0:
        modality = "SAR"
        evidence["reason"] = "sar_metadata_present_and_no_optical_band_names"
    elif optical_score > 0:
        modality = "MSI" if band_count >= 4 else "OPTICAL"
        evidence["reason"] = "optical_band_names_present"
    elif band_count == 1:
        # A single float band with a wide dynamic range is more likely SAR
        # backscatter than a panchromatic optical band.
        looks_float = dtype.startswith("float")
        if looks_float and sample is not None and _looks_like_backscatter(sample):
            modality = "SAR"
            evidence["reason"] = "single_float_band_with_backscatter_like_distribution"
            evidence["signals"].append("statistical_backscatter_match")
        else:
            modality = "PAN"
            evidence["reason"] = "single_band_defaulted_to_panchromatic"
    elif band_count >= 4:
        modality = "MSI"
        evidence["reason"] = "band_count_ge_4_no_names_assumed_multispectral"
    else:
        modality = "OPTICAL"
        evidence["reason"] = "band_count_lt_4_no_names_assumed_optical"

    evidence["modality"] = modality
    return modality, evidence


def _looks_like_backscatter(sample: np.ndarray) -> bool:
    """Heuristic: SAR amplitude/intensity is non-negative and right-skewed.

    Optical panchromatic data is comparatively symmetric once scaled. This is
    only ever used to break a genuine tie on a single float band, and the
    outcome is always recorded in modality_evidence.
    """
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return False
    if finite.min() < 0:
        return False
    mean = float(finite.mean())
    if mean <= 0:
        return False
    std = float(finite.std())
    if std == 0:
        return False
    # Speckled SAR intensity has a high coefficient of variation and a long
    # right tail; both are unusual for panchromatic optical.
    cov = std / mean
    skew = float(((finite - mean) ** 3).mean() / (std**3))
    return cov > 0.5 and skew > 1.0


def index_availability(
    band_presence: list[bool], modalities: list[str], polarisations: list[str]
) -> dict[str, bool]:
    """Which indices can actually be computed for this input set.

    Drives the SWIR-free fallback paths: if MNDWI/NDBI are unavailable the
    planner must route around them rather than fail (docs/02).
    """
    present = {
        band: flag for band, flag in zip(CANONICAL_BANDS, band_presence, strict=True)
    }
    has_sar = "SAR" in modalities
    pol_set = {p.upper() for p in polarisations}

    return {
        "ndvi": present["RED"] and present["NIR"],
        "ndwi": present["GREEN"] and present["NIR"],
        "mndwi": present["GREEN"] and present["SWIR1"],
        "ndbi": present["SWIR1"] and present["NIR"],
        "sigma0": has_sar,
        "vh_vv_ratio": has_sar and {"VH", "VV"}.issubset(pol_set),
        "glcm_texture": True,  # computable on any single band
        "cov": has_sar,
    }
