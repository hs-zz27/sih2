"""`index_engine_v1` - the first real tool (plan task 1.2).

Deterministic, no learned parameters. Computes every spectral and SAR index
the input actually supports, thresholds each adaptively, writes the results as
Cloud-Optimised GeoTIFFs, and reports which indices were unavailable and what
was substituted. Its confidence is `deterministic` because the arithmetic has
no uncertainty - the only uncertainty is in the thresholds, which is carried
separately in the per-index threshold report.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from satquery.contracts.input_manifest import ImageMeta, InputManifest
from satquery.contracts.tool_result import Artifact, ToolPayload, ToolResult
from satquery.ingest.reader import read_canonical_band
from satquery.tools.base import ToolProtocol
from satquery.verify.indices import (
    index_stats,
    mndwi,
    ndbi,
    ndvi,
    ndwi,
    polarisation_ratio_db,
    sigma0_db,
    swir_free_builtup_proxy,
)
from satquery.verify.texture import (
    coefficient_of_variation,
    glcm_features,
    local_variance,
)
from satquery.verify.thresholding import ThresholdResult, adaptive_threshold

TOOL_NAME = "index_engine"
TOOL_VERSION = "1.0.0"

# Literature default thresholds, used only when the data cannot support an
# adaptive estimate. Every use is recorded in the threshold report.
FIXED_PRIORS = {
    "ndvi": 0.3,
    "ndwi": 0.0,
    "mndwi": 0.0,
    "ndbi": 0.0,
    "builtup_proxy": 0.5,
}

OPTICAL_MODALITIES = {"OPTICAL", "MSI", "PAN"}


class IndexEnginePayload(ToolPayload):
    data: dict[str, Any]


def write_cog(path: Path, array: np.ndarray, reference: ImageMeta) -> Path:
    """Write a single-band float32 Cloud-Optimised GeoTIFF."""
    with rasterio.open(reference.path) as src:
        profile = {
            "driver": "COG",
            "dtype": "float32",
            "count": 1,
            "height": array.shape[0],
            "width": array.shape[1],
            "crs": src.crs,
            "transform": src.transform,
            "nodata": float("nan"),
            "compress": "DEFLATE",
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)
    return path


def _threshold_report(name: str, result: ThresholdResult) -> dict:
    return {
        "index": name,
        "value": round(result.value, 6),
        "method": result.method,
        "bimodal": result.bimodal,
        "separation": round(result.separation, 6),
        "n_pixels": result.n_pixels,
        "fallback_reason": result.fallback_reason,
    }


class IndexEngine(ToolProtocol):
    """Computes every index the input supports, with adaptive thresholds."""

    def run(self, manifest: InputManifest, params: dict[str, Any]) -> ToolResult:
        started = time.perf_counter()
        warnings: list[str] = []
        indices: dict[str, dict] = {}
        thresholds: list[dict] = []
        artifacts: list[Artifact] = []
        substitutions: list[str] = []

        write_artifacts = bool(params.get("write_artifacts", True))
        out_dir = Path(params.get("output_dir", "artifacts")) / manifest.run_id

        optical = next(
            (i for i in manifest.images if i.modality in OPTICAL_MODALITIES), None
        )
        sar = next((i for i in manifest.images if i.modality == "SAR"), None)
        avail = manifest.index_availability

        def record(name: str, arr: np.ndarray, reference: ImageMeta) -> None:
            stats = index_stats(arr)
            thr = adaptive_threshold(arr, fixed_prior=FIXED_PRIORS.get(name, 0.0))
            above = float(np.nansum(arr > thr.value))
            finite = float(np.sum(np.isfinite(arr)))
            indices[name] = {
                "stats": {k: round(v, 6) for k, v in stats.items()},
                "threshold": round(thr.value, 6),
                "threshold_method": thr.method,
                "fraction_above_threshold": round(above / finite, 6) if finite else 0.0,
            }
            thresholds.append(_threshold_report(name, thr))
            if thr.method == "fixed_prior":
                warnings.append(
                    f"{name}: adaptive thresholding failed ({thr.fallback_reason}); "
                    f"used fixed prior {thr.value}"
                )
            if write_artifacts:
                p = write_cog(out_dir / f"{name}.tif", arr, reference)
                artifacts.append(
                    Artifact(
                        key=name,
                        kind="cog",
                        path=p,
                        crs=reference.crs,
                        description=f"{name.upper()} index raster",
                    )
                )

        # --- Optical indices -------------------------------------------------
        if optical is not None:
            red = nir = green = swir1 = None
            if avail.get("ndvi"):
                red = read_canonical_band(optical, "RED")
                nir = read_canonical_band(optical, "NIR")
                record("ndvi", ndvi(red, nir), optical)

            if avail.get("ndwi"):
                green = read_canonical_band(optical, "GREEN")
                if nir is None:
                    nir = read_canonical_band(optical, "NIR")
                record("ndwi", ndwi(green, nir), optical)

            if avail.get("mndwi"):
                if green is None:
                    green = read_canonical_band(optical, "GREEN")
                swir1 = read_canonical_band(optical, "SWIR1")
                record("mndwi", mndwi(green, swir1), optical)
            elif "ndwi" in indices:
                # A substitution is a claim that something ran INSTEAD. This
                # branch used to fire whenever MNDWI was unavailable, including
                # when NDWI was unavailable too - so an RGB-only input, with no
                # NIR and no SWIR1, reported "NDWI used as the water index"
                # while computing no water index at all. The claim then reached
                # the verifier as a conflict, because the executor turns every
                # substitution into one.
                substitutions.append(
                    "MNDWI unavailable (no SWIR1); NDWI used as the water index"
                )
            else:
                # Neither water index is computable. Say so, rather than
                # claiming a substitution that did not happen or saying
                # nothing at all.
                warnings.append(
                    "no water index computed: MNDWI needs SWIR1 and NDWI needs "
                    "NIR; neither band is present in these inputs"
                )

            if avail.get("ndbi"):
                if swir1 is None:
                    swir1 = read_canonical_band(optical, "SWIR1")
                if nir is None:
                    nir = read_canonical_band(optical, "NIR")
                record("ndbi", ndbi(swir1, nir), optical)
            elif red is not None and nir is not None:
                # SWIR-free built-up path: combine low-NDVI, SAR brightness and
                # texture roughness in place of NDBI (docs/02).
                sigma_vv = None
                if sar is not None:
                    pols = sar.polarisations or []
                    co = next((p for p in ("VV", "HH") if p in pols), None)
                    if co is not None:
                        sigma_vv = sigma0_db(read_canonical_band(sar, co))
                texture = local_variance(nir)
                proxy = swir_free_builtup_proxy(
                    red, nir, sigma0_vv=sigma_vv, texture=texture
                )
                record("builtup_proxy", proxy, optical)
                terms = ["low_ndvi", "texture"] + (["sar_sigma0"] if sigma_vv is not None else [])
                substitutions.append(
                    "NDBI unavailable (no SWIR1); built-up estimated from "
                    + " + ".join(terms)
                )
                warnings.append(
                    "built-up derived from a SWIR-free proxy, not NDBI - "
                    "lower reliability"
                )
            else:
                # Same rule as the water index: the proxy needs RED and NIR,
                # so with neither NDBI nor the proxy computable the honest
                # trace says nothing was estimated - not nothing at all.
                warnings.append(
                    "no built-up index computed: NDBI needs SWIR1 and NIR, and "
                    "the SWIR-free proxy needs RED and NIR; these inputs "
                    "support neither"
                )

        # --- SAR indices -----------------------------------------------------
        if sar is not None:
            pols = sar.polarisations or []
            co = next((p for p in ("VV", "HH") if p in pols), None)
            cross = next((p for p in ("VH", "HV") if p in pols), None)

            if co is not None:
                co_arr = read_canonical_band(sar, co)
                record(f"sigma0_{co.lower()}", sigma0_db(co_arr), sar)
                cov = coefficient_of_variation(co_arr)
                indices["cov"] = {
                    "stats": {k: round(v, 6) for k, v in index_stats(cov).items()},
                    "window": 7,
                }
                if write_artifacts:
                    p = write_cog(out_dir / "cov.tif", cov, sar)
                    artifacts.append(
                        Artifact(
                            key="cov", kind="cog", path=p, crs=sar.crs,
                            description="SAR speckle coefficient of variation",
                        )
                    )

            if cross is not None and co is not None:
                cross_arr = read_canonical_band(sar, cross)
                co_arr = read_canonical_band(sar, co)
                record(
                    f"{cross.lower()}_{co.lower()}_ratio_db",
                    polarisation_ratio_db(cross_arr, co_arr),
                    sar,
                )
            elif cross is None:
                substitutions.append(
                    "cross-pol ratio unavailable (single-polarisation product)"
                )

        # --- Texture ---------------------------------------------------------
        texture_source = optical or sar
        glcm: dict[str, float] = {}
        if texture_source is not None:
            band = "NIR" if "NIR" in texture_source.bands else texture_source.bands[0]
            try:
                arr = read_canonical_band(texture_source, band)
                glcm = {k: round(v, 6) for k, v in glcm_features(arr).items()}
            except KeyError:
                warnings.append("GLCM skipped: no readable band")

        if not indices:
            warnings.append(
                "no indices computable from these inputs - check band availability"
            )

        payload = IndexEnginePayload(
            data={
                "indices": indices,
                "thresholds": thresholds,
                "glcm": glcm,
                "index_availability": avail,
                "substitutions": substitutions,
            }
        )

        return ToolResult(
            tool=TOOL_NAME,
            version=TOOL_VERSION,
            payload=payload,
            artifacts=artifacts,
            confidence=1.0,
            confidence_method="deterministic",
            model_card="index_engine_v1 (closed-form, no learned parameters)",
            runtime_ms=int((time.perf_counter() - started) * 1000),
            warnings=warnings,
        )

    def run_batch(
        self, manifests: list[InputManifest], params: dict[str, Any]
    ) -> list[ToolResult]:
        return [self.run(m, params) for m in manifests]
