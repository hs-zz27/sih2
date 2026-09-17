"""Cross-sensor generalisation test: Track A on real Cartosat imagery.

This is the evaluation that matters most for the project's central claim, and
the one that cannot be faked with a simulation: the encoder was trained on
Sentinel-2 and is asked to work on a Cartosat-2E product it has never seen,
held out per docs/03 section 4.3 and never trained on.

TWO PROBLEMS, AND HOW THEY ARE HANDLED

**No labels.** The Bhoonidhi product carries no land-cover ground truth, so
mAP is impossible. Instead the neural predictions are scored against the
*deterministic index engine* on the same pixels: patches the model calls water
should have high NDWI, patches it calls forest or arable land should have high
NDVI. That is not a substitute for labels, and it cannot detect an error the
physics shares - but it is an independent signal, computed by closed-form
arithmetic with no learned parameters, and disagreement is genuinely
informative. It is also exactly the verifier relationship the whole system is
built around.

**Two domain gaps at once.** Cartosat differs from Sentinel-2 in *both* band
set (4 VNIR vs 10) and resolution (1.6 m vs 10 m). Testing at native
resolution would confound them, so the scene is resampled to 10 m to isolate
the band gap, and both conditions are reported.

Radiometry is the residual uncertainty: Cartosat delivers 11-bit DN, not
Sentinel-2 surface reflectance. Training normalised to roughly zero-mean unit
variance per band, so the same distributional target is applied here via a
per-band z-score over the scene. That is the fairest available mapping without
radiometric cross-calibration, and it is an assumption, not a calibration.

Usage:
    python evaluation/cross_sensor.py --product data/bhoonidhi/cartosat2s_mx_5132611 \
        --checkpoint checkpoints/track_a_dropout --index data/bigearthnet_14k/index.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satquery.ingest.product import resolve as resolve_product  # noqa: E402
from satquery.verify.indices import ndvi, ndwi  # noqa: E402
from training.track_a_encoder import (  # noqa: E402
    BAND_NAMES,
    CARTOSAT_INDICES,
    N_CLASSES,
    build_model,
    load_index,
)

PATCH = 120                 # BigEarthNet patch size, in pixels
BEN_GSD = 10.0              # metres, the resolution the encoder was trained at
CARTOSAT_ORDER = ["BLUE", "GREEN", "RED", "NIR"]

# Vegetation group: FOREST ONLY, in both taxonomies.
#
# Cropland classes are deliberately excluded. WHU's "farmland" and
# BigEarthNet's "Arable land" both include bare and ploughed fields, which
# have LOW NDVI - so grouping them under "vegetation" and correlating against
# NDVI manufactures a negative correlation from the taxonomy rather than
# measuring the model. That is exactly what happened in the first Stage A2
# evaluation. Forest is unambiguously vegetated in both vocabularies, which
# makes the two models comparable on the same semantic content.
VEGETATION_CLASSES = [
    "Broad-leaved forest", "Coniferous forest", "Mixed forest",
]
# Kept for reference: the cropland classes excluded above, and why.
CROPLAND_CLASSES = [
    "Arable land", "Pastures", "Permanent crops", "Complex cultivation patterns",
    "Agro-forestry areas", "Transitional woodland, shrub",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
]
WATER_CLASSES = ["Inland waters", "Marine waters", "Coastal wetlands", "Inland wetlands"]
BUILT_CLASSES = ["Urban fabric", "Industrial or commercial units"]

# Stage A2 retrains the head on WHU-OPT-SAR's 8-class vocabulary, so the same
# semantic groups have different names. Keeping both lets one evaluator
# compare a BigEarthNet-head model against an A2-adapted one directly.
WHU_CLASSES = ["background", "farmland", "city", "village", "water", "forest",
               "road", "others"]
WHU_GROUPS = {
    # Forest only, matching VEGETATION_CLASSES: "farmland" covers bare and
    # ploughed fields and would poison an NDVI correlation.
    "vegetation": ["forest"],
    "water": ["water"],
    "built_up": ["city", "village", "road"],
}


def read_bands(product: Path, target_gsd: float | None) -> tuple[dict, float]:
    """Read the Cartosat bands, optionally resampled to `target_gsd` metres."""
    path, layout = resolve_product(product)
    with rasterio.open(path) as src:
        native_gsd = abs(src.transform.a)
        if target_gsd is None or target_gsd <= native_gsd:
            scale, out_h, out_w = 1.0, src.height, src.width
        else:
            scale = target_gsd / native_gsd
            out_h, out_w = int(src.height / scale), int(src.width / scale)

        bands = {}
        for i, name in enumerate(layout.band_names, start=1):
            arr = src.read(
                i, out_shape=(out_h, out_w),
                resampling=Resampling.average, masked=True,
            )
            bands[name] = np.ma.filled(arr.astype("float32"), np.nan)
    return bands, native_gsd * scale


def standardise(arr: np.ndarray) -> np.ndarray:
    """Per-band z-score, matching the distribution the encoder trained on.

    Cartosat ships 11-bit DN; Sentinel-2 training used reflectance/10000 then
    per-band standardisation. Matching the distribution is the closest fair
    mapping absent radiometric cross-calibration.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    mean, std = float(finite.mean()), float(finite.std())
    if std < 1e-6:
        std = 1.0
    out = (arr - mean) / std
    return np.where(np.isfinite(out), out, 0.0)


def build_patches(bands: dict, normalise: str = "scene") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile the scene into (N, 10, 120, 120) with a 4-band presence mask.

    Cartosat's bands are placed in their canonical slots so the learned band
    embedding sees BLUE where BLUE belongs. Placing them at 0..3 positionally
    would tell the model that RED is actually a red-edge band.
    """
    height, width = next(iter(bands.values())).shape
    rows, cols = height // PATCH, width // PATCH
    if rows == 0 or cols == 0:
        raise SystemExit(f"scene {width}x{height} is smaller than one {PATCH}px patch")

    # Normalisation must match how the checkpoint was trained. Track A used
    # dataset-level band statistics over whole 120px patches; Stage A2
    # standardises each tile independently. Feeding a per-scene distribution
    # to a per-tile-trained model is a distribution shift of our own making,
    # and it produced a spuriously strong negative correlation before this
    # was made explicit.
    scene_standardised = (
        {name: standardise(arr) for name, arr in bands.items()}
        if normalise == "scene" else bands
    )
    standardised = scene_standardised

    patches, coords = [], []
    for r in range(rows):
        for c in range(cols):
            y, x = r * PATCH, c * PATCH
            cube = np.zeros((len(BAND_NAMES), PATCH, PATCH), dtype="float32")
            raw = {}
            for name, slot in zip(CARTOSAT_ORDER, CARTOSAT_INDICES, strict=True):
                if name not in standardised:
                    continue
                window = standardised[name][y : y + PATCH, x : x + PATCH]
                cube[slot] = standardise(window) if normalise == "patch" else window
                raw[name] = bands[name][y : y + PATCH, x : x + PATCH]
            patches.append(cube)
            coords.append((r, c, raw))

    mask = np.zeros((len(patches), len(BAND_NAMES)), dtype="float32")
    mask[:, CARTOSAT_INDICES] = 1.0
    return np.stack(patches), mask, coords


def physics_reference(coords) -> dict[str, np.ndarray]:
    """Per-patch NDVI and NDWI from the deterministic index engine."""
    veg, water = [], []
    for _, _, raw in coords:
        if "RED" in raw and "NIR" in raw:
            v = ndvi(raw["RED"], raw["NIR"])
            veg.append(float(np.nanmean(v)))
        else:
            veg.append(np.nan)
        if "GREEN" in raw and "NIR" in raw:
            w = ndwi(raw["GREEN"], raw["NIR"])
            water.append(float(np.nanmean(w)))
        else:
            water.append(np.nan)
    return {"ndvi": np.array(veg), "ndwi": np.array(water)}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, robust to the non-linear neural-vs-index relationship."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype("float64")
    rb = np.argsort(np.argsort(b[ok])).astype("float64")
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def load_trained(checkpoint: Path, torch, device, n_classes: int = N_CLASSES):
    from training.common.checkpointing import find_latest_checkpoint, load_checkpoint

    import torch.nn as nn

    model = build_model(n_bands=len(BAND_NAMES), dim=64)
    if n_classes != N_CLASSES:
        model.head = nn.Linear(model.head.in_features, n_classes)
    model = model.to(device).to(device)
    latest = find_latest_checkpoint(checkpoint)
    if latest is None:
        raise SystemExit(f"no checkpoint found in {checkpoint}")
    load_checkpoint(latest, model, map_location=device)
    model.eval()
    return model, latest


def run(product: Path, checkpoint: Path, index_path: Path, target_gsd: float | None,
        class_set: str = "bigearthnet", normalise: str = "scene"):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if class_set == "whu":
        classes, groups = WHU_CLASSES, WHU_GROUPS
    else:
        classes = load_index(index_path)["classes"]
        groups = {
            "vegetation": VEGETATION_CLASSES,
            "water": WATER_CLASSES,
            "built_up": BUILT_CLASSES,
        }
    model, latest = load_trained(checkpoint, torch, device, len(classes))

    bands, effective_gsd = read_bands(product, target_gsd)
    patches, mask, coords = build_patches(bands, normalise)
    physics = physics_reference(coords)

    scores = []
    with torch.no_grad():
        for start in range(0, len(patches), 64):
            xb = torch.from_numpy(patches[start : start + 64]).to(device)
            mb = torch.from_numpy(mask[start : start + 64]).to(device)
            scores.append(torch.sigmoid(model(xb, mb)).cpu().numpy())
    scores = np.concatenate(scores)

    def group(names):
        idx = [classes.index(n) for n in names if n in classes]
        return scores[:, idx].max(axis=1)

    veg_score, water_score, built_score = (
        group(groups["vegetation"]), group(groups["water"]), group(groups["built_up"])
    )

    return {
        "checkpoint": str(latest),
        "effective_gsd_m": round(effective_gsd, 3),
        "normalisation": normalise,
        "n_patches": int(len(patches)),
        "bands_present": CARTOSAT_ORDER,
        "agreement": {
            "vegetation_vs_ndvi": round(spearman(veg_score, physics["ndvi"]), 4),
            "water_vs_ndwi": round(spearman(water_score, physics["ndwi"]), 4),
            "builtup_vs_ndvi_negated": round(spearman(built_score, -physics["ndvi"]), 4),
        },
        "mean_scores": {
            "vegetation": round(float(veg_score.mean()), 4),
            "water": round(float(water_score.mean()), 4),
            "builtup": round(float(built_score.mean()), 4),
        },
        "physics": {
            "ndvi_mean": round(float(np.nanmean(physics["ndvi"])), 4),
            "ndwi_mean": round(float(np.nanmean(physics["ndwi"])), 4),
        },
        "top_classes": [
            {"class": classes[i], "mean_score": round(float(scores[:, i].mean()), 4)}
            for i in np.argsort(-scores.mean(axis=0))[:6]
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--product", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--class-set", choices=["bigearthnet", "whu"], default="bigearthnet")
    p.add_argument(
        "--normalise", choices=["scene", "patch"], default="scene",
        help="must match how the checkpoint was trained (A2 uses per-patch)",
    )
    args = p.parse_args()

    results = {}
    for label, gsd in (("resampled_to_10m", BEN_GSD), ("native_resolution", None)):
        print(f"\n=== {label} ===")
        r = run(args.product, args.checkpoint, args.index, gsd, args.class_set,
                args.normalise)
        results[label] = r
        print(f"  effective GSD : {r['effective_gsd_m']} m   patches: {r['n_patches']}")
        print("  agreement with the deterministic index engine (Spearman):")
        for k, v in r["agreement"].items():
            print(f"    {k:<28} {v:+.4f}")
        print(f"  scene NDVI mean {r['physics']['ndvi_mean']:+.4f} | "
              f"NDWI mean {r['physics']['ndwi_mean']:+.4f}")
        print("  highest-scoring classes:")
        for row in r["top_classes"]:
            print(f"    {row['mean_score']:.4f}  {row['class'][:58]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
