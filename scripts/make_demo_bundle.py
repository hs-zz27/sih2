"""Build the curated demo bundle (plan task 4.1), and verify it still behaves.

The PS names exactly two deliverables, and one of them is *"Codes and models
including **test and demonstration**"*. This builds the demonstration's inputs.

Eight inputs, per `docs/04` §4.1: single optical, single SAR, a cross-modal
pair, a bi-temporal pair, a **deliberately incompatible** pair, a heavily
clouded optical, a large scene, and a low-confidence case. Real Bhoonidhi
products are used wherever they exist — a Cartosat-2E MX scene and an EOS-04
SAR scene are far more convincing on stage than synthetic rasters, and they
are the sensors the ISRO/SAC evaluation set actually uses. Synthetic scenes
from `evaluation/scenes.py` fill the cases no real product covers (a
controlled 41% overlap, a controlled 63% cloud fraction).

## Why this script verifies rather than only builds

A demo bundle that is built once and never re-checked is a bundle that breaks
silently between the last rehearsal and the venue. `--verify` runs every input
through the real controller and asserts the beat it is supposed to produce:
the mismatched pair must be *rejected*, the clouded optical must *abstain*,
the cross-modal pair must reach `XMODAL_JOINT_EXTRACT`. A demo beat that stops
working becomes a non-zero exit code instead of a surprise in front of judges.

Usage:
    python scripts/make_demo_bundle.py --out data/demo_bundle
    python scripts/make_demo_bundle.py --out data/demo_bundle --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.scenes import (  # noqa: E402
    build_msi_6band,
    build_msi_6band_t2,
    build_sar_dualpol,
    structured_scene,
    write_raster,
)
from training.common.paths import index_path  # noqa: E402

# Real products, held out from every training run (docs/03 §4.3). Paths are
# relative to the repo root; each is optional and the bundle degrades to a
# synthetic stand-in with the substitution recorded in the manifest.
# Vendor products ship one file per band, and `satquery.ingest.discover`
# assembles them - so these are **directories**, not single rasters. Passing
# the four BAND*.tif files individually is not the same thing: the controller
# reads that as four separate images and rejects it.
LEVIR_INDEX = Path("data/levircd/index.json")
CARTOSAT_MX = Path("data/bhoonidhi/cartosat2s_mx_5132611")
EOS04_FRS1 = Path("data/bhoonidhi/eos04_frs1_226981731")


@dataclass
class DemoInput:
    """One bundle entry, with the demo beat it exists to produce."""

    key: str
    beat: str
    query: str
    images: list[Path]
    expect: str
    real: bool = False
    notes: list[str] = field(default_factory=list)


def _cartosat() -> Path | None:
    """The Cartosat-2E MX product directory, if it is on disk."""
    return CARTOSAT_MX if (CARTOSAT_MX / "5132611" / "BAND1.tif").exists() else None


def _eos04_hh() -> Path | None:
    path = EOS04_FRS1 / "226981731" / "scene_HH" / "imagery_HH.tif"
    return path if path.exists() else None


def build_clouded_optical(path: Path, cloud_fraction: float = 0.63) -> Path:
    """A 6-band optical scene with a controlled fraction under bright cloud.

    The demo's abstention beat needs a *stated* cloud fraction, so this writes
    a known one rather than hoping a real scene happens to be cloudy. Cloud is
    modelled as high, spectrally flat, high-reflectance pixels - which is what
    makes it cloud to an index engine rather than merely bright ground.
    """
    scene = structured_scene(256, 256, seed=11)
    bands = np.stack([scene * (0.7 + 0.1 * i) for i in range(6)]).astype("float32")

    rng = np.random.default_rng(11)
    mask = np.zeros((256, 256), dtype=bool)
    # A few large blobs rather than salt-and-pepper: cloud is contiguous, and
    # a scattered mask would be filtered as noise instead of read as cover.
    while mask.mean() < cloud_fraction:
        cy, cx = rng.integers(0, 256, size=2)
        r = int(rng.integers(30, 70))
        yy, xx = np.ogrid[:256, :256]
        mask |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r

    top = float(bands.max()) * 1.6
    bands[:, mask] = top  # flat and bright across every band
    return write_raster(
        path,
        bands,
        band_names=["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
        gsd=10.0,
        tags={"cloud_fraction": round(float(mask.mean()), 4), "demo": "clouded"},
    )


def build_incompatible_pair(directory: Path) -> list[Path]:
    """An optical + SAR pair whose footprints barely overlap.

    Both members must be a **different modality**, or the pair is read as
    bi-temporal and the cross-modal task is excluded on configuration before
    overlap is ever considered - which tests the wrong gate. The SAR member is
    written 60 km east and 60 km south of the optical one, so the footprints
    are disjoint and no reading of the geometry makes this a valid pair.

    **This beat does not currently reject** - see limitation L16. The matrix
    declares `min_overlap_pct: 70` and nothing reads it. The input is kept, and
    `--verify` keeps failing on it, so the gap stays visible until it is fixed.
    """
    import rasterio
    from rasterio.transform import from_origin

    optical = write_raster(
        directory / "incompatible_optical.tif",
        np.stack([structured_scene(256, 256, seed=3) * (0.8 + 0.1 * i) for i in range(6)]),
        band_names=["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
        gsd=10.0,
        origin=(500000.0, 2000000.0),
    )
    sar_src = build_sar_dualpol(directory / "_sar_for_offset.tif")
    with rasterio.open(sar_src) as src:
        data, profile = src.read(), src.profile
    profile.update(transform=from_origin(560000.0, 1940000.0, 10.0, 10.0))
    sar_far = directory / "incompatible_sar.tif"
    with rasterio.open(sar_far, "w", **profile) as dst:
        dst.write(data)
        dst.descriptions = ("VV", "VH")
    Path(sar_src).unlink(missing_ok=True)
    return [optical, sar_far]


def build_benchmark_png(path: Path) -> Path:
    """An ungeoreferenced PNG, which is what the prescribed benchmarks ship.

    In operational mode this must be refused: the PS admits PNG/JPEG "only for
    the prescribed public benchmark datasets". In `IngestMode.BENCHMARK` the
    same file is accepted with `crs_present` recorded as WARN - which is what
    lets CDVQA run at all.
    """
    from PIL import Image

    rng = np.random.default_rng(7)
    tile = (structured_scene(256, 256, seed=7) * 255).clip(0, 255).astype("uint8")
    rgb = np.stack([tile, np.roll(tile, 8, axis=0), np.roll(tile, 8, axis=1)], axis=-1)
    Image.fromarray(rgb).save(path)
    return path


def build_change_pair(directory: Path) -> tuple[list[Path], bool, list[str]]:
    """A bi-temporal pair with change the detector can actually find.

    The synthetic pair does not work for this beat, and the rehearsal is what
    showed it: `build_msi_6band` and `build_msi_6band_t2` are *different
    random textures* rather than one area at two dates, so `change_mask_v1` -
    trained on real building change - reported `changed_fraction 0.0` and the
    captioner said "there is no difference". A change-analysis demo in which
    nothing changes is not a demo.

    So the pair comes from **LEVIR-CD** when it is on disk: real bi-temporal
    imagery, real building change, and the distribution the detector was
    trained on. The tile is chosen deterministically as the one with the
    largest labelled change fraction among the first 400 test tiles.

    **Disclosure:** LEVIR-CD tiles ship as ungeoreferenced PNGs. They are
    written here as GeoTIFFs at LEVIR-CD's nominal 0.5 m GSD with a **stated
    synthetic origin**, because the map and overlay need a CRS. The location
    is not real and the manifest says so; the imagery and the change are.
    """
    import json

    if not LEVIR_INDEX.exists():
        optical = build_msi_6band(directory / "optical_t1.tif")
        optical_t2 = build_msi_6band_t2(directory / "optical_t2.tif")
        return [optical, optical_t2], False, [
            "synthetic stand-in: LEVIR-CD not on disk. The trained change "
            "detector reports no change on synthetic textures, so this beat "
            "will show 'there is no difference'."
        ]

    from PIL import Image

    rows = json.loads(LEVIR_INDEX.read_text(encoding="utf-8"))["splits"]["test"]
    best = max(
        rows[:400],
        key=lambda r: float(
            (np.asarray(Image.open(index_path(r["label"])).convert("L")) > 127).mean()
        ),
    )
    changed = float(
        (np.asarray(Image.open(index_path(best["label"])).convert("L")) > 127).mean()
    )

    paths = []
    for role, key in (("t1", "a"), ("t2", "b")):
        rgb = np.asarray(Image.open(index_path(best[key])).convert("RGB")).transpose(2, 0, 1)
        paths.append(write_raster(
            directory / f"levir_{role}.tif",
            rgb,
            band_names=["RED", "GREEN", "BLUE"],
            gsd=0.5,
            origin=(500000.0, 2000000.0),
            tags={
                "source": f"LEVIR-CD {best['id']}",
                "labelled_change_fraction": round(changed, 4),
                "georeferencing": "SYNTHETIC origin; LEVIR-CD ships ungeoreferenced PNGs",
                "ACQUISITION_DATE": "2026-01-01" if role == "t1" else "2026-06-01",
            },
        ))
    return paths, True, [
        f"LEVIR-CD {best['id']}, {changed:.1%} of pixels labelled changed",
        "georeferenced with a SYNTHETIC origin for display; the location is "
        "not real, the imagery and the change are",
    ]


def build_bundle(out: Path) -> list[DemoInput]:
    out.mkdir(parents=True, exist_ok=True)
    synth = out / "synthetic"
    synth.mkdir(exist_ok=True)

    optical = build_msi_6band(synth / "optical_t1.tif")
    optical_t2 = build_msi_6band_t2(synth / "optical_t2.tif")
    sar = build_sar_dualpol(synth / "sar_dualpol.tif")
    change_pair, change_is_real, change_notes = build_change_pair(synth)

    cartosat = _cartosat()
    eos04 = _eos04_hh()

    inputs: list[DemoInput] = []

    # 1 - the rejection the demo opens on.
    inputs.append(DemoInput(
        key="incompatible_pair",
        beat="0:30 - the rejection. Footprint overlap below the 70% gate.",
        query="Use the optical and SAR images together to identify built-up and "
              "water-covered regions.",
        images=build_incompatible_pair(synth),
        expect="rejected_or_abstained",
    ))

    # 1b - a PNG in operational mode: answered, with the geospatial loss named.
    #
    # This beat used to expect a refusal, on two premises that have both since
    # expired:
    #
    #   * "the overlap rejection above is not enforced yet (L16)" - it has been
    #     enforced since 2026-08-30, and beat 1 now genuinely abstains, so the
    #     demo no longer needs a second rejection to open on.
    #   * "operational mode abstains with `crs_present` as the failing check" -
    #     pre-import:png-handling-change (2026-09-01) changed that deliberately, under PS item 10. A
    #     missing CRS in a container that CANNOT CARRY ONE is a format
    #     property, not a defect; a GeoTIFF with no CRS is still a defective
    #     product and still fails. So a PNG now WARNs instead of failing.
    #
    # The expectation was left behind by that change and asserted a behaviour
    # the system had deliberately stopped having, which is why `--verify` read
    # 8/9 while the documents said 9/9.
    #
    # Kept as a beat rather than deleted, because what it demonstrates now is
    # worth more than a second refusal: the system accepts the image, answers
    # the visual question from pixels, and states plainly in the same breath
    # that no coordinates, ground extent or areas in metres are available for
    # it. Capability and its limit in one answer.
    inputs.append(DemoInput(
        key="png_operational",
        beat="0:30 - PNG accepted for a visual question, geospatial loss named "
             "in the answer.",
        query="Describe the land-cover and major objects visible in this image.",
        images=[build_benchmark_png(synth / "benchmark_tile.png")],
        expect="answered",
        notes=[
            "PNG carries no CRS, so `crs_present` WARNs rather than failing "
            "(PS item 10); the answer says no georeferenced output is possible",
        ],
    ))

    # 2 - single optical, real if we have it.
    inputs.append(DemoInput(
        key="single_optical",
        beat="Single optical. Real Cartosat-2E MX where available.",
        query="Describe the land-cover and major objects visible in this image.",
        images=[cartosat] if cartosat else [optical],
        expect="answered",
        real=bool(cartosat),
        notes=[] if cartosat else ["synthetic stand-in: no Cartosat product on disk"],
    ))

    # 3 - single SAR, real if we have it.
    inputs.append(DemoInput(
        key="single_sar",
        beat="Single SAR. Real EOS-04 FRS-1 HH where available.",
        query="Describe the land-cover and major objects visible in this image.",
        images=[eos04] if eos04 else [sar],
        expect="answered",
        real=bool(eos04),
        notes=[] if eos04 else ["synthetic stand-in: no EOS-04 product on disk"],
    ))

    # 4 - the cross-modal flagship, PS representative query 4.
    inputs.append(DemoInput(
        key="crossmodal_pair",
        beat="1:10 - cross-modal flagship. PS query 4.",
        query="Use the optical and SAR images together to identify built-up and "
              "water-covered regions.",
        images=[optical, sar],
        expect="XMODAL_JOINT_EXTRACT",
    ))

    # 5 - bi-temporal, PS representative query 5.
    inputs.append(DemoInput(
        key="bitemporal_pair",
        # Real bi-temporal imagery where available - see build_change_pair.
        beat="3:10 - bi-temporal. PS query 5, a three-way direction answer.",
        query="Has the built-up area increased, decreased, or remained unchanged?",
        images=change_pair,
        expect="TEMPORAL_CHANGE_VQA",
        real=change_is_real,
        notes=change_notes,
    ))

    # 6 - what changed AND where, PS representative query 3 (limitation L13).
    inputs.append(DemoInput(
        key="change_what_and_where",
        # Real bi-temporal imagery where available - see build_change_pair.
        beat="3:10 - PS query 3. Must answer both what changed and where.",
        query="What changed between these two dates, and where did the change occur?",
        images=change_pair,
        expect="TEMPORAL_CHANGE_DESC",
        real=change_is_real,
        notes=change_notes,
    ))

    # 7 - the abstention the demo closes on.
    inputs.append(DemoInput(
        key="clouded_optical",
        beat="4:50 - the abstention. Cloud fraction is written into the tags.",
        query="Describe the land-cover and major objects visible in this image.",
        images=[build_clouded_optical(synth / "clouded_optical.tif")],
        # Was `answered_or_abstained`, which cannot fail. The module docstring
        # says this beat "must abstain", and until 2026-09-12 nothing in the
        # pipeline looked at cloud at all: with the learned tools on, the
        # scene was captioned at 0.88 confidence as "a white building near a
        # road lake". The abstention it appeared to produce came from the
        # land-cover head vetoing the plan with a 0.0 that meant "no claim".
        # `cloud_cover` is now a real check, so the beat can be a real check.
        expect="rejected_or_abstained",
    ))

    # 8 - the large scene, real Cartosat if we have it.
    if cartosat:
        inputs.append(DemoInput(
            key="large_scene",
            beat="Large scene. Real Cartosat-2E MX, 7687x7640 px - the tiling path.",
            query="Describe the land-cover and major objects visible in this image.",
            images=[cartosat],
            expect="answered",
            real=True,
        ))
    else:
        inputs.append(DemoInput(
            key="large_scene",
            beat="Large scene (synthetic stand-in).",
            query="Describe the land-cover and major objects visible in this image.",
            images=[optical],
            expect="answered",
            notes=["synthetic stand-in: no Cartosat product on disk"],
        ))

    return inputs


def verify(inputs: list[DemoInput], root: Path) -> tuple[int, list[dict]]:
    """Run every input through the real controller and check its beat."""
    from satquery.controller.pipeline import Controller

    controller = Controller()
    results: list[dict] = []
    failures = 0

    for item in inputs:
        record: dict = {"key": item.key, "beat": item.beat, "expect": item.expect}
        try:
            trace = controller.run(
                [Path(p) for p in item.images], item.query,
                run_id=f"demo_{item.key}",
            )
        except Exception as exc:  # noqa: BLE001 - a rejection is a valid beat
            record.update(outcome="rejected", detail=f"{type(exc).__name__}: {exc}")
            ok = item.expect in ("rejected_or_abstained",)
            record["ok"] = ok
            failures += not ok
            results.append(record)
            continue

        task = str(trace.routing.selected_task)
        record.update(
            outcome="abstained" if trace.abstained else "answered",
            task=task,
            confidence=round(trace.confidence.final, 4),
            answer=trace.answer[:120],
            tools=[s.tool for s in trace.execution],
        )
        if item.expect == "rejected_or_abstained":
            ok = trace.abstained or task == "CLARIFY_OR_ABSTAIN"
        elif item.expect == "answered":
            ok = not trace.abstained
        elif item.expect == "answered_or_abstained":
            ok = True
        else:
            ok = task == item.expect
        record["ok"] = ok
        failures += not ok
        results.append(record)

    return failures, results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/demo_bundle"))
    p.add_argument("--verify", action="store_true",
                   help="run every input through the controller and check its beat")
    args = p.parse_args()

    inputs = build_bundle(args.out)
    manifest = {
        "n_inputs": len(inputs),
        "n_real_products": sum(i.real for i in inputs),
        "inputs": [
            {
                "key": i.key, "beat": i.beat, "query": i.query,
                "images": [str(x) for x in i.images],
                "expect": i.expect, "real_product": i.real, "notes": i.notes,
            }
            for i in inputs
        ],
    }

    failures, results = (0, [])
    if args.verify:
        failures, results = verify(inputs, args.out)
        manifest["verification"] = results

    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"{len(inputs)} demo inputs -> {args.out}")
    print(f"  {manifest['n_real_products']} use real Bhoonidhi products")
    for i in inputs:
        mark = "real" if i.real else "synth"
        print(f"  [{mark:>5}] {i.key:<24} {len(i.images)} image(s)")
        for note in i.notes:
            print(f"          note: {note}")

    if args.verify:
        print("\nverification:")
        for r in results:
            print(f"  {'OK  ' if r['ok'] else 'FAIL'} {r['key']:<24}"
                  f" {r.get('task', r['outcome']):<22} {r['outcome']}")
            if not r["ok"]:
                print(f"        expected {r['expect']}, answer: {r.get('answer','')[:80]}")
        print(f"\n{len(results) - failures}/{len(results)} beats behave as scripted")

    print(f"\nmanifest -> {args.out / 'manifest.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
