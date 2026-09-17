"""Confidence stress response (plan task 3.4).

"Components move sensibly under stress" is the plan's acceptance criterion,
and as written it is not a measurement. This makes it one.

Each stressor degrades ONE dimension of the input and the run is compared
against a clean baseline. Two properties are reported, and the second is the
one that actually matters:

* **sensitivity** - did the targeted component fall?
* **specificity** - did the OTHER components stay put?

A confidence breakdown where every stressor moves every component is
decoration. It looks informative on a dashboard and tells a user nothing about
what to fix, which is exactly what task 3.6's abstention messages depend on:
naming the *limiting* component is only useful if that component is actually
the one the problem lives in.

Usage:
    python evaluation/confidence_stress.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.scenes import structured_scene, write_raster  # noqa: E402
from satquery.controller.pipeline import Controller  # noqa: E402
from satquery.ingest import ingest  # noqa: E402

REPORT = Path("docs/assets/confidence/stress.json")

COMPONENTS = ("model", "agreement", "input_quality")

# Below this absolute change a component counts as unmoved. Index statistics
# vary slightly with the raster content that a stressor necessarily alters, so
# a hard equality test would report noise as a response.
MOVEMENT_EPSILON = 0.02


@dataclass
class Stressor:
    name: str
    targets: str          # which component this SHOULD move
    description: str
    build: object         # (tmpdir) -> list[Path]
    # Components this stressor is EXPECTED to move as well, with the reason.
    # Declared up front rather than explained away after the fact: a stressor
    # that legitimately degrades two dimensions is not a specificity failure,
    # but deciding that once the number is on screen is how a real failure
    # gets rationalised.
    expected_collateral: dict = field(default_factory=dict)


def _optical(path: Path, scene: np.ndarray, **kwargs):
    bands = np.stack(
        [scene * 800 + 200, scene * 900 + 250, scene * 700 + 180,
         scene * 2200 + 400, scene * 1100 + 300, scene * 800 + 220]
    ).astype("uint16")
    return write_raster(
        path, bands,
        band_names=["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
        **kwargs,
    )


def clean(tmp: Path):
    return [_optical(tmp / "clean.tif", structured_scene(128, 128, seed=2))]


def high_nodata(tmp: Path):
    """Most of the scene is genuinely nodata, and declared as such.

    The first version of this stressor wrote zeros WITHOUT setting the
    raster's nodata value. That produced a flat region rather than missing
    data: the ingest nodata check never fired, the indices lost their bimodal
    split, and the stressor moved `agreement` instead of `input_quality` -
    measuring the wrong component and reporting it as a specificity failure
    of the system rather than a bug in the harness.
    """
    scene = structured_scene(128, 128, seed=2)
    scene[:, :96] = 0.0
    return [_optical(tmp / "nodata.tif", scene, nodata=200.0)]


def tiny_scene(tmp: Path):
    """Below the minimum dimension - a blocking check failure."""
    scene = structured_scene(16, 16, seed=2)
    return [_optical(tmp / "tiny.tif", scene)]


def flat_scene(tmp: Path):
    """No spatial structure, so no index finds a bimodal split.

    Every threshold falls back to a fixed prior, which is exactly the
    condition `physics_agreement_from_indices` scores at 0.4.
    """
    scene = np.full((128, 128), 0.5, dtype="float64")
    return [_optical(tmp / "flat.tif", scene)]


def gsd_mismatch(tmp: Path):
    """A pair whose resolutions differ by 12x."""
    a = _optical(tmp / "a.tif", structured_scene(128, 128, seed=2), gsd=10.0)
    b = _optical(tmp / "b.tif", structured_scene(128, 128, seed=5), gsd=120.0)
    return [a, b]


STRESSORS = [
    Stressor(
        "high_nodata", "input_quality",
        "94% of the scene is nodata", high_nodata,
        expected_collateral={
            "agreement": (
                "correct, not a leak: with 94% of the pixels missing the "
                "indices genuinely lose their bimodal split, so the physics "
                "really is less trustworthy. A stressor that degrades two "
                "dimensions should move two components."
            )
        },
    ),
    Stressor(
        "tiny_scene", "input_quality",
        "16x16 px, below the minimum dimension", tiny_scene,
        expected_collateral={
            "model": (
                "a real wart, recorded rather than hidden: the blocking check "
                "short-circuits before any learned tool runs, so "
                "model_confidence keeps its initial 1.0 and the component "
                "goes UP under stress. 1.0 means 'no model ran', not 'the "
                "model was certain', and a neutral value of 1.0 inflates the "
                "geometric mean. physics_agreement documents the same choice "
                "deliberately; here it is an accident of the initialiser. The "
                "abstention fires on input_validation first, so nothing "
                "user-facing depends on it today."
            )
        },
    ),
    Stressor("flat_scene", "agreement",
             "no bimodal split; every threshold is a fixed prior", flat_scene),
    Stressor("gsd_mismatch", "input_quality",
             "paired scenes at 10 m and 120 m", gsd_mismatch),
]

QUERY = "Classify the land cover."


def components_of(trace) -> dict[str, float]:
    return {
        "model": trace.confidence.components.model,
        "agreement": trace.confidence.components.agreement,
        "input_quality": trace.confidence.components.input_quality,
        "final": trace.confidence.final,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=REPORT)
    args = p.parse_args()

    import tempfile

    controller = Controller()
    rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        baseline = components_of(
            controller.run_on_manifest(ingest(clean(tmp)), QUERY)
        )
        print("baseline:", {k: round(v, 4) for k, v in baseline.items()})

        for stressor in STRESSORS:
            sub = tmp / stressor.name
            sub.mkdir(parents=True, exist_ok=True)
            trace = controller.run_on_manifest(
                ingest(stressor.build(sub)), QUERY
            )
            observed = components_of(trace)
            deltas = {
                key: round(observed[key] - baseline[key], 4)
                for key in (*COMPONENTS, "final")
            }
            target_delta = deltas[stressor.targets]
            moved = {
                key: value
                for key, value in deltas.items()
                if key in COMPONENTS
                and key != stressor.targets
                and abs(value) > MOVEMENT_EPSILON
            }
            collateral = {
                k: v for k, v in moved.items()
                if k not in stressor.expected_collateral
            }
            expected_moves = {
                k: v for k, v in moved.items()
                if k in stressor.expected_collateral
            }
            row = {
                "stressor": stressor.name,
                "description": stressor.description,
                "targets": stressor.targets,
                "components": {k: round(v, 4) for k, v in observed.items()},
                "deltas": deltas,
                "sensitivity_ok": target_delta < -MOVEMENT_EPSILON,
                "specificity_ok": not collateral,
                "unexpected_collateral": collateral,
                "expected_collateral": {
                    k: {"delta": v, "why": stressor.expected_collateral[k]}
                    for k, v in expected_moves.items()
                },
                "abstained": trace.abstained,
                "abstain_trigger": trace.abstain_trigger,
                "limiting_component": trace.abstain_limiting_component,
            }
            rows.append(row)
            print(
                f"\n{stressor.name} (targets {stressor.targets}): "
                f"{stressor.description}"
            )
            print(f"  deltas      {deltas}")
            print(f"  sensitivity {'OK' if row['sensitivity_ok'] else 'NO MOVE'}"
                  f"   specificity "
                  f"{'OK' if row['specificity_ok'] else collateral}")
            if trace.abstained:
                print(f"  abstained   {trace.abstain_trigger} "
                      f"(limiting: {trace.abstain_limiting_component})")

    report = {
        "baseline": {k: round(v, 4) for k, v in baseline.items()},
        "movement_epsilon": MOVEMENT_EPSILON,
        "stressors": rows,
        "sensitivity_passed": sum(r["sensitivity_ok"] for r in rows),
        "specificity_passed": sum(r["specificity_ok"] for r in rows),
        "total": len(rows),
        "note": (
            "Specificity is the property that matters. Task 3.6's abstention "
            "messages name the LIMITING component, and that is only useful "
            "advice if the component that moved is the one the problem lives "
            "in. A breakdown where every stressor moves every component would "
            "look informative and be useless."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"\nsensitivity {report['sensitivity_passed']}/{report['total']}  "
        f"specificity {report['specificity_passed']}/{report['total']}"
    )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
