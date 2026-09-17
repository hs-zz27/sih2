"""Labelled bench for the entailment gate (plan task 3.5).

The gate's statistics in a trace say what it *did*. They cannot say whether
it was right. This is a hand-written set of (payload, sentence, expected
status) cases so the two backends can be scored against a ground truth
instead of demonstrated on a favourable example.

PROVENANCE, recorded here for the same reason `satquery/synth/holdout.py`
records its own: these cases were written by hand while building task 3.5,
BEFORE either backend was scored on them, and the thresholds in
`entailment.py` were not tuned against the results. They are a smoke test
with a ground truth, not a benchmark - n is small and one author wrote both
the gate and the cases, which is exactly the setup that flatters a system.
Treat the direction of the numbers as informative and the absolute values as
soft.

The three statuses matter as much as the accuracy:

* marking a FALSE sentence `retained` is the dangerous error - the system
  asserts something its own measurements contradict;
* marking a TRUE sentence `flagged` destroys a correct answer;
* marking an UNCHECKABLE sentence `retained` is the quiet error, and the one
  a two-outcome gate cannot even express.

Usage:
    python evaluation/entailment_bench.py                     # deterministic
    SATQUERY_NLI=models/nli_deberta_mnli \
        python evaluation/entailment_bench.py --compare       # both backends
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from satquery.verify.entailment import (  # noqa: E402
    DeterministicBackend,
    NLIBackend,
    build_premises,
    run_gate,
)


def payload(**fractions: float) -> dict:
    return {
        "indices": {
            name: {"fraction_above_threshold": value}
            for name, value in fractions.items()
        }
    }


@dataclass(frozen=True)
class Case:
    sentence: str
    expected: str
    payload: dict
    why: str


# A scene that is mostly vegetation, barely any water, some built-up.
VEG = payload(ndvi=0.62, ndwi=0.05, ndbi=0.11)
# A flooded scene: mostly water, little vegetation.
WATER = payload(ndvi=0.08, ndwi=0.71, ndbi=0.03)
# A dense urban scene.
URBAN = payload(ndvi=0.09, ndwi=0.02, ndbi=0.58)
# A SWIR-free sensor: no NDBI, only the proxy.
NO_SWIR = payload(ndvi=0.44, ndwi=0.12, builtup_proxy=0.30)

TUNED_CASES: list[Case] = [
    # --- Quantitative claims that agree -------------------------------------
    Case("Vegetation covers 62% of the scene.", "retained", VEG,
         "exact match against NDVI"),
    Case("About 60% of the area is vegetated.", "retained", VEG,
         "within the 10-point tolerance"),
    Case("Water covers 71% of the scene.", "retained", WATER,
         "exact match against NDWI"),
    Case("Built-up land accounts for 58% of the image.", "retained", URBAN,
         "exact match against NDBI"),

    # --- Quantitative claims that disagree ----------------------------------
    Case("The scene is 90% water.", "flagged", VEG,
         "claims 90% where NDWI measures 5%"),
    Case("Vegetation covers 5% of the scene.", "flagged", VEG,
         "claims 5% where NDVI measures 62%"),
    Case("Roughly 80% of this image is built-up.", "flagged", VEG,
         "claims 80% where NDBI measures 11%"),
    Case("Only 10% of the area is water.", "flagged", WATER,
         "claims 10% where NDWI measures 71%"),

    # --- Qualitative claims that agree --------------------------------------
    Case("The scene is dominated by vegetation.", "retained", VEG,
         "NDVI at 62% supports dominance"),
    Case("Water covers a small part of the scene.", "retained", VEG,
         "NDWI at 5% is a small part"),
    Case("This is a densely built-up urban area.", "retained", URBAN,
         "NDBI at 58% supports it"),
    Case("Most of this scene is under water.", "retained", WATER,
         "NDWI at 71% supports it"),

    # --- Qualitative claims that disagree -----------------------------------
    # These are the cases a percentage parser cannot reach: no number appears,
    # so the deterministic backend can only ask "is the class present at all".
    Case("The scene is almost entirely covered by water.", "flagged", VEG,
         "NDWI is 5%, not almost everything"),
    Case("There is no vegetation anywhere in this image.", "flagged", VEG,
         "NDVI measures 62%"),
    Case("The area is overwhelmingly urban.", "flagged", VEG,
         "NDBI measures 11%"),
    Case("Vegetation dominates this scene.", "flagged", URBAN,
         "NDVI measures 9%"),
    Case("The image shows no water at all.", "flagged", WATER,
         "NDWI measures 71%"),
    Case("This region is heavily forested.", "flagged", WATER,
         "NDVI measures 8%"),

    # --- Sentences no premise covers ----------------------------------------
    Case("A large airport dominates the northern half.", "unverifiable", VEG,
         "no index measures airports"),
    Case("Several aircraft are parked near the terminal.", "unverifiable", VEG,
         "no index measures aircraft"),
    Case("The playground is next to the road.", "unverifiable", VEG,
         "no index measures playgrounds or roads"),
    Case("These classes are measured independently and may overlap.",
         "unverifiable", VEG, "a caveat, not a claim about the scene"),
    Case("The image was captured in the morning.", "unverifiable", VEG,
         "no index measures acquisition time"),
    Case("Two ships are visible in the harbour.", "unverifiable", WATER,
         "no index counts ships"),

    # --- Subjects with no available index -----------------------------------
    Case("Built-up land covers 30% of the scene.", "retained", NO_SWIR,
         "matches the SWIR-free proxy at 30%"),
]


# A coastal scene: substantial water, moderate vegetation, little built-up.
COAST = payload(ndvi=0.31, ndwi=0.44, ndbi=0.07)
# Arid: almost no vegetation or water, some bare/built structure.
ARID = payload(ndvi=0.03, ndwi=0.01, ndbi=0.22)
# A mixed peri-urban scene, using MNDWI rather than NDWI.
MIXED = payload(ndvi=0.40, mndwi=0.18, ndbi=0.35)
# Another SWIR-free sensor case.
PROXY2 = payload(ndvi=0.55, ndwi=0.09, builtup_proxy=0.14)

CLEAN_CASES: list[Case] = [
    # Quantitative, agreeing
    Case("Water covers 44% of this coastal scene.", "retained", COAST,
         "exact match against NDWI"),
    Case("Around 30% of the image is vegetated.", "retained", COAST,
         "within tolerance of NDVI at 31%"),
    Case("Built-up land makes up 22% of the area.", "retained", ARID,
         "exact match against NDBI"),
    Case("Roughly 35% of the scene is built-up.", "retained", MIXED,
         "exact match against NDBI"),

    # Quantitative, disagreeing
    Case("Vegetation covers 70% of this scene.", "flagged", COAST,
         "claims 70% where NDVI measures 31%"),
    Case("Water accounts for 60% of the image.", "flagged", ARID,
         "claims 60% where NDWI measures 1%"),
    Case("About 5% of this area is built-up.", "flagged", MIXED,
         "claims 5% where NDBI measures 35%"),
    Case("Vegetation covers 80% of the image.", "flagged", ARID,
         "claims 80% where NDVI measures 3%"),

    # Qualitative, agreeing
    Case("There is a substantial amount of water in this scene.", "retained",
         COAST, "NDWI at 44% supports it"),
    Case("This is a dry area with very little vegetation.", "retained", ARID,
         "NDVI at 3% supports it"),
    Case("The scene contains a mixture of vegetation and buildings.",
         "retained", MIXED, "NDVI 40% and NDBI 35% support both"),
    Case("Vegetation is the most prominent feature here.", "retained", PROXY2,
         "NDVI at 55% supports it"),
    Case("Nearly half of this scene is water.", "retained", COAST,
         "NDWI at 44% is nearly half"),

    # Qualitative, disagreeing
    Case("This scene is entirely desert with no water whatsoever.", "flagged",
         COAST, "NDWI measures 44%"),
    Case("The region is lush and densely vegetated.", "flagged", ARID,
         "NDVI measures 3%"),
    Case("There are no buildings anywhere in this image.", "flagged", MIXED,
         "NDBI measures 35%"),
    Case("The scene is completely covered by water.", "flagged", PROXY2,
         "NDWI measures 9%"),
    Case("This area shows no vegetation at all.", "flagged", COAST,
         "NDVI measures 31%"),

    # No premise covers these
    Case("A bridge spans the northern part of the image.", "unverifiable",
         COAST, "no index measures bridges"),
    Case("Three storage tanks are visible near the depot.", "unverifiable",
         ARID, "no index measures storage tanks"),
    Case("The photograph appears slightly overexposed.", "unverifiable", MIXED,
         "no index measures exposure"),
    Case("This tile was acquired by a Sentinel-2 sensor.", "unverifiable",
         COAST, "no index measures the platform"),
    Case("A road runs diagonally across the scene.", "unverifiable", PROXY2,
         "no index measures roads"),
    Case("Cloud cover obscures the upper left corner.", "unverifiable", MIXED,
         "no index measures cloud"),

    # SWIR-free proxy
    Case("Built-up land covers 14% of the scene.", "retained", PROXY2,
         "matches the SWIR-free proxy at 14%"),
]

SUITES = {"tuned": TUNED_CASES, "clean": CLEAN_CASES}


@dataclass
class Score:
    suite: str
    backend: str
    total: int
    correct: int
    dangerous: int          # false sentence marked retained
    destructive: int        # true sentence marked flagged
    overclaimed: int        # uncheckable sentence marked retained
    confusion: dict

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "backend": self.backend,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.correct / self.total, 4) if self.total else 0.0,
            "dangerous_false_retained": self.dangerous,
            "destructive_true_flagged": self.destructive,
            "overclaimed_unverifiable_retained": self.overclaimed,
            "confusion": self.confusion,
        }


def score_backends(backends, label: str, cases: list[Case], suite: str) -> Score:
    confusion: dict = {}
    correct = dangerous = destructive = overclaimed = 0

    for case in cases:
        result = run_gate(case.sentence, case.payload, backends=list(backends))
        got = result.verdicts[0].status if result.verdicts else "unverifiable"
        confusion.setdefault(case.expected, {}).setdefault(got, 0)
        confusion[case.expected][got] += 1

        if got == case.expected:
            correct += 1
        if case.expected == "flagged" and got == "retained":
            dangerous += 1
        if case.expected == "retained" and got == "flagged":
            destructive += 1
        if case.expected == "unverifiable" and got == "retained":
            overclaimed += 1

    return Score(suite, label, len(cases), correct, dangerous, destructive,
                 overclaimed, confusion)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nli", default=None,
                   help="path to an MNLI checkpoint (defaults to $SATQUERY_NLI)")
    p.add_argument("--compare", action="store_true",
                   help="score deterministic alone, NLI alone, and the hybrid")
    p.add_argument("--suites", nargs="+", default=["tuned", "clean"],
                   choices=sorted(SUITES))
    p.add_argument("--out", type=Path,
                   default=Path("docs/assets/entailment/bench.json"))
    args = p.parse_args()

    import os

    nli_path = args.nli or os.environ.get("SATQUERY_NLI")
    if args.compare and not nli_path:
        print("--compare needs --nli or SATQUERY_NLI", file=sys.stderr)
        return 1

    scores: list[Score] = []
    for suite in args.suites:
        cases = SUITES[suite]
        combos = [([DeterministicBackend()], "deterministic")]
        if args.compare:
            combos.append(([NLIBackend(nli_path)], "nli"))
            combos.append(
                ([DeterministicBackend(), NLIBackend(nli_path)],
                 "deterministic+nli")
            )
        for backends, label in combos:
            scores.append(score_backends(backends, label, cases, suite))

    for score in scores:
        d = score.to_dict()
        print(
            f"\n[{d['suite']}] {d['backend']}: {d['correct']}/{d['total']} "
            f"({d['accuracy']:.1%})"
        )
        print(f"  false sentence marked retained (dangerous) : "
              f"{d['dangerous_false_retained']}")
        print(f"  true sentence marked flagged  (destructive): "
              f"{d['destructive_true_flagged']}")
        print(f"  uncheckable marked retained   (overclaim)  : "
              f"{d['overclaimed_unverifiable_retained']}")
        for expected, got in sorted(d["confusion"].items()):
            print(f"    expected {expected:13s} -> {dict(sorted(got.items()))}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "suites": {k: len(v) for k, v in SUITES.items()},
                "provenance": {
                    "tuned": (
                        "Written before either backend was scored, but the "
                        "gate was then CHANGED twice in response to these "
                        "results - the strong/weak precedence rule and the "
                        "premise independence clause both came from failures "
                        "here. Optimistic; do not quote as a generalisation "
                        "estimate."
                    ),
                    "clean": (
                        "Written after the gate was final and never used to "
                        "change it. This is the honest number. Still small "
                        "(n=25) and single-author."
                    ),
                    "thresholds": (
                        "CONTRADICTION_THRESHOLD and ENTAILMENT_THRESHOLD were "
                        "never tuned against either suite."
                    ),
                },
                "scores": [s.to_dict() for s in scores],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
