"""Full instruction mix for the Track B retrain (plan task 3.1).

Track B v0 trained on RSVQA-LR alone: 2,000 optical VQA pairs, all of which
have an answer. Task 3.1 asks for the full mix, **SAR samples**, and **~5%
refusal examples**, with the acceptance criterion "model declines
appropriately".

## Why refusal examples need designing rather than generating

A refusal example is a (image, question, "I cannot answer that") triple, and
the obvious way to make 5% of them is to pair random images with random
unanswerable questions. That teaches the wrong thing. A model trained that way
learns *"questions phrased like this get refused"* - a lexical rule - and will
happily refuse an answerable question in the same register while still
answering an unanswerable one phrased normally.

The refusals here are built so that **the image is the reason**, not the
wording:

* `sensor_cannot_measure` - a question needing SWIR (built-up index, burn
  severity) asked of a 4-band VNIR image. Verification item 6 confirmed
  Cartosat-2E MX carries no SWIR, so this is the exact failure the deployed
  system will meet.
* `single_image_temporal` - a change question asked of one image. The router
  already blocks this structurally (task 3.8); training the model to decline
  it too means the two layers agree instead of the model fighting the gate.
* `not_in_image` - asking about a class the label mask says is absent from
  this specific tile. The refusal is grounded in that tile's annotation, so
  the same question is answerable for a different tile.
* `out_of_scope` - questions imagery cannot answer at all: ownership, price,
  weather, identity of people.

The first three are **image-conditional**: for `not_in_image` in particular,
the identical question is a normal answerable example on a tile where the
class IS present, and the mix includes those positives deliberately. A model
that learns the lexical shortcut will get them wrong, which is what makes the
refusal split measurable rather than assumed.

`out_of_scope` is the one category that is genuinely lexical, and it is capped
low for that reason.

## SAR

WHU-OPT-SAR supplies co-registered optical/SAR pairs with per-pixel labels.
The SAR half is included as its own examples so the model sees speckle and
backscatter rather than only reflectance, and the questions are built from the
label mask so the answers are measured, not written.

Usage:
    python training/prepare/instruction_mix.py --out data/instruct_mix
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.common.paths import index_path  # noqa: E402

REFUSAL_FRACTION = 0.05

# What the model should say when it declines. One canonical form per reason,
# so the refusal is recognisable to the pipeline as well as readable: the
# entailment gate and the abstention policy both key off content, and a model
# that invents a new phrasing each time makes that harder than it needs to be.
REFUSALS = {
    "sensor_cannot_measure": (
        "I cannot answer that from this image. The question needs a shortwave "
        "infrared band, and this sensor provides only visible and near-"
        "infrared bands."
    ),
    "single_image_temporal": (
        "I cannot answer that from this image. Detecting change requires two "
        "acquisitions of the same area, and only one image was provided."
    ),
    "not_in_image": (
        "I cannot answer that from this image. That feature is not present in "
        "this scene."
    ),
    "out_of_scope": (
        "I cannot answer that from this image. Satellite imagery does not "
        "carry that information."
    ),
}

# Questions that genuinely need SWIR. Asked of a 4-band VNIR image these are
# unanswerable for a physical reason, not a phrasing one.
SWIR_QUESTIONS = [
    "What is the NDBI value for this scene?",
    "Compute the built-up index using shortwave infrared.",
    "Use SWIR to separate bare soil from built-up land.",
    "What is the MNDWI water fraction here?",
    "How severe is the burn scar according to the shortwave bands?",
    "Give me the normalised burn ratio for this area.",
]

TEMPORAL_QUESTIONS = [
    "What changed between the two dates?",
    "How many new buildings appeared since the earlier image?",
    "Did the forest shrink over time?",
    "Produce a change mask for this area.",
    "By what percentage did the water extent change?",
    "Compare this scene with the previous acquisition.",
]

OUT_OF_SCOPE_QUESTIONS = [
    "Who owns this land?",
    "What is the market value of these buildings?",
    "What will the weather be here tomorrow?",
    "What is the name of the person who lives in that house?",
    "What is the population of this city?",
    "Which political party governs this region?",
]


@dataclass
class Example:
    image: str
    question: str
    answer: str
    source: str
    kind: str          # "vqa" | "caption" | "refusal"
    modality: str = "optical"
    refusal_reason: str | None = None

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None})


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rsvqa_examples(root: Path) -> list[Example]:
    """RSVQA-LR optical VQA - the whole of the v0 training set."""
    rows = load_jsonl(root / "instruct.jsonl")
    return [
        Example(
            image=str(root / index_path(row["image"])),
            question=row["question"],
            answer=str(row["answer"]),
            source="rsvqa_lr",
            kind="vqa",
        )
        for row in rows
    ]


def whu_sar_examples(root: Path, rng: random.Random) -> tuple[list[Example], list[Example]]:
    """Optical and SAR examples from WHU-OPT-SAR, with label-derived answers.

    Returns (answerable, not_in_image_refusals). The refusals come from the
    SAME label masks, so "is there water here" is answered for a tile that
    has water and refused for one that does not - which is the property that
    stops the model learning a lexical rule.
    """
    index_path = root / "index.json"
    if not index_path.exists():
        return [], []

    index = json.loads(index_path.read_text(encoding="utf-8"))
    classes = index.get("classes", [])
    answerable: list[Example] = []
    refusals: list[Example] = []

    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return [], []

    for split, rows in index.get("splits", {}).items():
        if split != "train":
            continue
        for row in rows:
            label_path = row.get("label") or row.get("lbl")
            if not label_path or not Path(label_path).exists():
                continue
            try:
                mask = np.asarray(Image.open(label_path))
            except Exception:  # noqa: BLE001
                continue

            present = {
                classes[i]
                for i in np.unique(mask).tolist()
                if 0 <= i < len(classes)
            }
            present.discard("background")
            absent = [c for c in classes if c not in present and c != "background"]

            for modality, key in (("optical", "optical"), ("sar", "sar")):
                image_path = row.get(key)
                if not image_path or not Path(image_path).exists():
                    continue
                if present:
                    target = rng.choice(sorted(present))
                    fraction = float((mask == classes.index(target)).mean())
                    answerable.append(Example(
                        image=str(image_path),
                        question=f"Is there {target} visible in this image?",
                        answer=(
                            f"Yes, {target} covers about {fraction:.0%} of the "
                            f"scene."
                        ),
                        source="whu_opt_sar",
                        kind="vqa",
                        modality=modality,
                    ))
                if absent:
                    missing = rng.choice(absent)
                    refusals.append(Example(
                        image=str(image_path),
                        # Deliberately the SAME question form as the
                        # answerable case above. Only the image differs.
                        question=f"Is there {missing} visible in this image?",
                        answer=REFUSALS["not_in_image"],
                        source="whu_opt_sar",
                        kind="refusal",
                        modality=modality,
                        refusal_reason="not_in_image",
                    ))
    return answerable, refusals


def synthetic_refusals(
    pool: list[Example], rng: random.Random, per_category: int
) -> list[Example]:
    """Sensor, temporal and out-of-scope refusals over real images."""
    if not pool:
        return []
    out: list[Example] = []
    plans = [
        ("sensor_cannot_measure", SWIR_QUESTIONS),
        ("single_image_temporal", TEMPORAL_QUESTIONS),
        # Capped at half: this is the one genuinely lexical category, and
        # over-weighting it is how a model learns to refuse on phrasing.
        ("out_of_scope", OUT_OF_SCOPE_QUESTIONS),
    ]
    for reason, questions in plans:
        count = per_category // 2 if reason == "out_of_scope" else per_category
        for _ in range(count):
            base = rng.choice(pool)
            out.append(Example(
                image=base.image,
                question=rng.choice(questions),
                answer=REFUSALS[reason],
                source="synthetic_refusal",
                kind="refusal",
                modality=base.modality,
                refusal_reason=reason,
            ))
    return out


def build_mix(
    data_root: Path, seed: int = 42, refusal_fraction: float = REFUSAL_FRACTION
) -> tuple[list[Example], dict]:
    rng = random.Random(seed)

    answerable = rsvqa_examples(data_root / "rsvqa_lr_2k")
    whu_answerable, whu_refusals = whu_sar_examples(
        data_root / "whu_opt_sar", rng
    )
    answerable += whu_answerable

    if not answerable:
        raise SystemExit(f"no answerable examples found under {data_root}")

    # Target refusal count from the ANSWERABLE total, so the fraction means
    # what it says regardless of how many refusals the label masks happened
    # to supply.
    target_refusals = max(1, round(len(answerable) * refusal_fraction / (1 - refusal_fraction)))

    rng.shuffle(whu_refusals)
    grounded = whu_refusals[: target_refusals // 2]
    remaining = target_refusals - len(grounded)
    synthetic = synthetic_refusals(answerable, rng, max(1, remaining // 3))

    examples = answerable + grounded + synthetic
    rng.shuffle(examples)

    stats = {
        "total": len(examples),
        "answerable": len(answerable),
        "refusals": len(grounded) + len(synthetic),
        "refusal_fraction": round(
            (len(grounded) + len(synthetic)) / len(examples), 4
        ),
        "by_source": dict(Counter(e.source for e in examples)),
        "by_kind": dict(Counter(e.kind for e in examples)),
        "by_modality": dict(Counter(e.modality for e in examples)),
        "by_refusal_reason": dict(
            Counter(e.refusal_reason for e in examples if e.refusal_reason)
        ),
        "seed": seed,
        "note": (
            "Refusals are image-conditional wherever possible: the "
            "not_in_image questions use the SAME wording as answerable ones "
            "and differ only in which tile they are asked about, so a model "
            "that learns a lexical refusal rule fails them. out_of_scope is "
            "the one genuinely lexical category and is capped at half the "
            "per-category count for that reason."
        ),
    }
    return examples, stats


def stratified_split(
    examples: list[Example], val_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    """Split holding the proportion of every refusal reason.

    A plain random split does NOT do this, and the first version of this file
    used one. The consequence was measured rather than theorised: of 244
    refusals, the val side received 17, and `sensor_cannot_measure` received
    ZERO - so one of the four refusal reasons could not be evaluated at all,
    and refusal recall moved 5.9 points per item.

    Stratifying by (kind, refusal_reason) puts a proportional share of every
    category on both sides. It does not manufacture data - 5% of a small
    corpus is still a small number of refusals - but it stops a whole
    category vanishing from the held-out set by chance.
    """
    rng = random.Random(seed)
    strata: dict[tuple[str, str | None], list[Example]] = {}
    for example in examples:
        strata.setdefault((example.kind, example.refusal_reason), []).append(example)

    train: list[Example] = []
    val: list[Example] = []
    for group in strata.values():
        rng.shuffle(group)
        # At least one item per stratum on the val side whenever the stratum
        # has more than one member: a category with zero held-out examples is
        # unmeasurable, which is the failure this function exists to prevent.
        n_val = max(1, round(len(group) * val_fraction)) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("data/instruct_mix"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refusal-fraction", type=float, default=REFUSAL_FRACTION)
    p.add_argument("--val-fraction", type=float, default=0.1)
    args = p.parse_args()

    examples, stats = build_mix(
        args.data_root, args.seed, args.refusal_fraction
    )

    args.out.mkdir(parents=True, exist_ok=True)

    # `track_b_vlm_qlora.load_examples` resolves each `image` against the
    # --data directory, so the paths written here must be relative to THIS
    # output directory, not to the repo root. Written absolute they silently
    # became data/instruct_mix/data/whu_opt_sar/... and every image was
    # reported missing by the dry run.
    import os

    for example in examples:
        example.image = os.path.relpath(
            Path(example.image).resolve(), args.out.resolve()
        ).replace(os.sep, "/")

    train, val = stratified_split(examples, args.val_fraction, args.seed)
    for name, subset in (("instruct", train), ("val", val)):
        path = args.out / f"{name}.jsonl"
        path.write_text(
            "\n".join(e.to_json() for e in subset) + "\n", encoding="utf-8"
        )
        print(f"wrote {path}  ({len(subset)} examples)")

    (args.out / "stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    for key, value in stats.items():
        if key != "note":
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
