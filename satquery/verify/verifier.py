"""Physics verifier: check text claims against the index engine (task 2.9).

Phase 1 scored "agreement" by whether a threshold was trustworthy. That says
nothing about whether the *answer* is true. This checks the claims themselves:
a sentence asserting "60% water" is compared against the measured NDWI
fraction, and a disagreement is named rather than averaged away.

The SWIR-free built-up path matters here. NDBI needs SWIR1, which Cartosat-2E
MX does not carry (verification item 6), so built-up claims are verified
against the low-NDVI + SAR + texture proxy instead - and every verdict records
which path was used, because a proxy-backed verdict deserves less weight than
an NDBI-backed one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# Words that map onto something the index engine actually measures.
SUBJECT_TERMS: dict[str, tuple[str, ...]] = {
    "vegetation": ("vegetation", "vegetated", "forest", "trees", "crops", "farmland",
                   "green", "arable"),
    "water": ("water", "river", "lake", "flooded", "wetland", "sea", "reservoir"),
    "built_up": ("built-up", "built up", "urban", "buildings", "settlement",
                 "industrial", "city", "construction"),
}

# Which index measures each subject, best first. The fallbacks are what make
# this work on a sensor with no SWIR.
SUBJECT_INDICES: dict[str, tuple[str, ...]] = {
    "vegetation": ("ndvi",),
    "water": ("mndwi", "ndwi"),
    "built_up": ("ndbi", "builtup_proxy"),
}

# A claimed fraction within this absolute tolerance of the measured one counts
# as full agreement; beyond FRACTION_CONFLICT it is a named conflict.
FRACTION_TOLERANCE = 0.10
FRACTION_CONFLICT = 0.25

# Proxy-backed verdicts are capped: the SWIR-free built-up estimate is a
# weaker instrument than NDBI and should not certify a claim outright.
PROXY_AGREEMENT_CAP = 0.7

_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


@dataclass
class Claim:
    kind: Literal["fraction", "presence"]
    subject: str
    value: float | None
    text: str


@dataclass
class Verdict:
    claim: Claim
    agreement: float
    index_used: str | None
    measured: float | None
    path: Literal["direct", "swir_free_proxy", "unverifiable"]
    note: str


def subject_of(fragment: str) -> str | None:
    """First subject mentioned anywhere in `fragment`."""
    lowered = fragment.lower()
    for subject, terms in SUBJECT_TERMS.items():
        if any(t in lowered for t in terms):
            return subject
    return None


def subjects_in(fragment: str) -> list[str]:
    """EVERY subject mentioned, in declaration order.

    `subject_of` returns only the first, and `extract_claims` used it for
    presence claims - so a sentence asserting several classes was checked on
    exactly one of them. The trained captioner exposed this immediately: "a
    bridge is over a river with some green trees on both sides" produced a
    single vegetation claim, and the water assertion was never verified at
    all. Returning every subject means each assertion in the sentence gets a
    verdict.
    """
    lowered = fragment.lower()
    return [
        subject
        for subject, terms in SUBJECT_TERMS.items()
        if any(t in lowered for t in terms)
    ]


def nearest_subject(sentence: str, position: int) -> str | None:
    """Subject term that a percentage at `position` refers to.

    English puts the class after the number - "65% vegetation, 28% water and
    14% built-up land" - so the subject that *follows* the figure is preferred,
    and only if none does are earlier ones considered. Plain nearest-by-distance
    gets this wrong: "14%" is equidistant from "water" and "built-up", so the
    tie broke toward whichever class was declared first and the built-up figure
    was silently attributed to water.
    """
    lowered = sentence.lower()
    after: list[tuple[int, str]] = []
    before: list[tuple[int, str]] = []
    for subject, terms in SUBJECT_TERMS.items():
        for term in terms:
            start = 0
            while (found := lowered.find(term, start)) != -1:
                target = after if found >= position else before
                target.append((abs(found - position), subject))
                start = found + 1
    if after:
        return min(after)[1]
    if before:
        return min(before)[1]
    return None


def extract_claims(answer: str) -> list[Claim]:
    """Pull checkable assertions out of an answer.

    Deliberately conservative: only claims tied to a subject the index engine
    can measure are returned. Inventing structure from free text would produce
    verdicts about things never actually asserted.
    """
    claims: list[Claim] = []
    if not answer:
        return claims

    for sentence in re.split(r"[.;\n]", answer):
        if not sentence.strip():
            continue
        # Percentages are attributed to the nearest preceding/following subject
        # word within the same clause.
        for match in _PERCENT_RE.finditer(sentence):
            subject = nearest_subject(sentence, match.start())
            if subject:
                claims.append(
                    Claim("fraction", subject, float(match.group(1)) / 100.0,
                          sentence.strip())
                )
        if not _PERCENT_RE.search(sentence):
            # One claim per subject mentioned, not just the first.
            for subject in subjects_in(sentence):
                claims.append(Claim("presence", subject, None, sentence.strip()))
    return claims


def _measure(subject: str, indices: dict) -> tuple[str | None, float | None, str]:
    """Measured fraction for a subject, and which path produced it."""
    for name in SUBJECT_INDICES.get(subject, ()):
        entry = indices.get(name)
        if entry and entry.get("fraction_above_threshold") is not None:
            path = "swir_free_proxy" if name == "builtup_proxy" else "direct"
            return name, float(entry["fraction_above_threshold"]), path
    return None, None, "unverifiable"


def verify_claim(claim: Claim, indices: dict) -> Verdict:
    index_used, measured, path = _measure(claim.subject, indices)

    if measured is None:
        return Verdict(
            claim, 0.5, None, None, "unverifiable",
            f"no index available to check a {claim.subject} claim; "
            "agreement is neutral, not confirmed",
        )

    if claim.kind == "presence":
        # A presence claim is supported when the class occupies a
        # non-negligible share of the scene.
        agreement = 1.0 if measured >= 0.05 else 0.3
        note = (
            f"{claim.subject} covers {measured:.0%} by {index_used}"
            if agreement == 1.0
            else f"{claim.subject} asserted but {index_used} measures only {measured:.0%}"
        )
    else:
        error = abs((claim.value or 0.0) - measured)
        if error <= FRACTION_TOLERANCE:
            agreement = 1.0
        elif error >= FRACTION_CONFLICT:
            agreement = 0.0
        else:
            span = FRACTION_CONFLICT - FRACTION_TOLERANCE
            agreement = 1.0 - (error - FRACTION_TOLERANCE) / span
        note = (
            f"claimed {claim.value:.0%}, {index_used} measures {measured:.0%} "
            f"(error {error:.0%})"
        )

    if path == "swir_free_proxy":
        agreement = min(agreement, PROXY_AGREEMENT_CAP)
        note += "; verified via the SWIR-free proxy, not NDBI"

    return Verdict(claim, round(agreement, 4), index_used, measured, path, note)


def verify(answer: str, index_payload: dict) -> dict:
    """Verify every checkable claim in `answer` against measured indices."""
    indices = index_payload.get("indices", {})
    verdicts = [verify_claim(c, indices) for c in extract_claims(answer)]

    agreements = {
        f"{v.claim.subject}:{v.claim.kind}": v.agreement for v in verdicts
    }
    conflicts = [
        f"{v.claim.subject}: {v.note}" for v in verdicts if v.agreement < 0.5
    ]

    # Report the index that backed the built-up verdict, not the generic
    # path label - "ndbi" and "swir_free_proxy" are what a reader needs.
    built_up = next(
        (
            (v.index_used or "not_computed") if v.path == "direct" else v.path
            for v in verdicts
            if v.claim.subject == "built_up"
        ),
        None,
    )
    if built_up is None:
        built_up = "swir_free_proxy" if "builtup_proxy" in indices else (
            "ndbi" if "ndbi" in indices else "not_computed"
        )

    return {
        "agreements": agreements,
        "conflicts": conflicts,
        "built_up_path": built_up,
        "n_claims": len(verdicts),
        "verdicts": [
            {
                "subject": v.claim.subject,
                "kind": v.claim.kind,
                "claimed": v.claim.value,
                "measured": v.measured,
                "index": v.index_used,
                "agreement": v.agreement,
                "path": v.path,
                "note": v.note,
            }
            for v in verdicts
        ],
    }
