"""Entailment gate: check each generated sentence against the payload (3.5).

Task 2.9's verifier checks *claims it can parse* against measured indices.
This generalises that to every sentence of an answer, and - the part that
matters - it refuses to call a sentence "retained" when nothing was able to
check it.

## Three outcomes, not two

The obvious design is retained/flagged, and it is wrong. A binary gate has to
put "we checked this and it holds" and "nothing in the payload says anything
about this" in the same bucket, and `retained` then silently means "not
caught". Judges and report readers would take a 95% retention rate as 95%
verified when most of it was never examined.

So every sentence lands in exactly one of:

* **retained** - a premise positively supports it
* **flagged** - a premise contradicts it
* **unverifiable** - no premise covers the claim at all

`retained + flagged + unverifiable == sentences`, and the trace reports all
four numbers. A scene caption like "the airport is very large" against a
premise of "index thresholds indicate 34% vegetation" is *unverifiable*, not
retained: the premise simply does not talk about airports.

## Two backends

* `deterministic` - always available, no model, no network. Reuses the 2.9
  verifier: a sentence is flagged when a claim it makes disagrees with the
  measured index beyond tolerance. This is the backend that carries the
  guarantee, because its premises are measurements.
* `nli` - optional, activated only when `SATQUERY_NLI` points at a local
  MNLI checkpoint, matching how `rs_vqa_v1` and `change_mask_v1` gate their
  models. It catches contradictions the parser cannot express - tense,
  negation, hedging, relations between clauses.

When both are present the deterministic verdict WINS wherever it has an
opinion, and NLI only fills in sentences it was silent on. A neural
entailment score does not get to overrule a measurement.

The NLI model is loaded with `trust_remote_code=False`; a checkpoint that
needs custom modeling code is refused, per the decision recorded for task 2.7.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from typing import Literal, Protocol

from satquery.verify.verifier import extract_claims, verify_claim

ENV_NLI = "SATQUERY_NLI"

Status = Literal["retained", "flagged", "unverifiable"]

# What the gate does with a flagged sentence. Dropping is the default: an
# answer that contradicts its own measurements should not be shown. The
# original text is preserved verbatim in the trace, so nothing is hidden -
# the user sees a corrected answer and the trace shows what was removed and
# why.
Action = Literal["drop", "annotate", "none"]
DEFAULT_ACTION: Action = "drop"

# An NLI head outputs three-way logits. These thresholds are deliberately
# asymmetric: flagging an answer sentence is a visible, disruptive action, so
# it needs strong evidence, while claiming a sentence is *entailed* is a
# positive assertion about correctness and needs to be strong too. Anything
# between the two is neutral, and neutral means unverifiable.
CONTRADICTION_THRESHOLD = 0.70
ENTAILMENT_THRESHOLD = 0.70

# Sentences shorter than this are punctuation artefacts, not claims.
MIN_SENTENCE_CHARS = 8

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences.

    A regex, not an NLP sentence splitter: answers here are short, generated
    from templates or small models, and adding a tokeniser dependency to the
    always-available path would defeat the point of having one.
    """
    if not text:
        return []
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text)]
    return [s for s in parts if len(s) >= MIN_SENTENCE_CHARS]


@dataclass(frozen=True)
class Premise:
    """One checkable fact drawn from the structured payload."""

    text: str
    subject: str | None = None


def build_premises(index_payload: dict) -> list[Premise]:
    """Verbalise the structured payload into premise sentences.

    An NLI model needs a text premise, but what the system actually knows is
    a dict of measured fractions. This is the bridge, and it is deliberately
    mechanical: every premise states one measured number, so a contradiction
    detected against a premise traces back to a specific measurement rather
    than to prose someone wrote.
    """
    from satquery.verify.verifier import SUBJECT_INDICES

    indices = index_payload.get("indices", {})
    index_to_subject = {
        name: subject
        for subject, names in SUBJECT_INDICES.items()
        for name in names
    }

    premises: list[Premise] = []
    for name, entry in indices.items():
        if not isinstance(entry, dict):
            continue
        fraction = entry.get("fraction_above_threshold")
        if fraction is None:
            continue
        subject = index_to_subject.get(name)
        label = (subject or name).replace("_", " ")
        premises.append(
            Premise(
                f"The {name.upper()} index measures {float(fraction):.0%} "
                f"{label} coverage in this scene. This index is thresholded "
                f"independently of the others and says nothing about how much "
                f"of any other class is present.",
                subject,
            )
        )
    return premises


# How much authority a verdict carries when backends disagree.
#
# A verdict is `strong` when it rests on a measured quantity: a claimed
# percentage compared against a measured fraction, either way it lands. It is
# `weak` when the check could not address what the sentence actually asserts.
#
# The presence check is the weak case, and measurement is how that was found.
# "The scene is almost entirely covered by water" against a measured 5% NDWI
# parses as a *presence* claim about water, water is indeed present, and the
# deterministic backend returns `retained` - for a plainly false sentence. It
# never asked about magnitude, because a presence check cannot. The same holds
# for negation: "there is no vegetation anywhere" is read as an assertion that
# vegetation is a subject of the sentence, not as a denial of it.
Strength = Literal["strong", "weak"]


@dataclass
class SentenceVerdict:
    sentence: str
    status: Status
    reason: str
    backend: str
    score: float | None = None
    strength: Strength = "strong"


class Backend(Protocol):
    name: str

    def judge(
        self, sentence: str, premises: list[Premise], index_payload: dict
    ) -> SentenceVerdict: ...


class DeterministicBackend:
    """Sentence-level wrapper around the task 2.9 physics verifier."""

    name = "deterministic"

    def judge(
        self, sentence: str, premises: list[Premise], index_payload: dict
    ) -> SentenceVerdict:
        indices = index_payload.get("indices", {})
        claims = extract_claims(sentence)
        if not claims:
            return SentenceVerdict(
                sentence, "unverifiable",
                "no claim the index engine can measure", self.name,
            )

        verdicts = [verify_claim(c, indices) for c in claims]
        checkable = [v for v in verdicts if v.path != "unverifiable"]
        if not checkable:
            subjects = ", ".join(sorted({v.claim.subject for v in verdicts}))
            return SentenceVerdict(
                sentence, "unverifiable",
                f"no index available to check the {subjects} claim", self.name,
            )

        worst = min(checkable, key=lambda v: v.agreement)
        if worst.agreement < 0.5:
            # A contradiction is a contradiction whichever claim kind found
            # it: presence flags only fire when the class is near-absent,
            # which is a genuine measurement.
            return SentenceVerdict(
                sentence, "flagged", worst.note, self.name, worst.agreement,
                strength="strong",
            )

        # Support from a presence claim establishes only that the class
        # exists, never that the sentence's magnitude is right, so it must
        # not block a backend that can read magnitude.
        strength: Strength = "strong" if worst.claim.kind == "fraction" else "weak"
        note = worst.note
        if strength == "weak":
            note += "; presence only - this check cannot read magnitude or negation"
        return SentenceVerdict(
            sentence, "retained", note, self.name, worst.agreement, strength=strength
        )


class NLIBackend:
    """Three-way MNLI head over (payload premise, answer sentence).

    Loaded lazily and only when `SATQUERY_NLI` names a local directory, so
    importing this module never touches the network or the GPU.
    """

    name = "nli"

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._label_order: list[str] = []

    def _load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            # trust_remote_code stays False: a checkpoint that needs custom
            # modeling code executes third-party Python, which this project
            # declined for Florence-2 in task 2.7 and declines here too.
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.path, local_files_only=True, trust_remote_code=False
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.path, local_files_only=True, trust_remote_code=False
            )
            self._model = model.eval()
            self._torch = torch

            # Label order differs between MNLI checkpoints, so it is read from
            # the config rather than assumed. Assuming it silently inverts
            # entailment and contradiction, which would flag exactly the
            # sentences it should retain.
            id2label = getattr(model.config, "id2label", {}) or {}
            self._label_order = [
                str(id2label.get(i, i)).lower() for i in range(len(id2label))
            ]
            if not any("entail" in lbl for lbl in self._label_order):
                raise ValueError(
                    f"{self.path} does not look like an NLI checkpoint: "
                    f"labels are {self._label_order}"
                )

    def judge(
        self, sentence: str, premises: list[Premise], index_payload: dict
    ) -> SentenceVerdict:
        if not premises:
            return SentenceVerdict(
                sentence, "unverifiable", "no premise available", self.name
            )
        self._load()
        torch = self._torch

        # Scored against each premise separately, then reduced. A single
        # concatenated premise lets an unrelated clause dilute a real
        # contradiction until it falls under the threshold.
        best_contra = 0.0
        best_entail = 0.0
        contra_premise = ""
        entail_premise = ""
        for premise in premises:
            inputs = self._tokenizer(
                premise.text, sentence, return_tensors="pt", truncation=True
            )
            with torch.no_grad():
                logits = self._model(**inputs).logits[0]
            probs = torch.softmax(logits.float(), dim=-1).tolist()
            for label, p in zip(self._label_order, probs, strict=False):
                if "contradict" in label and p > best_contra:
                    best_contra, contra_premise = p, premise.text
                elif "entail" in label and p > best_entail:
                    best_entail, entail_premise = p, premise.text

        # A sentence one measurement directly entails must not be flagged
        # because a DIFFERENT measurement conflicts with it by inference.
        # Without this, "most of this scene is under water" - entailed at 0.95
        # by a 71% NDWI premise - was flagged at 0.88 against an 8% NDVI
        # premise, because the model reasoned that little vegetation excludes
        # mostly-water. The indices are independently thresholded and overlap,
        # so that inference is not one the premises license.
        if best_entail >= best_contra and best_entail >= ENTAILMENT_THRESHOLD:
            return SentenceVerdict(
                sentence, "retained",
                f"entailed by a measured premise (p={best_entail:.2f}): "
                f"{entail_premise}",
                self.name, best_entail,
            )

        if best_contra >= CONTRADICTION_THRESHOLD:
            return SentenceVerdict(
                sentence, "flagged",
                f"contradicts a measured premise (p={best_contra:.2f}): "
                f"{contra_premise}",
                self.name, best_contra,
            )
        if best_entail >= ENTAILMENT_THRESHOLD:
            return SentenceVerdict(
                sentence, "retained",
                f"entailed by a measured premise (p={best_entail:.2f}): "
                f"{entail_premise}",
                self.name, best_entail,
            )
        return SentenceVerdict(
            sentence, "unverifiable",
            f"neither entailed nor contradicted "
            f"(entail {best_entail:.2f}, contradict {best_contra:.2f})",
            self.name, best_entail,
        )


@dataclass
class GateResult:
    sentences: int
    retained: int
    flagged: int
    unverifiable: int
    backend: str
    action: str
    answer: str
    original_answer: str
    verdicts: list[SentenceVerdict] = field(default_factory=list)

    @property
    def modified(self) -> bool:
        return self.answer != self.original_answer

    def counts_are_consistent(self) -> bool:
        return self.retained + self.flagged + self.unverifiable == self.sentences


def _nli_backend() -> NLIBackend | None:
    path = os.environ.get(ENV_NLI)
    return NLIBackend(path) if path else None


def run_gate(
    answer: str,
    index_payload: dict,
    *,
    action: Action = DEFAULT_ACTION,
    enabled: bool = True,
    backends: list[Backend] | None = None,
) -> GateResult:
    """Run the entailment gate over `answer`.

    `enabled=False` is the off arm of the verifier ablation (task 3.7): the
    answer passes through untouched and the counts report zeros with backend
    "disabled", so a disabled gate is never mistaken for a gate that ran and
    found nothing.
    """
    sentences = split_sentences(answer)

    if not enabled:
        return GateResult(
            0, 0, 0, 0, "disabled", "none", answer, answer, []
        )

    if backends is None:
        backends = [DeterministicBackend()]
        nli = _nli_backend()
        if nli is not None:
            backends.append(nli)

    names = "+".join(b.name for b in backends) or "none"

    premises = build_premises(index_payload)

    verdicts: list[SentenceVerdict] = []
    for sentence in sentences:
        verdict = SentenceVerdict(
            sentence, "unverifiable", "no backend available", names
        )
        for backend in backends:
            candidate = backend.judge(sentence, premises, index_payload)

            # Precedence, in order:
            #   1. nothing decided yet -> take whatever this backend says;
            #   2. a decision beats "unverifiable";
            #   3. a later backend may overturn a WEAK retain, but only by
            #      flagging - it may not upgrade a weak retain to a strong one,
            #      because that would let a neural score certify a sentence no
            #      measurement supports.
            # A strong verdict is never overturned: a measured contradiction
            # or a matched percentage is not up for a second opinion.
            if verdict.status == "unverifiable" and candidate.status != "unverifiable":
                verdict = candidate
            elif verdict.strength == "weak" and candidate.status == "flagged":
                verdict = candidate
            elif verdict.status == "unverifiable":
                verdict = candidate

            if verdict.status != "unverifiable" and verdict.strength == "strong":
                break
        verdicts.append(verdict)

    retained = sum(1 for v in verdicts if v.status == "retained")
    flagged = sum(1 for v in verdicts if v.status == "flagged")
    unverifiable = sum(1 for v in verdicts if v.status == "unverifiable")

    gated = answer
    if action == "drop" and flagged:
        kept = [v.sentence for v in verdicts if v.status != "flagged"]
        gated = " ".join(kept).strip()
        if not gated:
            # Dropping every sentence would leave an empty answer, which is
            # worse than a flagged one: the user gets nothing and no reason.
            # Abstention (task 3.6) is the right mechanism for that, and this
            # says so instead of silently returning "".
            gated = (
                "Every sentence of the draft answer contradicted the measured "
                "indices, so none of it is shown. See the trace for what was "
                "removed."
            )
    elif action == "annotate" and flagged:
        parts = []
        for v in verdicts:
            parts.append(
                f"{v.sentence} [unsupported: {v.reason}]"
                if v.status == "flagged" else v.sentence
            )
        gated = " ".join(parts).strip()

    return GateResult(
        sentences=len(sentences),
        retained=retained,
        flagged=flagged,
        unverifiable=unverifiable,
        backend=names,
        action=action if flagged else "none",
        answer=gated,
        original_answer=answer,
        verdicts=verdicts,
    )
