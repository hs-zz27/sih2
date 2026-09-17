# Which Track B adapter should be deployed?

**Written 2026-09-07. This is a decision document, not a decision.** Nothing
in the repository was changed to produce it. `docker-compose.yml` still points
`SATQUERY_VQA_ADAPTER` at `checkpoints/track_b_v2/adapter_final`, and it
should keep pointing there until the four preconditions in §5 are met.

The short version: **`track_b_v3` is the better model on every accuracy
measure the project has, in-domain and out, and it is not yet safe to
deploy.** Those two facts are both true and neither cancels the other.

---

## 1. What is deployed today

| | |
|---|---|
| Deployed | `checkpoints/track_b_v2/adapter_final` |
| Set in | `docker-compose.yml`, `docker-compose.gpu.yml` (both), and the `SATQUERY_VQA_ADAPTER` env var the CLI reads |
| Trainable params | 37,152,768 · LoRA r=16, α=32, 7 language-side projections |
| Steps | 300 · 6 h 26 m on the RTX 4050 |
| Status | known-good, reproducible, the artifact every published Track B number was measured on |

The candidate is `checkpoints/track_b_v3/adapter_final`: 82,726,912 trainable
parameters, r=32, α=64, eleven modules including the vision tower
(22,859,776 parameters, 3.42% of the 668.7M tower), 2,000 steps.

---

## 2. The accuracy case for swapping — it is strong

Every accuracy comparison the project has, points the same way.

| Measurement | `v2` (deployed) | **`v3`** | Source |
|---|---|---|---|
| RSVQA-LR **official test**, published convention (count excluded, n=7,057) | 0.7731 | **0.8923** [0.8849, 0.8993] | `phase4-results.md` §3.2 |
| RSVQA-LR official test, all types (n=10,004) | 0.5985 | **0.6966** | same |
| — presence / comparison | 0.8474 / 0.7216 | **0.8897 / 0.8946** | same |
| Beats a train-fitted per-type constant? | yes, χ²=47.6 (1,022 vs 732) | **yes, χ²=777.4 (1,675 vs 403)** | same |
| VRSBench **zero-shot**, strict (n=7,999) | 0.2045 | **0.2968** | `phase4-results.md` §4.5 |
| VRSBench: types where it beats the other arm | — | **12 of 12** | same |
| Held-out val, overall token F1 | 0.7927 | **0.8550** | `external_benchmark_results.json` |

**The out-of-domain result is the one that matters most.** `v3`'s in-domain
gain could have been overfitting to the adaptation set. It was not: the same
checkpoint is better on VRSBench, which neither arm has ever trained on,
**uniformly across all twelve question types**. That is the evidence that the
vision-tower adaptation improved genuine transfer rather than in-domain fit,
and it is the strongest argument on this page.

At 89.23 on the official RSVQA-LR test split, `v3` is statistically
indistinguishable from LHRS-Bot-Nova (89.61, 8B parameters) and within 3.7
points of the strongest published 2B model.

---

## 3. Why it is not a swap you can just make

### 3.1 The reliability metrics regress, and two of the three are the same fact

| Metric | direction | `v2` | `v3` | better |
|---|---|---|---|---|
| Refusal recall (17 unanswerable items) | higher better | **0.4118** | 0.3529 | **v2** |
| Lexical-shortcut probe (matched pairs, both right) | higher better | **0.1667** | 0.0833 | **v2** |
| — the same probe as counts | | **2 / 12** | 1 / 12 | **v2** |
| False-refusal rate | lower better | 0.0077 | **0.0039** | **v3** |

**Do not read the false-refusal row as independent evidence for `v3`.** A
model that refuses less has both a lower refusal recall *and* a lower
false-refusal rate; these are two views of one behaviour change, not two
findings. Netted out, **`v3` refuses less, and what it loses is concentrated
in the hardest and most important case** — the lexical-shortcut probe, where
the question wording is byte-identical and only the image differs, so a model
answering from text alone cannot get both of a pair right. `v3` gets one of
twelve. `v2` gets two.

For a system whose entire pitch is that a confident wrong answer is worse
than an abstention, that is the metric to be most reluctant about.

### 3.2 The run was defective and the ablation is confounded

`v3` was trained under a label-masking defect: the supervised span began
before the `<|image_pad|>` expansion, so roughly 89% of supervised tokens
were image placeholders and the loss sat at ~6.8 for 1,950 steps. It measures
best anyway — the defect cut the span *early*, so it supervised more than
intended rather than less.

But `v3` changes **three things at once** against `v2`: LoRA rank 16→32,
vision-tower targeting 0→22.9M parameters, and steps 300→2,000. The
`track_b_v3_probe` arm — the *identical* 82.7M configuration at 200 steps —
scores only 0.8039 on the official test split against `v3`'s 0.8923, which
points at **steps** rather than vision parameters doing much of the work.

**The isolating arm — rank-32, language-only, 2,000 steps — has not been
run.** Until it is, "the 82.7M visual adaptation is worthwhile" is not a
supported claim. "82.7M parameters at 2,000 steps is worth about +12 points
on the official test split" is.

### 3.3 What is *not* a reason to hesitate

**Counting.** `v3` scores 0.2280 on the 2,947 count questions, below the
0.2555 train-fitted constant. That is real and it is **W17** — but `v2`
scores 0.1802, which is worse, and the base model scores 0.0000. Counting is
negative-value under every checkpoint. It is an argument for routing count
questions away from the VLM entirely; it is not an argument between v2 and v3.

---

## 4. What has changed since the audit recommended this

`docs/external_benchmark_audit.md` §12 P2 listed three things in order. **The
first is now done.**

| P2 step | Status |
|---|---|
| 1. Commit the label-masking fix and its regression test | **DONE** — `pre-import:label-masking-fix`, "Supervise the answer, not 312 image placeholders per example", with `tests/test_vlm_label_masking.py`. The commit measures the defect precisely: 1,139 of 1,279 supervised tokens (89.1%) were `<|image_pad|>` |
| 2. Run the isolating ablation (rank-32, language-only, 2,000 steps) | **not run** |
| 3. Re-run `v3`'s recipe with the corrected mask | **not run** |

So the blocker is no longer the fix. It is two training runs.

Separately, **W16 is closed** (`pre-import:W16-fix`): the six specialist heads now have
`--eval-only`, so re-measurement no longer destroys the artifact of record.
That does not affect Track B — `evaluation/track_b_eval.py` was always
read-only — but it removes the general hazard from the surrounding work.

---

## 5. Preconditions for a safe swap

All four, in this order. Each is checkable.

1. **Re-run `v3`'s recipe with the corrected mask**, with held-out validation
   every N steps and validation-based checkpoint selection. If the corrected
   run beats the defective one, it becomes the candidate. **If it does not,
   that is itself a finding worth publishing** — and the defective run stays
   the best checkpoint the project has, which is a strange enough result to
   deserve saying out loud rather than burying.
2. **Run the isolating ablation** (rank-32, language-only, 2,000 steps).
   Without it the project cannot say what the vision tower bought.
3. **Recover refusal recall and the lexical-shortcut probe** to at least
   `v2`'s 0.4118 and 2/12. If the corrected run does not recover them on its
   own, this is a data question — the refusal set is 17 items in a 534-example
   mix — not a hyper-parameter one.
4. **Re-measure on the official RSVQA-LR test split and VRSBench** and record
   both with their dates, per `docs/code-freeze.md`. Not the 207-question
   validation slice, which cannot resolve differences smaller than ~6 points.

**Cost:** two overnight runs on the existing RTX 4050. `v2`'s 300-step run
took 6 h 26 m; a 2,000-step run is proportionally longer, and file timestamps
put `v3`'s original at roughly 14–27 hours with a step rate that halved
mid-run under thermal load. Budget one run per night, not two. Evaluation is
about 12 minutes per arm on the val split, plus roughly 3.5 hours for the
official RSVQA-LR test split at ~1.25 s/question, and VRSBench is longer
still — see the long-running-work rules before starting either.

**This is an unfreeze decision.** `docs/code-freeze.md` forbids retraining
that changes a published number, and every item above does. It needs an
explicit, recorded unfreeze from the team lead, the way Unfreeze 1 was
recorded for the v2 retrain.

---

## 6. If there is no time before the demo

Then **keep `v2` deployed and say why.** It is the reproducible, known-good
checkpoint, every published Track B number was measured on it, and its
refusal behaviour is the better of the two on the metric that matters most
for a system that argues abstention beats a confident wrong answer.

`v3`'s 89.23 is still quotable as a *measurement* — it is on the official
test split with a passed leakage check — provided it is quoted as what it is:
the best checkpoint the project has, from a defective run whose ablation is
confounded, not the deployed model. Saying that in one sentence is a stronger
position than either hiding it or shipping it.

---

## 7. Recommendation

**Do not swap yet.** Do the two runs, in the order in §5, under a recorded
unfreeze. Then swap on the corrected run if it holds the accuracy and
recovers the refusal behaviour — and if it holds the accuracy but not the
refusal, bring that trade back here as its own decision rather than resolving
it by picking the higher headline.
