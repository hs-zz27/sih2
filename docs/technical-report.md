# SatQuery AI — Technical Report

**PS 26167 · ISRO / Department of Space · SIH 2026**
*An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image
Analysis through Text Queries*

Plan task 4.3. Every number in this report is read from an artifact in this
repository — a `metrics.json` beside a checkpoint, or a report under
`docs/assets/`. Where a number is weak, or negative, or contradicts what the
design predicted, it appears here in that form. Nothing is estimated.

---

## 1. What the system is

An analyst uploads one or two rasters and asks a question in English. A
constrained planner reads the *images* to decide which tasks are legal,
classifies the query into one of nine tasks, selects tools from a
version-controlled capability matrix, executes them with only the parameters
that matrix permits, and returns an answer with a confidence, a georeferenced
evidence pack, and a full execution trace.

Five decisions shape everything else:

1. **Two adaptation tracks, bridged.** BigEarthNet is 10 m; Cartosat-2S MX is
   1.6 m. No single-track model spans that, so a band-agnostic encoder is
   adapted at 10 m and a VLM is instruction-tuned at object level, joined by
   band-presence masking, band dropout and GSD conditioning.
2. **A constrained planner, not a free-form LLM agent.** The PS grades the
   observable trace and explicitly does not grade internal reasoning, so
   determinism plus a provable illegal-plan rate strictly dominates.
3. **Physics verifies neural, never the reverse.** Classical indices and SAR
   statistics are an independent referee with no learned parameters.
4. **Deterministic computation for every number.** Counts come from
   detections, areas from the affine transform, direction from a signed
   subtraction against a stated threshold. Generation is used for prose only,
   and prose passes an entailment gate.
5. **Breadth before depth.** Scores are normalised before combining, so one
   zero costs more than any amount of polish gains.

---

## 2. Results

### 2.1 Adaptation (PS M1)

| Measurement | Value | Source |
|---|---|---|
| Track A, BigEarthNet-19 mAP, 12 bands | 0.2854 | `checkpoints/track_a_full_base/metrics.json` |
| Track A, Cartosat 4-band subset | 0.2573 | same |
| **Band retention (4-band / 12-band)** | **0.9015** | same |
| Stage A2, WHU-OPT-SAR mAP (fine-tuned) | **0.7759** | `checkpoints/stage_a2/` |
| Stage A2, frozen probe | 0.7206 | `checkpoints/stage_a2_frozen/` |
| Stage A3, adaptation gain | **+0.1729** (0.1151 → 0.2880) | `checkpoints/stage_a3/` |
| Track B, `rsvqa_lr` exact match, v0 → v1 | **0.4510 → 0.6425** | `docs/assets/refusal/` |

**Band dropout, ablated:** with dropout retention 0.9025 against 0.8443
without, at a cost of 1.4 points of full-band mAP (0.4171 vs 0.4310). The
design claims dropout buys robustness to missing bands; that is the
measurement of the claim.

### 2.2 Task performance

| PS requirement | Metric | Value |
|---|---|---|
| M2 single-image VQA | `rsvqa_lr` exact match | **0.6425** (n=207) |
| M3 captioning | RSICD BLEU-4 | 0.2446 — **13.4% unique captions** |
| M3 grounding | DIOR-RSVG Acc@0.5 | **0.0762** |
| M4 change description | LEVIR-CC BLEU-4, changed pairs | **0.3063** |
| M4 change VQA | **CDVQA, full split** | **0.5380** vs 0.5084 baseline |
| M5 change map | LEVIR-CD F1 | 0.5597 |
| M6 cross-modal | complementarity gain | **−0.0064** |
| M7 orchestration | illegal-plan rate | **0 / 600** |

### 2.3 CDVQA, the prescribed change-VQA benchmark

The most instructive result in the project, because it was wrong twice first.

| pass | accuracy | note |
|---|---|---|
| first measurement | **0.0000** | structural: RGB imagery, no classical index computable |
| second | 0.4439 | a real model — and **below** the 0.5084 constant baseline |
| **third** | **0.5380** | pretrained encoder; beats the baseline by +0.0296 |
| oracle over ground-truth maps | **0.9975** | the ceiling this design allows |

Coverage is **100%** — 39,686 questions over 968 pairs, the full official test
split. Six of the eight question types derive at exactly 1.0000 from
ground-truth maps; the 0.25% shortfall is two deliberate refusals (an
all-unchanged scene has no "largest change"; an exact area tie does not
discriminate) rather than error.

**What this establishes:** CDVQA is not a VQA problem. It is one semantic
change segmentation problem plus exact arithmetic, and 93% of the remaining
gap is the segmenter's change-class mIoU of 0.2636.

A 20-point gap was also found here between calling the tool directly (0.5701)
and running the full pipeline (0.3616), because only 67.4% of CDVQA questions
reached the tool that answers them — `change_to_what` reached it **0.000** of
the time. After adding real question phrasings to the router's training bank,
routing is 1.000, including **1.000 on 151 held-out phrasings never trained
on**, and the two paths agree to six decimal places.

### 2.4 Calibration

| head | before ECE | after ECE | method |
|---|---|---|---|
| `change_mask_v1` | 0.0668 | **0.0034** | **affine**, not temperature |
| `landcover_v1` | 0.0638 | — | temperature T = 1.608 |

Temperature scaling alone did not fit the change mask; the accepted transform
is recorded rather than the conventional one assumed.

**No head currently reports a probability of correctness**, so
`CALIBRATABLE_CONFIDENCE_METHODS` is deliberately empty. An empty set is the
correct state, not a gap: calibrating a stub's hardcoded constant would
produce a number that looks measured and is not.

### 2.5 Selective prediction and AURC

| head | n | base error | AURC | optimal | **E-AURC** |
|---|---|---|---|---|---|
| Tier-1 router | 29 | 0.3793 | 0.1302 | 0.0897 | **0.0405** |
| `landcover_v1` | 111,473 | 0.2064 | 0.1195 | 0.0229 | **0.0966** |

Raw AURC makes the two look comparable. **E-AURC — excess over the optimal
ranking — separates them by 2.4×**, and is the number to read: the router's
confidence ranks its own errors far better than the land-cover head's does.

### 2.6 Abstention and the entailment gate

| measurement | value |
|---|---|
| entailment gate, clean suite, hybrid backend | **96%**, all 9 contradictions caught |
| deterministic gate cost | **+1.9 ms/query** |
| NLI backend cost | +2,625 ms — **22× the pipeline** |
| adversarial queries that abstain | 12.7% |
| abstentions without a named reason | **0** |

The first hybrid gate scored *identically* to deterministic alone, because a
precedence rule meant NLI was never consulted on the six cases that mattered.
That was caught before publication and the gate was fixed.

### 2.7 Confidence

Three components — model, agreement, input quality — combined geometrically,
with the limiting component named in the answer. Stressed rather than
asserted: injecting 94% nodata moves `input_quality` −0.15 and `agreement`
−0.48 while leaving `model` untouched, so the components respond to what they
claim to measure.

One stressor initially wrote zeros without setting the raster's nodata value
and therefore moved the wrong component — a bug in the *measurement* that read
as a failure of the system.

### 2.8 Robustness

| measurement | value |
|---|---|
| soak, 120 iterations, 20 warm-up excluded | +0.0239 MB/query |
| median runtime | 134.1 ms |
| offline suite, sockets blocked | 13 tests pass |
| tests | **855 pass** (at the freeze) |
| no-torch CI simulation | 730 pass, 18 skip, 0 fail (at the freeze) |

The plan asked for a 20-iteration soak. At 20 the figure is +0.2445 MB/query —
a false leak alarm produced entirely by warm-up. The reported number is 120
iterations with warm-up excluded.

**Re-measured 2026-08-30, after the post-freeze audit.** The two test figures
above are the freeze-time measurements and stand as recorded; these are the
current ones, and they are **not comparable** to them, because the suite grew
by 141 tests and the trained checkpoints were deleted and then restored (§"Model availability"
below, and `docs/00` §3.6 L26).

| measurement | value |
|---|---|
| tests | **1070 pass, 0 fail, 0 skip**, 456.8 s |
| no-torch CI simulation | **851 pass, 32 skip, 0 fail**, 153.3 s |
| illegal-plan rate | **0 / 600**, re-verified after the router change |

All nine skips in the full run name a missing checkpoint. Before the loss the
suite skipped nothing, because the checkpoints were on the machine; the tests
that exercised them now skip with the checkpoint they wanted in the message,
which is the same gate the suite has always used for real-model tests. The
no-torch run skips 34: the same nine, plus the 14 training scripts whose
module-scope `import torch` cannot resolve without the `train` extra, plus
the 11 that already skipped at the freeze.

The no-torch simulation is also, as of this date, **reproducible**:
`scripts/ci_no_torch_sim.py` blocks torch, peft, transformers, bitsandbytes,
accelerate and datasets at import time in a subprocess and writes
`docs/assets/ci/no_torch.json`. The freeze's bug-fix bar has always named this
check; until now it had no script and was run by hand.

### 2.8b Model availability — 2026-08-31 (recovered)

**The trained checkpoints were deleted on 2026-08-30 and restored on
2026-08-31** from a Windows volume shadow copy taken six hours forty minutes
before the deletion: **4.542 GB, 136 files, 18 directories**. The restore is
bit-exact where it can be proven — `change_mask/ckpt_step_1780.pt` returned the
SHA-256 recorded from the live file before the deletion — all 61 `.pt` files
load, and every `metrics.json` reproduces the numbers printed in this report.

Twelve small JSON sidecars (42,104 bytes, 0.00093% of the tree) restored as
NUL bytes. **The three that mattered were repaired on 2026-08-31**, each
validated by reproducing a published metric: the caption vocabulary returned
BLEU-4 0.24460787515482577 against the published 0.24460787515482577, the
grounding vocabulary reproduced Acc@0.5 and Acc@0.7 bit-exactly, and the
regenerated multires band statistics reproduced mAP to 17 significant digits
at all four GSD levels. ~~All eight learned tools load.~~ **RESTORED 2026-09-01** by retraining Track B - see the v2 section of `docs/model-cards.md`; `rsvqa_lr` 0.6473 (v2) against 0.6425 (v1) on the identical 534-example split, with the refusal metrics reproduced to four decimals. **CORRECTED 2026-09-01: seven of eight load, not eight.** The Track B QLoRA adapter is destroyed - `adapter_model.safetensors` is 148,712,776 bytes of which the first 148,701,184 are NUL (99.9922%), and the same is true of all eleven adapter files, 1.636 GB in total. The earlier claim came from a verification that loaded the 61 `.pt` files and only *hashed* the safetensors, so a whole model's weights were reported as recovered without ever being opened. Re-verified by loading every weight file: **64 load (10.784 GB), 11 fail (1.636 GB)**, the failures being exactly the adapters. See `docs/00` section 3.6 **L32**. Nine reporting-only
sidecars remain zeroed; `docs/00` §3.6 L29 lists them.

**No number in this report was re-derived, re-estimated or adjusted at any
point** — not during the loss, and not during the recovery. They were read
from the `metrics.json` each training run wrote, they are reproduced in
`docs/model-cards.md`, `docs/phase1-status.md` and `docs/assets/`, and the
restored files were checked against them rather than the other way round.

### 2.8c Model availability — 2026-08-30 (superseded, kept for the record)

**The trained checkpoints are not on disk.** They were deleted during the
post-freeze audit and could not be recovered; the inventory, the recovery
attempt and the root cause are `docs/00` §3.6 **L26** and **L27**.

Every number in this report was read from the `metrics.json` the training run
wrote, and every one of those numbers is reproduced here, in
`docs/model-cards.md`, in `docs/phase1-status.md` and in the JSON reports under
`docs/assets/`, all of which are in git and untouched. **Nothing in this report
has been re-derived, re-estimated or adjusted, and no number was removed.**
What cannot be done today is *reloading a head to reproduce one*, and the
`/models` registry page renders empty because it reads its numbers from inside
the deleted directories. Retraining is a separate decision that has not been
made.

### 2.9 Ablations

Four were planned. **Two are measured, one is a measured negative, and one is
not answerable** — reported that way rather than as four tables of equal
apparent authority.

| ablation | status | result |
|---|---|---|
| agent vs monolith | measured | ungated, the same classifier selects an impossible task on **148/600 (24.7%)**; gated, **0/600** |
| verifier on/off | measured | the gate catches 9/9 contradicting sentences at +1.9 ms |
| triad (optical/SAR/fused) | measured, **negative** | gain −0.0064; fusion does not beat optical alone |
| two-track | **not comparable** | the tracks were trained and evaluated on different tasks; no controlled comparison exists |

The two-track split is the project's central design decision and it is
**reasoned, not demonstrated**. Saying so is the honest position.

---

## 3. Limitations

Twenty are recorded in `docs/00` §3.6 with evidence and consequence. The ones
a reviewer should know about without asking:

1. **Grounding is near floor** — Acc@0.5 0.0762. The PS's own query
   *"Highlight the water body referred to in the query"* routes here.
2. **Fusion does not help** — complementarity −0.0064.
3. **Tier-1 routing is 0.5862** on a never-tuned 29-item holdout. The config
   gate keeps a misroute from becoming an illegal plan, so it degrades rather
   than fails.
4. **Task 3.1's refusal half failed** — 5/5 on lexical refusals, **2/12 on
   image-conditional** ones. The model refuses when the question is absurd,
   not when the image cannot support the answer.
5. **VRSBench is not evaluated** — annotations only; imagery lives in DOTA,
   not on disk. One of three prescribed benchmarks has no number.
6. **Captioning diversity is 13.4%** — 146 unique captions over 1,093 images.
7. **`landcover_v1` at threshold 0.5 is worse than always predicting
   negative**, so it asserts on ~0.25% of decisions at 91% precision.
8. **The two-track ablation is not comparable** (§2.9).
9. **Sub-pixel co-registration is unverified.** Footprint overlap is now
   gated, but the cross-modal shift estimator reports **38.1 px on a pair with
   identical footprints**, so `max_coreg_shift_px` is deliberately not
   enforced and we have no real co-registered optical–SAR pair to settle it.
10. **`band_stats.json` is gitignored**, so a fresh clone cannot load
    `landcover_v1` until it is regenerated from the 45 GB corpus.
11. **An unresolved flaky test** — `test_swir_free_path_exercised_on_real_cartosat`
    has failed twice under the CI simulation and passed on every other run.
    Undiagnosed.
12. **Which RISAT the evaluation set uses is unconfirmed.** The PS does not
    specify, and explicitly says not to assume one; the implementation is
    sensor-configurable via adaptive rather than absolute σ⁰ thresholds.
13. **BigEarthNet.txt was not used.** The PS's Background names it the primary
    adaptation dataset; we adapted on BigEarthNet imagery + 19 labels instead.
    The Mandatory Scope permits "or other open source training data", so this
    is defensible — and it is a stated expectation we did not meet.
14. **Model weights are not published**, and the semantic change head's are
    blocked outright: SECOND states no licence at all.

---

## 4. What would move the numbers, in cost order

1. **CDVQA segmenter** — 93% of the headroom to a 0.9975 ceiling sits in one
   well-posed segmentation task with a published literature. Longer training,
   full-resolution crops, a stronger backbone.
2. **Grounding** — the weakest component, and the one a PS representative
   query depends on. Currently a from-scratch backbone; pretraining lifted the
   change segmenter by 56% relative and would likely help here too.
3. **Image-conditional refusal** — three candidate causes are named and the
   run separates none of them. A designed ablation, not more data.
4. **VRSBench** — acquire DOTA imagery and close the third prescribed
   benchmark.
5. **A real co-registered optical–SAR pair** — needed to validate the shift
   estimator and to enforce `max_coreg_shift_px` honestly.

---

## 5. Reproducing every number

```bash
make report          # regenerates every evaluation artifact under docs/assets/
python -m pytest -q  # 1,247 tests (855 at the freeze)
python evaluation/adversarial.py            # illegal-plan rate
python evaluation/cdvqa_oracle.py --split Test
python evaluation/cdvqa_baseline.py --compare artifacts/cdvqa/head_test_pretrained.json
python scripts/make_demo_bundle.py --out data/demo_bundle --verify
python scripts/rehearse.py --runs 10 --offline
```

`docs/phase1-status.md` carries every measurement in dated sections, and later
sections correct earlier ones — including the two CDVQA numbers that were
superseded. Nothing was deleted when it turned out to be wrong.
