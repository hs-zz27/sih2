# Model cards — SatQuery AI

**Plan task 4.5.** One card per trained component. Every number is read from
the `metrics.json` and `run_metadata.json` written by the training run that
produced the checkpoint; nothing here is estimated, rounded up, or carried
over from a paper.

**Read the weakest line in each card.** Several of these models are honestly
poor, and the card says so rather than reporting only the flattering figure.
A reviewer who finds a weakness we did not disclose has learned something
about our reporting, not about the model.

## Weight availability — 2026-08-31 (recovered)

**The checkpoints described below are on disk again.** They were deleted on
2026-08-30 and **restored in full on 2026-08-31** from a volume shadow copy —
4.542 GB, 136 files, bit-exact against a digest recorded before the deletion.
Every `.pt` file loads and every `metrics.json` matches the numbers printed in
these cards.

~~Every component listed below now loads.~~ **CORRECTED 2026-09-01: seven of eight load, not eight.** The Track B QLoRA adapter is destroyed - `adapter_model.safetensors` is 148,712,776 bytes of which the first 148,701,184 are NUL (99.9922%), and the same is true of all eleven adapter files, 1.636 GB in total. The earlier claim came from a verification that loaded the 61 `.pt` files and only *hashed* the safetensors, so a whole model's weights were reported as recovered without ever being opened. Re-verified by loading every weight file: **64 load (10.784 GB), 11 fail (1.636 GB)**, the failures being exactly the adapters. See `docs/00` section 3.6 **L32**.

**`rs_vqa_v1` (Track B) cannot be loaded and cannot be published.** Its card below records a run that happened; the weights behind it no longer exist. Twelve small JSON sidecars came
back from the shadow copy as NUL bytes; the three that mattered were repaired
on 2026-08-31, each validated by reproducing a published metric rather than by
inspection (`docs/00` §3.6 L29):

| Sidecar | Repair | Validation |
|---|---|---|
| `caption/vocab.json` | regenerated from `data/rsicd`, 1,781 tokens | BLEU-4 **0.24460787515482577** vs published **0.24460787515482577**; `n` 1093, `unique_captions` 146 — all exact |
| `grounding/vocab.json` | regenerated from `data/dior_rsvg`, 85 tokens | Acc@0.5 and Acc@0.7 bit-exact; mIoU to 4.2e-9 |
| `track_a_full_multires/band_stats.json` | `compute_stats(sample=2000, seed=0)` over 60,000 patches | multires mAP identical to 17 digits at all four GSD levels |

The recovered NUL files are kept beside their replacements as
`*.zeroed-2026-08-30`, and **all 61 `.pt` digests were re-verified unchanged**
after the repairs. The regenerated `band_stats.json` is not claimed as a
byte-identical restoration — its statistics are identical, its provenance
string records the regeneration.

**Nine sidecars remain zeroed** and are reporting files only: two
`metrics.json` (`track_a_full_multires`, `track_a_dropout`), one
`cross_sensor.json`, one `run_metadata.json`, and four `adapter_config.json`
under the `killtest`, `smoke` and `track_b_v1/adapter_step_275` scratch runs.
Their numbers are published in this document. The consequence is cosmetic: the
`/models` page shows 18 checkpoints, 13 of them carrying metrics.

**Every number in these cards was already verified against the restored
`metrics.json` files and none of them changed.**

---

## Weight availability — 2026-08-30 (superseded, kept for the record)

**The checkpoint files described in these cards are no longer on disk.** On
2026-08-30 `training/run_checkpoint_test.py` deleted `checkpoints/` and the
weights could not be recovered from any source; the full account is `docs/00`
§3.6 **L26**, and the root cause and its containment are **L27**.

What this does and does not mean:

* **Every number in these cards stands.** They were read from the
  `metrics.json` and `run_metadata.json` each training run wrote, and they are
  reproduced here, in `docs/phase1-status.md`, in `docs/technical-report.md`
  and in the JSON reports under `docs/assets/`. Those files are in git and are
  untouched. **Nothing below has been re-derived, re-estimated or adjusted.**
* **The `Checkpoint` row in each card is a historical path, not a claim that
  the file is present.** Every one of them is currently missing. Do not read
  these cards as an inventory of what can be loaded today.
* **`/models` renders empty.** The registry page reads `metrics.json` and
  `run_metadata.json` from inside those directories, so it has nothing to show
  until checkpoints exist again. This is the accurate state, not a bug in the
  page.
* **No learned tool can be enabled.** Every `SATQUERY_*` checkpoint variable
  points at a path that no longer exists, so the registry serves stubs and the
  deterministic index engine — the configuration CI has always run, the test
  suite exercises, and the demo bundle was verified under.
* **The third-party base models survived.** `models/qwen25_vl_3b` and
  `models/nli_deberta_mnli` are intact and their digests are in
  `configs/model_lock.json`. What was lost from Track B is the **QLoRA
  adapter**, which was under `checkpoints/track_b_v1/adapter_final`.

**Retraining is a separate decision and has not been made.** It costs
GPU-hours; it cannot reproduce a previous run bit-for-bit; and under
`docs/code-freeze.md` any number it produced would have to be published as a
new dated section rather than as an edit to the numbers below.

---

## Publication status — read before distributing anything

| Component | Weights publishable? | Why |
|---|---|---|
| `change_vqa_v1` semantic head | **BLOCKED** | Trained on **SECOND**, which states **no licence at all** — not a restrictive one, none. See `docs/verification.md` §"NEW RISK — SECOND states no licence". |
| Everything else | **Undecided — do not publish yet**, and as of 2026-08-30 **not available to publish** (see the availability note above) | Each was trained on an openly licensed corpus, but no licence has been chosen for our own weights and no per-dataset redistribution check has been done. The licence question is unchanged by the loss and still has to be answered before any future weights are distributed. |

The PS's deliverable is *"codes and models including test and demonstration"*.
Code and tests are in the repository. **Weights are not published**, and the
blocker above is the reason for one of them; the rest await a decision that is
the team's to make. This is recorded rather than quietly skipped.

---

## Track A — band-agnostic encoder → `landcover_v1`

| | |
|---|---|
| Checkpoint | `checkpoints/track_a_full_base/` |
| Task | Multi-label land cover, BigEarthNet-19 |
| Training data | BigEarthNet v2 imagery + 19 labels, 30,000 patches, 3 epochs, lr 1e-3, dim 64, seed 42 |
| Key design | Band-presence masking + **random band dropout 0.3** + GSD conditioning |
| **mAP, all 12 bands** | **0.2854** |
| **mAP, Cartosat 4-band subset** | **0.2573** |
| **Retention (4-band / 12-band)** | **0.9015** |

**What the retention number is for.** The model must survive losing eight of
twelve bands, because Cartosat-2S MX has four. Band dropout is what buys that,
and it was ablated rather than assumed:

| arm | mAP all bands | mAP 4-band | retention |
|---|---|---|---|
| with dropout (`track_a_dropout`) | 0.4171 | 0.3765 | **0.9025** |
| without (`track_a_nodropout`) | 0.4310 | 0.3639 | 0.8443 |

Dropout costs 1.4 points of full-band mAP and buys 5.8 points of retention.
That is the trade the design claims, measured.

**Weakness:** mAP 0.2854 is low. At threshold 0.5 the head is **worse than
always predicting negative** (0.2064 against 0.1834), which is why
`landcover_v1` asserts on only ~0.25% of decisions at 91% precision and the
narrative synthesiser carries land-cover answers. See limitation L8.

**Not shipped in git:** `band_stats.json` lives beside the weights and
`checkpoints/` is gitignored, so a fresh clone cannot load this head until the
statistics are regenerated with `compute_stats(seed=0, sample=2000)` over the
BigEarthNet train shards — which needs the 45 GB corpus. Limitation L2.

### Stage A2 — WHU-OPT-SAR transfer

| | fine-tuned | frozen probe |
|---|---|---|
| mAP | **0.7759** | 0.7206 |

Fine-tuning beats a frozen probe by 5.5 points, which is the evidence that the
transfer is doing work rather than the head fitting the labels alone.

### Stage A3 — high-resolution transfer

| | |
|---|---|
| Frozen probe mAP | 0.1151 |
| Fine-tuned mAP | **0.2880** |
| **Adaptation gain** | **+0.1729** |

Ran **optical-only**, on evidence: verification item 8 measured every
accessible high-resolution SAR source as X-band against EOS-04's C-band. That
was the plan's documented fallback, chosen because the measurement said so and
not because the data was unavailable.

---

## Track B — `rs_vqa_v1` (QLoRA adapter)

| | |
|---|---|
| Checkpoint | `checkpoints/track_b_v1/adapter_final` |
| Base | Qwen2.5-VL-3B-Instruct, 4-bit |
| Training data | 4,806-example RS instruction mix, lr 1e-4, effective batch 8, 300 steps, seed 42 |

### Retrain, 2026-09-01 — v2

The v1 weights were destroyed (`docs/00` §3.6 **L32**) and no intact copy
existed anywhere: 22 candidates across two trees, git history, LFS, the
remote, the HF cache, Docker images and archives, **0 loadable**. Retraining
was authorised by the team lead and is recorded as Unfreeze 1 in
`docs/code-freeze.md`.

**The recipe was not redesigned.** Every field of `run_metadata.json` is
identical to v1's: `models/qwen25_vl_3b`, `data/instruct_mix`, 4,806 examples,
300 steps, lr 1e-4, effective batch 8, seed 42, LoRA r=16 / α=32 /
dropout=0.05 over the same seven target modules.

| | v1 (2026-08-29) | v2 (2026-09-01) | Δ |
|---|---|---|---|
| **`rsvqa_lr` exact match** | **0.6425120773** | **0.6473429952** | **+0.0048** |
| `whu_opt_sar` exact match | 0.2064516129 | 0.2000000000 | −0.0065 |
| overall exact match | 0.3810444874 | 0.3791102515 | −0.0019 |
| refusal recall | 0.4118 | 0.4117647059 | ±0 |
| false-refusal rate | 0.0077 | 0.0077369439 | ±0 |
| lexical-shortcut probe | 0.1667 | 0.1666666667 | ±0 |

Both rows are scored on the **identical** 534-example `val` split, with the
same 207 `rsvqa_lr` rows. Artifacts: `docs/assets/refusal/track_b_fullval.json`
(v1, untouched) and `docs/assets/refusal/track_b_v2_fullval.json` (v2).

**Read the deltas as reproduction, not improvement.** +0.0048 on 207 examples
is one extra correct answer; it is not evidence that v2 is better. The three
refusal metrics reproducing to four decimals is the more interesting result:
it says the recipe is reproducible under its seed, and it means **L3 stands
unchanged** - refusal recall is still 0.4118, still decomposing into 100% on
lexical refusals and 16.7% on the image-conditional ones. The retrain restored
the capability; it did not fix the known negative result, and was not expected
to.

| | |
|---|---|
| Checkpoint | `checkpoints/track_b_v2/adapter_final` |
| sha256 | `10f4830141237846a439f9166acc21eef0be050c5580381e2e66256cf7041174` |
| Integrity | 696 tensors, 0.34% NUL (the corrupted v1 files were 99.9922%) |
| Wall time | 6 h 26 m on an RTX 4050 Laptop (6 GiB), 4-bit NF4 |
| Final loss | 6.63 at step 300, flat from ~step 45 - the plateau L3 already records |

The eleven corrupted v1 adapters remain on disk as evidence and must not be
deleted.

| **`rsvqa_lr` exact match** | **v1: 0.6425** (n=207, 2026-08-29) · **v2: 0.6473** (n=207, 2026-09-01 retrain) |
| Full held-out val, exact match | 0.3810 (n=534) |
| Full held-out val, token F1 | 0.7913 |

**The comparison that matters** — v0 and v1 on an *identical* held-out split:

| | v0 | v1 |
|---|---|---|
| `rsvqa_lr` exact match | 0.4510 | **0.6425** |

**Weakness — refusal is a negative result.** Recall **0.4118** decomposes into
**5/5 (100%) on lexical refusals** and **2/12 (16.7%) on image-conditional
ones**. The model learned to refuse when the *question* is impossible on its
face and did not learn to refuse when the *image* is the reason — the harder
and more useful half. False-refusal rate 0.0077; lexical-shortcut probe
0.1667. Limitation L3.

**Not evaluated on VRSBench**, which the PS assigns to VQA alongside RSVQA:
VRSBench ships annotations only and its imagery lives in DOTA, not on disk.
Limitation L11.

---

## `caption_v1`

| | |
|---|---|
| Checkpoint | `checkpoints/caption/` |
| Training data | RSICD, 8,734 examples, 8 epochs, dim 192, vocab 1,781 |
| **BLEU-4 (sentence mean)** | **0.2446** (n=1,093) |
| Unique captions | **146 of 1,093 — 13.4%** |

**Weakness, and it is the diversity number, not the BLEU.** The model emits
146 distinct captions across 1,093 images. It has learned the corpus's common
sentences well enough to score, and it is not describing each scene
individually. Quote the 13.4% alongside the 0.2446 or the BLEU misleads.

**Not on the prescribed split:** the PS assigns captioning to VRSBench, which
is not evaluated. This number is on RSICD.

---

## `grounding_v1`

| | |
|---|---|
| Checkpoint | `checkpoints/grounding/` |
| Training data | DIOR-RSVG, 6,359 examples, 5 epochs, dim 128, **backbone from scratch** |
| **Acc@0.5** | **0.0762** (n=1,141) |
| Acc@0.7 | 0.0088 |
| mIoU | 0.1405 |

**This is the weakest model in the system and it should not be presented as
working.** Acc@0.5 of 0.0762 means roughly nine in ten referring expressions
are not localised. It satisfies the PS's M3 only because M3 requires
captioning **or** grounding and captioning is the stronger arm — but the PS's
own representative query *"Highlight the water body referred to in the query"*
routes here, so the routing is right and the answer is usually not.

`run_metadata` records `split_note: NO published split in this mirror`, so the
split is ours and the number is not comparable to published DIOR-RSVG results.

---

## `change_mask_v1`

| | |
|---|---|
| Checkpoint | `checkpoints/change_mask/` |
| Architecture | TinyCD-style siamese, **49,185 parameters** |
| Training data | LEVIR-CD, 7,120 tiles, 4 epochs, `pos_weight` 10.11 |
| **F1 (change class)** | **0.5597** |
| IoU | 0.3886 |
| Precision / Recall | 0.4426 / **0.7613** |

Scored on the **change class only**: LEVIR-CD is heavily imbalanced and
overall pixel accuracy would sit near 0.98 for a model that predicts "nothing
changed" everywhere.

The precision/recall split is a design choice worth stating: at 0.44/0.76 the
detector over-calls change. For a screening tool that surfaces candidates to
an analyst, recall is the cheaper error.

**Calibration:** ECE **0.0668 → 0.0034** after an *affine* fit. Temperature
scaling alone did not work on this head, and the calibration report records
which transform was accepted rather than assuming the usual one.

---

## `change_caption_v1`

| | |
|---|---|
| Checkpoint | `checkpoints/change_caption/` |
| Training data | LEVIR-CC, 6,815 examples, 6 epochs, dim 128, vocab 377 |
| **BLEU-4, changed pairs** | **0.3063** (n=964) |
| BLEU-4, unchanged pairs | 0.9706 (n=965) |
| BLEU-4, aggregate | 0.5686 |

**Quote 0.3063, never 0.5686.** The aggregate is inflated by the trivially
unchanged half, where the reference is a fixed "no difference" sentence the
model learns to emit verbatim. The checkpoint's own `metrics.json` carries
this warning in a `note` field. 85 unique captions.

---

## `change_vqa_v1` — semantic change head

| | |
|---|---|
| Checkpoint | `checkpoints/change_vqa/best.pt` |
| Architecture | Siamese **ImageNet-pretrained ResNet-18** encoder, two per-date decoders |
| Training data | SECOND via **CDVQA's own train ids** — 1,600 pairs, 400 val, **968 test ids never read** |
| Pixel accuracy | 0.7528 |
| mIoU | 0.3323 |
| **Change-class mIoU** | **0.2636** |

Pretraining ablation, same data and schedule:

| encoder | change-class mIoU |
|---|---|
| from scratch (`change_vqa_scratch`) | 0.1691 |
| **ImageNet ResNet-18** | **0.2636** (+56% relative) |

**End-to-end on the PS's prescribed benchmark**, full split, 39,686 questions
over 968 pairs at **99.82% coverage** *(corrected 2026-09-07 from 0.5380 /
"100% coverage" — see `docs/research/cdvqa-baseline-correction-2026-09-03.md`,
which established 0.6061 by A/B across two commits and found the documented
100% was never reproducible at either. This figure comes from the evaluation
artifact, not from `checkpoints/change_vqa/metrics.json`, which records only
the segmentation metrics and is unchanged)*:

| | accuracy |
|---|---|
| per-type majority baseline (fitted on train, applied to test) | 0.5084 |
| **system** | **0.6061** |
| oracle over ground-truth change maps | **0.9975** |

**The three numbers must be read together.** +9.8 points over a constant is a
real win — and materially larger than the +3.0 the superseded 0.5380 showed,
which is the main practical consequence of the correction. The 0.9975 oracle
says the answer layer contributes no measurable error and **nearly all of the
remaining headroom is this segmenter**. *(The original "93%" was derived from
0.5380 and has not been recomputed; the direction is unchanged.)*
Earlier iterations scored 0.0000 and then 0.4439 — *below* the baseline — and
both are recorded in `docs/phase1-status.md` rather than deleted.

**Weights are not publishable.** See the table at the top.

---

## `optsar_fusion_v1`

| | |
|---|---|
| Checkpoint | `checkpoints/optsar_fusion/` |
| Training data | WHU-OPT-SAR, 1,548 tiles, 5 epochs, dim 32 |
| Optical only | **0.7778** |
| SAR only | 0.7410 |
| Fused | 0.7714 |
| **Complementarity gain** | **−0.0064** |

**A negative result, reported as one.** Fusion does not beat optical alone on
this dataset. The PS's M6 asks the system to *extract complementary
information from a co-registered pair*, and it does — the triad runs, the
per-modality numbers are produced, and the complementarity score is reported
in the trace. What is **not** supported is the claim that fusing helps, and
that claim must not be made on stage.

`run_metadata` records `split_method: deterministic random by tile; NOT
geographic`, so tiles from one scene can span train and test and the absolute
numbers are optimistic. The *comparison* between the three arms is unaffected,
which is what the ablation is for.

---

## `index_engine_v1` — not a model

Deterministic: NDVI, NDWI, MNDWI, NDBI, σ⁰, VH/VV, GLCM texture, CoV, adaptive
Otsu/GMM thresholding, SWIR-free fallbacks. No learned parameters, no training
data, no failure mode beyond bad input. It is the independent referee the
neural outputs are checked against, and it is why quantitative answers in this
system come from arithmetic rather than generation.

---

# Phase 5 cards — 2026-09-12 (v2, trained on an L40S)

**These are additions.** Every card above is the v1 record and is unchanged;
it is the baseline each card below is measured against. The rules from
`docs/code-freeze.md` are honoured the same way: new checkpoint directories
(`checkpoints/v2/`), new architectures behind `--arch v2`, new dated numbers.
Every figure is read from a `metrics.json` in `docs/assets/phase5/`.

**Weight availability.** All Phase 5 checkpoints are on the AI Lab server at
`/scratch/home/<cluster-user>/satquery/checkpoints/v2/`. None have been copied
elsewhere yet and none are published; the licence position in §"Publication
status" is unchanged, and `change_vqa` weights remain unpublishable.

**Which checkpoint each tool should load** is stated per card. Where a
pretrained arm exists it is the recommendation, and the from-scratch v2 is
kept as the comparison that justifies it.

---

## `rs_vqa_v1` — v2 adapter

| | |
|---|---|
| Checkpoint | `checkpoints/v2/track_b_vqa/adapter_final` |
| Base | Qwen2.5-VL-3B-Instruct, 4-bit NF4, unchanged |
| Adapter | LoRA r=16 on the language tower, 37.15M trainable (0.98%) |
| Training | 6,000 steps on the 4,806-row instruction mix, 3.79 GPU-h |
| **RSVQA-LR official test, published convention** | **0.8947** — 95% CI [0.8873, 0.9017] |
| RSVQA-LR official test, all types | 0.6958 |
| Base model, same split | 0.3717 / 0.2622 |
| Train-fitted per-type constant | 0.7006 / 0.5695 |

**Why this is the card to lead with.** The v1 adapter was destroyed (§"Weight
availability — 2026-08-31") and could not be scored. This one was retrained
from the same recipe and is the first Track B number measured on the
**official** RSVQA-LR test split rather than the 2,000-question validation
slice, where a constant scored identically to the model. On the official
split the adapter beats the constant by 19 points and the base model by 52.

**Deployment note, corrected.** An early-stopped rerun
(`checkpoints/v2/track_b_vqa_es/adapter_best`, val loss 0.1301 vs 0.2529)
scores 0.6958 all-types — identical — and 0.8851 on the published
convention, marginally behind. Held-out loss on the instruction mix did not
predict benchmark accuracy, and the earlier note recommending `adapter_best`
on that basis was wrong. Either adapter is acceptable; `adapter_final` is
listed because it is marginally ahead on the metric that matters.

**Count questions score 0.22** and are excluded from the published
convention, as the literature does. Both figures are reported.

---

## `change_mask_v1` — v2

| | |
|---|---|
| Checkpoint | `checkpoints/v2/change_mask/` |
| Architecture | siamese residual encoder, multi-scale absolute difference, concat-skip decoder, **7.395M parameters** |
| Training | LEVIR-CD, 7,120 tiles, 60 epochs, `pos_weight` 10.1, 1.86 GPU-h |
| **F1 (change class)** | **0.8550** (v1 0.5597) |
| IoU | **0.7467** (v1 0.3886) |
| Precision / Recall | 0.8182 / 0.8952 (v1 0.4426 / 0.7613) |

Precision and recall both rose, so this is a better detector and not a moved
threshold. v1's over-calling (0.44 precision) is gone without giving up the
recall a screening tool needs. The two things v1 reasoned about correctly —
one shared encoder, absolute difference — are kept; what changed is that the
difference is taken at every scale and the decoder has skips, so small
buildings survive to the output.

**Recommendation:** load this checkpoint. Calibration has not been refitted
for v2 and the v1 affine transform should not be assumed to carry over.

---

## `grounding_v1` — v2, pretrained

| | |
|---|---|
| Checkpoint | `checkpoints/v2/grounding_pre/` |
| Architecture | ImageNet ResNet-50 backbone → 1×1 projection → phrase cross-attends over the feature map → box read from the attention; **32.4M parameters** |
| Training | DIOR-RSVG, all ~38k referring expressions, 40 epochs |
| **Acc@0.5** | **0.1604** (v1 0.0762; v2 from-scratch 0.1262) |
| Acc@0.7 | 0.0543 (v1 0.0088) |
| mIoU | 0.1974 (v1 0.1405) |

The v1 card names the defect — global-average-pooling before regressing the
box — and removing it accounts for 0.0762 → 0.1262. The pretrained backbone
accounts for the rest. Acc@0.5 has doubled and Acc@0.7 has risen six-fold.

**It is still far below the published 0.70–0.80**, and the reason is now
narrowed: the from-scratch run's training loss reached 0.0054, so the model
memorised the training set, and more epochs would not help. What remains is
the gap between an ImageNet backbone and the pretrained detection backbones
and BERT-class text encoders the literature uses.

**Recommendation:** load `grounding_pre`. Keep the confidence-gated
abstention the v1 card describes; at 16% Acc@0.5 most boxes are still wrong.

---

## `caption_v1` — v2, pretrained

| | |
|---|---|
| Checkpoint | `checkpoints/v2/caption_pre/` |
| Architecture | ImageNet ResNet-50 → 1×1 projection → transformer decoder with cross-attention; **43.3M parameters** |
| Training | RSICD, 8,734 captions, 40 epochs |
| **BLEU-4** | **0.2658** (v1 0.2446; v2 from-scratch 0.2255) |
| Unique captions | 764 / 1,093 (69.9%) — v1 146 (13.4%) |

The from-scratch v2 **regressed** on BLEU-4 while producing five times as
many distinct captions. That pattern is consistent with v1 collapsing onto a
few safe captions BLEU rewards, but it is not claimed as a hidden win — the
from-scratch samples inspected were diverse and wrong. The pretrained
backbone recovers BLEU-4 past v1 while keeping the diversity.

**Recommendation:** load `caption_pre`. The from-scratch checkpoint is kept
as the comparison and should not be deployed.

---

## Track A — v2 → `landcover_v1`

| | |
|---|---|
| Checkpoint | `checkpoints/v2/track_a/` |
| Architecture | band-agnostic residual encoder, per-band stem, masked mean, FiLM GSD conditioning at every stage, **12.04M parameters** |
| Training | 65,867 prepared BigEarthNet patches (11% of v2), 40 epochs, band dropout 0.3, multi-resolution, bf16 |
| **mAP, 12 bands** | **0.3150** (v1 0.2854) |
| mAP, Cartosat 4-band | 0.2917 |
| **Retention** | **0.9262** (v1 0.9015) |

| ablation arm | epochs | mAP 12-band | retention |
|---|---|---|---|
| dropout 0.3 | 12 → 40 | 0.3127 → 0.3150 | — → 0.9262 |
| dropout 0.0 | 12 → 40 | 0.3213 → 0.3015 | 0.8392 → 0.8775 |

**The ablation is now controlled** — both arms, identical 40-epoch schedule —
and it says more than the single-seed v1 version did. Band dropout costs
nothing on 12 bands and buys ~5 points of 4-band retention, as before; but it
also **prevents the overfit** the no-dropout arm shows between epochs 12 and
40. It is a regulariser as well as a robustness mechanism.

**mAP is data-limited, not schedule-limited.** 28 extra epochs moved it by
0.0023 while training loss fell tenfold. The published 0.65–0.85 is measured
on ~549k patches; this is measured on 65,867, and no schedule will close that.

**Recommendation:** load `track_a`. The v1 calibration and the 0.70 assertion
threshold were fitted to v1 scores and must be refitted before the selective
prediction in `satquery/tools/landcover.py` is trusted on v2.

---

## `change_vqa_v1` — v2 (pretrained stem, as v1)

| | |
|---|---|
| Checkpoint | `checkpoints/v2/change_vqa/` |
| Architecture | as v1 — ImageNet ResNet-18 stem, two unshared per-date decoders |
| Training | SECOND, 60 epochs |
| **Change-class mIoU** | **0.2933** (v1 0.2636) |
| mIoU, all classes | 0.3623 |
| Ablation, from-scratch v2 | 0.1730 |

The ablation arm answers the question it was run for: on ~1,600 pairs a
from-scratch encoder loses **0.12 mIoU** to a pretrained stem, same data and
schedule. That measurement is what justified the pretrained arms above.

**Publication:** still **BLOCKED** — SECOND states no licence. Retraining
does not change that.

---

## `change_caption_v1` — v2 (NOT deployed)

| | |
|---|---|
| Checkpoint | `checkpoints/v2/change_caption/` — **kept as the comparison; the tool loads v1** |
| Architecture | siamese residual encoder → difference + mask → transformer decoder; 25.39M parameters |
| Training | LEVIR-MCI, 50 epochs |
| **BLEU-4, changed pairs** | **0.1641** (v1 0.3063) — **regression** |
| BLEU-4, unchanged pairs | 0.9846 (v1 0.9706) |
| BLEU-4, aggregate | 0.5746 (v1 0.5686) |
| Unique captions | 638 (v1 85) |

**Quote 0.1641, never 0.5746** — the same rule the v1 card states, and the
reason this card exists in this form. The aggregate rose while the changed
half fell by 0.14, because v2 is slightly better at the fixed "there is no
difference" sentence that half the test set expects. The evaluator had
dropped the split; the first Phase 5 write-up read the aggregate as "not
worse" and was wrong. Reinstated and re-scored 2026-09-12.

**Recommendation:** load `checkpoints/change_caption` (v1). This is the one
tool where Phase 5 produced a worse model, and the one where the
architecture changed without a pretrained backbone; a pretrained arm is the
obvious next attempt and has not been run.

---

## `optsar_fusion_v1` — v2

| | |
|---|---|
| Checkpoint | `checkpoints/v2/optsar_fusion/` |
| Architecture | separate optical (4-band) and SAR (1-band) residual encoders, bidirectional cross-attention **before** pooling, three heads; 6.26M parameters |
| Training | WHU-OPT-SAR, 40 epochs, 0.09 GPU-h |
| optical-only mAP | 0.7722 (v1 0.7778) |
| SAR-only mAP | 0.7369 (v1 0.7410) |
| fused mAP | 0.7420 (v1 0.7714) |
| **complementarity gain** | **−0.0301** (v1 −0.0064) |

**The fused head is worse than optical alone, for the second architecture in
a row.** The three-head design exists so this cannot be hidden: the
per-stream heads read pre-attention features, so the comparison is honest.
Cross-attention before pooling — the textbook remedy for v1's concatenation —
did not help and slightly hurt.

**Recommendation:** do not deploy the `fused` head on the strength of this
number. The tool remains available and its per-stream heads are sound; the
claim that the fusion *adds* information is not supported on WHU-OPT-SAR and
the report should say so. This is a PS-mandatory capability and the finding
is about the corpus, not a bug.

---

## `index_engine_v1` — unchanged

No training run, by design. Deterministic NumPy. The fallback every other
tool degrades to, and the reason `change_vqa` cannot score zero.
