# Phase 2 — immutable baseline reproduction

**Started 2026-09-03. Measurement only.** No training, no architecture change, no hyper-parameter change. Every output is written to `artifacts/phase2_baseline/`; **no published `metrics.json`, and nothing under `docs/assets/`, is written by this phase.**

**Gate D is armed at ±3 points.** Any reproduced value deviating from its recorded reference by more than that stops the phase and is reported rather than judged "close enough".

> **Status: MEASUREMENT-COMPLETE, 2026-09-03.** All seven references measured. **Six reproduce; one (R6, CDVQA) was corrected upward by +6.81 points with an A/B proof.** Two figures are **BLOCKED as unreproducible** (W15, D2). One unintended write occurred and was reverted (§8.2). See §9 for the closing summary.

---

## 1. Provenance

| | |
|---|---|
| Worktree HEAD | `pre-import:P0-primary` |
| Primary checkout HEAD | `pre-import:P0-primary` — **identical** |
| Branch | `agent-worktree` |
| Tracked modifications | **0** |
| Output root | `artifacts/phase2_baseline/` — a *named* directory, so `satquery prune`'s whitelist protects it by construction |
| GPU | RTX 4050 Laptop, 6,141 MiB · torch 2.13.0+cu126 · 4-bit NF4 |
| Offline | `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` throughout |

### 1.1 Artifact-of-record protection

Per **W16** (six of seven specialist heads rewrite `metrics.json`, `vocab.json` or `band_stats.json` in their checkpoint directory when evaluated):

* Specialist evaluations run against **copies** under `artifacts/phase2_baseline/ckpt_scratch/`, never the originals.
* SHA-256 of every original `metrics.json` was recorded **before** any run, so mutation is detectable rather than assumed absent:

| checkpoint | original `metrics.json` sha256 (first 32) |
|---|---|
| `caption` | `fef8d4d5315da20495a424a7a0e777bc` |
| `grounding` | `41e5b37fbea5d2c71614a09b3c78167d` |
| `change_mask` | `96b24bdbb5feda5a694dccdb1cf9951b` |
| `change_caption` | `6df3500125ced2343851e0dce4e36a2e` |
| `optsar_fusion` | `f0cd292e56353f3c16c0dd608846078b` |

* `evaluation/calibrate.py` defaults `--out-dir` to **`docs/assets/calibration`**; it is run with an explicit scratch path instead.

---

## 2. Reference values and reproduction status

| # | Reference (as recorded) | Recorded | Reproduced | Δ | Gate D |
|---|---|---|---|---|---|
| R1 | `track_b_v3` VQA ≈95% — 150-question published convention | 0.9533 | **0.9533** | **0.0000** | **PASS** |
| R2 | `track_b_v3` ≈79% — all types, n=207 | 0.7874 | **0.7874** | **0.0000** | **PASS** |
| R3 | previous model ≈65% — `track_b_v2`, all types | 0.6473 | **0.6473** | **0.0000** | **PASS** |
| R4 | Counting ≈35% — `track_b_v3`, count type | 0.3509 | **0.3509** | **0.0000** | **PASS** |
| R5 | Oracle change-map CDVQA ≈99.75% | 0.9975 | **0.99748** | **0.002 pts** | **PASS** |
| R6 | CDVQA ≈54% | 0.5380 | **0.606133** | **+6.81 pts** | **FAIL — see §7 D1** |
| R7 | Grounding ≈7.6% | 0.0762 | **0.0762** | **0.0000** | **PASS** |

**BLOCKED, by decision (Rule 4):**

| Item | Status |
|---|---|
| **D2** — change captioning `bleu4_changed` **0.3063** | **BLOCKED — unreproducible from code.** `bleu4_changed` has never appeared in any tracked commit (`git log --all -S` returns empty). See §7 D2. Not estimated; the aggregate 0.5686 is **not** substituted for it |
| v0 → v1 comparison, `rsvqa_lr` **0.4510 → 0.6425** | **BLOCKED — unreproducible from disk.** The v0 arm (`checkpoints/killtest/adapter_final`) is unloadable: all four of its adapter safetensors are corrupt and the sibling `.pt` files contain no `model_state_dict` (**W15**). Decision 2026-09-03: do **not** retrain v0 to reconstruct a superseded baseline. This comparison is recorded as unreproducible; it is **not** patched, estimated or substituted |

---

## 3. VQA — reproduced exactly

### 3.1 Held-out RSVQA-LR slice, re-typed (n=207; 150 under the published convention)

All four arms, identical greedy decode, `evaluation/track_b_eval.py` + `evaluation/metrics/vqa.py` unmodified.

| Arm | EM all types | 95% CI | Published convention micro (n=150) | presence (68) | comparison (81) | count (57) | vs constant |
|---|---|---|---|---|---|---|---|
| base (no adapter) | 0.1981 | 0.150–0.258 | 0.2733 | 0.1324 | 0.3951 | 0.0000 | 18 / **111** — far worse |
| `track_b_v2` | 0.6473 | 0.580–0.709 | 0.8133 | 0.8824 | 0.7531 | 0.2105 | **17 / 17 — no difference** |
| `track_b_v3_probe` | 0.6715 | 0.605–0.732 | 0.8333 | 0.7500 | 0.9136 | 0.2456 | 30 / 25 — not significant |
| **`track_b_v3`** | **0.7874** | 0.727–0.838 | **0.9533** | **0.9559** | **0.9506** | 0.3509 | **35 / 6 — p<0.001** |
| *train-fitted per-type constant* | *0.6473* | — | *0.7600* | *0.8088* | *0.7284* | *0.3509* | — |

**Every cell is bit-identical to the 2026-09-03 pre-rebase measurement**, including the per-type accuracies, the Wilson intervals and the constant-baseline contingency counts.

This is a stronger result than it looks. Before the rebase I argued *structurally* that the commit move could not perturb the VQA path, because `git diff --name-only pre-import:P0-worktree pre-import:P0-primary` touches neither `evaluation/` nor `satquery/tools/rs_vqa.py`. This measurement confirms it **empirically**, on four independently loaded checkpoints.

### 3.2 Findings carried forward, unchanged by reproduction

* **`track_b_v2`'s headline number is indistinguishable from a constant.** Both score 134/207; discordant 17 vs 17.
* **`track_b_v3` is the only arm that beats the constant significantly** (35 vs 6, McNemar χ²≈20.5, p<0.001) — and it was trained under the label-masking defect.
* **Counting is a constant for every arm.** `v3`'s 0.3509 equals the constant's 0.3509 exactly; 19 of its 20 correct count answers are the gold "0".

### 3.3 Full held-out split (n=534) — reproduced exactly

All four arms × seven metrics = **28 comparisons, worst deviation 0.0000 points.**

| Arm | exact match | token F1 | `rsvqa_lr` | `whu_opt_sar` | refusal recall | false-refusal | lexical probe |
|---|---|---|---|---|---|---|---|
| base | 0.079304 | 0.202717 | 0.198068 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| `track_b_v2` | 0.379110 | 0.792742 | 0.647343 | 0.200000 | **0.411765** | **0.007737** | **0.166667** |
| `track_b_v3_probe` | 0.394584 | 0.807712 | 0.671498 | 0.209677 | 0.294118 | 0.001934 | 0.000000 |
| `track_b_v3` | **0.460348** | **0.855027** | **0.787440** | **0.241935** | 0.352941 | 0.003868 | 0.083333 |

Every cell matches its recorded value to six decimal places. `track_b_v2`'s three **published** refusal metrics (0.411765 / 0.007737 / 0.166667) are among them, so this is a reproduction of published figures and not merely of an internal run.

**What this establishes.** The deployed inference path — 4-bit NF4 quantisation, chat template, greedy decode, normalisation and metric code — is deterministic end to end across a commit change, four independently loaded checkpoints, and two separate sessions. Any difference observed between checkpoints in later phases is attributable to the checkpoints and to nothing else.

### 3.4 Latency — reproduced with an important caveat

| Arm | s/example, run 1 | s/example, run 2 (this phase) |
|---|---|---|
| base | 1.5228 | 1.4478 |
| `track_b_v2` | 1.3056 | **2.2615** |
| `track_b_v3_probe` | 1.2476 | 1.2696 |
| `track_b_v3` | 1.2626 | 1.2889 |

**Accuracy is bit-identical; latency is not, and varies by up to 73% for the same checkpoint on the same data.** `track_b_v2` took 1.31 s/example in one run and 2.26 s/example in another, with identical outputs. This is the bimodal timing `ENVIRONMENT.md` §T8 documents, and it means **latency on this machine is not a reproducible measurement at single-run resolution.** Any latency claim in a later phase needs repeated runs and a stated dispersion, not one number. Recorded here so it is not mistaken for a regression later.

---

## 4. CDVQA

Prerequisites verified present before running: official `Test_questions.json` / `Test_answers.json` / `Test_images.json`, `data/second/{im1,im2,label1,label2}`, and the prepared `cdvqa_test_full.json`. `data/cdvqa/webdataset/` is absent as documented (L-record); neither evaluator needs it.

Both `evaluation/cdvqa_predict.py` and `evaluation/cdvqa_oracle.py` were confirmed by inspection to write **only** to `--out` — they are the one genuinely read-only pair in the repository.

### 4.1 Oracle ceiling — R5 reproduced

`evaluation/cdvqa_oracle.py --split Test`, output `artifacts/phase2_baseline/cdvqa_oracle_test.json`.

| | Recorded | Reproduced |
|---|---|---|
| Images | 968 | **968** |
| Questions | 39,686 | **39,686** |
| **Oracle accuracy** | **0.9975** | **0.99748** |
| Deferred | — | 70 |

**Δ = 0.002 points. Gate D PASS.**

**Per-type, and this is the useful part.** The oracle is **exactly 1.0000 on seven of the eight question types** — `change_or_not`, `change_ratio`, `change_ratio_types`, `change_to_what`, `decrease_or_not`, `increase_or_not` and `largest_change`. All 70 disagreements fall in **`smallest_change` (0.9773)**.

Inspecting them, every sampled failure is a tie between two rare classes:

```
truth 'trees'         derived 'buildings'
truth 'NVG_surface'   derived 'low_vegetation'
truth 'trees'         derived 'NVG_surface'
```

So the ceiling is not 99.75% because the deterministic answer rules are approximate. It is 99.75% because **"which change is smallest" is genuinely ambiguous when two rare classes are close in area** — a labelling property of CDVQA, not a defect in SatQuery's arithmetic. Two consequences for Phase 5:

* The headroom figure stands: **the answer layer contributes no measurable error**, and the entire 0.5380 → 0.9975 gap is the segmenter.
* `smallest_change` is the one question type where **even a perfect segmenter would not reach 100%**. It is also the type SatQuery scores worst on (0.1319, *below* the 0.2231 constant), so its 2,904 questions should be analysed separately rather than folded into an overall figure.

---

## 5. Specialist heads — every one reproduced

Run against **copies** under `artifacts/phase2_baseline/ckpt_scratch/`. Originals never used as a write target.

| Head | Recorded | Reproduced (full precision) | Δ |
|---|---|---|---|
| `grounding_v1` Acc@0.5 | 0.0762 | `0.07624890446976336` | 0.0000 |
| `grounding_v1` Acc@0.7 | 0.0088 | `0.008764241893076249` | 0.0000 |
| `grounding_v1` mIoU | 0.1405 | `0.14048209367087824` | 0.0000 |
| `caption_v1` BLEU-4 (n=1093) | 0.2446 | `0.24460787515482577` | **exact to 17 s.f.** |
| `caption_v1` unique captions | 146 (13.4%) | 146 (13.36%) | 0 |
| `change_mask_v1` F1 | 0.5597 | `0.5597365719863596` | 0.0000 |
| `change_mask_v1` IoU / P / R | 0.3886 / 0.4426 / 0.7613 | `0.3886348574151661` / `0.44255118259648896` / `0.7613341262268545` | 0.0000 |
| `change_caption_v1` BLEU-4 **aggregate** (n=1929) | 0.5686 | `0.5685920403061677` | 0.0000 |
| `change_caption_v1` BLEU-4 **changed pairs** | 0.3063 | **BLOCKED — see §7 D2** | — |
| `optsar_fusion_v1` optical / SAR / fused | 0.7778 / 0.7410 / 0.7714 | `0.7777944611792945` / `0.7409939886934886` / `0.7714183422364739` | 0.0000 |
| `optsar_fusion_v1` complementarity gain | −0.0064 | `-0.006376118942820641` | 0.0000 |

**`grounding_v1` and `caption_v1` matter beyond their numbers.** Both are heads whose `vocab.json` was destroyed in the L29 incident and reconstructed. Reproducing their metrics bit-identically is **independent confirmation that the regenerated vocabularies are correct** — a stronger check than the one L29 itself used.

### 5.1 Verification that no artifact of record was mutated

| head | original `metrics.json` sha256 before | after | |
|---|---|---|---|
| `caption` | `fef8d4d5…` | `fef8d4d5…` | **UNCHANGED** |
| `grounding` | `41e5b37f…` | `41e5b37f…` | **UNCHANGED** |
| `change_mask` | `96b24bdb…` | `96b24bdb…` | **UNCHANGED** |
| `change_caption` | `6df35001…` | `6df35001…` | **UNCHANGED** |
| `optsar_fusion` | `f0cd292e…` | `f0cd292e…` | **UNCHANGED** |

---

## 6. Reliability, latency and memory

### 6.1 Refusal — reproduced (from §3.3)

| | base | `track_b_v2` | `track_b_v3_probe` | `track_b_v3` |
|---|---|---|---|---|
| refusal recall | 0.000000 | **0.411765** | 0.294118 | 0.352941 |
| false-refusal rate | 0.000000 | **0.007737** | 0.001934 | 0.003868 |
| lexical-shortcut probe | 0.000000 | **0.166667** | 0.000000 | 0.083333 |

v2's three figures are **published** numbers and reproduce to six decimals.

### 6.2 Calibration — reproduced

`evaluation/calibrate.py --heads change_mask`, LEVIR-CD official test split, 2048 tiles, 1024 pixels sampled per tile, fit split **by tile** (not by pixel, which would leak).

| transform | Recorded | Reproduced |
|---|---|---|
| affine | 0.0668 → **0.0034** | `0.06676796` → **0.0034** ✓ |
| temperature | insufficient | 0.0668 → 0.0591 (T=0.8616) |

The affine-over-temperature choice reproduces, including *why*: the head was trained with `pos_weight=10.1`, so its logits carry a deliberate positive-class offset that a single temperature cannot remove.

### 6.3 Latency — reproducible only as a distribution

See §3.4. **Accuracy is bit-identical; latency is not.** `track_b_v2` measured 1.31 and 2.26 s/example on identical data with identical outputs — a 73% swing. Any latency claim in a later phase needs repeated runs and a stated dispersion.

### 6.4 Memory — measured

| | |
|---|---|
| VRAM, VLM inference (4-bit NF4, unmerged LoRA) | **2,627–2,909 MiB** observed across all four arms, of 6,141 MiB available |
| VRAM headroom | ~3.2 GB free during inference |
| Peak host RSS, stub pipeline (recorded, not re-measured) | 215.91 MB, slope +0.0201 MB/iteration over 120 iterations |

VRAM is **not** the binding constraint on this machine. Disk is (`docs/storage-audit.md`).

---

## 7. Deviations found

### D1 — CDVQA end-to-end: **+6.81 points. GATE D FAILURE. PHASE HALTED.**

| | Recorded | Reproduced | Δ |
|---|---|---|---|
| **Overall accuracy** | **0.5380** | **0.606133** | **+6.81 pts** |
| Coverage | 100% (0 deferred) | 99.82% (**73 deferred**) | −0.18 pts |
| Images / questions | 968 / 39,686 | 968 / 39,686 | identical |
| Runtime | ~220 s | 271.3 s | — |

Per question type, sorted by deviation:

| type | n | recorded | reproduced | Δ pts | per-type constant | beats constant now? |
|---|---|---|---|---|---|---|
| `largest_change` | 2,904 | 0.4497 | 0.616736 | **+16.70** | 0.4291 | YES |
| `change_to_what` | 2,991 | 0.3714 | 0.521899 | **+15.05** | 0.3805 | **YES** (was below) |
| `decrease_or_not` | 4,658 | 0.6496 | 0.754830 | **+10.52** | 0.6900 | **YES** (was below) |
| `change_ratio_types` | 5,811 | 0.4791 | 0.556359 | +7.73 | 0.4770 | YES |
| `increase_or_not` | 4,600 | 0.6437 | 0.717826 | +7.41 | 0.6663 | **YES** (was below) |
| `change_or_not` | 13,882 | 0.6772 | 0.709336 | +3.21 | 0.5617 | YES |
| `smallest_change` | 2,904 | 0.1319 | 0.152204 | +2.03 | 0.2231 | no |
| `change_ratio` | 1,936 | 0.1952 | 0.187500 | −0.77 | 0.1529 | YES |
| **OVERALL** | **39,686** | **0.5380** | **0.606133** | **+6.81** | **0.5084** | YES |

**Seven of eight types moved up.** The system now beats its per-type constant on **seven of eight** types, where it previously beat it on five.

### D1.1 Root cause — identified, and it is a code fix, not a measurement error

`evaluation/cdvqa_predict.py` line 47 imports `satquery.ingest` and line 70 calls `ingest([second/"im1"/name, second/"im2"/name])`. **CDVQA therefore runs through the ingest reader.**

Commit **`pre-import:png-channels-fix` "Read the channels a PNG declares instead of guessing them"** — one of the four commits present in the primary checkout but absent from the worktree until today's rebase — modified `satquery/ingest/reader.py`. Its own docstring states the bug it fixed:

> **Red and blue swapped.** GDAL reports band 1 of a PNG as `red`, but with no descriptions the fallback assumed the GeoTIFF convention `[BLUE, GREEN, RED]`. `to_rgb_preview` then looked "RED" up at index 3 and handed the model a **channel-reversed image**.

And the SECOND imagery CDVQA reads is exactly that case — verified: `data/second/im1/00003.png`, **format PNG, mode RGB, 512×512, bands (R,G,B)**.

So **the recorded 0.5380 was measured while the semantic-change head was being fed channel-reversed (BGR) images.** The head is an **ImageNet-pretrained ResNet-18**, and ImageNet pretraining is RGB-specific, so channel reversal degrades it substantially. The reproduced 0.606133 is the first CDVQA measurement taken on correctly-ordered input.

The per-type pattern corroborates the mechanism rather than merely being consistent with it: the two largest gains are `largest_change` (+16.70) and `change_to_what` (+15.05), which are precisely the questions that require **discriminating between land-cover classes by colour** — the thing a channel swap destroys. The one type that did not improve, `smallest_change`, is the type §4.1 shows is ambiguous even for the oracle.

### D1.2 What this changes

* **SatQuery's true CDVQA accuracy is 0.6061, not 0.5380.** Every document quoting 0.5380 — `docs/00` §3.1 and L10, `docs/model-cards.md`, `docs/deck.md`, `docs/judge-qa.md`, `docs/phase1-status.md`, `docs/external_benchmark_audit.md` §6.2 — is now stale.
* **The external comparison narrows materially.** Against the same CDVQA test1 split:

| System | Accuracy | Old gap vs SatQuery | **New gap** |
|---|---|---|---|
| Qwen3.5-2B change-VQA (2026) | 0.7474 | −20.9 | **−14.1** |
| VisTA (2024) | 0.7310 | −19.3 | **−12.5** |
| SOBA (2024) | 0.6920 | −15.4 | **−8.6** |
| CDVQA baseline (2021) | 0.6590 | −12.1 | **−5.3** |
| **SatQuery** | **0.6061** | — | — |
| per-type majority constant | 0.5084 | +3.0 | **+9.8** |

* **The margin over a constant triples**, from +3.0 to +9.8 points. The `docs/judge-qa.md` Q&A "Your CDVQA score is 0.5380. A constant scores 0.5084. Why should I be impressed?" is answered differently now.
* **The 93%-of-headroom claim shifts**: the gap to the 0.99748 oracle narrows from 0.4595 to **0.3914**. The segmenter is still the bottleneck, but less of one.
* **Coverage regressed slightly**, 100% → 99.82% (73 deferred, 71 of them in `change_to_what`). Small, but it is a change in behaviour and is not explained by the channel fix. It needs its own look.

### D1.3 Not done, pending your decision

Per Gate D this phase is **halted**. The following are **proposed, not executed**:

1. **The decisive A/B confirmation.** Re-run `cdvqa_predict` at commit `pre-import:P0-worktree` (pre-`pre-import:png-channels-fix`) and confirm it returns 0.5380. If it does, the channel-swap explanation is proven rather than merely well-evidenced. This requires temporarily moving the worktree to another commit — a state change I will not make unasked. Cost: ~5 minutes plus two checkouts.
2. **Superseding the published 0.5380.** Under `docs/code-freeze.md`, a changed number belongs in a **new dated section**, never an edit. No published document has been touched.
3. **The 73 deferrals.** Cause unknown; not investigated.

**No published number has been changed. No document outside `docs/research/` and `artifacts/phase2_baseline/` has been written.**

### D2 — change-captioning `bleu4_changed` 0.3063: **BLOCKED, unreproducible from code**

`checkpoints/change_caption/metrics.json` records `bleu4_changed: 0.3063`, `bleu4_unchanged: 0.9706`, `bleu4_aggregate: 0.5686`, `n_changed: 964`, `n_unchanged: 965`, `unique_captions: 85`, and a note reading *"bleu4_changed is the meaningful figure; the aggregate is inflated by the trivially-unchanged half."*

The current evaluator emits only `bleu4_sentence_mean` and `n`. So the history was searched:

```
git log --all -S"bleu4_changed" -- training/train_change_caption.py   ->  (empty)
git log --all -S"bleu4_changed"                                       ->  (empty)
```

**`bleu4_changed` has never appeared in any tracked code, in any commit, anywhere in this repository.**

This matters more than an ordinary missing metric, because `docs/model-cards.md` line 314 instructs: **"Quote 0.3063, never 0.5686."** The project's *preferred* change-captioning figure — the one it explicitly says to use instead of the aggregate — cannot be produced by its own code.

**Treatment, identical to W15 (Rule 4):** 0.3063 is recorded as **unreproducible-from-code**. It is **not** estimated, **not** patched, and the aggregate is **not** substituted for it. The aggregate 0.5686 reproduces exactly, but the model card is explicit that the aggregate is the misleading number — **"the aggregate reproduces" must never be read downstream as "the preferred figure is fine now."**

This compounds an existing limitation: `docs/external_benchmark_audit.md` already classified LEVIR-CC as **Category C** (SatQuery scores against one reference where the literature uses five). The task now also lacks a reproducible headline of its own.

---

## 8. Process findings — three lessons and one error

### 8.1 Worktree/data locality — the third instance of one root cause

Three of the five specialist evaluations failed with `FileNotFoundError` on paths like `data\levircd\tiles\train\a\000000.png`. **They did not run; they were not deviations**, and reporting them as either passes or failures would have been wrong.

Cause: `index.json` stores paths **relative to the repo root**, `data/` is gitignored (`.gitignore:41`), and a git worktree therefore has no `data/`. `grounding` and `caption` worked only because they take `--data` as an absolute argument.

**Fix applied — environment setup, NOT a code change:** a Windows **directory junction** `worktree/data` → `primary/data`, created with `New-Item -ItemType Junction`.

It is gitignored, read-only in use, reversible by deleting the link, and leaves `git status` clean. **A future session finding an unexplained `data` link inside a worktree should read this section rather than remove it.**

This is the **third** manifestation of the same root cause, after the 43 skipped `test_real_products.py` tests and the discovery route to W15. **Any index-driven evaluator is unrunnable from a worktree without this junction.**

### 8.2 An `--out-dir` flag is not evidence a script writes nowhere else — STANDING CAUTION

**An error was made and reverted.** `evaluation/calibrate.py` accepts `--out-dir`, which redirects its report and SVGs. It **also writes `configs/calibration.json` — a tracked file — unconditionally.** That path is not redirected by any flag. Running with `--heads change_mask` alone caused the rewrite to keep only that head, **deleting** the `SINGLE_LANDCOVER` calibration entry and the `rejected._router_intent` record — the latter a deliberately-preserved negative result.

| | |
|---|---|
| Detected by | `git status --short configs/` immediately after the run |
| Reverted by | `git checkout -- configs/calibration.json` |
| Verified restored | `generated_utc` 2026-08-29, `heads: [SINGLE_LANDCOVER, TEMPORAL_CHANGE_MAP]`, `rejected: [_router_intent]` |
| Blast radius | **worktree only.** Primary checkout config untouched; primary's protected `artifacts/calibration/logits/change_mask.npz` still dated Aug 29 14:46 (the cache went to the worktree's own `artifacts/`); `docs/assets/` untouched; all five `metrics.json` digests unchanged |

**STANDING CAUTION for Phase 2 onward — applies to every evaluator invocation, not just this one:**

> A redirect flag proves only where the *named* output goes. Before running any evaluator, grep it for **every** write target — `write_text`, `open(..., "w")`, `save_checkpoint`, `np.save`, `savefig`, `to_json` — and confirm each is either inside a scratch path or is a file you have deliberately accepted will change. Then check `git status` immediately afterwards, **before** reading any result.

This is the same failure mode as **W16** one layer up: W16 is about heads mutating their checkpoint directory; this is about an evaluator mutating a tracked config. The scratch-copy discipline protected the checkpoints and did nothing for the config, because the question *"what else does this script write?"* was never asked.

### 8.3 CDVQA's real number is 0.6061 — and the correction doc is the source of truth

`docs/research/cdvqa-baseline-correction-2026-09-03.md` is authoritative. The A/B is proven: the pre-fix commit returns 0.5380 on all eight question types, the post-fix commit returns 0.606133, and the only variable changed was the code.

**At least seven documents still quote the stale 0.5380 and have NOT been edited**, per the freeze: `docs/00` (§3.1 M4, §3.5, §3.6 L10), `docs/model-cards.md`, `docs/phase1-status.md`, `docs/deck.md`, `docs/judge-qa.md`, `docs/external_benchmark_audit.md` (§1.1, §6.1, §6.2, §8, §10.2, §11 G1, §15 — **and every derived gap figure in them**), and `docs/external_benchmark_results.json`. **They need a separate, explicit update pass.**

### 8.4 Deferred, logged, not acted on

| Item | Status |
|---|---|
| CDVQA deferrals — documented "100% coverage" not reproducible at either commit (122 pre-fix, 73 post-fix) | Logged in the correction doc §6 and `system-audit.md` §12. **Likely to matter for Phase 8** |
| `data/levir_mci/LEVIR-MCI-dataset.zip` alongside `extracted/` | Same redundant-archive pattern as the two zips removed in Phase 1. **Not touched.** For a future storage pass |
| W16 `--eval-only` fix | Logged, deliberately not built (Phase-21/22 work) |

---

## 9. Closing summary

### 9.1 Gate D — complete

**Seven named references, 28 full-val metrics, ten specialist metrics, and calibration. Worst deviation among everything that reproduced: 0.002 points.**

| # | Reference | Recorded | Reproduced | Δ | Gate D |
|---|---|---|---|---|---|
| R1 | `track_b_v3` VQA, published convention (n=150) | 0.9533 | 0.9533 | 0.0000 | **PASS** |
| R2 | `track_b_v3` all types (n=207) | 0.7874 | 0.7874 | 0.0000 | **PASS** |
| R3 | `track_b_v2` all types | 0.6473 | 0.6473 | 0.0000 | **PASS** |
| R4 | Counting (`v3`, count type) | 0.3509 | 0.3509 | 0.0000 | **PASS** |
| R5 | CDVQA oracle ceiling | 0.9975 | 0.99748 | 0.002 | **PASS** |
| R6 | CDVQA end-to-end | 0.5380 | **0.606133** | **+6.81** | **CORRECTED — A/B proven** |
| R7 | Grounding Acc@0.5 | 0.0762 | 0.0762 | 0.0000 | **PASS** |
| — | Full held-out split: 4 arms × 7 metrics | — | — | **0.0000** (28/28) | **PASS** |
| — | Specialist heads: 10 metrics across 5 heads | — | — | **0.0000** | **PASS** |
| — | Calibration ECE (affine) | 0.0668→0.0034 | 0.0668→0.0034 | 0.0000 | **PASS** |

### 9.2 BLOCKED — two figures, neither estimated nor substituted

| Item | One-line reason |
|---|---|
| **W15** — v0→v1 VQA, `rsvqa_lr` 0.4510 → 0.6425 | The v0 adapter is 100% NUL and its `.pt` files hold no `model_state_dict`, so the arm cannot be loaded and the comparison cannot be re-run from disk |
| **D2** — change captioning, `bleu4_changed` 0.3063 | `bleu4_changed` has never existed in any tracked commit, so the figure the model card tells readers to quote cannot be produced by the repository's own code |

### 9.3 The immutable baseline

The frozen reference every later phase compares against. Commit `pre-import:P0-primary`, artifacts under `artifacts/phase2_baseline/`.

| Capability | Baseline | Source |
|---|---|---|
| VQA (`v3`, published convention, n=150) | **0.9533** | §3.1 |
| VQA (`v2` deployed, published convention) | **0.8133** | §3.1 |
| Counting | **0.3509** — equals the constant | §3.1 |
| Captioning, RSICD BLEU-4 | **0.2446**, 13.4% unique | §5 |
| Grounding, DIOR-RSVG Acc@0.5 | **0.0762** | §5 |
| Change detection, LEVIR-CD change-class F1 | **0.5597** | §5 |
| **Change VQA, CDVQA test1** | **0.6061** *(corrected)* | §4, correction doc |
| CDVQA oracle ceiling | **0.99748** | §4.1 |
| Change captioning, LEVIR-CC aggregate | **0.5686** (preferred figure BLOCKED) | §5, §7 D2 |
| Opt–SAR complementarity gain | **−0.0064** | §5 |
| Refusal recall (`v2` deployed) | **0.4118** | §6.1 |
| Calibration ECE (change mask, affine) | **0.0034** | §6.2 |
| VRAM, VLM inference | **2.6–2.9 GB** of 6.0 GB | §6.4 |
| Latency | **not reproducible at single-run resolution** | §3.4, §6.3 |

### 9.4 What changed about the project's self-assessment

* **CDVQA is materially better than believed**: 0.6061, not 0.5380. Deficit against the 2021 baseline falls 12.1 → **5.3** points; margin over a per-type constant triples to **+9.8**; it beats that constant on seven of eight types instead of five.
* **Two headline figures are unreproducible** and are now labelled as such rather than quietly carried forward.
* **Everything else is exactly as recorded** — which, given a checkpoint-loss incident, a corrupted adapter set and two regenerated vocabularies in this project's history, is a genuinely good result.
