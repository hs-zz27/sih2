# Phase 4 — official benchmarks: requests, not actions

**Written 2026-09-03. NOTHING WAS DOWNLOADED.** Every size below came from a **metadata-only API query** (HuggingFace `paths-info`, Zenodo `records`) — no file content was transferred.

This document contains three items: one already-satisfied benchmark, and **two requests awaiting explicit individual approval**.

---

## 1. CDVQA — already satisfied by Phase 2, no new work

| | |
|---|---|
| Benchmark | CDVQA test1 |
| Split | **official**, 968 image pairs / 39,686 questions |
| Coverage | 99.82% (73 deferred) |
| Metric | overall accuracy |
| **Result** | **0.6061** *(measured 2026-09-03, corrected from 0.5380, A/B proven)* |
| Oracle ceiling | **0.99748** |
| Artifacts | `artifacts/phase2_baseline/cdvqa_head_test.json`, `cdvqa_oracle_test.json`, `cdvqa_head_test_AB_a93982d.json` |
| Record | `docs/research/phase2-baseline.md` §4, `docs/research/cdvqa-baseline-correction-2026-09-03.md` |

**Phase 4's CDVQA requirement is met.** Data was already on disk, the official protocol was used, no download is needed, and the result is the strongest Category-A comparison the project has.

---

## 2. REQUEST A — VRSBench: unfreeze + download

### 2.1 A finding that changes the cost

`docs/00` §3.6 **L11** records VRSBench as unevaluable because *"its 142k rows reference images that live in the separate DOTA and DIOR datasets"* — implying the imagery must be assembled from two large third-party datasets.

**That is not the case.** A metadata query of the VRSBench HuggingFace repository shows it hosts the imagery **directly**:

```
xiang709/VRSBench   (11 files)
  Images_train.zip          8.359 GB
  Images_val.zip            3.977 GB
  Annotations_train.zip     0.029 GB   [already on disk]
  Annotations_val.zip       0.013 GB   [already on disk]
  VRSBench_EVAL_vqa.json    0.009 GB   [already on disk]
  VRSBench_EVAL_Cap.json    0.005 GB   [already on disk]
  VRSBench_EVAL_referring.json 0.010 GB [already on disk]
```

**No DOTA or DIOR download is required.** L11's premise should be corrected in the same pass that updates the CDVQA figure.

### 2.2 What evaluation actually needs

Verified against the annotations already on disk:

| | |
|---|---|
| `VRSBench_EVAL_vqa.json` | **37,409 entries over 9,349 unique images** — matches the published 37,408 VQA / 9,350 images |
| Image references | `P2571_0006.png`, `11293_0000.png`, … i.e. the **val** imagery |
| **Required for evaluation** | **`Images_val.zip` only — 3.977 GB** |
| Not required | `Images_train.zip` (8.359 GB) — needed only for fine-tuning, which Phase 4 does not do |

**Requested download: 3.977 GB.** Extracted size unknown; a PNG archive typically expands modestly, so budget ~4–8 GB total. Free space is **57.97 GB**.

### 2.3 UNFREEZE REQUEST (Rule 12)

`docs/code-freeze.md` §"Explicitly out of scope after freeze" lists **VRSBench** with the reason *"Needs the DOTA download"* — a reason that §2.1 shows no longer holds.

**1. Exact files / components affected**

| Component | Change |
|---|---|
| `data/vrsbench/` | **Data only.** Add `Images_val.zip` and its extraction. No existing file modified |
| A new evaluation entry point | **New file**, e.g. `evaluation/vrsbench_eval.py`, writing only to `--out`. **No existing evaluator touched** |
| `docs/00` §3.6 **L11** | Correction of the DOTA premise — **a new dated note, not an edit** |

**2. Minimum required scope**

- Download **`Images_val.zip` only**. Not the train images.
- Add **one new read-only evaluator**. Modify **zero** existing files.
- Evaluate the **`v3` and `v2` adapters** on VRSBench-VQA. Captioning and grounding are a separate decision — VRSBench-Cap needs a corpus-BLEU implementation this repo does not have (`ENVIRONMENT.md` §T7), and VRSBench-VG needs the grounding head, which is separately frozen.
- **No training. No fine-tuning. No checkpoint written.**

**3. Reason, with measured evidence**

- VRSBench is **prescribed by the problem statement** and is the only prescribed benchmark still unevaluated.
- It is the **only public benchmark scoring captioning, grounding and VQA under one protocol**, and it is the direct fix for the audit's single largest comparability hole.
- Its test split is **9,350 images / 37,408 VQA pairs** — large enough to eliminate the ±6.5-point uncertainty that makes the current 207-question slice unable to resolve anything (Phase 2 §2, `ENVIRONMENT.md` §T4).
- **GeoChat publishes both zero-shot and fine-tuned numbers** (VQA 40.8 / 60.6; Cap BLEU-4 1.4 / 13.8; VG Acc@0.5 12.9 / 39.6), giving a graded ladder rather than a single unreachable target.
- Four of the five Top-5 models report it.

**4. Rollback plan**

| Step | Rollback |
|---|---|
| Download + extract | `rm -rf data/vrsbench/Images_val*` — restores the exact current state; the annotations already on disk are untouched |
| New evaluator file | `git rm` / delete — it is a new file, so removal is complete by construction |
| Results | Written to `artifacts/phase4_vrsbench/` only. No `docs/assets/`, no `metrics.json`, no `configs/` |
| Re-freeze | Component returns to the frozen list immediately on completion |

**5. What is NOT requested**

Training on VRSBench · the 8.359 GB train images · touching the grounding head · touching any existing evaluator · editing any published document.

### 2.4 Pre-run safety check that will be applied

Per the Phase-2 **standing caution** (§8.2): before running any new evaluator I will grep it for **every** write target (`write_text`, `open(...,"w")`, `save_checkpoint`, `np.save`, `savefig`, `to_json`), confirm each lands in a scratch path, and check `git status` immediately after the run and **before** reading any result.

---

## 3. REQUEST B — official RSVQA-LR test split: download only

**No unfreeze needed.** RSVQA is **not** on the frozen list; only VRSBench is.

### 3.1 Exact source and size

Zenodo record **6344334**, *"Remote Sensing VQA - Low Resolution (RSVQA LR)"*, DOI `10.5281/zenodo.6344334`, licence **CC-BY-4.0**.

| File | Size | Needed for test eval? |
|---|---|---|
| `Images_LR.zip` | **95.0 MB** | **yes** (all splits share one image archive) |
| `LR_split_test_questions.json` | 2.7 MB | **yes** |
| `LR_split_test_answers.json` | 1.9 MB | **yes** |
| `LR_split_test_images.json` | 0.1 MB | **yes** |
| `LR_split_train_*` / `LR_split_val_*` / `all_*` | 50.7 MB | no |
| **Full record** | **150.5 MB** | — |

**Requested download: ~99.7 MB** (or the whole 150.5 MB record, which is simpler and still negligible).

### 3.2 Why this is the highest value-per-byte item in the entire programme

The audit's single most damaging caveat is that SatQuery's headline VQA number is measured on the wrong data:

| | Current | Official |
|---|---|---|
| Split | **validation** subsample | **test** |
| Size | **207** questions | ~10,000 |
| Count questions | **included** (27.5%) | excluded by convention |
| 95% interval | **±6.5 points** | ~±1 point |
| Comparable to published figures? | **no (Category C)** | **yes (Category A)** |

Phase 2 also established that a **train-fitted per-type constant scores 0.6473 on that slice — identical to `track_b_v2`'s headline**. The current benchmark cannot distinguish the deployed model from a constant.

**For ~100 MB, the project's weakest comparability claim becomes its strongest.** Nothing else in the plan has this ratio.

### 3.3 Scope and rollback

| | |
|---|---|
| Scope | Data download + **one new read-only evaluator**. No existing file modified. No training |
| Evaluate | `base`, `v2`, `v3_probe`, `v3` on the official test split, reported **per question type** with a train-fitted per-type constant baseline alongside — the discipline Phase 2 §7.4 established |
| Output | `artifacts/phase4_rsvqa/` only |
| Rollback | delete `data/rsvqa_lr_official/` and the new evaluator; the existing `data/rsvqa_lr_2k` is untouched, so every Phase-2 number remains reproducible |

---

## 4. Summary — three items, two awaiting approval

| # | Item | Download | Unfreeze needed? | Status |
|---|---|---|---|---|
| 1 | **CDVQA** | none | no | **DONE** — 0.6061, Phase 2 |
| 2 | **VRSBench** | **3.977 GB** (`Images_val.zip`) | **YES — Rule 12 request in §2.3** | **AWAITING APPROVAL** |
| 3 | **RSVQA-LR official test** | **~99.7 MB** | no | **AWAITING APPROVAL** |

**Combined download if both approved: ~4.08 GB** against 57.97 GB free — leaving ~53.9 GB.

**Recommended order:** RSVQA-LR first. It is 40× smaller, needs no unfreeze, and fixes the audit's worst comparability problem. VRSBench is the larger prize but carries the unfreeze decision.

**Nothing will be downloaded, and no evaluator will be written, until each is approved individually.**
