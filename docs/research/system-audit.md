# SatQuery AI — system audit (Phase 0, repository forensics)

**Written 2026-09-03. Batch 1 of the Master Research & Engineering Directive.**

**Nothing was modified.** No code, no config, no checkpoint, no dataset, no artifact. The only file created is this one. Every measurement below was taken read-only on this machine during this session; where a value is quoted from a document rather than measured, it is marked `(doc)` and treated as a claim, not a state.

This audit **builds on** `docs/research/ENVIRONMENT.md` (hardware, software, paths, commands, budgets, traps) rather than repeating it. Read that file first; this one maps the *system*, not the machine.

---

## 0. Scope and method

| | |
|---|---|
| Inspected | `docs/HANDOFF.md`, `docs/code-freeze.md`, `docs/verification.md`, `docs/00-…`, `docs/phase1-status.md`, `docs/model-cards.md`, `docs/external_benchmark_audit.md`, `docs/research/ENVIRONMENT.md`, all of `satquery/`, `evaluation/`, `training/`, `configs/`, `tests/`, `frontend/app/`, `scripts/` |
| Measured | directory sizes, checkpoint sizes and parameter counts, dataset sizes, artifact counts, free disk, git state of both checkouts, `satquery prune --dry-run` |
| Not run | any training, any `make report`, any script that mutates a checkpoint directory (see ENVIRONMENT §T2) |
| Method note | Sizes were measured with PowerShell recursive `Measure-Object Length -Sum`; parameter counts by reading safetensors headers and `model_state_dict` tensors directly |

---

## 1. Current architecture

SatQuery is **not a single model.** It is a five-layer constrained-planner agent with one remote-sensing-adapted VLM and seven small specialist heads inside it. Understanding this is a prerequisite for every later phase, because "SatQuery's score" on a task can come from the VLM, from a specialist, or from deterministic arithmetic, and the three must never be conflated.

### 1.1 Data flow

```
files + free-text query
        │
        ▼
L0  INGEST                     satquery/ingest/
    reader → modality → checks → coreg → tiling → InputManifest
    • modality inferred from band count, dtype, histogram, local CoV,
      metadata — never the filename
    • config ∈ {SINGLE, CROSSMODAL_PAIR, BITEMPORAL_PAIR}
    • IngestMode ∈ {OPERATIONAL (GeoTIFF), BENCHMARK (PNG/JPEG, named benchmark only)}
    • emits index_availability: ndvi/ndwi/mndwi/ndbi/sigma0/vh_vv/glcm/cov
        │
        ▼
L1  ROUTE                      satquery/controller/router.py + intent.py
    Tier-1 tf-idf + logreg intent classifier (`tfidf_logreg_v1`)
        → legal task set from configs/capability_matrix.yaml (cm-2026.11.02)
        → Plan validated against permitted_params
    • 9 tasks; illegal-plan rate 0/600 (doc)
    • confidence gate 0.35: below it the router ignores the classifier and
      falls back to the configuration default
        │
        ▼
L2  EXECUTE                    satquery/controller/executor.py
    ordered tool calls from the matrix; each returns a ToolResult
        │
        ▼
L3  VERIFY                     satquery/verify/
    indices → thresholding → texture → semantic_change → verifier
    • deterministic NDVI/NDWI/MNDWI/NDBI/σ⁰/VH-VV/GLCM/CoV referee
    • entailment gate (DeBERTa-MNLI) over every generated sentence
    • three-component confidence: model · agreement · input_quality
        │
        ▼
L4  REPORT                     satquery/report/
    narrative synth → evidence pack (GeoJSON + COG + evidence.json ZIP)
    → PDF → Trace (Pydantic-validated, 36 golden traces)
        │
        ▼
    API (FastAPI + SSE)  →  Next.js frontend (query / runs / models / benchmarks)
```

### 1.2 The nine tasks in the capability matrix

`configs/capability_matrix.yaml`, version `cm-2026.11.02`:

| Task | Tools | Optional | Forbidden |
|---|---|---|---|
| `SINGLE_VQA` | `rs_vqa_v1` | `grounding_v1` | all change + fusion tools |
| `SINGLE_CAPTION` | `index_engine_v1`, `landcover_v1`, `caption_v1` | — | all change + fusion tools |
| `SINGLE_GROUND` | `grounding_v1` | — | all change + fusion tools |
| `SINGLE_LANDCOVER` | `index_engine_v1`, `landcover_v1` | — | all change + fusion tools |
| `XMODAL_JOINT_EXTRACT` | `index_engine_v1`, `optsar_fusion_v1`, `rs_vqa_v1` | `grounding_v1` | all change tools |
| `TEMPORAL_CHANGE_DESC` | `index_engine_v1`, `change_mask_v1`, `change_caption_v1` | — | `optsar_fusion_v1` |
| `TEMPORAL_CHANGE_VQA` | `index_engine_v1`, `change_mask_v1`, `change_vqa_v1` | `change_caption_v1` | `optsar_fusion_v1` |
| `TEMPORAL_CHANGE_MAP` | `index_engine_v1`, `change_mask_v1` | — | `optsar_fusion_v1` |
| `CLARIFY_OR_ABSTAIN` | — | — | — |

**Observation for later phases.** There is **no counting task, no referring-expression task distinct from `SINGLE_GROUND`, no multi-image (T1..Tn) task beyond pairs, and no geospatial-metadata task.** The directive's target capability space is wider than the matrix. Adding tasks touches the router, which is frozen (§8).

---

## 2. Component inventory

Tool ↔ checkpoint ↔ activation ↔ published metric. **Every learned tool is opt-in by environment variable and silently falls back to a stub when unset** — this is what keeps CI green without a GPU, and it is also the single easiest way to publish a stub run as a model result. `Trace.weights_hashes` is the check: stubs get no hash.

| Tool | Module | Env var | Checkpoint | Params (measured) | Published metric |
|---|---|---|---|---|---|
| `rs_vqa_v1` | `tools/rs_vqa.py` | `SATQUERY_VQA_BASE` + `_ADAPTER` | `track_b_v2/adapter_final` | 3,754,622,976 base + **37,152,768** LoRA | `rsvqa_lr` EM 0.6473 |
| `caption_v1` | `tools/caption.py` | `SATQUERY_CAPTION` | `caption/` | **1,556,442** | RSICD BLEU-4 0.2446, 13.4% unique |
| `grounding_v1` | `tools/grounding.py` | `SATQUERY_GROUNDING` | `grounding/` | **549,865** | DIOR-RSVG Acc@0.5 0.0762 |
| `landcover_v1` | `tools/landcover.py` | `SATQUERY_LANDCOVER` | `track_a_full_base/` | **422,231** | BigEarthNet-19 mAP 0.2854 |
| `change_mask_v1` | `tools/change_mask.py` | `SATQUERY_CHANGE_MASK` | `change_mask/` | **49,543** | LEVIR-CD change-class F1 0.5597 |
| `change_caption_v1` | `tools/change_caption.py` | `SATQUERY_CHANGE_CAPTION` | `change_caption/` | **292,911** | LEVIR-CC BLEU-4 0.3063 (1 ref) |
| `change_vqa_v1` | `tools/change_vqa.py` | `SATQUERY_CHANGE_VQA` | `change_vqa/best.pt` | **6,257,001** | CDVQA test1 OA 0.5380 |
| `optsar_fusion_v1` | `tools/optsar_fusion.py` | `SATQUERY_FUSION` | `optsar_fusion/` | **132,030** | complementarity gain **−0.0064** |
| `index_engine_v1` | `tools/index_engine.py` | — (always on) | none | **0** — deterministic | no failure mode beyond bad input |
| entailment gate | `verify/entailment.py` | `SATQUERY_NLI` | `models/nli_deberta_mnli` | **184,424,963** (frozen, third-party) | see `docs/assets/entailment/bench.json` |

**Totals.** System = **4,031,034,874** parameters in the v3 configuration, of which **91,986,935 (2.28%)** were trained by this project. The seven specialist heads together are **9,260,023** parameters — 0.23% of the system — and they carry captioning, grounding, land cover, change detection, change captioning, change VQA and fusion. That ratio is the headline structural fact of this repository.

---

## 3. Checkpoint inventory

Measured 2026-09-03 in the primary checkout. `checkpoints/` totals **8.56 GB** and is **gitignored with no backup** (`checkpoints_backup/` is 0.00 GB and holds only resume-test scratch — the name is misleading).

| Directory | Size | Modified | Classification | Note |
|---|---|---|---|---|
| `track_b_v3` | 1.706 GB | 2026-09-03 | **ACTIVE EXPERIMENT** | 82,726,912 params, 2,000 steps. Best accuracy, worse refusal, trained under the label-masking defect |
| `track_b_v2` | 1.386 GB | 2026-09-01 | **PRODUCTION** | 37,152,768 params. sha256 `10f48301…41174`. Reproduces to 12 d.p. |
| `track_b_v1` | 1.386 GB | 2026-08-29 | **EVIDENCE — DO NOT TOUCH** | 99.9922% NUL. Retained deliberately as evidence for `docs/00` L32 |
| `track_b_v3_probe` | 1.240 GB | 2026-09-03 | **ACTIVE EXPERIMENT** | Same 82.7M config, 200 steps. The step-count control arm |
| `killtest` | 1.386 GB | 2026-08-29 | **BASELINE (v0)** | The v0 adapter `track_b_eval.py`'s docstring names as the v0 arm |
| `smoke` | 0.970 GB | 2026-08-29 | REGENERABLE? | Scratch run; zeroed `adapter_config.json` (`docs/00` L29) |
| `change_vqa` | 0.233 GB | 2026-08-30 | **PRODUCTION** | 6.26M ResNet-18 semantic-change head; **the CDVQA bottleneck** |
| `change_vqa_scratch` | 0.085 GB | 2026-08-30 | **EVIDENCE** | The from-scratch arm of the published +56% pretraining ablation |
| `caption` | 0.052 GB | 2026-08-31 | **PRODUCTION** | vocab.json was regenerated and validated (L29) |
| `grounding` | 0.019 GB | 2026-08-31 | **PRODUCTION** | vocab.json regenerated and validated (L29) |
| `track_a_full_base` | 0.014 GB | 2026-08-29 | **PRODUCTION** | `landcover_v1`. `band_stats.json` gitignored (L2) |
| `track_a_full_multires` | 0.014 GB | 2026-08-31 | EVIDENCE | zeroed `metrics.json` (L29); numbers published in model cards |
| `track_a_nodropout` / `track_a_dropout` | 0.014 GB each | 2026-08-29 | **EVIDENCE** | The two arms of the band-dropout ablation (0.9025 vs 0.8443 retention) |
| `stage_a2` / `stage_a2_frozen` | 0.014 / 0.005 GB | 2026-08-29 | **EVIDENCE** | Fine-tuned 0.7759 vs frozen probe 0.7206 |
| `track_a_smoke` | 0.005 GB | 2026-08-29 | REGENERABLE? | Scratch |
| `change_caption` | 0.010 GB | 2026-08-29 | **PRODUCTION** | |
| `optsar_fusion` | 0.005 GB | 2026-08-29 | **PRODUCTION** | |
| `change_mask` | 0.002 GB | 2026-08-29 | **PRODUCTION** | 49,543 params |
| `stage_a3` | 0.002 GB | 2026-08-29 | **EVIDENCE** | Adaptation gain +0.1729 |
| loose `ckpt_step_*.pt` at root | ~0.18 MB | 2026-08-28 | REGENERABLE | 10 files from the resume test |

**Classification is preliminary and is NOT a delete list.** Phase 1 will produce the dry-run list Gate A requires.

---

## 4. Dataset inventory

`data/` totals **79.50 GB** against **53.2 GB free** on the only volume. **Disk, not GPU, is this project's binding constraint.**

| Dataset | Size | Role | Used by | Notes |
|---|---|---|---|---|
| `ben_full` | **44.13 GB** | BigEarthNet 12-band HDF5 shards | `track_a_full.py` | 4 × 10.4 GB train shards + `test_p8.hdf5` (3.78 GB) **and** its own `.gz` (1.69 GB). Track A trained on **30,000 patches** of ~590k |
| `whu_opt_sar` | **9.22 GB** | optical+SAR land cover | `optsar_fusion`, `instruct_mix` | `whu-opt-sar-512.zip` 6.38 GB **plus** `prepared/` 2.84 GB — the zip is post-extraction |
| `bigearthnet_14k` | **8.08 GB** | BigEarthNet 14k subset | Track A smoke/ablations | `BigEarthNet_14K.zip` 2.92 GB **plus** `extracted/` 5.16 GB — same duplication pattern |
| `bhoonidhi` | **6.58 GB** | real Cartosat-2E MX + EOS-04 products | `test_real_products.py`, demo | **Never trained on. Held out.** Irreplaceable without a Bhoonidhi account |
| `levir_mci` | **5.28 GB** | LEVIR-MCI / LEVIR-CC | `train_change_caption.py` | |
| `second` | **2.25 GB** | SECOND semantic change | `change_vqa` training | **Licence: none stated.** Weights blocked from publication |
| `dior_rsvg` | **1.87 GB** | referring grounding | `train_grounding.py` | No published split in this mirror |
| `levircd` | **0.65 GB** | LEVIR-CD tiles | `train_change_mask.py` | Official split, 7,120/1,024/2,048 at 256px |
| `rsicd` | **0.49 GB** | RSICD captions | `train_caption.py` | Official test split, n=1,093 |
| `rsvqa_lr_2k` | **0.32 GB** | RSVQA-LR **validation** 2k subset | `instruct_mix` | **Not the official test split** (ENVIRONMENT §T4) |
| `vrsbench` | **0.31 GB** | VRSBench **annotations only** | nothing yet | `VRSBench_EVAL_vqa.json`, `_Cap.json`, `_referring.json` and `VRSBench_train.json` **are on disk**. Only the imagery (DOTA/DIOR) is missing |
| `whu_opt_sar_lbl` | 0.19 GB | labels | fusion | |
| `cdvqa` | 0.12 GB | CDVQA Q/A/image indices | `cdvqa_predict`, oracle | Apache-2.0. Test1 = 39,686 questions / 968 pairs. Imagery resolves through `second/` |
| `demo_bundle` | 0.01 GB | demo | `make_demo_bundle.py` | |
| `instruct_mix` | 0.001 GB | the Track B training/val mix | Track B | 4,806 train / 534 val |

**Finding that changes the VRSBench plan.** `docs/00` L11 records VRSBench as unevaluable because "imagery lives in DOTA, not on disk". That is still true, but **the annotations are already here** — all three EVAL files. The remaining gap is imagery only, which sharpens the Phase-4 cost estimate considerably.

---

## 5. Benchmark and evaluation inventory

| Capability | Script | Protocol as implemented | External comparability |
|---|---|---|---|
| VQA | `evaluation/track_b_eval.py` | normalised exact match + token F1 over `instruct_mix/val.jsonl` (n=534) | **C** as published; **B** once re-typed per question type |
| Per-type VQA / counting | **none** | must be recovered from templated wording | classifier + finding in `external_benchmark_audit.md` §7.4 |
| Captioning | `training/train_caption.py` (eval block only) | sentence-mean **add-one-smoothed** BLEU-4, 5 refs, RSICD official test | **B** — metric ≠ corpus BLEU |
| Grounding | `training/train_grounding.py` (eval block only) | Acc@0.5 / Acc@0.7 / mIoU, **self-made split** | **C** |
| Land cover | `training/track_a_full.py` (eval block only) | macro mAP, partition shard | **B** |
| Change mask | `training/train_change_mask.py` (eval block only) | change-class F1/IoU/P/R, **official LEVIR-CD split** | **A** |
| Change caption | `training/train_change_caption.py` | sentence-mean BLEU-4, **1 reference** | **C** |
| Change VQA | `evaluation/cdvqa_predict.py` / `satquery eval --benchmark CDVQA` | overall accuracy, **CDVQA test1, 39,686 Q, 100% coverage** | **A** |
| CDVQA ceiling | `evaluation/cdvqa_oracle.py` | oracle over ground-truth change maps | — |
| CDVQA diagnosis | `evaluation/cdvqa_diagnosis.py` | per-question-type stratification | — |
| CDVQA routing | `evaluation/cdvqa_routing.py`, `cdvqa_baseline.py` | routing check + majority baseline | — |
| Refusal | `evaluation/refusal.py` | recall, false-refusal rate, lexical-shortcut probe | none published anywhere |
| Calibration | `evaluation/calibrate.py`, `calibration.py` | ECE, temperature vs affine | none published anywhere |
| Selective risk | `evaluation/selective.py` | risk-coverage, AURC | — |
| Abstention | `evaluation/abstention.py` | — | — |
| Adversarial / illegal plans | `evaluation/adversarial.py` | 200 queries × 3 configs | none published anywhere |
| Entailment | `evaluation/entailment_bench.py` | retained/flagged/unverifiable | — |
| Cross-sensor | `evaluation/cross_sensor.py` | GSD generalisation | — |
| Ablations | `evaluation/run_ablations.py` | 4 ablations | — |
| Soak / memory | `evaluation/soak.py` | RSS slope over 120 iterations | — |
| Confidence stress | `evaluation/confidence_stress.py` | — | — |
| Harness / schemas | `evaluation/harness.py`, `schemas.py` | four annotation types | — |

**Gap that matters most.** Five of the seven specialist heads have **no read-only evaluator** — their eval lives after the training loop and re-running it **overwrites `metrics.json` in place** (ENVIRONMENT §T2). Any phase that re-measures captioning, grounding, land cover, change mask or fusion must first copy the checkpoint directory. This is the highest-probability way a later batch destroys published evidence.

---

## 6. Keyword coverage map

Files matching each directive keyword across `satquery/ evaluation/ training/ tests/ configs/ docs/ scripts/ frontend/` (data excluded):

| Keyword | Files | Reading |
|---|---|---|
| `calibration` | 87 | deeply wired — a genuine cross-cutting concern, not a bolt-on |
| `checkpoint` | 69 | |
| `grounding` | 52 | well-wired, badly performing — the gap is quality, not integration |
| `CDVQA` | 38 | |
| `RSVQA` | 37 | |
| `refusal` | 36 | |
| `VRSBench` | 29 | referenced everywhere, evaluated nowhere |
| `captioning` | 26 | |
| `metrics.json` | 25 | |
| `oracle` | 14 | |
| `counting` | **10** | **thin.** No task, no tool, no metric — counting exists only as prose |
| `change detector` / `change_detection` | 3 / 1 | the component is named `change_mask_v1`; "change detector" is not repo vocabulary |
| `track_b_v3` | **3** | only `training/track_b_vlm_qlora.py`, `tests/test_vlm_label_masking.py` (both **uncommitted in the primary checkout**) and this audit's predecessor |

---

## 7. Repository state — and a divergence that must be resolved before Phase 2

| | |
|---|---|
| Primary checkout | `C:\Users\<dev>\Desktop\sih2` — branch `phase-0-closeout`, HEAD **`pre-import:P0-primary`** |
| Agent worktree | `.claude/worktrees/agent-worktree` — branch `agent-worktree`, HEAD **`pre-import:P0-worktree`** (the `main` review merge 2) |
| Relationship | **DIVERGED.** `pre-import:P0-primary` is *not* a descendant of `pre-import:P0-worktree` |

Four commits exist in the primary checkout and **not** in the worktree:

```
pre-import:P0-primary  Add 8-bit Adam so vision adaptation fits in 6 GB     ← trained track_b_v3
pre-import:geo-answering  Answer "where is this?" from the georeferencing the file already carried
pre-import:png-channels-fix  Read the channels a PNG declares instead of guessing them
pre-import:gpu-compose-fix  Give the GPU compose file an actual GPU
```

Plus two uncommitted, load-bearing changes in the primary checkout: ` M training/track_b_vlm_qlora.py` (the label-masking fix + validation loop) and `?? tests/test_vlm_label_masking.py`.

**What this does and does not invalidate.**

* **It does not invalidate the 2026-09-03 VQA baseline.** `git diff --name-only pre-import:P0-worktree pre-import:P0-primary` touches only `satquery/contracts/input_manifest.py`, `satquery/controller/executor.py`, `satquery/ingest/reader.py`, `docker-compose.gpu.yml`, and tests/golden traces. **`evaluation/` is byte-identical and `satquery/tools/rs_vqa.py` is byte-identical**, so the four-arm sweep in `docs/external_benchmark_audit.md` §7 is unaffected.
* **It does invalidate any end-to-end pipeline measurement** taken in the worktree: ingest, the executor, golden traces and the CDVQA end-to-end path all differ. `pre-import:geo-answering` in particular adds geospatial answering — a Phase-14 capability that **already exists in the primary checkout and is absent from the worktree.**
* **`track_b_v3` was trained by code that does not exist in this worktree** (8-bit Adam, `pre-import:P0-primary`). Reproducing that run from the worktree is impossible as things stand.

**Recommendation, for approval:** before Phase 2, either rebase the agent worktree onto `pre-import:P0-primary`, or run all Phase-2 baselines from a worktree created off `phase-0-closeout`. Doing neither means the immutable baseline is taken against code that is not the code the project runs. **This is flagged, not acted on.**

---

## 8. Frozen components

`docs/code-freeze.md` §"Explicitly out of scope after freeze" — six items, **must not** be started without an unfreeze request under Rule 12:

| Frozen item | Current measured state | Directive phase that needs it |
|---|---|---|
| **CDVQA segmenter** (0.2636 change-class mIoU, 0.9975 ceiling) | 93% of CDVQA headroom | **Phase 5 + 6 — Gate B** |
| **Grounding** (Acc@0.5 0.0762) | weakest component | **Phase 9 — Gate C** |
| **Image-conditional refusal** (2/12) | open negative result | Phase 8 |
| **VRSBench** | annotations on disk, imagery missing | **Phase 4** |
| **`max_coreg_shift_px` enforcement** | estimator unvalidated (L16) | Phase 13 |
| **Tier-1 routing** (0.5862 CLEAN_HOLDOUT) | weakest measured number; touching it risks the 0/600 guarantee | Phase 15 |

**Four of the directive's first ten phases land on frozen components.** Phases 5 (diagnosis only), 7 (v3 ablation — training code is *not* on the frozen list), 10 (counting), 11 (captioning) and 12 (temporal) can proceed without an unfreeze; Phases 4, 6, 9 cannot.

The freeze also carries a precedent: **Unfreeze 1** was granted for the Track B v2 retrain and is recorded in `docs/code-freeze.md`. That is the shape any request should take.

---

## 9. Known weaknesses — measured, not suspected

Every row is a number this repository or the 2026-09-03 audit produced, with its comparability class.

| # | Weakness | Measured | Class | Where |
|---|---|---|---|---|
| W1 | **CDVQA is 20.9 points behind SOTA and 12.1 behind a 2021 baseline** | 0.5380 vs 0.7474 / 0.6590, identical test1 split | **A** | audit §6.2 |
| W2 | **LEVIR-CD change detection 36.3 points behind** | F1 0.5597 vs ≈0.9227 | **A** | audit §6.3 |
| W3 | **Grounding near floor** | Acc@0.5 0.0762; 2B models reach 0.749 | C | model cards |
| W4 | **Land-cover head worse than always-predicting-negative at τ=0.5** | 0.2064 vs 0.1834; asserts on ~0.25% of decisions | B | model cards, L8 |
| W5 | **Counting is a constant** | 0.3509 — exactly the train-fitted constant; 19 of 20 correct answers are the gold "0" | B | audit §7.4 |
| W6 | **Caption collapse** | 146 unique captions across 1,093 images (13.4%) | — | model cards |
| W7 | **v2's headline VQA number equals a per-type constant** | both 134/207; discordant 17 vs 17 | A (internal) | audit §7.4 |
| W8 | **Image-conditional refusal barely works** | 2/12 (v2), 1/12 (v3); lexical refusals 10/10 | — | L3 |
| W9 | **Fusion does not beat optical alone** | complementarity −0.0064, where EarthMind and Earth-OneVision both report positive | C | model cards, M6 |
| W10 | **Routing accuracy weak** | 0.5862 on CLEAN_HOLDOUT (n=29) | — | `docs/00` §3.1 |
| W11 | **Change-caption BLEU not comparable** | 1 reference vs the published 5 | C | ENVIRONMENT §T7 |
| W12 | **v3 trades reliability for accuracy** | +0.1401 RSVQA, −0.0589 refusal recall | A (internal) | audit §7.6 |
| W13 | **No SAR benchmark result of any kind** | SARLANG-Bench and equivalents not on disk | — | audit §13 |
| W14 | **Training set is 4,806 examples** | vs 3M–34M for every Top-5 competitor | — | audit §11 G5 |
| **W15** | **The v0 baseline arm is dead, and the command that uses it does not say so.** `evaluation/track_b_eval.py` names `checkpoints/killtest/adapter_final` as the `v0=` arm in two places (docstring line 34, `--adapters` help line 248). **All four of `killtest`'s adapter safetensors are corrupt**, and the sibling `.pt` files cannot rescue them: for a PEFT run `save_checkpoint()` writes `['step','is_peft','adapter_dir','optimizer_state_dict','scheduler_state_dict','training_state','rng_state','extra']` — **no `model_state_dict`**. The weights only ever lived in the `adapter_dir`, which is 100% NUL. Verified 2026-09-03 by loading a representative `.pt` and by opening every safetensors header | v0 arm **unloadable**; 2.38 GB of orphaned optimiser state | — | `docs/storage-audit.md` §3; `docs/00` L32 |

| **W16** | **Six of the seven specialist heads have no read-only evaluator, and evaluating one overwrites the artifact of record.** Every one of `train_caption.py`, `train_grounding.py`, `train_change_mask.py`, `track_a_full.py`, `train_optsar_fusion.py` and `train_change_caption.py` calls `save_checkpoint(args.ckpt_dir, …)` **and** rewrites `args.ckpt_dir / "metrics.json"` unconditionally *after* the training loop — so the documented "`--resume --epochs <already-trained>`" trick for reaching the eval block also rewrites the published metrics in place. Only `change_vqa_v1` has a genuine read-only evaluator (`evaluation/cdvqa_predict.py`, which writes solely to `--out`) | Re-measuring any of six heads destroys the number being checked | — | verified line-by-line 2026-09-03; `ENVIRONMENT.md` §T2 |

**W16 is worse than `ENVIRONMENT.md` §T2 records, in two ways.** T2 says *five* heads; it is **six**. And the blast radius is not only `metrics.json`: `train_caption.py`, `train_grounding.py` and `train_change_caption.py` also rewrite **`vocab.json`**, and `track_a_full.py` rewrites **`band_stats.json`** — which are precisely the sidecars that were destroyed in the L29 incident and had to be reconstructed and validated against published metrics. A routine re-measurement can therefore overwrite the very files whose recovery is one of this project's documented near-misses.

**Mitigation in force from Phase 2 onward** (approved 2026-09-03): every specialist evaluation runs against a **copy** of the checkpoint directory under `artifacts/phase2_baseline/ckpt_scratch/`, never the original; the SHA-256 of each original `metrics.json` is recorded before the run so any mutation is detectable; and no published `metrics.json` is overwritten in place under any circumstance, even temporarily. Superseding a published number is a separate, explicit decision.

**Recommended fix — logged, deliberately NOT built now.** Give each head an `--eval-only` flag, or a small `evaluation/eval_<head>.py` that loads the checkpoint and writes only to `--out`, matching what `cdvqa_predict.py` already does. Six small files, not on the frozen list, and it removes the single most likely way a future session silently destroys evidence.

**Decision 2026-09-03: deferred.** Phase 2 is measurement-only, and building this would be an engineering change inside a baseline-reproduction phase — exactly the kind of scope drift Rule 3 (one major variable at a time) exists to prevent. It belongs with the Phase-21/22 experiment-discipline and efficiency work. The scratch-copy mitigation above carries the risk until then.

| **W17** | **Counting is negative-value, not merely weak — and this is now resolved, not suspected.** On the **official RSVQA-LR test split** (2,947 count questions, Phase 4), **every checkpoint scores BELOW a train-fitted constant**: `v3` **0.2280** · `v3_probe` 0.1944 · `v2` 0.1802 · base 0.0000, against a constant of **0.2555**. Answering "0" to every count question outperforms the deployed model and the best model alike | The counting capability **subtracts** accuracy. A system that routed count questions to a constant would score higher than one that routes them to the VLM | **A** (official test split) | Phase 4 official RSVQA-LR run |

**W17 — the exact framing that must carry into Phase 10.** Phase 2 could only say *"counting equals a constant"*, on 57 questions from a validation slice where the 95% interval was ±13 points. **Phase 4 resolves it at n=2,947 on the official test split: counting is not weak, it is worse than useless.** The distinction matters for how Phase 10 is scoped:

* The Phase-10 question is **not** *"how do we improve counting from 0.23?"* — improving to 0.25 would still be worthless.
* It is *"can a learned counter beat a constant at all, and if not, should count questions be routed away from the VLM entirely?"* The second option is cheap, requires no training, and would raise the all-types score immediately.
* Supporting evidence already gathered: of `v3`'s correct count answers on the 207-question slice, **19 of 20 were the gold "0"** — the model has learned *"is this class absent?"*, not *how many*. The official split's 0.2280-vs-0.2555 result is the same phenomenon measured properly.
* The literature routinely **excludes count** from RSVQA-LR reporting for this reason. SatQuery is one of the few systems that reports it, which is to its credit — and the number says the capability should be redesigned or bypassed, not tuned.

**W15 is an open decision, not just a finding.** `evaluation/track_b_eval.py` will raise on the `v0=` arm rather than silently emit a number — which is the safe failure — but the *published* v0→v1 comparison (`rsvqa_lr` 0.4510 → 0.6425, quoted in `docs/00` §3.4b and `docs/model-cards.md`) can no longer be reproduced from disk. Before Phase 2 (baseline reproduction) or Phase 7 (the v3 ablation) runs, one of two things must happen:

1. **Re-establish a real v0** — retrain the RSVQA-LR-only adapter under its recorded recipe. This is a training run and therefore a freeze decision.
2. **Mark every v0 comparison `BLOCKED`** in those phases, per Rule 4 (missing result = `BLOCKED`, not an estimate).

**What must not happen is the third option:** letting the arm fail at runtime and quietly dropping it from the results table, or substituting a different checkpoint and continuing to call it v0. Recorded here so the choice is made deliberately rather than discovered mid-phase.

---

## 10. Suspected bottlenecks, ranked

Ranked by *evidence strength × expected system-level gain ÷ cost*, not by how large the number looks.

| Rank | Bottleneck | Evidence it is the bottleneck | Expected gain | Cost | Frozen? |
|---|---|---|---|---|---|
| **1** | **The 6.26M ResNet-18 semantic-change segmenter** | Oracle over ground-truth maps scores **0.9975** while the system scores 0.5380 — the answer layer contributes no measurable error. Per-class IoU 0.068 water / 0.071 playgrounds / 0.098 trees are exactly the classes the three worst question types ask about. The project's own ablation already showed **+56% relative** from swapping scratch→ImageNet ResNet-18 | Largest and best-bounded in the repo | Hours, one GPU | **YES — Gate B** |
| **2** | **The evaluation protocol itself** | A per-type constant scores identically to the production model on the headline VQA metric; the RSVQA slice is a 207-question *validation* subsample; VRSBench annotations sit unused on disk | Converts ~7 ⚪ verdicts into numbers. Cannot tell whether *any* later phase worked without it | Disk for DOTA/DIOR imagery; no modelling | **VRSBench: YES** |
| **3** | **Training-data scale and curriculum** | 4,806 examples against 3M–34M. Upstream of the VQA, captioning and counting gaps simultaneously | Broad but diffuse | High | No |
| **4** | **The grounding architecture** | A text-conditioned box regressor with a **from-scratch** visual backbone on 6,359 examples. Not a tuning problem | 0.0762 → plausibly >0.5 | Rebuild | **YES — Gate C** |
| **5** | **Absence of an explicit counting mechanism** | Counting is not a task, not a tool, and not a metric anywhere in the repo (10 keyword hits, all prose). Autoregressive generation is being asked to count | Bounded but real | Medium | No |
| **6** | **The 49,543-parameter change detector** | Deliberate design choice (screening, precision 0.44 / recall 0.76). The gap is capacity, not method | Improves W2 and feeds W1 | Medium | Adjacent to Gate B |
| **7** | **Tier-1 router at 0.5862** | Weakest measured number, but touching it risks the 0/600 illegal-plan guarantee, which is the project's strongest uncontested result | Low gain, high risk | Medium | **YES** |

---

## 11. Storage risks — a preview for Phase 1, not a delete list

**Nothing in this section may be acted on until Gate A produces an approved dry-run list.**

| | Measured 2026-09-03 |
|---|---|
| Volume | C: — single volume, shared with system temp |
| Free | **53.2 GB** |
| `data/` | 79.50 GB |
| `checkpoints/` | 8.56 GB |
| `models/` | 7.36 GB |
| `artifacts/` | 4.68 GB, **359 directories** |

**Candidate reclaim, in descending confidence.** Every figure is measured; every classification is provisional.

| Candidate | Size | Provisional class | Why, and what must be verified first |
|---|---|---|---|
| `artifacts/run_*` beyond the newest 20 | **3.35 GB** | **REGENERABLE — highest confidence** | `satquery prune --dry-run` reports 343 generated run dirs, 323 deletable, **16 named directories protected** by the whitelist. The repo's own tool, already tested, already whitelist-shaped. Still requires Gate A |
| `data/whu_opt_sar/whu-opt-sar-512.zip` | **6.38 GB** | REGENERABLE? | `prepared/` (2.84 GB) exists alongside it. **Verify** nothing reads the zip and that `index.json` points only at `prepared/` |
| `data/bigearthnet_14k/BigEarthNet_14K.zip` | **2.92 GB** | REGENERABLE? | `extracted/` (5.16 GB) exists alongside. Same verification |
| `data/ben_full/bigearthnet_test_p8.hdf5.gz` | **1.69 GB** | REGENERABLE? | `test_p8.hdf5` (3.78 GB) is the decompressed form. Same verification |
| `data/ben_full` train shards p1–p3 | **~31 GB** | **DO NOT TOUCH without a decision** | Track A trained on 30,000 patches, which likely came from p0 alone — but `band_stats.json` regeneration (L2) is documented as needing "the four BigEarthNet train shards". Removing them may make `landcover_v1` unrestorable on a fresh clone |
| `checkpoints/killtest`, `smoke` | 1.386 + 0.970 GB | **KEEP for now** | `killtest` is the **v0 baseline arm** named in `track_b_eval.py`'s docstring. `smoke` is scratch but carries a zeroed sidecar documented in L29 |
| `checkpoints/track_b_v1` | 1.386 GB | **DO NOT TOUCH** | Corrupted deliberately-retained evidence for L32 |
| `data/bhoonidhi` | 6.58 GB | **DO NOT TOUCH** | Real ISRO products, never trained on, not re-downloadable without an account |
| `.pytest_cache`, `__pycache__`, `satquery.egg-info` | <0.02 GB | REGENERABLE | Trivial; not worth the risk budget |

**Storage risks that are not about space.**

* **`checkpoints/` has no backup at all** and a script in this repo destroyed it once already (ENVIRONMENT §T1). 8.56 GB against 53.2 GB free — **an off-volume copy is affordable and should precede any Phase-1 deletion.**
* **Every dataset's HF cache lives inside its own folder** as `data/<name>/.cache`, so a naive "clear the HF cache" would hit dataset directories.
* **Phase 4 needs DOTA + DIOR imagery** for VRSBench — tens of GB — against 53.2 GB free. Phase 1 is a prerequisite for Phase 4, not an optional tidy-up.

---

## 12. BLOCKED

Carried forward from `docs/research/ENVIRONMENT.md` and confirmed unchanged, plus one new item.

| Item | What is missing |
|---|---|
| Verified "all tests pass" | Not run since 2026-09-01 (doc); the tree has changed twice since. Needs one `pytest tests/ -q` — read ENVIRONMENT §T1 first |
| VRSBench evaluation | **Annotations are on disk.** DOTA + DIOR **imagery** is not. Network available; disk is the constraint |
| Official RSVQA-LR test split | Only the 2k validation subset is present |
| Any SAR VQA / captioning benchmark | Nothing on disk. SatQuery has no SAR benchmark result of any kind |
| Local evaluation of any competitor | Nothing downloaded. EarthDial (public, CC BY 4.0) and TinyRS-R1 (public, 2B) are the two feasible targets |
| `band_stats.json` on a fresh clone | Gitignored; regeneration needs `data/ben_full` |
| Publishing any trained weights | Licence undecided; `change_vqa_v1` blocked outright (SECOND states no licence) |
| ~~**NEW — reproducing `track_b_v3`**~~ | ~~It was trained by `pre-import:P0-primary`, a commit absent from the agent worktree~~ **RESOLVED 2026-09-03** — the worktree was rebased onto `pre-import:P0-primary`; see `docs/research/checkpoint-protective-copy-2026-09-03.md` §2. Reproducing v3 *with the label-masking fix* still needs the two uncommitted files in the primary checkout |
| **NEW — the v0 baseline arm** | `checkpoints/killtest/adapter_final` is unloadable and unrescuable (**W15**). **RESOLVED by decision 2026-09-03: marked BLOCKED per Rule 4.** The `rsvqa_lr` 0.4510 → 0.6425 comparison is recorded as unreproducible-from-disk and is not patched, estimated or substituted |
| **NEW — CDVQA deferrals unexplained** | The documented *"100% coverage, 39,686 / 39,686"* **is not reproducible at either commit.** Measured 2026-09-03: `pre-import:P0-worktree` defers **122** questions (99.69% coverage), `pre-import:P0-primary` defers **73** (99.82%). All deferrals at both commits fall in `change_to_what`. The `pre-import:png-channels-fix` fix **improved** coverage; it did not regress it. Why the recorded figure says zero is unknown. **Logged, deliberately not investigated** — a secondary behaviour change, not a Gate D violation, and chasing it now would drift into Phase 5 / Phase 8 scope. **Likely to matter for Phase 8 (reliability gate)**, where coverage-aware scoring is the subject. See `docs/research/cdvqa-baseline-correction-2026-09-03.md` §6 |

---

## 13. What Phase 1 will need from the human

Phase 1 (Storage Cleanup) stops at **Gate A** and will produce a dry-run table of `path · size · classification · reason` for approval. Before that list can be honest, three questions need answering, and they are decisions rather than measurements:

1. **May `checkpoints/` (8.56 GB) be copied off-volume first?** There is room. There is currently no backup, and this repository has already lost every checkpoint once.
2. **Are `data/ben_full` train shards p1–p3 (~31 GB) still needed**, or was Track A's 30,000-patch sample drawn from p0 alone? This is the single largest reclaim available and the only one that could make a published number unreproducible.
3. **Should the agent worktree be rebased onto `pre-import:P0-primary`** before Phase 2, so the immutable baseline is taken against the code the project actually runs? (§7)

---

## 14. Deliverable status

| Directive deliverable | Status |
|---|---|
| `docs/research/system-audit.md` | **This file — complete** |
| `docs/storage-audit.md` | Phase 1 |
| `docs/research/leaderboard-matrix.md` | Phase 3 — a substantial head start exists in `docs/external_benchmark_audit.md` §4, §6, §14 |
| `docs/research/cdvqa-bottleneck.md` | Phase 5 |
| `docs/research/grounding-rebuild-plan.md` | Phase 9 |
| `docs/research/ENVIRONMENT.md` | Pre-existing, verified current |
