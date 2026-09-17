# SatQuery AI — As-Built Technical Status (revised 17 September 2026)

**SIH26167 · Indian Space Research Organisation (ISRO/SAC)**
*An Interactive Vision-Language Assistant for Multimodal Remote-Sensing Image Analysis through Text Queries*

| | |
|---|---|
| Prepared for | Internal evaluation |
| Prepared by | [Team Name / Team Leaders] |
| Reviewed by | [Faculty Guide / Professor Name] |
| Institution | [Institution Name] |
| Repository | github.com/hs-zz27/sih2 |
| Supersedes | *SatQuery-AI As-Built Status & Proposal*, dated 17 Sep 2026, whose figures were measured on 7 Sep |

Every number below is read from a file in this repository, and the file is named
beside it. Where a figure is older than the deployed model, the text says so.

---

## 0. What changed since the 17 September document

The earlier document reported the system as measured on **7 September**. Phase 5
(retraining all tools on an NVIDIA L40S, deployed 12 September) and the fixes made
on 17 September moved many of its numbers. Some got better, one got worse, and
several "open" items are closed.

| Item | 17 Sep document said | As built now | Source |
|---|---|---|---|
| Single-image VQA | RSVQA-LR 0.6425–0.6473 (n=207 slice) | **0.8947** on the official RSVQA-LR test split, published convention, 95% CI [0.8873, 0.9017], n=7,057; base model without adapter 0.3717 | `docs/assets/phase5/rsvqa_lr_official_test.json` |
| Change mask | LEVIR-CD F1 0.5597 | **F1 0.8550**, IoU 0.7467 | `docs/assets/phase5/change_mask/metrics.json` |
| Grounding | Acc@0.5 0.0762 | **0.1604** (deployed pretrained v2); 0.1262 from scratch | `docs/assets/phase5/grounding_pre/metrics.json` |
| Captioning | BLEU-4 0.2446 | **0.2658** (deployed pretrained v2), 764 unique captions (69.9%) | `docs/assets/phase5/caption_pre/metrics.json` |
| Land-cover head | mAP 0.2854 | **0.3150**, 4-band retention 0.9262 | `docs/assets/phase5/track_a/` |
| Cross-modal fusion | fused 0.7714 vs optical 0.7778 | **still negative, and worse in v2:** fused 0.7420 vs optical 0.7722 (gain −0.0301) | `docs/assets/phase5/optsar_fusion/` |
| VRSBench | "not evaluated" | **VQA evaluated** zero-shot on a pre-Phase-5 adapter (`track_b_v3`): 0.2968 (n=7,999) — above the honest train-fitted floor (0.2400), below GeoChat's 0.408 and below the optimistic test-fitted constant (0.3463). Caption and referring evaluators added on 17 Sep, **not yet run** | `docs/research/phase4-results.md`, `evaluation/vrsbench_tasks.py` |
| Routing | 0.5862, "weakest component" | Raw classifier 0.5862 did not reproduce (0.6897 at the same commit on scikit-learn 1.8.0). **What the router actually selects**, on a new sealed holdout: **0.8254** [0.714, 0.900], n=63 | `docs/assets/routing/revised.json` |
| L36 abstention demo beat | open | **closed 12 Sep**: cloud cover is a real ingest check. The clouded scene abstains for that reason with the learned tools (Phase 5 record) and, re-run on 17 Sep, on the CPU/stub configuration too (`cloud_cover` FAIL at ~71%, trigger `input_validation`) | `docs/phase5-full-training.md` §7 |
| Executor warnings | not mentioned | **fixed 17 Sep**: warnings were collected and never written to the trace; now in `Trace.warnings` | `satquery/contracts/trace.py` |
| Grounding confidence | not mentioned | **fixed 17 Sep**: the deployed v2 grounder reported v1's 0.0762 and blamed pooling it no longer does | `satquery/tools/grounding.py` |
| Automated tests | 1,070+ | **1,374 collected**; CI runs them on every push | `pytest --co tests/` |

---

## 1. Problem statement

Unchanged from the earlier document; the authoritative copy is `docs/ps-26167.md`.
In short, the problem statement requires:
- a remote-sensing-adapted vision or vision-language component;
- single-image VQA, plus captioning or grounding;
- bi-temporal change analysis;
- joint optical + SAR extraction;
- an agentic controller with an auditable execution summary.

Judging uses VRSBench, RSVQA and CDVQA, plus a private ISRO/SAC set of co-registered Cartosat-2S and RISAT pairs.

---

## 2. Executive summary

**Headline.** All mandatory capabilities are built, wired end to end and measured:

- **Single-image VQA is now the strongest result.** 0.8947 on the official RSVQA-LR test split, inside the 89–93% literature range.
- **Change detection jumped** to F1 0.855.
- **Grounding improved but is still weak** at 0.16.
- **Cross-modal fusion remains negative** after two different architectures. This is disclosed as a finding about the only openly licensed paired corpus, not hidden.
- **VRSBench:** VQA is measured on a pre-Phase-5 adapter; it clears the honest floor but not the optimistic constant. Caption and referring have evaluators ready to run.

### 2.1 Mandatory capability scorecard

| # | PS requirement | Status | Headline evidence |
|---|---|---|---|
| M1 | Remote-sensing adaptation | **VERIFIED** | QLoRA adapter lifts Qwen2.5-VL-3B from 0.3717 to 0.8947 on official RSVQA-LR; pretrained vs scratch on SECOND mIoU 0.2933 vs 0.1730 |
| M2 | Single-image VQA (mandatory) | **VERIFIED — strong** | RSVQA-LR official 0.8947 (published convention), all-types 0.6958. VRSBench zero-shot 0.2968 (pre-Phase-5 adapter) is weak |
| M3 | Captioning or grounding | **PARTIAL** | Caption BLEU-4 0.2658 on RSICD; grounding Acc@0.5 0.1604, Acc@0.7 0.0543 on DIOR-RSVG, far below published 0.70–0.80 |
| M4 | Bi-temporal change (mandatory) | **VERIFIED** | CDVQA 0.6061 vs majority 0.5084 (99.82% coverage, measured on the v1 change-VQA head); change caption BLEU-4 0.3063 on changed pairs |
| M5 | Change map (optional) | **VERIFIED — strong** | LEVIR-CD F1 0.8550, IoU 0.7467, precision 0.8182, recall 0.8952 |
| M6 | Cross-modal optical + SAR | **VERIFIED (negative)** | Fusion below optical alone in v1 (−0.0064) and v2 (−0.0301) |
| M7 | Agentic orchestration | **VERIFIED** | Illegal plans 0 / 600; system routing 0.8254 on the sealed holdout |
| M8 | Auditable execution trace | **VERIFIED** | Pydantic-validated trace incl. warnings; 36 golden traces; checkpoint SHA-256 per tool |
| M9 | Combine outputs, confidence, evidence | **VERIFIED** | Three-component confidence with limiting component named; GeoJSON + COG + PDF evidence pack |

### 2.2 Input scope and deliverables

| Item | Status | Note |
|---|---|---|
| Single image (I1) | VERIFIED | Real Cartosat-2E MX and EOS-04 products |
| Cross-modal pair (I2) | PARTIAL | Footprint overlap gated (≥70%); sub-pixel co-registration unverified — the estimator reports ~38 px on identically-footprinted pairs, so it is not gated |
| Bi-temporal pair (I3) | PARTIAL | Footprint overlap gated (≥80%); dates disclosed, not enforced (CDVQA ships undated PNGs) |
| GeoTIFF / PNG rules (I4/I5) | VERIFIED | PNG admitted only in benchmark mode |
| Interactive GUI + agentic backend | VERIFIED | Next.js + FastAPI/SSE, containerised; run page now shows execution warnings |
| Codes and models | PARTIAL | Code and tests public; `change_vqa_v1` weights unpublishable (SECOND has no licence) — see §9 |

---

## 3. System architecture (as built)

Four stages, unchanged in structure (`docs/01-Solution-Architecture-and-System-Design.md`):

1. **Ingestion and compatibility gate.**
   - Named checks: `crs_present`, `nodata_fraction`, `min_dimension`, `footprint_overlap`.
   - **`cloud_cover`**, where ≥50% blocks (added 12 Sep).
   - Modality is inferred from pixels and metadata, never from filenames.
2. **Controller.**
   - A TF-IDF + logistic-regression Tier-1 classifier proposes a task.
   - `configs/capability_matrix.yaml` restricts it to legal tasks with permitted parameters.
   - A confidence gate falls back to a configuration default.
   - Two rules added 17 Sep: a query with no textual evidence abstains, and the "your inputs cannot support X" notice requires a majority probability.
3. **Tool registry and physics verifier.**
   - Nine tools.
   - Every neural output is cross-checked against NDVI/NDWI/MNDWI/NDBI and SAR statistics.
   - Counts and areas are computed deterministically.
4. **Confidence, trace, evidence.**
   - Calibrated three-component confidence.
   - An entailment gate on generated prose.
   - A structured trace that now also carries the executor's `warnings`.
   - A GeoJSON + COG + PDF evidence pack.

**Deployed checkpoints** (`docker-compose.yml`):

| Tool | Deployed checkpoint |
|---|---|
| `rs_vqa_v1` | Phase 5 `adapter_final` |
| `caption_v1` | pretrained v2 |
| `grounding_v1` | pretrained v2 |
| `landcover_v1` | v2 |
| `change_mask_v1` | v2 |
| `change_vqa_v1` | v2 pretrained stem |
| `optsar_fusion_v1` | v2 |
| `change_caption_v1` | **v1** — v2 regressed on changed pairs, 0.1641 vs 0.3063 |

`docker-compose.v1.yml` reverts everything.

---

## 4. Technical specifications

Software stack as in the earlier document (Pydantic, rasterio/GDAL, scikit-image,
scikit-learn, PyTorch + PEFT, FastAPI/SSE, Next.js, reportlab, Docker Compose,
GitHub Actions). CI now runs on current action majors (checkout/setup-python/
setup-node v7) with pip caching.

| Component | Deployed approach | Training data |
|---|---|---|
| `rs_vqa_v1` | QLoRA (4-bit) on Qwen2.5-VL-3B-Instruct | RSVQA-LR instruction mix |
| `caption_v1` | Transformer decoder, ImageNet ResNet-50 backbone | RSICD |
| `grounding_v1` | Phrase cross-attention over the feature map (no pooling), ImageNet ResNet-50 | DIOR-RSVG |
| `landcover_v1` | Residual band-agnostic trunk, FiLM per stage, band dropout 0.3 | BigEarthNet subset (65,867 patches) |
| `change_mask_v1` | Multi-scale differences, concat-skip decoder | LEVIR-CD |
| `change_caption_v1` | v1 change captioner | LEVIR-CC |
| `change_vqa_v1` | Siamese ResNet-18 stem, per-date decoders | SECOND via CDVQA ids |
| `optsar_fusion_v1` | Bidirectional cross-attention before pooling, triad heads | WHU-OPT-SAR |

---

## 5. Measured results

| Task | Metric | Result | Source |
|---|---|---|---|
| Single-image VQA | RSVQA-LR official test, published convention / all types | **0.8947** / 0.6958 (base model 0.3717 / 0.2622) | `phase5/rsvqa_lr_official_test.json` |
| Single-image VQA | VRSBench VQA zero-shot, n=7,999, `track_b_v3` (pre-Phase-5) | 0.2968 [0.2869, 0.3069]; train-fitted floor 0.2400; test-fitted constant (optimistic) 0.3463 | `docs/research/phase4-results.md` §4 |
| Captioning | BLEU-4, RSICD | **0.2658** (v1 0.2446) | `phase5/caption_pre/metrics.json` |
| Grounding | Acc@0.5 / Acc@0.7 / mIoU, DIOR-RSVG n=1,141 | **0.1604** / 0.0543 / 0.1974 | `phase5/grounding_pre/metrics.json` |
| Land cover | BigEarthNet mAP 12-band / 4-band retention | 0.3150 / 0.9262 | `phase5/track_a/` |
| Change mask | LEVIR-CD F1 / IoU | **0.8550** / 0.7467 | `phase5/change_mask/metrics.json` |
| Change VQA | CDVQA accuracy / majority / oracle (v1 head) | 0.6061 / 0.5084 / 0.9975 | `docs/research/cdvqa-baseline-correction-2026-09-03.md` |
| Change VQA head | change-class mIoU, pretrained / scratch | 0.2933 / 0.1730 | `phase5/change_vqa*/` |
| Change captioning | BLEU-4 changed half (deployed v1) | 0.3063 | `docs/phase1-status.md` Phase 5 |
| Cross-modal fusion | optical / SAR / fused (v2) | 0.7722 / 0.7369 / 0.7420 | `phase5/optsar_fusion/` |
| Orchestration | illegal plans | **0 / 600** | `docs/assets/adversarial/report.json` |
| Routing | system accuracy, sealed n=63 | **0.8254** [0.7138, 0.8996] (baseline before 17 Sep revision 0.8095) | `docs/assets/routing/` |
| Calibration | ECE change-mask, before → after | 0.0668 → 0.0034 (v1; refit for v2 in `configs/calibration.v2.json`) | `docs/assets/calibration*/` |
| Refusal | recall / matched-pair probe / false refusals | 0.41 (7/17) / 0.167 (2/12) / 0.8% — **pre-Phase-5 adapter, not re-measured** | `docs/assets/refusal/track_b_v2_fullval.json` |

**How to read the routing number.**
- *Raw* is the classifier over all nine tasks.
- *System* is what the router actually selects: the prediction restricted to legal tasks, after the confidence gate.
- The sealed holdout was written and committed before the 17 Sep query-bank revision and was not used for it.
- The revision improved sealed system accuracy by one query (0.8095 → 0.8254), which is **within noise**. It improved the holdouts used for diagnosis much more (clean 0.793 → 0.828), which is the expected sign that adding templates is at diminishing returns.

---

## 6. Engineering rigour and verification

- **1,374 automated tests** collected; the full suite runs in GitHub Actions on every push, alongside the adversarial gate, a dependency audit, the frontend typecheck/build and both Docker image builds.
- **36 golden execution traces**, including all five PS representative queries, asserted behaviourally. Every golden change on 17 Sep was diffed field by field, and each changed field is named in its commit message.
- **Provenance:** each learned tool records the SHA-256 of the checkpoint it loaded.
- **Adversarial:** 200 queries × 3 configurations = 600 plans, 0 illegal, re-measured after every routing change.
- **Sealed routing holdout** with a test that fails if any substantive sealed query appears in the training bank.
- **Honest-number discipline:**
  - Phase 5 results were appended as dated sections, never overwriting v1 figures.
  - A claim disproved by measurement (early-stopped adapter "should be deployed") was withdrawn.
  - An inflated aggregate (change-caption BLEU) was replaced by the split that matters.

---

## 7. Limitations and risk register

| Limitation | Evidence | Why it matters |
|---|---|---|
| Cross-modal fusion is negative under two architectures, and **the deployed fusion checkpoint is the worse v2** | fused 0.7420 (v2) vs 0.7714 (v1); optical alone 0.7722 | M6 is mandatory. Either revert fusion to v1 or state plainly that the optical head carries the answer |
| Grounding far below published results | Acc@0.5 0.1604 vs ~0.70–0.80 | The PS's only grounding query ("Highlight the water body…") routes correctly but localises weakly; lead with captioning in demos |
| VRSBench VQA below the optimistic constant, and not re-measured on the deployed adapter | 0.2968 vs test-fitted constant 0.3463 (floor 0.2400), on `track_b_v3` | Weak transfer from RSVQA-LR to VRSBench's question types; the deployed Phase 5 adapter may score differently |
| VRSBench caption and referring not yet measured | evaluator exists, needs val imagery + checkpoints | Two of VRSBench's three PS roles have no number yet |
| Out-of-scope questions are answered, not refused | "what's the weather in delhi tomorrow" routes to VQA on the sealed holdout; image-conditional refusal 2/12 | A judge can probe this in seconds |
| Refusal not re-measured on the deployed adapter | refusal figures are pre-Phase-5 | Quote them as such or re-run `evaluation/track_b_eval.py` |
| CDVQA not re-measured end to end with the v2 change-VQA head | 0.6061 is the v1 head | The deployed head's mIoU is higher (0.2933 vs 0.2636); the end-to-end number may move |
| `change_vqa_v1` weights cannot be published | SECOND has no licence | "Codes and models" deliverable is partial — options in §9 |
| Routing measured on n=63 | CI ±9 points | Treat 0.83 as "around 0.8" |
| Raw routing figure depends on solver version | 0.5862 documented, 0.6897 reproduced | scikit-learn version is now recorded with every routing number |
| Sub-pixel co-registration unverified | shift estimator reports ~38 px on identical footprints | `max_coreg_shift_px` stays ungated |

The full dated register (L1–L37) is `docs/00-README-and-Requirement-Traceability.md` §3.6.

---

## 8. Real-world use case per capability

- **Disaster triage (ingestion):**
  - Mismatched optical/SAR footprints are rejected before any model runs.
  - A ≥50% clouded optical scene now abstains, naming cloud cover as the reason.
- **Agricultural monitoring (single-image VQA):** the strongest capability (0.8947 official RSVQA-LR), and the one to lead a live demo with.
- **Urban encroachment (change):**
  - The change mask is now strong (F1 0.855).
  - Compound "what changed and where" questions return both prose and the mask.
- **Monsoon flood mapping (cross-modal):**
  - The controller shifts weight toward SAR when cloud cover exceeds 40%.
  - Present fusion as a disclosed negative result, not a solved capability.
- **Regulatory accountability (trace):**
  - Every answer carries task, tools, checkpoint hashes, three-component confidence and, since 17 Sep, the executor's warnings.
  - A degraded run is now visibly different from a healthy one.

---

## 9. Remaining work plan

| Priority | Item | Why |
|---|---|---|
| P0 | Rehearse the demo on the actual presentation hardware | Environment drift caused L34 and L36 |
| P0 | Decide the fusion checkpoint: revert to v1 (fused 0.7714) or keep v2 and say so | The deployed v2 is measurably worse |
| P1 | Run `evaluation/vrsbench_tasks.py` for caption and referring, and re-run `evaluation/vrsbench_eval.py` on the deployed adapter | Closes the remaining two VRSBench roles and updates the VQA figure |
| P1 | Re-run CDVQA end to end with the deployed v2 change-VQA head | The quoted 0.6061 predates it |
| P1 | Re-measure refusal on the deployed adapter; add out-of-scope refusal | Weakest judge-visible behaviour |
| P2 | Resolve the SECOND licence for `change_vqa_v1` (options below) | Needed for "codes and models" |
| P2 | Routing beyond templates (e.g. a small sentence-embedding classifier), measured on the sealed holdout once | Templates are at diminishing returns |
| P3 | Confirm the RISAT product/mode for the private set | Narrowed to C-band (EOS-04), not confirmed |

**SECOND licence: options, in order of preference.**
1. Ask the SECOND authors for written permission to redistribute derived weights.
2. Publish code and the training recipe, and ship `change_vqa_v1` with its deterministic index path only. The semantic head stays available to evaluators on request.
3. Retrain the semantic head on a licensed change corpus and re-measure CDVQA.

---

## 10. Conclusion

SatQuery AI meets every mandatory capability of SIH26167, and the Phase 5 retrain
changed which ones lead:
- **Single-image VQA** is now competitive with published results on the official benchmark.
- **Change detection** is strong.
- **The orchestration guarantee** (0 illegal plans in 600) held through every routing change.

The weak points are measured and disclosed: fusion under two architectures,
grounding, VRSBench transfer and out-of-scope refusal.

The greatest remaining risk is still presentation discipline. Quote the numbers in
this document rather than older slides, and rehearse on the hardware the judges
will see.

## References

1. SIH26167 problem statement — `docs/ps-26167.md`.
2. Requirement traceability and limitations register — `docs/00-README-and-Requirement-Traceability.md`.
3. Phase 5 results — `docs/phase5-full-training.md`, `docs/phase1-status.md` §Phase 5, `docs/assets/phase5/`.
4. Lobry et al., "RSVQA: Visual Question Answering for Remote Sensing Data", IEEE TGRS 2020.
5. Yuan et al., "Change Detection Meets Visual Question Answering", arXiv:2112.06343.
6. Li, Ding, Elhoseiny, "VRSBench", NeurIPS 2024, arXiv:2406.12384.
7. Qwen2.5-VL-3B-Instruct (base model for the VQA adapter).
8. WHU-OPT-SAR, LEVIR-CD / LEVIR-CC, DIOR-RSVG, RSICD, BigEarthNet.
