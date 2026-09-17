# SatQuery AI — Implementation Plan

**PS 26167 · ISRO / Department of Space · SIH 2026**
Document 4 of 6 · Written 2026-08-27

> Weeks are **relative** (W0 = the week you start), because the SIH 2026 calendar could not be verified in this environment — item 10 in the week-0 gate. If the real runway is shorter than 14 weeks, compress by cutting from the **bottom of the descope ladder in §7**, never by delaying Phase 1.

---

## 1. The three organising principles

### Principle 1 — Contracts in week 1, so four people never wait on a GPU

The team is six people, of whom **one or two can fine-tune models**. If the software depends on trained models existing, four people idle for two months and then everything integrates in the final week. That is the standard way SIH projects fail.

**Freeze the Pydantic contracts — `InputManifest`, `Plan`, `ToolResult`, `Trace` — by end of W1, and write a stub implementation of all nine tools that returns plausible, schema-valid fake data.** Then the API, the executor, the capability matrix, the frontend, the trace panel, the PDF report, the evidence pack, the eval CLI and the golden tests are all buildable *immediately*, in parallel, with no GPU. When real models land in W6–W10 they are dropped in behind an unchanged interface.

This is the single highest-leverage decision in the plan, and it is the one that plays directly to this team's actual strength.

### Principle 2 — Every mandatory capability reaches end-to-end "ugly but working" before anything is optimised

By end of **W8** all five PS-mandatory areas must produce a real (possibly poor) answer through the real pipeline. Only then do you improve quality. A team that perfects VQA through W10 and starts change detection in W11 will ship a zero in a mandatory area — and under normalised scoring, one zero costs more than any amount of depth elsewhere gains.

### Principle 3 — Demo-day robustness is engineered in W12, not hoped for

Offline operation, the `lite` profile, the fallback paths, the curated demo bundle, and the rehearsed script are **scheduled work items**, not a final-night scramble.

---

## 2. Team allocation

Six members. Names replaced with roles; adjust to the actual team.

| Role | Owns | Primary skills | Load |
|---|---|---|---|
| **M1 — ML lead** | Track B (VLM QLoRA), `rs_vqa_v1`, `caption_v1`, instruction-mix construction, calibration | PyTorch, transformers, peft | Heaviest GPU |
| **M2 — ML #2** | Track A (encoder), `landcover_v1`, `optsar_fusion_v1`, `change_mask_v1`, `change_caption_v1`, `grounding_v1` | PyTorch, segmentation, detection | Heavy GPU |
| **M3 — Geospatial / data** | Layer 0 entirely, `index_engine_v1`, physics verifier, dataset acquisition + splits, Bhoonidhi, COG/evidence pipeline | rasterio, GDAL, NumPy | No GPU |
| **M4 — Backend / orchestration** | Controller, capability matrix, planner, executor, VRAM manager, FastAPI, SSE, trace, eval CLI | Python, API design | No GPU |
| **M5 — Frontend** | Next.js app, OpenLayers views, swipe comparator, trace panel, confidence card, registry + benchmark pages | React/Next/TS/Tailwind | No GPU |
| **M6 — Evaluation / docs / pitch** | Harness, metrics, ablations, adversarial suite, report, deck, demo script, video | Python, writing | Light GPU |

**The team lead's likely seat is M4 (or M4+M5).** That is the correct placement and it is not a consolation: the orchestration layer, the trace, and the eval CLI are the components the PS *separately evaluates* and the components most competing teams will treat as an afterthought. The differentiation lives there.

M3 is the quietly critical role. Layer 0 plus the index engine plus the verifier is a large amount of high-value, zero-GPU work that almost nobody else will do properly, and it is what makes the physics story credible.

---

## 3. Phase 0 — Foundations (W0–W1)

**Goal: nobody is blocked after this, and no unverified assumption is still load-bearing.**

| # | Task | Owner | Done when |
|---|---|---|---|
| 0.1 | Run the **12-item week-0 verification gate** (doc `03` §6) | all, per table | Every row is resolved with a written answer in `docs/verification.md` |
| 0.2 | Bhoonidhi registration + download real Cartosat-2S MX/PAN and RISAT products | M3 | Products on disk; **band composition and RISAT band read from actual metadata** |
| 0.3 | Repo scaffold, `docker-compose.yml`, `Makefile`, CI, pre-commit | M4 | `make dev` boots api + web |
| 0.4 | **Freeze the Pydantic contracts** | M4 + M3 | `satquery/contracts/` complete, reviewed, tagged `contracts-v1` |
| 0.5 | **Stub all nine tools** returning schema-valid fake data | M4 | `pytest tests/test_stubs.py` green; full pipeline runs on stubs |
| 0.6 | `capability_matrix.yaml` v1 + `satquery matrix --validate` in CI | M4 | Nine tasks defined; validator passes |
| 0.7 | `scripts/fetch_datasets.py` — download + mirror all P0 datasets | M3 + M2 | P0 datasets in shared storage, checksummed |
| 0.8 | `scripts/fetch_models.py` — download + sha256-verify base checkpoints | M1 | `models/` populated; `HF_HUB_OFFLINE=1` smoke test passes |
| 0.9 | Decide **LangGraph vs hand-rolled executor** — once, permanently | M4 | Written decision in `docs/adr/001-orchestration.md` |
| 0.10 | Verify a T4 QLoRA run starts, checkpoints, and **resumes** | M1 | `--resume` demonstrated after a deliberate kill |
| 0.11 | GPU account inventory: how many accounts, quotas, shared-storage plan | M1 + M6 | Written budget; weekly hour tracking sheet started |

**Exit criteria:** the full pipeline runs end to end on stubs and emits a valid trace. Every verification item answered. A T4 training run has been started, killed, and resumed successfully.

> Do not skip 0.10. Discovering in W7 that your training script cannot resume, after Kaggle killed a 10-hour run at hour 11, is the most common and most avoidable disaster in free-tier ML work.

---

## 4. Phase 1 — Vertical slice (W2–W5)

**Goal: one mandatory capability works end to end with a real model, and the entire application shell is complete against stubs.**

Choose **single-image VQA** as the slice. It is PS-mandatory, it has the clearest benchmarks, and it exercises every layer.

| # | Task | Owner | Done when |
|---|---|---|---|
| 1.1 | Layer 0 complete: reader, adaptive modality inference, all checks, co-registration (incl. gradient-phase for optical–SAR), normalisation, band harmonisation | M3 | `InputManifest` produced correctly for 20 hand-picked real files incl. Cartosat + RISAT |
| 1.2 | `index_engine_v1` **real** (first real tool): NDVI, NDWI, MNDWI, NDBI, σ⁰, VH/VV, GLCM, CoV, **adaptive Otsu/GMM thresholding**, SWIR-free fallback paths | M3 | Index rasters as COGs; unit tests on synthetic arrays with known answers |
| 1.3 | Controller: config gating, Tier-1 intent classifier, planner, validation, executor, VRAM manager | M4 | 9 tasks routable; **illegal-plan rate 0** on a 50-query suite |
| 1.4 | Synthetic query bank: ~60 templates/task → 3–5 k paraphrases; train + evaluate Tier-1 | M4 + M6 | Held-out accuracy reported with a confusion matrix |
| 1.5 | FastAPI + SSE trace streaming + SQLite run store | M4 | Trace streams live to a curl client |
| 1.6 | Frontend shell: upload, OpenLayers viewer, live trace panel, confidence card, answer view | M5 | Full UI usable against stubs |
| 1.7 | **Track B v0**: QLoRA on VRSBench + RSVQA subset (short run, quality irrelevant) | M1 | Adapter trains, loads, and answers through the real pipeline |
| 1.8 | `satquery eval` CLI + `--dry-run` + prediction schemas for all four annotation types | M4 + M6 | Batch run over 200 RSVQA items produces a valid predictions file |
| 1.9 | Eval harness v1 with VQA metrics | M6 | One command → one JSON report |
| 1.10 | **Track A v0**: encoder + land-cover head on a BigEarthNet subset | M2 | mAP reported on the official test split |
| 1.11 | Golden trace tests for 10 cases | M4 | `pytest tests/golden_traces` green |

**Exit criteria (end of W5):** a real image is uploaded through the real UI, routed by the real controller, answered by a real fine-tuned VQA model, verified by the real index engine, and displayed with a real streamed trace. **Everything else still stubbed — and that is fine.** The spine exists; the rest is filling it in.

---

## 5. Phase 2 — Breadth (W6–W9)

**Goal: all five PS-mandatory areas produce real answers. Quality is explicitly not the objective yet.**

This is the phase where breadth beats depth. Resist the pull toward polishing VQA.

| # | Task | Owner | Done when |
|---|---|---|---|
| 2.1 | **Track A full**: BigEarthNet(.txt) adaptation with band-presence masking, **random band dropout**, GSD conditioning | M2 | Land-cover mAP on official test; **band-dropout ablation shows 4-band inference works** |
| 2.2 | **Stage A2**: WHU-OPT-SAR ~5 m transfer | M2 | mIoU on WHU-OPT-SAR test |
| 2.3 | `optsar_fusion_v1` real, dual-stream + cross-attention, **triad mode** | M2 | Complementarity score computed per query; PS-mandatory area live |
| 2.4 | `change_mask_v1` real (Change-Agent MCI, or TinyCD) | M2 | F1/IoU on LEVIR-CD test |
| 2.5 | `change_caption_v1` real, **mask-conditioned** | M2 | BLEU-4/CIDEr on LEVIR-CC test |
| 2.6 | `change_vqa_v1`: **deterministic template path first**, then the CDVQA head | M1 | Template answers correct on the area-delta query type; head accuracy on CDVQA |
| 2.7 | `grounding_v1` real (Florence-2 on DIOR-RSVG + VRSBench referring) | M2 | Acc@0.5 reported; boxes exported as GeoJSON |
| 2.8 | `caption_v1` adapter + `landcover_v1` narrative grounded in index statistics | M1 | Caption metrics on VRSBench |
| 2.9 | **Physics verifier** with SWIR-free built-up path (SAR σ⁰ + variance + optical texture) | M3 | Per-claim agreement scores in the trace; conflicts named |
| 2.10 | **Tile pyramid + coarse-to-fine retrieval** for large scenes | M3 + M4 | 8000×8000 Cartosat scene answered in bounded time; retrieval logged in the trace |
| 2.11 | Evidence pack: GeoJSON + GeoTIFF + COG + `evidence.json` + ZIP | M3 + M4 | Mask opens correctly georeferenced in QGIS |
| 2.12 | Bi-temporal swipe comparator + optical–SAR blend in the UI | M5 | Both interactions smooth on a real pair |
| 2.13 | `run_batch` implemented on all nine tools | M4 + M1 + M2 | Batch throughput measured on T4 |
| 2.14 | Harness covers all nine tasks + all prescribed benchmarks | M6 | Single JSON report with every metric row |

**Exit criteria (end of W9):** every row of the PS mandatory-scope table in doc `00` maps to a working component with a measured number. **No zeros anywhere.** Ugly is acceptable; missing is not.

---

## 6. Phase 3 — Depth, calibration and hardening (W10–W12)

**Goal: quality, trustworthiness, and demo-day survivability.**

| # | Task | Owner | Done when |
|---|---|---|---|
| 3.1 | Full instruction-mix retrain of Track B, including **SAR samples and ~5 % refusal examples** | M1 | Improved metrics across VQA/caption; model declines appropriately |
| 3.2 | **Stage A3** high-res transfer (SpaceNet 6 / Umbra, or optical-only with the limitation documented) | M2 | Qualitative improvement on Cartosat-scale imagery |
| 3.3 | **Calibration**: temperature scaling per head; ECE before/after; reliability diagrams | M1 + M6 | ECE improvement table + plots in the report |
| 3.4 | **Three-component confidence** wired end to end with the geometric-mean combiner and fitted weights | M4 + M1 | UI shows the breakdown; components move sensibly under stress |
| 3.5 | **Entailment gate** (NLI over generated sentences vs structured payloads) | M1 | Gate statistics in the trace; verifier-on/off ablation quantified |
| 3.6 | **Abstention**: thresholds, named reasons, risk–coverage curve, AURC | M6 | AURC reported; every abstention message names the resolving input |
| 3.7 | **Four ablations** (two-track, triad, agent-vs-monolith, verifier on/off) | M6 + M2 | Four tables in the report |
| 3.8 | **Adversarial routing suite** (200 queries) | M4 + M6 | Illegal-plan rate 0; every rejection has a named reason |
| 3.9 | **Offline hardening**: `make offline-test` passes with networking disabled | M4 | Cold boot, no network, full run succeeds |
| 3.10 | **`lite` profile** genuinely working on CPU / low VRAM | M4 + M1 | Every task answers in `lite`, degraded but never failing |
| 3.11 | **Soak test**: 20 consecutive mixed queries, no OOM, no leak | M4 | Memory profile flat across the run |
| 3.12 | PDF report generation + model registry page + benchmark page | M5 + M6 | PDF renders with maps, indices, trace and confidence |
| 3.13 | Fault injection: kill a tool mid-plan, corrupt a file, mismatch CRS, feed a 1-band PNG in operational mode | M4 + M3 | Graceful degradation everywhere; zero stack traces surfaced to the user |
| 3.14 | Golden traces expanded to ~30 cases across all nine tasks and all hard routing cases | M4 | Green in CI |

**Exit criteria (end of W12):** the system is calibrated, abstains correctly, boots offline, degrades gracefully, and every headline claim in the report has a number behind it.

---

## 7. Phase 4 — Submission and finale prep (W13–W14)

| # | Task | Owner |
|---|---|---|
| 4.1 | **Curated demo bundle**: 8 inputs covering single optical, single SAR, cross-modal pair, bi-temporal pair, a deliberately incompatible pair, a heavily clouded optical, a large Cartosat scene, a low-confidence case | M3 + M6 |
| 4.2 | **Rehearse the 7-minute demo ten times**, including on the actual venue laptop with networking off | all |
| 4.3 | Technical report: architecture, methods, all metric tables, all ablations, calibration plots, AURC, limitations | M6 |
| 4.4 | **Requirement traceability matrix** — every PS clause → component → test → metric | M6 + M4 |
| 4.5 | Model cards + published weights (the PS lists models as a deliverable) | M1 + M2 |
| 4.6 | Deck + recorded backup video of the full demo | M6 + M5 |
| 4.7 | Freeze code. **Only bug fixes after W13.** | M4 |
| 4.8 | Prepare answers to the ten hardest anticipated judge questions | all |

---

## 8. The descope ladder

If time compresses, cut **strictly from the top of this list**, and never touch the four never-cuts. Decide descopes deliberately at the W9 checkpoint rather than by drift.

**Cut in this order:**

1. Semantic (multi-class) change → binary change only.
2. `TEMPORAL_CHANGE_MAP` as a distinct task → mask becomes an internal artifact only (it is a PS *bonus*, not a requirement).
3. Self-consistency paraphrase sampling in the confidence estimator.
4. Stage A3 high-resolution SAR transfer → run A2 only, document the gap.
5. Tile pyramid retrieval → whole-scene downsample only (hurts large-scene grounding; acceptable).
6. `change_caption_v1` **or** `change_vqa_v1` — the PS requires only one. **Keep change-VQA**, because its template path guarantees a non-zero floor.
7. `SINGLE_CAPTION` **or** `SINGLE_GROUND` — the PS requires only one. **Keep grounding**, because visual bounding boxes are far more convincing in a demo and grounding feeds the counting query.
8. Tier-2 LLM tie-break → Tier-1 argmax only.
9. PDF report → ZIP evidence pack only.
10. SAM/Lang-SAM mask upgrade on grounding boxes.

**Never cut, under any circumstances:**

- **The physics verifier and `index_engine_v1`.** Zero GPU cost, and it is the entire trustworthiness story.
- **The audit trace.** Directly and separately evaluated by the PS.
- **The compatibility gate.** An explicit PS deliverable, and the demo's opening move.
- **Offline operation.** Everything else is worthless if it does not boot on finale night.

---

## 9. Risk register

| # | Risk | Likelihood | Impact | Mitigation | Trigger to act |
|---|---|---|---|---|---|
| R1 | **BigEarthNet.txt is not as reported** (item 1 of the gate) | Medium | High | BigEarthNet v2 + GeoChat-Instruct + VRSBench instruction mix; Track A unaffected | W0 verification fails |
| R2 | **GPU quota exhausted mid-project** | Medium-High | High | Weekly hour tracking; multiple accounts; nothing >3B trained; resumable runs; `lite` profile as insurance | Weekly burn exceeds 45 h |
| R3 | **Cartosat has no SWIR → built-up detection is weak** | **High (assume true)** | Medium | Already designed for: SAR-primary + optical-texture path, `degraded_if` rule, honest confidence penalty | Confirmed at W0 item 6 |
| R4 | **RISAT is C-band, not X-band** (or vice versa) | Medium | Medium | All σ⁰ thresholds adaptive (Otsu/GMM), never absolute dB; CoV adapts to look count | W0 item 5 |
| R5 | fp16 instability / NaN losses on T4 | Medium | Medium | Clipping at 1.0, warmup, checkpoint every 200 steps, resume | First NaN |
| R6 | OOM at inference with multiple tools loaded | Medium | Medium | VRAM manager + LRU eviction + LoRA adapter swap + declared budgets + soak test | Soak test fails |
| R7 | **Integration crunch in the final weeks** | Medium | High | Stub-first architecture; contracts frozen W1; every tool integrated the day it exists | Any tool is "nearly done" for two weeks |
| R8 | Change-Agent / MCI weights unavailable | Medium | Low | TinyCD (~1 GPU-h) + separate caption head | W0 item 4 |
| R9 | Demo-day network failure | **High** | High | Full offline operation, tested; `lite` profile; recorded backup video | Assume it will happen |
| R10 | Judges question ML depth given the orchestration emphasis | Medium | Medium | Lead with the two-track adaptation argument and the ablation table; the ablations *are* the ML depth | Rehearse this answer |
| R11 | Team member drops out or goes dark | Medium | Medium | Contracts mean any component is resumable by another member; no undocumented work | Two missed standups |
| R12 | Eval harness format does not match ISRO's expectations | Medium | Medium | `--dry-run` schema validation; support all four annotation types; keep the writer pluggable | On receiving any harness spec |

---

## 10. The 7-minute demo script

Rehearsed ten times. Timings are hard. **Bracket the demo with the system's limits** — that is what distinguishes an operational tool from a hackathon toy.

| Time | Beat | What happens on screen |
|---|---|---|
| 0:00–0:30 | **The problem, in one sentence** | "An analyst has an optical image, a SAR image, and a question. Today that takes three tools and an expert. Watch." |
| 0:30–1:10 | **The rejection** | Upload a mismatched pair. The gate rejects it: *"footprint overlap 0.41, below the required 0.70."* Then a PNG in operational mode: rejected, with the PS's own rule quoted back. **Open on refusal — it establishes that everything after it is trustworthy.** |
| 1:10–2:20 | **Cross-modal, the flagship** | Upload the Cartosat + RISAT pair. Query 3 from the PS. Trace panel fills live: `index_engine_v1` → `optsar_fusion_v1` (triad) → narrative. Point at the trace line reading *"NDBI unavailable on this 4-band product; built-up via SAR σ⁰ + optical texture."* Then the complementarity number - **and say the measured one**: the triad scores optical 0.7778, SAR 0.7410, fused 0.7714, a gain of **-0.0064**. Fusion does *not* beat optical alone on WHU-OPT-SAR. (This line previously read *"SAR contributed +14 % IoU on built-up"*, which was written at plan time before the ablation ran and is contradicted by it. Corrected 2026-08-30; see `docs/deck.md`.) |
| 2:20–3:10 | **Counting, verifiably** | Query 1: *"How many aircraft are visible?"* Answer "8" with **eight boxes drawn on the map**. State plainly: we detect then count arithmetically — we never ask a language model to count. |
| 3:10–4:10 | **Bi-temporal** | Swipe comparator. Query 5. Deterministic answer: *"Increased — built-up grew 4.7 hectares (+18 %) in the south-east, centroid 17.42° N 78.51° E."* Overlay the change mask. |
| 4:10–4:50 | **Evidence, in QGIS** | Download the evidence pack. Open the mask GeoTIFF in QGIS beside the source scene. *"This is not a screenshot. It is a georeferenced product an analyst can ingest."* |
| 4:50–5:40 | **The abstention** | Feed the heavily clouded optical. The system answers with low confidence, shows the three-component breakdown, and states: *"optical member 63 % cloud-obscured; SAR-weighted; confidence 0.41 — below threshold. Supply a clearer optical acquisition, or accept the SAR-only classification."* Say the line: **"A system that knows what it cannot see is the one you can actually deploy."** |
| 5:40–6:30 | **The engineering** | Open `capability_matrix.yaml` on screen. Show the model registry page with versions and weights hashes. Show the metric table with the ablations, the reliability diagram, and the illegal-plan rate of zero. |
| 6:30–7:00 | **Close** | "Two-track adaptation because the training data is 10 m and your data is 1.6 m. A constrained planner because you grade the trace, not the reasoning. Physics verification because a confident wrong answer is worse than an abstention. It runs offline, on a T4." |

Two deliberate structural choices. **Open with a rejection** so every subsequent success is credible. **Close the interactive portion with an abstention** so the final impression is of a system with judgment. Any team can show five successes; showing where the boundary is, and that the system finds it by itself, is the harder and more memorable thing.

---

## 11. Weekly cadence

- **Monday standup, 20 minutes.** Blockers only. Anything "nearly done" for two weeks becomes an explicit descope decision.
- **Wednesday integration.** Every real tool that exists is merged and running through the pipeline. Nothing is allowed to live on a branch for more than a week.
- **Friday metrics.** M6 runs the harness and posts the single JSON report. **The metric table only ever grows — a row that regresses is a bug, not a fluctuation.**
- **GPU hours logged weekly** against the 35–45 h/week budget.
- **W5, W9, W12 are hard checkpoints** with go/no-go on descope items.

The Friday metrics ritual is what keeps the project honest. A visible table where every mandatory area has a number, updated weekly, makes "we have no change-detection number yet" impossible to ignore until week 11.

---

*Continues in `05-Innovation-and-Extra-Features.md`.*
