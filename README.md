# SatQuery AI — Build Plan Index & Requirement Traceability

**PS 26167 · ISRO / Department of Space · SIH 2026 · Software · Space Technology**
**SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**

Document 0 of 6 · Written 2026-08-27 · **Read this first.**

---

## 1. What this document set is

Six documents. A consolidated build plan produced by merging five independent design passes over PS 26167 and resolving their disagreements, then adding the decisions none of them made.

| Doc | Contents | Primary reader |
|---|---|---|
| **`00`** (this file) | Index, **PS requirement traceability matrix**, unverified-claims register, week-0 checklist, first-two-weeks action list | Everyone, day 1 |
| **`01-Solution-Architecture-and-System-Design.md`** | Six design axioms, five-layer architecture, ingest & compatibility gate, tool registry, verification & confidence, trace, dual-mode operation, tech stack, repo layout, non-functional targets | Backend, geo, frontend |
| **`02-Agentic-Workflow-and-Orchestration.md`** | Why a constrained planner beats free-form tool-calling, nine-task taxonomy, routing, the capability matrix, plan validation, executor, worked traces for all five PS representative queries, hard cases, testing | Backend lead |
| **`03-Models-and-Datasets.md`** | Compute reality and the three hard rules, two-track resolution-bridged adaptation, per-tool model recommendations with fallbacks, dataset plan, splits and leakage, evaluation plan, ablations, **week-0 verification gate** | ML leads |
| **`04-Implementation-Plan.md`** | Three organising principles, team allocation, four phases W0–W14 with exit criteria, descope ladder, 12-row risk register, 7-minute demo script, weekly cadence | Team lead, everyone |
| **`05-Innovation-and-Extra-Features.md`** | Extras beyond mandatory scope, tiered by marks-per-hour, with an explicit declined list and a "build only these five" summary | Team lead, M6 |

**Reading order for a new team member:** this file, then `04` §2 (your role) and `04` §3–7 (the phase you are in), then whichever of `01`–`03` covers your components.

---

## 2. The submission in five sentences

An interactive assistant that accepts single optical/SAR imagery, co-registered optical–SAR pairs, or bi-temporal pairs, plus a free-text question, and answers it by orchestrating a registry of small remote-sensing-adapted specialist models. Adaptation runs on **two tracks** — a band-agnostic multi-sensor encoder adapted on BigEarthNet at 10 m, and high-resolution object-level instruction tuning — bridged by band-presence masking, random band dropout and GSD-conditioned augmentation, because the mandated training data and the ISRO evaluation data are 10–20× apart in ground sample distance. Orchestration is a **constrained planner over a version-controlled capability matrix**, not free-form LLM tool-calling, giving a provable illegal-plan rate of zero — which matters because the PS grades the observable trace and explicitly does not grade internal reasoning. Every neural output is independently checked by a **physics verifier** built from classical remote-sensing indices and SAR backscatter statistics, with documented fallbacks for the fact that Cartosat-2S has no SWIR band and RISAT's frequency is unconfirmed. Confidence is calibrated and reported as three separate components, quantitative answers come from deterministic computation rather than generation, prose passes an entailment gate against structured evidence, and every answer ships a georeferenced evidence pack an analyst can open in QGIS.

---

## 3. PS requirement traceability matrix

**The matrix lives in [`docs/00-README-and-Requirement-Traceability.md`](docs/00-README-and-Requirement-Traceability.md) §3, and that is the only authoritative copy.**

It used to be duplicated here verbatim. That is how a traceability matrix goes
stale: the copy nobody edits is the copy a reader trusts. One matrix, one
place, refreshed 2026-08-30 with measured results and a status per row.

Current headline status (see `docs/00` §3.1 for the evidence behind each):

| | Requirement | Status |
|---|---|---|
| M1 | RS adaptation of a visual/VL component | **MET** |
| M2 | Single-image VQA *(mandatory)* | **MET** |
| M3 | Captioning **or** grounding | **MET (weak)** — both built; grounding near-floor |
| M4 | Bi-temporal change description **or** change-VQA *(mandatory)* | **MET** |
| M5 | Change map *(optional)* | **MET** |
| M6 | Cross-modal optical + SAR extraction | **MET (negative)** — fusion does not beat optical alone |
| M7 | Agentic orchestration | **MET** — illegal-plan rate 0/600; routing accuracy weak |
| M8 | Auditable execution summary | **MET** |

| M9 | Combine outputs, estimate confidence, return visual evidence | **VERIFIED** |

Inputs I1, I4 and I5 are **VERIFIED**; **I2 and I3 are PARTIAL**. The pair
configurations are detected, and since 2026-08-30 the capability matrix's
`min_overlap_pct` and `min_bands_optical` gates are **enforced** — a pair of
images of different places now abstains instead of being fused. Two gates stay
deliberately unenforced on evidence: `max_coreg_shift_px`, because the
cross-modal shift estimator reports 38.1 px on a pair with identical
footprints, and `require_dates`, because enforcing it would refuse the
prescribed benchmarks, which ship undated (limitation L16). I6 (large scenes)
is built but is **not a PS clause**. **All three prescribed benchmarks are now
evaluated** — RSVQA-LR on its official test split, CDVQA on official test1,
and VRSBench zero-shot (updated 2026-09-07; the earlier claim that VRSBench's
imagery "lives in DOTA, not on disk" was false — the HuggingFace repo hosts it
directly, see L11). Twenty-eight known limitations are recorded in `docs/00` §3.6 —
open and closed alike, with dates — rather than left to be discovered.

The matrix was checked clause-by-clause against
[`docs/ps-26167.md`](docs/ps-26167.md), the authoritative PS text, on
2026-08-30.

### Model availability — 2026-08-31

**The trained checkpoints were deleted on 2026-08-30 and recovered on
2026-08-31** from a Windows volume shadow copy: 4.542 GB, 136 files, verified
bit-exact against a SHA-256 recorded from the live file before the deletion.
All 61 `.pt` files load, and every `metrics.json` matches the numbers
published in `docs/model-cards.md`. **No number was re-derived or adjusted at
any point.** The account is `docs/00` §3.6 **L26**; the root cause and its
containment are **L27**.

**Update 2026-09-07 — the three are no longer unavailable, and every learned
tool answers.** All seven `is_available()` checks return ready against the
checkpoints on disk, and two real queries were put through the controller end
to end: a caption (`SINGLE_CAPTION`, confidence 0.702 MEDIUM) and a grounding
query (`SINGLE_GROUND`, confidence 0.402 LOW — consistent with that head's
measured Acc@0.5 of 0.0762). The three components below are exactly the three
sidecars **L29** records as repaired on 2026-08-31, each validated by
reproducing its published metric: `caption/vocab.json`, `grounding/vocab.json`
and `track_a_full_multires/band_stats.json`.

**Superseded, kept because it was true when written (2026-08-31).** Twelve
small JSON sidecars came back as NUL bytes — their size reached the volume,
their contents did not — which left `caption_v1` and `grounding_v1` on their
stubs and the Track A *multires* variant unnormalisable (the *base* variant
was unaffected). See **L29**, and **L30** for the availability check that used
to report such a tool as ready and then fail inside its loader.

Everything loads and answers: `landcover_v1` (base), `caption_v1`,
`grounding_v1`, `change_mask_v1`, `change_caption_v1`, `optsar_fusion_v1`,
`change_vqa_v1` (semantic path), and the Track B QLoRA adapter.

## 4. Unverified claims register

No network access was available while writing these documents. Most items below remain unconfirmed. **Update 2026-08-27:** the user supplied the BigEarthNet.txt paper (`2603.29630v2`) and the dataset's HuggingFace card directly, so **item 1 is now fully verified** — contents *and* licence (**CDLA-Permissive-1.0**, a permissive open-data licence; see the row and `03` §4.1). The remaining load-bearing unknowns are items 2 (Cartosat SWIR) and 3 (RISAT band), both needing a real Bhoonidhi product.

| # | Claim | Status | Consequence if false |
|---|---|---|---|
| 1 | BigEarthNet.txt at `txt.bigearth.net`: 464,044 co-registered S1/S2 pairs, 9.6 M annotations incl. captions + VQA (binary + MCQ) + referring expressions; benchmark subset 1,082 pairs / 15,029 annotations; arXiv 2603.29630 | **VERIFIED 2026-08-27 — paper + HF card; licence = CDLA-Permissive-1.0** | Fully resolved; licence permits use. Download = 467 MB Parquet (9,553,962 rows = 1 per annotation, **text only**); S1/S2 imagery is a separate reBEN pull (large — size storage around it). Format-only fallback if that pull is impractical: BigEarthNet v2 + GeoChat-Instruct + VRSBench; Track A unaffected |
| 2 | Cartosat-2S MX is 4-band VNIR with **no SWIR** → MNDWI and NDBI unavailable | **Assumption, high confidence** | **Load-bearing.** If SWIR exists it is pure upside — enable both index paths |
| 3 | The RISAT in the ISRO set: RISAT-1 (C-band, ~3–50 m) vs RISAT-2B/2BR1 (X-band, ~0.35–4 m) | **Unknown** | **Load-bearing.** Already mitigated by adaptive rather than absolute σ⁰ thresholds |
| 4 | CROMA / DOFA checkpoints downloadable under a permissive licence | Unverified | Fall back to torchgeo SSL weights |
| 5 | Change-Agent / LEVIR-MCI weights available | Unverified | TinyCD + separate caption head, ~1 extra GPU-h |
| 6 | Qwen2.5-VL-3B is still the best ≤4B VLM in Aug 2026 | **Open — knowledge ends May 2025.** Note: the BigEarthNet.txt paper (early 2026) built its RS baseline on **InternVL3-1B** — evaluate it as an alternate | Substitute per the six criteria in `03` §2.1; drop-in by design |
| 7 | SpaceNet 6 / Umbra / Capella high-res SAR accessible and licensed | Unverified | Stage A3 runs optical-only; document the limitation |
| 8 | GeoChat-7B / RS-LLaVA / LHRS-Bot downloadable for zero-shot baselines | Unverified | Baseline against the un-finetuned base VLM only |
| 9 | SIH 2026 calendar: internal deadline, finale dates, submission format | **Unknown** | Compress the `04` phase plan proportionally, cutting from the top of the descope ladder |
| 10 | Published benchmark numbers cited anywhere in these docs | Unverified | Treat every number as directional until the harness produces your own |

**Rule for the team: no GPU-hour is spent on any path whose enabling claim is still unverified.** The full 12-item gate with owners and deadlines is in `03` §6.

---

## 5. The first two weeks, as a checklist

Copy this into your tracker. Nothing here needs a GPU except item 11.

**Week 0**

- [ ] Run the 12-item verification gate (`03` §6); write every answer into `docs/verification.md`
- [ ] Register for **ISRO Bhoonidhi**; download real Cartosat-2S MX + PAN and RISAT products
- [ ] **Open a real Cartosat product and read its band list from the metadata** — resolves register item 2
- [ ] **Determine which RISAT and which mode** — resolves register item 3
- [ ] Confirm the SIH 2026 timeline and submission format — resolves register item 9
- [ ] Repo scaffold, `docker-compose.yml`, `Makefile`, CI, pre-commit
- [ ] **Freeze the Pydantic contracts** (`InputManifest`, `Plan`, `ToolResult`, `Trace`); tag `contracts-v1`
- [ ] Write ADR 001: LangGraph vs hand-rolled executor. **Decide once.**
- [ ] Inventory GPU accounts and quotas; start the weekly hour-tracking sheet
- [ ] Assign the six roles (`04` §2)

**Week 1**

- [ ] **Stub all nine tools** with schema-valid fake data; full pipeline runs end to end on stubs
- [ ] `capability_matrix.yaml` v1 with all nine tasks; `satquery matrix --validate` wired into CI
- [ ] `scripts/fetch_datasets.py` — download and mirror every P0 dataset to shared storage
- [ ] `scripts/fetch_models.py` — download and **sha256-verify** base checkpoints; `HF_HUB_OFFLINE=1` smoke test
- [ ] **Prove a T4 QLoRA run starts, checkpoints, and resumes after a deliberate kill** — do not skip this
- [ ] Synthetic query bank: ~60 templates per task, expanded to 3–5 k paraphrases
- [ ] Frontend shell running against stubs: upload, OpenLayers viewer, trace panel

**Gate to leave Phase 0:** stubbed pipeline produces a valid trace end to end; every verification item answered in writing; a training run has been resumed successfully.

---

## 6. The six decisions that define this submission

If you remember nothing else from these documents, remember these. Each is a deliberate choice against a plausible alternative, and each is defensible out loud.

1. **Two adaptation tracks, bridged.** Because BigEarthNet is 10 m and Cartosat-2S is ~1.6 m, and no single-track model spans that gap. The two-track ablation is the empirical proof, not a claim.

2. **A constrained planner, not a free-form LLM agent.** Because the PS grades the observable trace and explicitly does not grade reasoning, so determinism plus a provable illegal-plan rate of zero strictly dominates. The LLM is retained as a tie-breaker inside an already-legal set.

3. **Physics verifies neural, not the reverse.** Classical indices and SAR statistics are the independent referee — including a SAR-primary built-up path because **Cartosat-2S has no SWIR**, and adaptive rather than absolute σ⁰ thresholds because **RISAT's frequency is unconfirmed**. This is the trustworthiness story, and it costs zero GPU.

4. **Deterministic computation for every number; generation only for prose, and gated.** Counts come from detections, areas from the affine transform, increase/decrease from a signed subtraction against a stated significance threshold. Prose is generated once, from structured facts, and each sentence must be entailed by the evidence.

5. **Contracts in week 1 so four non-ML members never wait on a GPU.** Nine stub tools returning schema-valid data means the API, the executor, the matrix, the frontend, the trace, the evidence pack, the eval CLI and the golden tests are all built in parallel from week 2. This is the decision that plays to the team's real strength and it is the one that prevents the standard SIH failure.

6. **Breadth before depth, always.** Five mandatory areas, normalised scores. Every mandatory capability produces a real answer by W9 before anything is optimised. One zero costs more than any amount of polish gains.

---

*End of index. Begin with `01-Solution-Architecture-and-System-Design.md`.*
