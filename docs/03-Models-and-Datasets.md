# SatQuery AI — Model & Dataset Recommendations

**PS 26167 · ISRO / Department of Space · SIH 2026**
Document 3 of 6 · Written 2026-08-27

> **Verification notice.** This document was written without network access, so no checkpoint availability, licence, benchmark number, or paper claim below could be independently confirmed. Every item marked **[VERIFY]** must be checked by a human in week 0 before any compute is spent on it. Items marked **[VERIFY-HARD]** are load-bearing — if they turn out false, the plan changes. Knowledge here is reliable to roughly **May 2025**; it is now August 2026, so assume at least one newer small VLM exists that beats the named ones. §2.1 gives selection *criteria* precisely so that a better model can be substituted without redesigning anything.

---

## 1. The compute reality, stated first

Every model decision below is downstream of this. The team has **free-tier Colab and Kaggle only — T4 (16 GB) and occasionally P100 (16 GB)**, one or two members with hands-on fine-tuning experience.

### 1.1 Hardware constraints that are not optional

| Constraint | Consequence | Non-negotiable action |
|---|---|---|
| T4 is Turing (sm75), P100 is Pascal (sm60) | **No bf16. No FlashAttention-2** (needs Ampere sm80+). Anyone who copies a bf16 recipe from a blog post will waste a day on cryptic errors. | `bnb_4bit_compute_dtype=torch.float16`; `attn_implementation="sdpa"`; fp16 + `GradScaler` + gradient clipping at 1.0 |
| fp16 is less numerically forgiving than bf16 | Loss spikes and NaNs mid-run | Aggressive clipping, warmup, `eps=1e-6` on AdamW, and **checkpoint every 200 steps** so a NaN costs minutes not hours |
| Kaggle sessions hard-stop at 12 h; Colab kills idle sessions | An 18-hour training run cannot exist | **Every training script is resumable from day one** — save adapter + optimiser + scheduler + step index; `--resume` flag; test the resume path in week 2 before you need it |
| Kaggle offers **2×T4** | ~1.7× throughput via `accelerate launch` DDP | Prefer Kaggle 2×T4 over a single P100 for anything long |
| P100 has no tensor cores | fp16 gives far less speedup than on T4 | Use P100 for data prep, evaluation, and CPU-adjacent work; use T4 for training |
| Free-tier quota | ~30 h/week Kaggle GPU + variable Colab ≈ **35–45 GPU-h/week aggregate** across accounts | Budget it; do not discover the ceiling in week 10 |

### 1.2 The budget arithmetic

Roughly **500 GPU-hours** available across a 14-week runway. Estimated need: **55–95 GPU-h for a first complete pass** across the four main trainings plus the small change-VQA head, **150–220 GPU-h with realistic iteration and failed runs**. That is comfortable — *if and only if* the three rules below hold.

**Rule 1 — Nothing above 3B parameters is ever fine-tuned.**

The arithmetic: a 7B model in 4-bit is ~3.5 GB of weights. QLoRA training adds gradients, optimiser state, and activations. Vision-language models make it worse because image tokens dominate the sequence — a high-resolution image can produce 1,000–2,000 visual tokens, and attention memory grows with sequence length. On a 16 GB T4 a 7B VLM QLoRA run at useful resolution sits at batch size 1 with gradient checkpointing and still risks OOM. Even when it fits, throughput is so low that a single epoch consumes days.

7B models (GeoChat-7B, RS-LLaVA, LHRS-Bot, EarthGPT) are therefore **zero-shot inference baselines for the comparison table**, never training targets. Reporting "our 3B fine-tune beats GeoChat-7B zero-shot on RSVQA" is a *stronger* result than a half-trained 7B, and it costs a tenth of the compute.

**Rule 2 — Visual token count is a first-class hyperparameter.**

Cap it explicitly. For Qwen2.5-VL, `max_pixels = 512 * 28 * 28` bounds visual tokens at ~512, which is the difference between fitting on a T4 and not. Large scenes are handled by the tiling pyramid in doc `01` §2.7, not by raising the token budget.

**Rule 3 — Four trainings, not nine.**

| Training run | Produces | Est. GPU-h (first pass) |
|---|---|---|
| **A. Band-agnostic encoder** on BigEarthNet | `landcover_v1` + `optsar_fusion_v1` (shared encoder) | 15–25 |
| **B. VLM instruction tuning** (QLoRA) | `rs_vqa_v1` + `caption_v1` (shared base, two adapters) | 20–35 |
| **C. Change model** (multi-task mask + caption) | `change_mask_v1` + `change_caption_v1` | 10–18 |
| **D. Grounding fine-tune** | `grounding_v1` | 8–14 |
| *(E. Change-VQA head — small, often folded into C)* | `change_vqa_v1` | 2–4 |
| `index_engine_v1` | deterministic, no training | **0** |

Weight sharing is not a compromise; it is what makes nine tools affordable.

---

## 2. The two-track adaptation strategy

This is the central technical argument of the submission, and no competing analysis reviewed for this project made it. It exists because of a hard fact: **the mandated adaptation data and the actual evaluation data are 10–20× apart in ground sample distance.**

### 2.0 The gap, quantified

| | BigEarthNet (the PS's dataset link) | ISRO/SAC private set |
|---|---|---|
| Optical | Sentinel-2, 12 bands incl. SWIR | **Cartosat-2S**: PAN ~0.65 m (1 band), MX ~1.6 m (**4-band VNIR, no SWIR**) |
| SAR | Sentinel-1 C-band, VV+VH, 10 m | **RISAT** — C-band ~3–50 m (RISAT-1) or X-band ~0.35–4 m (RISAT-2B/2BR1) **[VERIFY-HARD]** |
| Patch geometry | 120×120 px @ 10 m = 1.2 km tile | Full scenes, sub-metre to few-metre |
| Semantic level | land-cover classes over a 1.2 km tile | individual aircraft, buildings, vehicles |

At 10 m GSD an aircraft is a fraction of one pixel. A model trained only on BigEarthNet cannot answer *"how many aircraft are visible"* — not because it is undertrained, but because the concept is unrepresentable at that scale. Conversely, a model trained only on high-resolution RGB benchmarks has never seen SAR, never seen 12-band multispectral, and cannot satisfy the optical–SAR mandate.

**Therefore: two tracks, plus an explicit bridge.**

### 2.1 Track A — Band-agnostic multi-sensor encoder

**Purpose:** satisfies the PS's adaptation mandate literally (it uses the linked dataset), and carries land-cover, multi-sensor handling, and optical–SAR fusion.

**Base model candidates, in preference order:**

| Candidate | Why | Risk |
|---|---|---|
| **CROMA** | Contrastive optical–SAR masked autoencoder; pretrained *jointly* on Sentinel-1+2, which is exactly the cross-modal structure required | **[VERIFY]** checkpoint availability and licence |
| **DOFA** | Wavelength-conditioned dynamic weights — natively handles arbitrary band sets, which is precisely the 12-band → 4-band problem | **[VERIFY]** checkpoint + licence |
| **Prithvi-2.0 / TerraMind** (IBM–NASA / ESA) | Strong geospatial foundation models, TerraTorch tooling | **[VERIFY]** size fits T4; some variants are large |
| **SatMAE / Scale-MAE** | Well-established MAE-style RS pretraining; Scale-MAE explicitly scale-aware | Optical-only; needs a separate SAR stream |
| **torchgeo ResNet50/ViT SSL weights** | **The safe fallback.** Ships in torchgeo, small, reliable, Sentinel-pretrained | Weaker than the above, but *guaranteed to work* |

Decide this in week 0 after verifying availability. The fallback row exists because a plan that depends on an unverified checkpoint is a plan with a single point of failure — **if none of the top four verify by end of week 1, go with torchgeo weights and move on.** A working weaker encoder beats a perfect one you never downloaded.

**Adaptation recipe:**

- Dual-stream architecture: optical stream + SAR stream, fused by cross-attention (with a `concat` mode as the ablation baseline and the permitted-parameter alternative).
- **Band-presence masking:** map every input onto the canonical band vocabulary from doc `01` §2.6 with a presence mask; missing bands are masked, not zero-filled. Zero-filling teaches the model that "absent" means "zero reflectance," which is actively wrong.
- **Random band dropout during training** (p ≈ 0.3 per band, always retaining at least 3): forces the encoder to work with arbitrary subsets. **Measured 2026-08-29 (Track A v0) - this claim was too strong.** An ablation on the official BigEarthNet splits found band dropout raises 4-band retention from 84.4% to 90.2%, a real but modest gain, and it costs 0.0139 mAP at full bands. The control, trained with every band present, still worked at 4 bands. The actual enabler is the **masked-mean band-agnostic architecture** - a fixed-channel convolution could not run on 4 bands at all. There are two mechanisms, not one. See docs/phase1-status.md; note the result is single-seed and not yet established as significant.
- **GSD-conditioned scale augmentation:** random rescaling over roughly 0.5×–4× with the effective GSD supplied as a conditioning input (a scalar embedding, or FiLM). The model learns that scale is a variable, not a constant.
- Heads: multi-label land-cover classification (BigEarthNet's 19 or 43 classes) + a lightweight segmentation decoder (UPerNet or a simple FPN).

**The resolution ladder** — the bridge itself, run as three short stages rather than one long one:

```
Stage A1: BigEarthNet, 10 m, S1+S2          -> multi-sensor + land-cover + optical-SAR fusion  (~10-15 GPU-h)
Stage A2: WHU-OPT-SAR, ~5 m, optical+SAR     -> mid-resolution transfer                          (~3-6 GPU-h)
Stage A3: high-res optical (+ any 1 m SAR)   -> approach Cartosat/RISAT scale                    (~2-4 GPU-h)
```

Stage A3's data is the weak link, because openly-licensed sub-metre SAR is scarce. Options: **Umbra Open Data** (~0.25–1 m, permissive) and **Capella Open Data** **[VERIFY]** licences and current availability; **SpaceNet 6** (Rotterdam, optical + Capella X-band SAR, ~0.5 m) is the best-known fit if it is still accessible **[VERIFY]**. If Stage A3 SAR data cannot be obtained, run A3 on high-resolution *optical* only and rely on the GSD conditioning plus band dropout to carry the SAR stream. State that limitation openly in the report — a documented, quantified limitation reads as competence; a hidden one reads as an oversight when a judge probes it.

**Also, in week 1:** register for **ISRO Bhoonidhi** and download real Cartosat-2S and RISAT products. Two people, one afternoon. This gives you actual target-sensor imagery for calibration, threshold tuning, qualitative demo material, and — most importantly — verification of Axiom 2 (band composition) and the RISAT band question, both of which currently sit on assumption. There is no substitute for opening the real product's metadata and reading what bands it actually has.

### 2.2 Track B — High-resolution object-level instruction tuning

**Purpose:** VQA, captioning, grounding, and counting at the scale where objects are individually visible.

**Base VLM — criteria before names**, because reliable knowledge here ends May 2025 and it is now August 2026:

1. **≤ 4B parameters** (Rule 1).
2. **Dynamic / high native resolution** with a controllable visual-token budget (Rule 2).
3. **Native grounding output format** — the model should already speak bounding boxes, so you fine-tune a capability rather than invent one.
4. **First-class PEFT/LoRA support** in `peft` with a documented target-module list.
5. **Permissive licence** (Apache-2.0 or similar), because the PS requires the models themselves as deliverables.
6. **Turing-compatible attention** — works with `sdpa`, does not require FA2.

**Candidates as of the May-2025 knowledge horizon:**

| Candidate | Why | Risk |
|---|---|---|
| **Qwen2.5-VL-3B-Instruct** | Meets all six criteria: dynamic resolution, native bbox/point grounding, excellent `peft` support, Apache-2.0, strong OCR/detail | **Primary recommendation.** Check for a newer Qwen-VL small variant first **[VERIFY]** |
| **Florence-2-large (0.77B)** | Purpose-built for detection/grounding/region tasks; tiny; trains fast | Weaker at free-form VQA — use it for `grounding_v1`, not for VQA |
| **InternVL2/2.5-2B**, **Phi-3.5-vision (4.2B)**, **PaliGemma-3B** | Credible alternates | Verify licence and grounding format **[VERIFY]** |
| **GeoChat-7B / RS-LLaVA / LHRS-Bot / EarthGPT / EarthDial** | Already RS-domain-tuned | **Inference-only baselines.** Too large to fine-tune here (Rule 1) **[VERIFY]** which exist and are downloadable |

**Recommended split:** Qwen2.5-VL-3B for `rs_vqa_v1` + `caption_v1` (two LoRA adapters, one base). **Florence-2 for `grounding_v1`** — it is 0.77B, purpose-built for region tasks, and trains in hours. Using the right small specialist for grounding rather than forcing the VQA model to do it is both cheaper and better.

**QLoRA recipe (T4-safe):**

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,     # NOT bfloat16 on T4/P100
)
LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
# Vision tower: freeze initially. Unfreeze the last 2-4 blocks in a second short stage
# only if RS-domain gains have plateaued -- this is where OOM comes from, so gate it.

TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,           # effective batch 16
    gradient_checkpointing=True,
    fp16=True,
    optim="paged_adamw_8bit",
    learning_rate=1e-4, warmup_ratio=0.03, lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    save_steps=200, save_total_limit=3,       # 12-hour session insurance
    logging_steps=10,
    attn_implementation="sdpa",
)
# max_pixels = 512 * 28 * 28   -> caps visual tokens at ~512
```

**Instruction-mix composition** matters as much as the hyperparameters. Mix, per epoch: single-image VQA (~35 %), captioning (~20 %), referring/grounding (~20 %), land-cover description grounded in `index_engine` statistics (~10 %), change description and change-VQA (~10 %), and **explicit refusal/abstention examples (~5 %)**.

That last 5 % is not filler. If you never train the model to say "the image does not show this," it will never say it, and your abstention mechanism will be fighting the model instead of expressing it. Generate refusal examples programmatically: ask about objects verified absent from the annotations, ask about SWIR-dependent properties on a 4-band image, ask change questions over a single image.

**Also mix in SAR and pseudo-RGB SAR samples** even for the VQA model. A VQA model that has literally never seen speckle will produce confident nonsense the first time it does, and it *will* see SAR during evaluation.

### 2.3 Where the tracks meet

They are not merged into one model — they are two tools the agent orchestrates, which is exactly what the PS's "predefined registry of specialised models/tools" describes. The bridges between them:

- **Shared canonical band vocabulary and presence masking** so both consume the same normalised representation.
- **`index_engine_v1` as a common grounding signal** — both tracks receive deterministic index statistics as context, so both are anchored to the same physical measurements.
- **GSD conditioning in both**, so both know what scale they are looking at.
- **The verifier checks both** against the same physics, producing comparable agreement scores.

---

## 3. Model recommendations per tool

| Tool | Primary | Fallback (if primary fails/unavailable) | Training | Risk |
|---|---|---|---|---|
| `rs_vqa_v1` | Qwen2.5-VL-3B + QLoRA on RS instruction mix | Florence-2 VQA head; or Qwen2-VL-2B | Run B, 20–35 h | Medium — main quality driver |
| `caption_v1` | Same base, caption LoRA | Same | Run B (shared) | Low |
| `grounding_v1` | **Florence-2-large** fine-tuned on DIOR-RSVG + VRSBench referring | Qwen2.5-VL grounding mode; or a plain YOLOv8/DINO detector for closed classes | Run D, 8–14 h | Medium — bbox mAP is the visible metric |
| `landcover_v1` | Track-A encoder + multi-label head + FPN decoder | torchgeo ResNet50 SSL + linear head | Run A | Low — most reliable component |
| `optsar_fusion_v1` | Track-A encoder, dual-stream + cross-attention | `concat` fusion; ultimate fallback `index_engine_v1` | Run A (shared) | Medium — PS-mandatory, so no gaps allowed |
| `change_mask_v1` | **Change-Agent / LEVIR-MCI** multi-task (mask + caption in one) **[VERIFY]** | **TinyCD** (~0.3 M params, trains in ~1 h) or ChangeFormer-b0 | Run C, 10–18 h | Low — TinyCD is a genuinely strong cheap fallback |
| `change_caption_v1` | Change-Agent MCI captioning branch | RSICCformer / Chg2Cap on LEVIR-CC | Run C (shared) | Medium |
| `change_vqa_v1` | Classification head on CDVQA, conditioned on the change mask | Template answers from mask statistics (**always available**) | Run E, 2–4 h | Low — the template path guarantees a floor |
| `index_engine_v1` | NumPy: NDVI, NDWI, MNDWI, NDBI, σ⁰, VH/VV, GLCM, CoV, adaptive thresholding | — | **None** | **Zero** |

Two notes worth internalising.

**`change_vqa_v1` cannot score zero**, because the template path from mask statistics always produces an answer. Given that change-VQA is one of the two PS-mandatory temporal capabilities, guaranteeing a non-zero floor there is disproportionately valuable under normalised scoring.

**Change-Agent / LEVIR-MCI is attractive because it produces mask and caption from one model** — that is two mandatory-adjacent capabilities from one training run. But it is **[VERIFY]**, and if it does not materialise, TinyCD plus a separate caption head is a perfectly respectable path that costs about one extra GPU-hour. Do not let an unverified dependency sit on the critical path; decide by end of week 2.

---

## 4. Datasets

### 4.1 The mandated dataset

The PS's dataset link is `https://txt.bigearth.net` with arXiv reference `2603.29630` — a **text-annotated extension of BigEarthNet**. The paper (`2603.29630v2`, read directly) confirms: 464,044 co-registered Sentinel-1/Sentinel-2 patch pairs, ~9.6 M text annotations spanning **captions, VQA pairs (binary + MCQ), and referring expressions**, plus a manually-verified benchmark subset of 1,082 pairs and 15,029 annotations.

**VERIFIED 2026-08-27** — the full paper (`2603.29630v2`) was read directly. Every headline figure is exact: **464,044** co-registered S1/S2 pairs, **9.6 M** annotations (captions + binary VQA + MCQ + referring-expression *and* point detection), benchmark split of **1,082 pairs / 15,029 annotations** drawn from the test split and manually verified on four quality criteria. Four facts from the paper that shape how we use it:

- **It is European-only and single-timestamp.** Images come from BigEarthNet v2.0 (reBEN) over ten European countries; there are **no bi-temporal / change pairs**. It therefore fully serves M1, M2, M3 and M6 but **does nothing for M4/M5** — the change datasets (LEVIR-CC, CDVQA, LEVIR-CD) stay mandatory, and the Bhoonidhi Indian-context set remains the only in-region data we will have.
- **Bands: 12-band S2 + S1, SWIR present.** Confirms the training side of Axiom 2. Their own model uses only the **10 m + 20 m bands (10 bands; the three 60 m atmospheric bands are dropped)** — a sensible selection to mirror for Track A. SWIR (B11/B12, both 20 m) is included, so our **band-dropout curriculum** (train with SWIR, degrade gracefully to Cartosat's 4-band VNIR) is exactly right and now empirically grounded, not just asserted.
- **The benchmark VQA uses deliberate hard negatives** — "no" answers where the queried class is present but the stated count/size is wrong, and presence/adjacency negatives drawn from semantically similar CORINE classes. Our VQA tool must beat class-absence heuristics, not merely detect presence; evaluate against exactly this.
- **Mirror their metrics** so our numbers sit directly beside theirs: captioning = BLEU-4 / ROUGE / METEOR / CIDEr / BERTScore / SBERT-Cosine / CLAIR; binary VQA and MCQ = accuracy; referring = mIoU + Acc@{25,50,75,90}.

**Targets to beat (their benchmark split, %).** Off-the-shelf VLMs are *weak* here — that is the paper's whole point — and their fine-tuned 1 B baseline crushes them, which is our thesis in a single table:

| Model | Caption BLEU-4 | Binary VQA | MCQ | Referring mIoU |
|---|---|---|---|---|
| Best SOTA RS VLM (7B / 4B) | 1.66 | 58.38 | 35.26 | 16.18 |
| Best SOTA CV VLM (incl. GPT-5.2, 2 T) | 0.96 | 61.96 | 37.55 | 31.73 |
| **RS-InternVL** (their 1 B, LoRA + S1/S2 branches) | **34.04** | **73.29** | **51.49** | **65.84** |

A **1 B** model with **5.8 M** trainable parameters, fine-tuned on this dataset, beats a **2 T** model by wide margins (**+31.5 % average**) — the strongest external evidence in existence for "adapt a small VLM, don't prompt a giant one." **Cost caveat that constrains us:** they spent **~2 days on 4× H200** on that 1 B model over the full train+val set. On free-tier T4/P100 we cannot match that data scale, so our lever is **aggressive subsampling** of the instruction mix (tens of thousands of examples, not millions). Also check whether they release the RS-InternVL checkpoint — if so, build on it rather than from the base InternVL.

The implication is significant, and now confirmed rather than assumed: **BigEarthNet.txt alone can carry both Track A encoder adaptation *and* Track B VLM instruction tuning**, and it does so using the PS's own linked dataset, which is the cleanest possible answer to "did you adapt on the specified data?" The 120×120 @ 10 m geometry still forces Track B's high-resolution component to exist — you cannot learn to count aircraft from it — but it becomes the backbone of the instruction mix rather than just the encoder's diet.

**Week-0 action — licence already confirmed.** The HF dataset card (`BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`) states the licence as **CDLA-Permissive-1.0** — a fully permissive open-data licence: use, modify and redistribute freely, no non-commercial or share-alike clause; the only obligation is preserving the attribution notice *if you redistribute the raw data*, and derived results (including trained models) carry no obligation at all. So there is no licence risk to the adaptation mandate. What remains is mechanical, and it hides the real storage cost: the HF repo is **467 MB of Parquet, 9,553,962 rows = one row per annotation, text only** — the rows carry the query/answer text keyed to a patch (`ID`, `s1_name`, `patch_id`, `input`, …). **The imagery is not in this file.** The actual Sentinel-1/Sentinel-2 rasters must be pulled separately from BigEarthNet v2.0 / reBEN (tens–hundreds of GB), so size the fetch script and shared storage around the imagery, not the 467 MB of text. Everything else in §4.2 is sized around this.

### 4.2 Full dataset plan

| Dataset | Provides | Used for | Size | Priority |
|---|---|---|---|---|
| **BigEarthNet.txt** | S1+S2 pairs + captions/VQA/referring @10 m | Track A + Track B backbone; **the adaptation mandate** | text 467 MB (Parquet); **+ reBEN imagery, large** | **P0** |
| **BigEarthNet v2 (base)** | S1+S2 patches, 19/43 land-cover labels | Track A land-cover head | ~66 GB (S2) | **P0** |
| **VRSBench** | High-res VQA + captions + referring, ~29 k images | Track B; **prescribed eval split** | moderate | **P0** |
| **RSVQA-LR / RSVQA-HR** | VQA @ Sentinel-2 and high-res aerial | Track B; **prescribed eval split** | moderate | **P0** |
| **CDVQA** | Change-VQA over bi-temporal pairs | `change_vqa_v1`; **prescribed eval split** | small | **P0** |
| **LEVIR-CC** | ~10 k bi-temporal pairs with change captions | `change_caption_v1` | moderate | **P0** |
| **LEVIR-CD / WHU-CD / S2Looking / SECOND** | Bi-temporal change masks | `change_mask_v1`; SECOND adds semantic change | moderate | **P0** |
| **DIOR-RSVG** | ~38 k referring expressions with boxes, high-res | `grounding_v1` | moderate | **P0** |
| **WHU-OPT-SAR** | ~5 m co-registered optical+SAR, land-cover labels | **Stage A2 — the resolution bridge** | ~10 GB | **P0** |
| **SEN12MS** | ~180 k S1/S2 triplets, global | Track A augmentation if BigEarthNet.txt disappoints | ~500 GB (subset it) | P1 |
| **GeoChat-Instruct** | ~318 k RS instruction pairs | Track B instruction diversity | moderate | P1 |
| **SpaceNet 6** | Rotterdam optical + Capella X-band SAR @ ~0.5 m | **Stage A3 — high-res SAR** | moderate | P1 **[VERIFY]** |
| **Umbra / Capella Open Data** | 0.25–1 m X-band SAR | Stage A3; RISAT-like appearance | curated subset | P1 **[VERIFY]** |
| **DOTA / FAIR1M / xView** | High-res object detection, many small objects | Grounding + counting robustness | large — subset | P2 |
| **ISRO Bhoonidhi** (Cartosat-2S, RISAT) | **Real target-sensor products** | Threshold calibration, qualitative demo, Axiom-2 verification | small curated set | **P0 — week 1** |

Budget roughly 150–250 GB of working storage after subsetting. **Write `scripts/fetch_datasets.py` to download and mirror everything to a shared Drive/S3 location in week 1**, because free-tier notebook filesystems are ephemeral and re-downloading 60 GB in week 9 is a wasted day. Store as WebDataset shards or pre-tiled `.npy`/`.pt` — the notebook GPU should never wait on decompression.

### 4.3 Splits — the leakage trap

BigEarthNet patches are tiled from larger Sentinel scenes, so **adjacent patches are near-duplicates**. A random split puts neighbouring patches on both sides and inflates validation accuracy by a wide margin. The same applies to LEVIR-CD tiles and to any tiled high-resolution dataset.

**Use geographic / spatial-block splits**: partition by scene ID, tile grid block, or S2 cell so that no two patches from the same source scene appear in different splits. Where the dataset publishes official splits (BigEarthNet, LEVIR-CD, VRSBench, RSVQA, CDVQA all do), **use the official ones** — you must anyway, since the PS says evaluation uses "prescribed public benchmark test subsets."

Additionally hold out a **cross-sensor generalisation split**: never train on the Bhoonidhi Cartosat/RISAT products. Keep them purely as a qualitative out-of-distribution check. A model that transfers to real ISRO imagery it never trained on is the most convincing evidence you can present to an ISRO reviewer, and it costs nothing to preserve.

---

## 5. Evaluation plan

| Capability | Dataset / split | Metrics | Baseline to beat |
|---|---|---|---|
| Single-image VQA | RSVQA-LR/HR test, VRSBench VQA test | Accuracy overall + per question type (presence, count, comparison, rural/urban) | GeoChat-7B zero-shot, base VLM zero-shot |
| Captioning | VRSBench caption test | BLEU-4, METEOR, ROUGE-L, CIDEr, SPICE | base VLM zero-shot |
| Grounding | DIOR-RSVG test, VRSBench referring | Acc@0.5, Acc@0.7, mIoU, mAP | Florence-2 zero-shot |
| Land-cover | BigEarthNet official test | mAP (micro/macro), F2, per-class F1 | torchgeo ResNet50 linear probe |
| Optical–SAR fusion | WHU-OPT-SAR test + BigEarthNet | mIoU, per-class IoU, **plus the complementarity triad** | optical-only and SAR-only ablations |
| Change mask | LEVIR-CD / WHU-CD test | F1, IoU, precision, recall | TinyCD published numbers |
| Change caption | LEVIR-CC test | BLEU-4, METEOR, CIDEr | RSICCformer published numbers |
| Change VQA | CDVQA test | Accuracy per question type | published CDVQA baselines |
| **Calibration** | all validation splits | **ECE, MCE, reliability diagram, before/after temperature scaling** | uncalibrated |
| **Abstention** | all test splits | **Risk–coverage curve, AURC, accuracy on answered subset** | always-answer |
| **Orchestration** | 200-query adversarial suite | **Routing accuracy, illegal-plan rate (target 0), abstention precision** | — |
| **Latency / throughput** | curated 50-item set | p50/p95 interactive latency, batch items/s | — |

The bottom four rows are where the submission separates. Almost no competing team will produce a reliability diagram, an AURC number, or an illegal-plan-rate table. Those are cheap to compute and disproportionately persuasive because they demonstrate you evaluated *the system*, not just the models.

### 5.1 Ablations to run and report

Four, and each one directly defends a design decision that a judge is likely to challenge:

1. **Two-track vs single-track** — train Track A only, and Track B only, then evaluate both on high-res object tasks and on optical–SAR tasks. This is the empirical justification for the whole architecture.
2. **Complementarity triad** — optical-only vs SAR-only vs fused, per class. This is *also* the PS-required proof of complementary extraction, so it does double duty.
3. **Agent vs monolith** — route everything to one VLM versus using the orchestrated registry. Expected result: the monolith is far worse on grounding, change masks and quantitative answers.
4. **Verifier on/off** — accuracy and calibration with and without the physics verifier and entailment gate. Quantifies the anti-hallucination claim instead of asserting it.

Run each ablation on a reduced split if compute is tight. A directional result on 2,000 items is worth far more than a perfect result you did not have time to produce.

---

## 6. Week-0 verification gate

Nothing below was confirmable when this document was written. **Assign each item an owner and a deadline before spending GPU time.** A plan built on unverified checkpoints is a plan with hidden single points of failure.

> **Update 2026-08-27:** the user supplied the BigEarthNet.txt paper (`2603.29630v2`) and the dataset's HuggingFace card, so items 1 and 2 are resolved (details in §4.1) — including item 1's licence, confirmed as **CDLA-Permissive-1.0** (permissive open data). The only plan-changing unknowns left are the **Cartosat SWIR band (item 6)** and the **RISAT frequency (item 5)** — both require opening a real Bhoonidhi product.

| # | Claim to verify | If false, do this instead | Owner | Deadline |
|---|---|---|---|---|
| 1 | **VERIFIED incl. licence 2026-08-27.** Paper read (464,044 pairs, 9.6 M annotations, captions + VQA + referring) **and** HF card confirms **licence = CDLA-Permissive-1.0** (permissive — usable), **467 MB Parquet, 9,553,962 rows = 1 per annotation, text only** (S1/S2 imagery is a separate reBEN download) | None on licence grounds. Format-only fallback if the reBEN imagery pull is impractical: BigEarthNet v2 + GeoChat-Instruct + VRSBench; Track A unaffected | ML lead | Done |
| 2 | **VERIFIED** — `2603.29630v2` read; every headline figure exact (464,044 / 9.6 M / 1,082-pair benchmark) | — | ML lead | Done |
| 3 | CROMA and/or DOFA checkpoints are downloadable with a permissive licence | Use torchgeo SSL weights (guaranteed fallback) | ML #2 | Day 4 |
| 4 | Change-Agent / LEVIR-MCI weights available | TinyCD + separate caption head | ML #2 | Day 7 |
| 5 | **Which RISAT and which mode ISRO/SAC will use (C-band vs X-band, look count)** | Support both; keep all σ⁰ thresholds adaptive (already the design) | Geo lead | Day 7 |
| 6 | **Cartosat-2S MX band composition — confirm 4-band VNIR with no SWIR** (open a real Bhoonidhi product and read the metadata) | If SWIR exists, enable MNDWI/NDBI paths — pure upside | Geo lead | Day 5 |
| 7 | Newer ≤4B VLM available (Aug 2026) that beats Qwen2.5-VL-3B on the six criteria in §2.1. *Note: the BigEarthNet.txt group used **InternVL3-1B** as their RS backbone in early 2026 — evaluate it as an alternate* | Substitute it; §2.1 criteria make this a drop-in | ML lead | Day 3 |
| 8 | SpaceNet 6 / Umbra / Capella high-res SAR accessible and licensed | Run Stage A3 optical-only; document the limitation | Geo lead | Day 10 |
| 9 | Prescribed benchmark test splits downloadable (VRSBench, RSVQA, CDVQA) | Use published splits from the papers' repos | Eval lead | Day 5 |
| 10 | SIH 2026 timeline: internal deadline, grand finale dates, submission format | Compress the phase plan in doc `04` proportionally | Team lead | Day 1 |
| 11 | Bhoonidhi registration approved; Cartosat-2S + RISAT products downloaded | Use any open Indian-context high-res imagery for qualitative work | Geo lead | Day 7 |
| 12 | GeoChat-7B / RS-LLaVA / LHRS-Bot downloadable for zero-shot baselines | Baseline against the un-finetuned base VLM only | ML #2 | Day 10 |

Items **5 and 6** (RISAT band, Cartosat SWIR) are now the only plan-changing unknowns left — item 1 is fully resolved (contents *and* licence: **CDLA-Permissive-1.0**, permissive). The rest have fallbacks that cost hours, not weeks.

---

*Continues in `04-Implementation-Plan.md`.*
