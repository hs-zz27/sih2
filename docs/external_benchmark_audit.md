# External benchmark audit — SatQuery AI against the strongest comparable remote-sensing VLMs

**Written 2026-09-03.** Research and evaluation task. **No model architecture, training code or inference code was changed to produce it, and no training was run.** The only additions to the repository are this document and its machine-readable companion, `docs/external_benchmark_results.json`.

---

> ## ⚠ CORRECTION BANNER — added 2026-09-07. Read before any CDVQA number below.
>
> **Every CDVQA figure in this document is 0.5380. The correct figure is 0.6061.** Nothing below has been edited: this document is a dated audit, and rewriting its numbers in place would destroy the record of what was measured on 2026-09-03. This banner supplies the corrected values instead.
>
> **Cause.** `docs/research/cdvqa-baseline-correction-2026-09-03.md` established by A/B across two commits that the 0.5380 run fed **channel-reversed (BGR) images** to the semantic-change head. That head is an **ImageNet-pretrained ResNet-18**, and ImageNet pretraining is RGB-specific, so channel reversal degrades it substantially. The pre-fix commit reproduces 0.5380 exactly on all eight question types and overall; the post-fix commit returns **0.606133**. The only variable changed was the code. **0.6061 is the first CDVQA measurement taken on correctly-ordered input.**
>
> ### Figures that change
>
> | Quantity | This document says | Corrected |
> |---|---|---|
> | CDVQA test1 overall accuracy | 0.5380 | **0.6061** |
> | Coverage | "100%" | **99.82%** (73 deferrals, all in `change_to_what`; the documented 100% was never reproducible at either commit) |
> | Deficit vs SOTA (Qwen3.5-2B, 0.7474) | −20.9 pts | **−14.1 pts** |
> | Deficit vs the 2021 CDVQA baseline (0.6590) | −12.1 pts | **−5.3 pts** |
> | Deficit vs VisTA (0.7310) | −19.3 pts | **−12.5 pts** |
> | Deficit vs SOBA (0.6920) | −15.4 pts | **−8.6 pts** |
> | Margin over the per-type majority constant (0.5084) | +3.0 pts | **+9.8 pts** |
> | Types where SatQuery beats its own per-type constant | 5 of 8 | **7 of 8** |
> | `largest_gaps[G1].delta` in the JSON companion | −0.2094 | **−0.1413** |
>
> ### Per-type, re-diffed against Qwen3.5-2B
>
> | question type | n | this doc (0.5380 run) | **corrected** | Qwen3.5-2B | corrected Δ |
> |---|---|---|---|---|---|
> | `change_or_not` | 13,882 | 0.6772 | **0.7093** | 0.8395 | −0.1302 |
> | `change_ratio_types` | 5,811 | 0.4791 | **0.5564** | 0.7986 | −0.2422 |
> | `decrease_or_not` | 4,658 | 0.6496 | **0.7548** | 0.8270 | −0.0722 |
> | `increase_or_not` | 4,600 | 0.6437 | **0.7178** | 0.8324 | −0.1146 |
> | `change_to_what` | 2,991 | 0.3714 | **0.5219** | 0.6172 | −0.0953 |
> | `largest_change` | 2,904 | 0.4497 | **0.6167** | 0.6446 | −0.0279 |
> | `smallest_change` | 2,904 | 0.1319 | **0.1522** | 0.3523 | −0.2001 |
> | `change_ratio` | 1,936 | 0.1952 | **0.1875** | 0.5755 | −0.3880 |
> | **overall** | **39,686** | **0.5380** | **0.6061** | **0.7474** | **−0.1413** |
>
> ### What the correction does *not* change
>
> **§1.1's verdict stands.** SatQuery still loses the CDVQA Category-A comparison to all four published comparators, and still **loses on all eight question types** — the gaps narrow, none closes, and the nearest (`largest_change`, −0.028) is the only one within striking distance. The audit's central claim, that the two Category-A comparisons are both losses, is unaffected. **§11 G1 stands too:** the cause is still the segmenter, and the oracle ceiling is still 0.9975.
>
> What *does* change is the margin over a constant. At 0.5380 the system beat a per-type constant by 3.0 points on five of eight types; at 0.6061 it beats it by 9.8 points on seven of eight. The "beats a constant by only 3.0 points" characterisation in §10.2 is superseded.

---

Read §1 and §8 first. If you read only one table, read the one in §6.1 — and read its *comparable?* column before you read its numbers.

**Two rules govern every number below.**

1. **Ours and theirs are always labelled.** A number marked *(measured here)* was produced by this repository's own evaluation code during this audit. A number marked *(published)* is quoted from a paper and was **not** reproduced by us. We ran no external model — see §13.
2. **No comparison is asserted unless the protocol matches.** Where it does not, the row is labelled **B** (partially comparable) or **C** (not comparable) and the mismatch is named. There are exactly **two** Category-A external comparisons in this audit, and SatQuery loses both.

---

## 1. Executive summary

### 1.1 Where SatQuery stands against the literature

**On the two benchmarks where SatQuery's protocol and the published protocol genuinely coincide, SatQuery is substantially behind.**

- **CDVQA test1** — identical dataset, identical split (39,686 questions over 968 image pairs), identical overall-accuracy metric. SatQuery **0.5380** against **0.6590** for the 2021 CDVQA baseline, **0.7310** for VisTA and **0.7474** for a 2B Qwen change-VQA model published in April 2026. A **12.1-point deficit against a five-year-old baseline**, a **20.9-point deficit against the state of the art**, and a loss on **all eight** question types.
- **LEVIR-CD** — official split, the standard 256px tiling (7,120 / 1,024 / 2,048 tiles), change-class F1. SatQuery **0.5597** against **≈0.9227** (PhyUnfold-Net, 2026). **−36.3 points.**

Both gaps are explained rather than mysterious, and both are budget-and-capacity gaps: a **49,543-parameter** change detector trained for 4 epochs, and a **6.26M-parameter** ResNet-18 semantic-change segmenter trained on 1,600 pairs.

### 1.2 Where the comparison cannot be made, and why

Most of the remaining rows are **⚪ cannot be determined**, because SatQuery's harness and the literature disagree on split, question-type filtering, reference count or metric definition — often all four. The three that matter most:

- SatQuery's headline **`rsvqa_lr` 0.6473** is measured on a **207-question slice of a 2,000-question HuggingFace redistribution of the RSVQA-LR _validation_ split**, by whole-string exact match, with **count questions included (27.5% of the slice)**. Every published 89–93% RSVQA-LR figure reports per-type accuracy on the **official test split** with **count excluded**. `docs/00` §3.5 calls this slice the "official split"; §5.2 sets out why that description does not hold.
- SatQuery's **RSICD caption BLEU-4 0.2446** is *sentence-mean, add-one-smoothed* BLEU. The repository's own implementation says so in a comment and warns: "Compare models against each other, not against a paper's corpus BLEU." Published RSICD BLEU-4 (RSGPT 65.74) is corpus BLEU.
- SatQuery's **LEVIR-CC change-caption BLEU-4 0.3063** is scored against **one** reference; the published protocol uses **five**.

### 1.3 The three findings this audit produced by measurement

**Finding 1 — the un-adapted base model is far worse, so the adaptation is unambiguously worthwhile.** Scoring `Qwen2.5-VL-3B-Instruct` with **no adapter** through the repository's own eval code on the identical 534-item held-out split:

| | base *(measured here)* | `track_b_v2` deployed *(measured here; reproduces the published run exactly)* | `track_b_v3` *(measured here)* |
|---|---|---|---|
| Held-out RSVQA-LR exact match | **0.1981** | 0.6473 | **0.7874** |
| `whu_opt_sar` exact match | **0.0000** | 0.2000 | **0.2419** |
| Refusal recall (17 unanswerable items) | **0.0000** | **0.4118** | 0.3529 |
| Overall token F1 | 0.2027 | 0.7927 | **0.8550** |

The base model **refuses nothing** — 0/17 — and scores **zero** exact match on the optical/SAR rows. Both capabilities are created entirely by the adaptation. Caveat in §7.3: roughly **63%** of the raw base→v2 exact-match gain is answer-*format* compliance rather than perception, and the audit measures that share instead of assuming it.

**Finding 2 — `track_b_v2`'s headline RSVQA-LR number is statistically indistinguishable from a constant.** The most common **training** answer per question type, applied to the val slice, scores **134/207 = 0.6473** — numerically identical to v2's published 0.6473. The two disagree symmetrically (17 items each way). The aggregate exact-match number therefore contains **no evidence** that v2 beats a per-type constant. It is the wrong instrument, not a wrong model: under the literature's own per-type convention v2 scores **0.8133** and clearly beats the constant on presence and comparison. This is the same failure mode the project already found and documented for CDVQA.

**Finding 3 — `track_b_v3`, the 2,000-step 82.7M-parameter adapter, is the best checkpoint, and this was not expected.** It was trained under a label-masking defect (the supervised span was ~89% `<|image_pad|>` tokens) and its loss sat at ~6.8 for 1,950 steps, so the reasonable prior was that it had learned nothing. Measured:

| | v2 (37.2M, 300 steps) | **v3 (82.7M, 2,000 steps)** |
|---|---|---|
| Held-out RSVQA-LR, all types | 0.6473 | **0.7874** |
| Published-convention micro (presence + comparison + rural-urban, n=150) | 0.8133 | **0.9533** |
| presence / comparison | 0.8824 / 0.7531 | **0.9559 / 0.9506** |
| Beats the per-type constant? | **no** (17 vs 17) | **yes** — 35 vs 6, McNemar χ²≈20.5, **p<0.001** |

The likely explanation is that the defective mask cut the supervised span **early**, so it supervised *more* than intended: every answer token was still supervised, merely swamped by placeholder targets. Flat loss reflected the placeholders, not the absence of learning.

**This does not license quoting v3 as a result yet.** The comparison confounds three changes at once (rank 16→32, vision tower 0→22.9M parameters, steps 300→2,000), and the 200-step probe with the *same* 82.7M configuration scores only 0.6715 — which suggests **steps, not vision parameters, are doing much of the work.** The isolating ablation (rank-32, language-only, 2,000 steps) has not been run. And v3 is **worse** than v2 on refusal recall (0.3529 vs 0.4118) and on the image-conditional refusal probe (1/12 vs 2/12).

### 1.4 What may and may not be claimed about RSVQA-LR

Under the literature's own convention — presence + comparison + rural-urban, count excluded — SatQuery's checkpoints score **0.8133 (v2)** and **0.9533 (v3)** on the held-out slice, against published figures of 89.61 (LHRS-Bot-Nova), 90.30 (RingMo-Agent), 90.70 (GeoChat), 92.70 (EarthDial) and 92.91 (Earth-OneVision).

**v3's 0.9533 must not be reported as beating those models.** It is measured on **150 questions from the validation split**; they are measured on the **official test split** with roughly 60× more questions. That is a Category-B comparison. What it does support is a much weaker and still useful statement: *SatQuery's VQA component is plausibly in the same band as published RS-VLMs, and the way to find out is to run the official test split* (§12, P3).

### 1.5 Top-5 selection and reproducibility

The Top 5 are **EarthDial, Earth-OneVision, RingMo-Agent, EarthMind and EarthGPT** — all compact (2–7B), all Earth-observation-specialised, all handling optical **and** SAR, selected by the scored criteria in §3.2 rather than by size or citation count. **None of them can be evaluated locally:** Earth-OneVision and RingMo-Agent have released nothing, EarthMind's weights are unconfirmed, and EarthDial and EarthGPT would need multi-GB downloads that were not made. **Every external number in this document is published and unreproduced.**

---

## 2. SatQuery: architecture, checkpoints, and what it actually measures

### 2.1 System shape

SatQuery is **not** a single VLM. It is a constrained-planner agent over a registry of small specialists, with one remote-sensing-adapted VLM inside it. This matters for every comparison below: the Top-5 models are single end-to-end VLMs, so comparing "SatQuery" to them conflates model-level and system-level performance. §10.3 keeps the two apart.

| Layer | Component | Params (measured this audit) | Role |
|---|---|---|---|
| Base VLM | `Qwen2.5-VL-3B-Instruct`, 4-bit NF4 at inference | **3,754,622,976** total; vision tower **668,684,288** (17.8%) | Free-text VQA |
| RS adaptation | Track B QLoRA adapter | **v2: 37,152,768** · **v3: 82,726,912** trainable | The remote-sensing adaptation |
| Specialists | 7 learned heads | **9,260,023** total | caption, grounding, land-cover, change mask, change caption, change VQA, opt–SAR fusion |
| Entailment gate | `nli_deberta_mnli` (third-party, frozen) | **184,424,963** | Faithfulness gate on generated prose |
| Verifier | `index_engine_v1` | **0** — deterministic | NDVI / NDWI / MNDWI / NDBI / σ⁰ / GLCM referee |

Counts were read directly from safetensors headers and `model_state_dict` tensors on disk during this audit, not from documentation.

**System total (v3 configuration): 4,031,034,874 parameters, of which 91,986,935 (2.28%) were trained by this project.**

### 2.2 Checkpoint inventory, as found on disk 2026-09-03

| Checkpoint | Trainable params | LoRA config | Steps | Status |
|---|---|---|---|---|
| `track_b_v1/adapter_final` | — | r=16, α=32, 7 LLM modules | 300 | **Unloadable.** 99.9922% NUL bytes; retained as evidence (`docs/00` L32) |
| `track_b_v2/adapter_final` | **37,152,768** (696 tensors) | r=16, α=32, 7 LLM modules | 300 | **Deployed / known-good.** 0.34% NUL |
| `track_b_v3_probe/adapter_final` | **82,726,912** (828 tensors) | r=32, α=64, 11 modules incl. vision tower | 200 | Corrected-configuration probe |
| `track_b_v3/adapter_final` | **82,726,912** (828 tensors) | r=32, α=64, 11 modules incl. vision tower | 2,000 | **Trained under the label-masking defect — yet measures best (§7).** Not a validated result: the ablation is confounded |

v3's target set adds the vision tower (`attn.qkv`, `attn.proj`, `merger.mlp.0`, `merger.mlp.2`) to v2's seven language-side projections. Of v3's 82.7M trainable parameters, **22,859,776 (27.6%) sit on the vision side** — 3.42% of the 668.7M-parameter vision tower. **This is the "82.7M visual adaptation" the brief asks about**, and the 22.9M vision-side share is the genuinely *visual* part of it. No checkpoint was overwritten, moved or deleted during this audit.

### 2.3 Every metric SatQuery currently measures

Read from the code, not from the documentation.

| Task | Implementation | Metric as implemented | Published figure |
|---|---|---|---|
| VQA | `evaluation/metrics/vqa.py`, `evaluation/track_b_eval.py` | normalised **exact match** + **token F1**, coverage-aware | `rsvqa_lr` 0.6473; full-val 0.3791 / F1 0.7927 |
| Captioning | `evaluation/metrics/all_tasks.py:bleu` | **sentence-mean BLEU-4, add-one smoothed**, multi-reference | RSICD 0.2446 (n=1,093), 13.4% unique captions |
| Change captioning | `training/train_change_caption.py` | same BLEU, **single reference** | LEVIR-CC 0.3063 changed / 0.5686 aggregate (n=1,929) |
| Grounding | `all_tasks.py:score_grounding` | **Acc@0.5, Acc@0.7, mIoU**, top-scoring box | DIOR-RSVG 0.0762 / 0.0088 / 0.1405 (n=1,141) |
| Land cover | `training/track_a_full.py` | **mAP** (macro over per-class average precision) | BigEarthNet-19 0.2854 all-band / 0.2573 4-band, retention 0.9015 |
| Change mask | `training/train_change_mask.py` | **F1 / IoU / P / R on the change class only** | LEVIR-CD F1 0.5597, IoU 0.3886 |
| Change VQA | `evaluation/cdvqa_predict.py` | **overall accuracy** over the full test split | CDVQA 0.5380 (39,686 Q / 968 pairs, 100% coverage) |
| Opt–SAR | `training/train_optsar_fusion.py` | **mAP** per arm + complementarity gain | optical 0.7778 / SAR 0.7410 / fused 0.7714 → **−0.0064** |
| Refusal | `evaluation/refusal.py` | **refusal recall, false-refusal rate, lexical-shortcut probe** | 0.4118 / 0.0077 / 0.1667 |
| Calibration | `evaluation/calibrate.py` | **ECE** before/after temperature and affine fits | change mask 0.0668 → 0.0034 |
| Entailment | `evaluation/entailment_bench.py` | retained / flagged / unverifiable sentence counts | `docs/assets/entailment/bench.json` |
| Routing | `evaluation/adversarial.py` | **illegal-plan rate**, routing accuracy | 0/600 illegal; CLEAN_HOLDOUT accuracy 0.5862 (n=29) |
| Latency | trace `runtime_ms` | wall-clock per tool step | `rs_vqa_v1` **2,670 ms** for 14 tokens, 256×256, RTX 4050 |

**Three of these have no published counterpart anywhere** — complementarity gain, illegal-plan rate, and the lexical-shortcut probe. They are this project's own instruments: a strength for review, a limitation for benchmarking, because nothing external can be laid beside them.

---

## 3. Top-5 selection methodology

### 3.1 The model class SatQuery belongs to

From §2, the comparison class is: **a compact (≤8B) vision-language model, specialised to Earth observation, that performs image VQA, captioning and grounding, understands land cover and bi-temporal change, and handles more than one sensor modality (optical *and* SAR).**

That exclusion is deliberate and removes most of the field. A generic VLM with a large parameter count is *not* comparable merely for being large; an RS model that handles only optical RGB fails SatQuery's mandatory M6 (cross-modal optical + SAR extraction); and a grounding-only or ultra-high-resolution-VQA-only model cannot be laid against a system that must answer five task families.

### 3.2 Scoring

Eight criteria from the brief, each scored 0–3, equally weighted, maximum 24. Scores are judgements from the evidence collected in §4, shown so a reader can disagree with a specific cell rather than with the conclusion.

| Model | RS spec. | VQA | Caption | Ground | Multi-sensor | Bench. strength | Reproducible | Recency | **Total** |
|---|---|---|---|---|---|---|---|---|---|
| **EarthDial** | 3 | 3 | 3 | 2 | 3 | 3 | 3 | 2 | **22** |
| **Earth-OneVision** | 3 | 3 | 3 | 3 | 3 | 3 | 0 | 3 | **21** |
| **RingMo-Agent** | 3 | 3 | 3 | 2 | 3 | 3 | 1 | 3 | **21** |
| **EarthMind** | 3 | 2 | 2 | 2 | 3 | 2 | 2 | 2 | **18** |
| **EarthGPT** | 3 | 2 | 3 | 2 | 3 | 2 | 2 | 1 | **18** |
| LHRS-Bot-Nova | 3 | 3 | 1 | 3 | 0 | 3 | 3 | 1 | 17 |
| TinyRS-R1 | 3 | 2 | 1 | 2 | 0 | 2 | 3 | 2 | 15 |
| VHM | 3 | 2 | 2 | 2 | 0 | 2 | 3 | 1 | 15 |
| GeoChat | 3 | 2 | 2 | 2 | 0 | 2 | 3 | 1 | 15 |
| GeoEyes | 3 | 3 | 0 | 1 | 0 | 2 | 1 | 3 | 13 |

**Ties are broken explicitly.** Earth-OneVision is ranked above RingMo-Agent on modality breadth (six modalities vs three) and benchmark coverage. EarthMind is ranked above EarthGPT because it is built on the **same LLM family as SatQuery** (Qwen2.5-3B) and is 15 months more recent.

**Earth-OneVision leads on published capability and would be #1 on the model alone.** It falls to #2 because nothing has been released — no code, no weights, no dataset — so its numbers cannot be checked by anyone. That is the reproducibility criterion doing exactly the work it should.

### 3.3 The Top 5

1. **EarthDial** (4B) — 22
2. **Earth-OneVision** (2B) — 21
3. **RingMo-Agent** (3B) — 21
4. **EarthMind** (4B) — 18
5. **EarthGPT** (7B) — 18

Three further models are carried through the tables as clearly-labelled **reference rows**, not Top-5 members, because they are the anchors the literature itself uses:

- **GeoChat** (7B) — the universal baseline; nearly every RS-VLM paper reports against it.
- **LHRS-Bot-Nova** (8B) — the strongest published optical-only VQA + grounding numbers.
- **TinyRS-R1** (2B) — the only model found in SatQuery's exact compute class that also publishes VRAM and latency.

### 3.4 Considered and excluded, with reasons

| Model / system | Why excluded |
|---|---|
| **LHRS-Bot-Nova**, **LHRS-Bot** | Optical RGB only. Fails the multi-sensor criterion SatQuery's M6 requires. Kept as a reference row. |
| **TinyRS-R1 / TinyRS** | Optical only, and captioning is not a headline capability. Kept as a reference row because it is the closest compute-class match. |
| **GeoChat**, **RS-LLaVA**, **RSGPT**, **SkyEyeGPT**, **H2RSVLM**, **SkySenseGPT** | 2023–2024 optical-only generation; superseded on nearly every published metric by the Top 5. GeoChat kept as a reference row. |
| **VHM** | Optical only. Genuinely relevant on *one* axis — its HnstD honesty data is the nearest published analogue to SatQuery's refusal metric — and it is cited in §6.4 for that. |
| **GeoEyes**, **ZoomEarth**, **GeoVista** | Ultra-high-resolution VQA specialists. No captioning, no SAR, no change. Different problem. |
| **GeoGround**, **RSGround-R1**, **TIDM** | Grounding only. Used in §6.3 as the grounding ceiling, not as system comparators. |
| **RSUniVLM**, **SkyMoE**, **UniRS**, **GeoPix**, **GeoVLM-R1** | Plausible, but each covers a strict subset of the required capability set with weaker or narrower published evidence. |
| **MLRS ("More with Less")**, July 2026 | Multi-sensor and very recent, but its published metrics are GEOBench-VLM / XLRS-Bench segmentation and detection scores that do not intersect SatQuery's crosswalk at all, and no release is stated. |
| **GPT-4V / Qwen3-VL-235B and other generic large VLMs** | Excluded on the brief's own instruction: not RS-specialised, and parameter count is not relevance. |
---

## 4. Top-5 model profiles

Every field below is from the primary source listed in §14. Where a paper does not state something, the cell says **not stated** rather than an estimate.

### 4.1 #1 — EarthDial (score 22)

| | |
|---|---|
| Full name | EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues |
| Organisation | Mohamed bin Zayed University of AI · IBM Research · Linköping University · ANU |
| Date | arXiv 2412.15190; v2 dated 7 April 2025 |
| Parameters | **4B** total |
| Base LLM | Phi-3-mini |
| Vision encoder | InternViT-300M (distilled from InternViT-6B) + MLP projector + Adaptive High-Resolution block + Data Fusion module |
| Training | Three stages, **>11.11M instruction pairs**: 7.67M pretraining (NAIP, Sentinel-2, Landsat, SkyScript), 1.85M RGB/temporal, 2.4M multispectral/SAR |
| Modalities | RGB, **SAR**, multispectral (S2, L8), NIR/infrared, hyperspectral, **bi-temporal and multi-temporal** |
| Weights / code | **Public** — `github.com/hiyamdebary/EarthDial`, CC BY 4.0 |
| Locally evaluable? | **In principle yes** at 4-bit on the 6 GiB RTX 4050; requires a multi-GB download that has **not** been made |

**Why comparable.** It is the closest published match to SatQuery's *capability contract*: single-image VQA, captioning, grounding-as-referred-detection, bi-temporal change, multispectral land cover, and SAR — the same five families SatQuery's M1–M6 require, in one model, at 4B. It is also the only Top-5 model with both public weights and an explicit licence.

**Key published numbers.** RSVQA-LRBEN presence 92.58 / comparison 92.75 / **average 92.70**; RSVQA-HRBEN average 72.45; BigEarthNet RGB 68.82 / MS 69.94 accuracy; SoSAT-LCZ42 60.72; xBD image classification recall 96.37; LEVIR-MCI change captioning ROUGE-1 33.78 / ROUGE-L 30.47; RSICD ROUGE-1 33.77 / ROUGE-L 27.61; UCM-Captions ROUGE-1 40.0; NWPU-Captions ROUGE-1 45.84; SAR ship detection mAP@0.5 6.06 (single) / 26.02 (multiple); QuakeSet 57.53.

**Caveat on its captioning table.** The extracted METEOR column carries values above 50 (e.g. NWPU 80.61), which is outside METEOR's normal range and is most likely a column-label problem in the extraction rather than in the paper. **This audit therefore quotes only EarthDial's ROUGE columns** and treats its METEOR values as unverified.

### 4.2 #2 — Earth-OneVision (score 21)

| | |
|---|---|
| Full name | Earth-OneVision: Extending Remote Sensing MLLMs to More Sensor Modalities and Tasks |
| Organisation | National Key Laboratory of Science and Technology on Space-Born Intelligent Information Processing, Beijing Institute of Technology |
| Date | arXiv 2606.10819v1, **9 June 2026** — the most recent model in the Top 5 |
| Parameters | **2B** |
| Base LLM | Qwen3 (2B) |
| Vision encoder | SigLIP-2 (0.3B), native dynamic resolution |
| Training | Two-stage progressive cross-modality adaptation on **MMRS-OneVision, ~34M QA pairs**, 6 modalities, 9 task categories, 25 subtasks; 8×H100-80GB, DeepSpeed ZeRO-2 |
| Modalities | Optical, **SAR**, infrared, multispectral, temporal, video, **cross-sensor fusion** — six, the broadest found |
| Weights / code | **None released.** The paper states no GitHub or HuggingFace URL; v1 is the only version |
| Locally evaluable? | **No** |

**Why comparable.** At 2B it is *smaller* than SatQuery's 3.75B base and yet covers a strict superset of SatQuery's capability set, including exactly the optical + SAR fusion that M6 requires. It is the fairest available answer to "what does a well-executed model of this size achieve?"

**Key published numbers.** **RSVQA-LR 92.91%**; RSVQA-HR 86.36%; VRSBench-VQA 80.32%; DIOR-RSVG **P@0.5 94.41%**; OPT-RSVG 87.52%; VRSBench-VG 90.77%; RSVG-HR 82.36%; RSICD METEOR 33.98%; SARLANG-Bench VQA **80.68%**; SARLANG complex captioning CIDEr 110.24; BigEarthNet-MS recall 75.74%; BigEarthNet-RGB recall 78.03%; LEVIR-CDC F1 73.28; WHU-CDC F1 70.91; LEVIR-MCI mF1 85.85; EarthMind-Bench optical 80.70 / SAR 76.10 / fusion 81.94 MCQ.

**Caveat.** Its LEVIR-MCI "METEOR 76.45%" and xBD "ROUGE-1 92.33%" are far outside the usual ranges for those metrics on those datasets; this audit does not quote them.

### 4.3 #3 — RingMo-Agent (score 21)

| | |
|---|---|
| Full name | RingMo-Agent: A Unified Remote Sensing Foundation Model for Multi-Platform and Multi-Modal Reasoning |
| Organisation | Chinese Academy of Sciences (Aerospace Information Research Institute) group — Hu, Wang, Feng, Wei, Yin, Diao, Wang, Bi, Kang, Ling, Fu, Sun |
| Date | arXiv 2507.20776; **v3 dated 18 August 2026** |
| Parameters | **3B** (DeepSeekMoE variant), LoRA rank 64 |
| Base LLM | DeepSeek-VL2 (3B) |
| Vision encoder | SigLIP-SO400M-384 |
| Training | **RS-VL3M**, >3M image–text pairs, 8 tasks, 2 platforms, 3 modalities |
| Modalities | Optical, **SAR**, infrared; **satellite and UAV platforms** |
| Weights / code | **Not stated** in the paper |
| Locally evaluable? | **No** |

**Why comparable.** Same parameter class as SatQuery (3B vs 3.75B), same LoRA-on-a-frozen-VLM adaptation strategy, and the same optical + SAR requirement — plus it is the only Top-5 model that publishes a *fine-tuned* RSVQA-LR presence number that is currently the highest we found.

**Key published numbers.** RSVQA-LR (fine-tuned) presence **93.10%** / comparison 87.50% / **average 90.30%**; RSVQA-HR (zero-shot) presence 75.24 / comparison 83.92 / average 79.58; UCM captioning BLEU-4 77.63, METEOR 51.79, ROUGE-L 85.51, CIDEr 373.68; **SARDet-100k mAP@50 53.84**; IR-DET mAP@50 59.88.

### 4.4 #4 — EarthMind (score 18)

| | |
|---|---|
| Full name | EarthMind: Multi-Granular and Multi-Sensor Earth Observation with Large Multimodal Models |
| Organisation | University of Trento · University of Pisa · INSAIT Sofia · TU Munich · TU Berlin |
| Date | arXiv 2506.01667v1, 2 June 2025 |
| Parameters | **4B**, on **Qwen2.5-3B** |
| Vision encoders | InternVL2 multi-scale patches; SAM2 for grounding; GPT4RoI region encoder |
| Training | 1.7M general image–text, 1M EO multimodal (EarthGPT, VRSBench, DIOR-RSVG, RRSIS-D, RefSegRS), 500K multispectral (BigEarthNet, SoSAT-LCZ42), **20K synthetic RGB–SAR paired dialogues** |
| Modalities | Optical, **SAR** (single/dual channel), multispectral, explicit **RGB–SAR fusion** via Modality Alignment + Modality Mutual Attention |
| Weights / code | **Code public** — `github.com/shuyansy/EarthMind`; weight release not explicitly stated |
| Locally evaluable? | **Uncertain** — code yes, weights unconfirmed |

**Why comparable.** It is the only Top-5 model built on the **same LLM base family as SatQuery** (Qwen2.5-3B), and it is the closest published attempt at the exact hypothesis SatQuery's M6 tests: that fusing a co-registered optical + SAR pair beats the better single modality. **EarthMind reports that fusion does help** (MCQ average 69.0 RGB / 67.5 SAR / **70.6 fused**), where SatQuery measured a complementarity gain of **−0.0064**. That contrast is the single most useful external data point in this audit for M6, and §11 treats it as a gap.

**Key published numbers.** EarthMind-Bench MCQ average 69.0 / 67.5 / 70.6 (RGB / SAR / fusion); AID 97.2; UC-Merced 95.0; RSVQA-HRBEN 74.0; VRSBench-VQA 78.9; VRSBench visual grounding Acc@0.5 55.6; DIOR-RSVG region captioning CIDEr 428.2; RRSIS-D 82.2 mIoU; RefSegRS 62.6 mIoU; BigEarthNet 70.4; SoSAT-LCZ42 58.3; SAR ship detection mAP 13.58 / 28.55 / 36.78 (small / medium / large).

### 4.5 #5 — EarthGPT (score 18)

| | |
|---|---|
| Full name | EarthGPT: A Universal Multimodal Large Language Model for Multisensor Image Comprehension in Remote Sensing |
| Organisation | Beijing Institute of Technology |
| Date | arXiv 2401.16822, March 2024; IEEE TGRS |
| Parameters | Not stated; LLaMA-2 backbone (7B class) |
| Vision encoders | DINOv2 ViT-L/14 + frozen CLIP ConvNeXt-L (dual, transformer + CNN) |
| Training | **MMRS-1M**, >1M image–text pairs aggregated from 34 RS datasets, including 3 SAR and 5 infrared detection sets |
| Modalities | Optical, **SAR**, infrared |
| Weights / code | **Public** — `github.com/wivizhang/EarthGPT`, "code and dataset are available" |
| Locally evaluable? | **Marginal** at 4-bit on 6 GiB with dual vision encoders; not attempted |

**Why comparable.** The earliest model to do what SatQuery's M6 asks — one model across optical, SAR and infrared — and still the most-cited multi-sensor reference. It is the historical anchor for the multimodal column.

**Key published numbers.** NWPU-RESISC45 supervised top-1 93.84%; CLRS zero-shot 77.37%; NaSC-TG2 zero-shot 74.72%; **NWPU-Captions BLEU-4 65.5, METEOR 44.5, ROUGE-L 78.2, CIDEr 192.6**; CRSVQA supervised overall 82.00%; **RSVQA-HR zero-shot 72.05%**; DIOR-RSVG mIoU 69.34 / cIoU 81.54; MAR20 HBB zero-shot AP@40 90.47.

### 4.6 Reference rows (not Top-5 members)

| Model | Params / base | Date | Modalities | Weights | Headline numbers used in this audit |
|---|---|---|---|---|---|
| **GeoChat** | 7B, LLaVA-1.5 | CVPR 2024 | optical | **Public** (`MBZUAI/geochat-7B`) | RSVQA-LR fine-tuned presence 91.09 / comparison 90.33 / avg 90.70; RSVQA-HR zero-shot avg 70.82; VRSBench (fine-tuned) caption BLEU-4 13.8, grounding all Acc@0.5 39.6, VQA avg 60.6; VRSBench zero-shot caption BLEU-4 1.4, grounding Acc@0.5 12.9, VQA avg 40.8 |
| **LHRS-Bot-Nova** | 8B, LLaMA-3 + SigLIP-L/14@336 | Nov 2024 | optical only | **Public** (`NJU-LHRS/LHRS-Bot`) | RSVQA-LR presence 89.11 / comparison 89.00 / rural-urban 90.71 / **avg 89.61**; RSVQA-HR avg 92.06; **DIOR-RSVG Acc@0.5 92.87**, RSVG 81.85; classification avg 76.60; LHRS-Bench overall 34.93 |
| **TinyRS-R1** | **2B**, Qwen2-VL-2B, 4-stage + GRPO | Oct 2025 | optical only | **Public** (`aybora/TinyRS`) | VQA avg: TinyRS **83.5** / TinyRS-R1 76.0 (LR-presence 90.4 / 78.1, LR-compare 89.9 / 84.0, LR-rural 92.0 / 76.0, HR-presence 64.5 / 68.6, HR-compare 80.6 / 73.5); **DIOR-RSVG 69.4 / 74.9**; classification avg 81.0 / 85.6; **latency 90 ms / 689 ms, VRAM 4.4 GB / 4.6 GB** |

TinyRS-R1's resource table is the most useful external row in this whole audit for §9, because it is the only one that reports latency and VRAM at all.
---

## 5. Benchmark crosswalk

### 5.1 The crosswalk table

Each row maps one thing SatQuery measures to the nearest public benchmark, then states the **verified** protocol difference. Comparability class: **A** same dataset, split, task, metric, protocol · **B** same task and dataset family, one or more protocol differences · **C** not comparable.

| SatQuery metric | Value | SatQuery protocol (verified in code/data) | Nearest public benchmark | Public protocol | Class | Mismatch |
|---|---|---|---|---|---|---|
| VQA exact match, `rsvqa_lr` | **0.6473** (v2) · **0.7874** (v3) | 207 questions; stratified 10% of `dmarsili/RSVQA-LR-2k`, itself a 2,000-question subset of the RSVQA-LR **validation** split; whole-string exact match over **all** types, **count included (27.5%)** | RSVQA-LR | Official **test** split (100 images, ~10k questions); per-type accuracy averaged; **count normally excluded** | **B** once re-typed (§7.4); **C** as published by SatQuery | split (val vs test), n (207 vs ~10k), question-type filtering, metric aggregation |
| VQA exact match, full val | **0.3791** | 534 questions, project-built mix (207 RSVQA + 322 WHU-OPT-SAR + 5 refusal) | none | — | **C** | benchmark does not exist outside this repo |
| VQA token F1 | **0.7927** | same 534 | none | — | **C** | no RS-VLM paper reports token F1 |
| Caption BLEU-4 | **0.2446** | **RSICD official test split, n=1,093**; 5 references; **sentence-mean, add-one-smoothed** BLEU | RSICD captioning | Same split; **corpus** BLEU-4 over 5 references | **B** | metric implementation (sentence-mean + smoothing vs corpus); repo's own comment forbids the direct comparison |
| Caption diversity | **13.4% unique** (146/1,093) | — | none | — | **C** | not a published metric; a genuine SatQuery instrument |
| Change caption BLEU-4 | **0.3063** changed / **0.5686** aggregate | **LEVIR-CC test, n=1,929** (964 changed + 965 unchanged); **one** reference; sentence-mean smoothed BLEU | LEVIR-CC change captioning | Same split; **five** references; corpus BLEU-4 and CIDEr | **C** | reference count (1 vs 5) alone makes the numbers incommensurable |
| Grounding Acc@0.5 / Acc@0.7 / mIoU | **0.0762 / 0.0088 / 0.1405** | DIOR-RSVG, n=1,141; `run_metadata` records `split_note: NO published split in this mirror` — **self-made, image-grouped split**; backbone trained **from scratch** | DIOR-RSVG referring grounding | Official test split; Acc@0.5 (a.k.a. P@0.5) | **C** | split is not the published one; the repo says so |
| Land-cover mAP | **0.2854** all-band / **0.2573** Cartosat 4-band; retention **0.9015** | BigEarthNet **19-label** nomenclature, 12 bands, macro mAP; trained on **30,000 patches, 3 epochs, dim 64**; evaluated on an HDF5 **partition shard** (`test_p8`), not the officially recommended split | BigEarthNet-19 multi-label | Official recommended split, full ~590k train patches, macro mAP | **B** | training-set size (30k vs ~590k), model capacity, split provenance |
| Band-dropout retention | **0.9015** | mAP(4-band) / mAP(12-band) | none | — | **C** | SatQuery's own instrument; no published counterpart found |
| Change-mask F1 (change class) | **0.5597**, IoU 0.3886, P 0.4426, R 0.7613 | **LEVIR-CD official split, 256px tiling: 7,120 train / 1,024 val / 2,048 test tiles** — the standard tiling of the official 445/64/128 pairs. Change-class F1, threshold 0.5. Model is **49,543 parameters**, 4 epochs | LEVIR-CD binary change detection | Identical split, identical tiling convention, change-class F1 | **A** | none material — this is directly comparable, and the gap is large |
| Change-VQA overall accuracy | **0.5380** | **CDVQA test1: 39,686 questions over 968 image pairs, 100% coverage.** Per-type majority baseline 0.5084; ground-truth-map oracle 0.9975 | CDVQA | **test1: 39,686 QA pairs** — the identical split; overall accuracy (OA) | **A** | none — identical dataset, split, question set and metric |
| Opt–SAR complementarity gain | **−0.0064** (optical 0.7778, SAR 0.7410, fused 0.7714) | WHU-OPT-SAR, 1,548 tiles, **tile-level multi-label mAP**; `split_method: deterministic random by tile; NOT geographic` | WHU-OPT-SAR fusion | **Semantic segmentation**: mIoU / OA / mF1 over 100 scenes | **C** | different task (tile multi-label vs pixel segmentation) and different metric; only the *direction* of the fusion result is comparable |
| Refusal recall / false-refusal / lexical-shortcut probe | **0.4118 / 0.0077 / 0.1667** | 17 refusal items in the 534-question val split; image-conditional refusals share wording with answerable ones | VHM HnstD (deceptive questions on non-existent objects) | Accuracy on factual vs deceptive question pairs | **C** | different construction and metric; conceptually the nearest published analogue |
| Calibration ECE | **0.0668 → 0.0034** (change mask, affine fit) | 15-bin ECE on cached logits | none | — | **C** | **no RS-VLM in the Top 5 or the reference rows reports ECE at all** |
| Entailment gate | retained/flagged/unverifiable counts | DeBERTa-MNLI over generated sentences vs structured evidence | none | — | **C** | no counterpart |
| Illegal-plan rate | **0 / 600** | 200 adversarial queries × 3 configurations | none | — | **C** | no counterpart |
| Routing accuracy | **0.5862** (n=29 CLEAN_HOLDOUT) | never-tuned holdout | none | — | **C** | no counterpart |
| VQA latency | **2,670 ms** / 14 tokens, 256×256 | RTX 4050 Laptop 6 GiB, 4-bit NF4, greedy | TinyRS-R1 90 ms (2B, no reasoning) | unstated GPU | **B** | different hardware and token budget |

### 5.2 Why the RSVQA-LR row is the most important caveat in this document

`docs/00` §3.5 records RSVQA-LR as evaluated on the "official split, n=207". Four things verified during this audit contradict that description, and all four push in the same direction — **towards SatQuery's number being harder than the published ones on some axes and easier on others, so the net direction is unknown**:

1. **It is the validation split, not the test split.** The data on disk is `data/rsvqa_lr_2k`, whose own `README.md` says: "A 2k subset of the validation split of the RSVQA LR dataset ported to HF." RSVQA-LR's official partition is 572 / 100 / 100 images with 77,232 questions total.
2. **n = 207, not ~10,000.** The 2,000 questions were split 90/10 by `training/prepare/instruction_mix.py` (`stratified_split`, seed 42, `val_fraction` 0.1). At n=207 the 95% binomial interval around 0.6473 is roughly ±0.065 — wider than most of the differences this project has been reading as signal.
3. **Count questions are included.** Classifying the 207 by RSVQA's own templated wording (§7.4) gives **presence 68 · comparison 81 · count 57 · rural-urban 1**. The published 89–93% figures are averages over presence / comparison / rural-urban; the RSVQA literature explicitly notes that "some researchers removed area and count questions during training and testing because the answers … are numerical". Count is the hardest type — the **original RSVQA baseline scored 67.01% on count against 87.46% presence and 90.00% rural/urban** — so including 27.5% count questions scores SatQuery on a harder distribution than the published averages. **§7.4 re-scores the slice under the published convention**, which is the correct fix.
4. **The metric is whole-string exact match over all types at once, not per-type accuracy.** These coincide for `yes`/`no` and short numerals and diverge for everything else — and §7.4 shows the aggregate is exactly the number a per-type constant achieves.

**What is clean about it.** Train/val image overlap for `rsvqa_lr` is **0 of 207** (verified this audit) — there is no image-level leakage on this row. On `whu_opt_sar`, 22 of 322 val images (**6.8%**) also appear in train, so that row is mildly optimistic.

**The one comparison that survives on the count-inclusive metric.** The nearest published number that *does* include count is the original RSVQA baseline's **overall accuracy 79.08%** (average accuracy 81.49%) on the official test split. `track_b_v2`'s 64.73% sits **14.4 points below** it; `track_b_v3`'s **78.74%** sits **0.3 points below** it. Both are on a different split at n=207, so **Category B**, not a verdict.

### 5.3 Benchmarks the PS prescribes that are still not evaluated

- **VRSBench** — annotations only on disk; its imagery lives in DOTA/DIOR, which are not present. This is the largest crosswalk hole, because VRSBench is the *one* public benchmark that scores captioning, grounding **and** VQA under a single protocol, and every Top-5 model except EarthDial and EarthGPT reports on it. `docs/00` L11.
- **RSVQA-HR** — not evaluated. Four of the five Top-5 models report it.
- **BigEarthNet.txt** — the image–text corpus the PS names as primary for adapting image–text representations was not used; Track A used BigEarthNet imagery + 19 labels instead. Permitted by the mandatory clause, at odds with the background clause.
---

## 6. Published benchmark comparison

**Every external number in this section is published and was not reproduced by us.** SatQuery's numbers are its own measurements. `N/R` = not reported by that paper. `N/E` = not evaluated by SatQuery. The **Cls** column is the comparability class from §5.

### 6.1 Master table

| Task | Dataset | Metric (whose?) | SatQuery | EarthDial (4B) | Earth-OneVision (2B) | RingMo-Agent (3B) | EarthMind (4B) | EarthGPT (7B) | Cls | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| VQA | RSVQA-LR | accuracy — **protocol differs**, see note ¹ | **0.6473** v2 / **0.7874** v3, all types ¹ | **92.70** (P 92.58 / C 92.75) | **92.91** | **90.30** (P 93.10 / C 87.50) | N/R | N/R | **C** as published | ¹ |
| VQA | RSVQA-LR, **re-scored to the published convention** (§7.4) | presence+comparison+rural-urban micro, count excluded | **0.8133** v2 / **0.9533** v3 (n=150, **validation** slice) | 92.70 | 92.91 | 90.30 | N/R | N/R | **B** | §7.4 |
| VQA | RSVQA-HR | avg accuracy | **N/E** | 72.45 | 86.36 | 79.58 (zero-shot) | 74.0 (HRBEN) | 72.05 (zero-shot) | **C** | papers |
| VQA | VRSBench-VQA | accuracy | **N/E** | N/R | 80.32 | N/R | 78.9 | N/R | **C** | papers |
| Captioning | RSICD | BLEU-4 | **0.2446** *sentence-mean, smoothed, 5 refs* | N/R (ROUGE-1 33.77 / ROUGE-L 27.61) | N/R (METEOR 33.98) | N/R | N/R | N/R | **B** | ² |
| Captioning | NWPU-Captions | BLEU-4 (corpus) | **N/E** | N/R (ROUGE-1 45.84) | N/R | N/R | N/R | **65.5** (METEOR 44.5, CIDEr 192.6) | **C** | EarthGPT |
| Captioning | UCM-Captions | BLEU-4 (corpus) | **N/E** | N/R (ROUGE-1 40.0) | N/R | **77.63** (CIDEr 373.68) | N/R | N/R | **C** | RingMo-Agent |
| Grounding | DIOR-RSVG | Acc@0.5 / P@0.5 | **0.0762** *self-made split* | N/R | **94.41** | N/R | N/R | N/R (mIoU 69.34) | **C** | ³ |
| Grounding | VRSBench-VG | Acc@0.5 | **N/E** | N/R | 90.77 | N/R | 55.6 | N/R | **C** | papers |
| Land cover | BigEarthNet-19 | **mAP** (SatQuery) vs accuracy/recall (theirs) | **0.2854** all-band / 0.2573 4-band | 69.94 *accuracy*, MS | 75.74 *recall*, MS | N/R | 70.4 *accuracy* | N/R | **C** | ⁴ |
| Change detection | LEVIR-CD | change-class F1 | **0.5597** | N/R | N/R (LEVIR-CDC F1 73.28 — different dataset) | N/R | N/R | N/R | **A** vs specialists, see §6.3 | ⁵ |
| Change captioning | LEVIR-CC | BLEU-4 | **0.3063** *1 reference* | N/R (LEVIR-MCI ROUGE-1 33.78) | N/R | N/R | N/R | N/R | **C** | ⁶ |
| **Change VQA** | **CDVQA test1** | **overall accuracy** | **0.5380** | N/R | N/R | N/R | N/R | N/R | **A** vs CDVQA literature, see §6.2 | ⁷ |
| Multimodal / SAR | SARLANG-Bench VQA | accuracy | **N/E** | N/R | **80.68** | N/R | N/R | N/R | **C** | Earth-OneVision |
| Multimodal / SAR | SAR detection | mAP@0.5 | **N/E** | 6.06 / 26.02 (ship, single / multiple) | N/R | **53.84** (SARDet-100k mAP@50) | 13.58 / 28.55 / 36.78 (small/med/large) | N/R | **C** | papers |
| Multimodal / SAR | **does fusion beat the best single modality?** | direction of the effect | **NO — −0.0064 mAP** | N/R | **YES** — fusion 81.94 vs optical 80.70 (EarthMind-Bench MCQ) | N/R | **YES** — fusion 70.6 vs RGB 69.0 / SAR 67.5 | N/R | **C** *(direction only)* | ⁸ |
| Reliability | calibration ECE | ECE | **0.0034** (change mask, after affine) | **none reported** | **none reported** | **none reported** | **none reported** | **none reported** | — | SatQuery only |
| Reliability | refusal recall | recall on unanswerable | **0.4118** | **none reported** | **none reported** | **none reported** | **none reported** | **none reported** | — | SatQuery only |
| Orchestration | illegal-plan rate | illegal plans / total | **0 / 600** | **none reported** | **none reported** | **none reported** | **none reported** | **none reported** | — | SatQuery only |

**Notes.**
¹ SatQuery: 207 questions from the RSVQA-LR **validation** subset, count questions included (27.5%), whole-string exact match over all types. Published figures: official **test** split, per-type accuracy, count excluded. **These are not the same measurement** — §5.2. The row below re-scores SatQuery's slice under the published convention, which makes the comparison **B** rather than **C**; it is still a 150-question validation slice against test splits ~60x larger, so it supports "plausibly in the same band" and nothing stronger. The only published count-inclusive figure is the original RSVQA baseline's **OA 79.08 / AA 81.49**, which v3's 0.7874 sits 0.3 points below and v2's 0.6473 sits 14.4 points below.
² SatQuery's BLEU is sentence-mean with add-one smoothing; the code comment says explicitly not to compare it to a paper's corpus BLEU. The RSICD corpus-BLEU ceiling in the literature is RSGPT at **65.74**.
³ `run_metadata` records `split_note: NO published split in this mirror`.
⁴ Metric mismatch is fatal here: SatQuery reports macro mAP; EarthDial/EarthMind report accuracy; Earth-OneVision reports recall. The nearest same-metric external anchor is the BigEarthNet-19 CNN literature — **ResNet50 mAP 79.98**, SeCo average precision 82.62 — see §6.3.
⁵ Same official split and the standard 256px tiling (7,120 / 1,024 / 2,048 tiles) and the same change-class F1. External comparators are change-detection specialists, not VLMs — §6.3.
⁶ One reference vs the standard five. Published LEVIR-CC BLEU-4 is ~65 (SAGE-CC 65.50, KCFI 65.30) with CIDEr ~137–140 (SAT-Cap 140.23).
⁷ Identical dataset, identical test1 split (39,686 QA pairs / 968 pairs), identical OA metric. §6.2.
⁸ Not a numeric comparison — the metrics differ entirely. What *is* comparable is that two published multi-sensor models report fusion helping and SatQuery measured it not helping on its own dataset and metric.

### 6.2 The one Category-A comparison: CDVQA test1

Identical dataset, identical split (**39,686 questions over 968 image pairs**), identical metric (overall accuracy). This is the comparison to trust.

| System | Year | Test1 OA | Test1 AA | Δ vs SatQuery (OA) |
|---|---|---|---|---|
| **SatQuery** *(measured here, 100% coverage)* | 2026 | **0.5380** | N/R | — |
| per-type majority constant *(SatQuery's own baseline)* | — | 0.5084 | — | −0.0296 |
| **CDVQA baseline** (Yuan et al.) | 2021 | **0.6590** | 0.5530 | **+0.1210** |
| **SOBA** | 2024 | 0.6920 | 0.6030 | **+0.1540** |
| **VisTA** | 2024 | 0.7310 | 0.6590 | **+0.1930** |
| **Qwen3.5-2B change-VQA** (Bazi et al., Apr 2026) | 2026 | **0.7474** | 0.6859 | **+0.2094** |
| SatQuery oracle over ground-truth change maps | — | 0.9975 | — | (ceiling) |

Per question type, on the same 39,686 questions:

| Question type | n | SatQuery majority baseline | **SatQuery** | **Qwen3.5-2B (published)** | Gap |
|---|---|---|---|---|---|
| `change_or_not` | 13,882 | 0.5617 | 0.6772 | **0.8395** | −0.1623 |
| `change_ratio_types` | 5,811 | 0.4770 | 0.4791 | **0.7986** | −0.3195 |
| `decrease_or_not` | 4,658 | 0.6900 | 0.6496 | **0.8270** | −0.1774 |
| `increase_or_not` | 4,600 | 0.6663 | 0.6437 | **0.8324** | −0.1887 |
| `change_to_what` | 2,991 | 0.3805 | 0.3714 | **0.6172** | −0.2458 |
| `largest_change` | 2,904 | 0.4291 | 0.4497 | **0.6446** | −0.1949 |
| `smallest_change` | 2,904 | 0.2231 | 0.1319 | **0.3523** | −0.2204 |
| `change_ratio` | 1,936 | 0.1529 | 0.1952 | **0.5755** | −0.3803 |
| **overall** | **39,686** | **0.5084** | **0.5380** | **0.7474** | **−0.2094** |

**SatQuery loses on every one of the eight question types.** It also sits below its own majority constant on three of them. Its oracle result of 0.9975 says the arithmetic answer layer is not the problem — a ground-truth change map plus SatQuery's deterministic rules answers 99.75% of CDVQA correctly. **The whole deficit is the 6.26M-parameter ResNet-18 semantic-change segmenter**, whose change-class mIoU is 0.2636.

### 6.3 Category-A and near-A comparisons against task specialists

The Top-5 models do not report LEVIR-CD, BigEarthNet mAP or RSICD corpus BLEU, so the honest comparator for those three rows is the task-specialist literature.

| Task | Dataset & split | Metric | SatQuery *(measured)* | Best published | Δ | Cls |
|---|---|---|---|---|---|---|
| Binary change detection | **LEVIR-CD official split, 256px tiling** | change-class F1 | **0.5597** (49,543-param model, 4 epochs) | **≈0.9227** (PhyUnfold-Net, 2026); ChangeRWKV-B 0.8601; ConvFormer-CD/48 0.8530 | **−0.363** | **A** |
| Binary change detection | as above | change-class IoU | **0.3886** | ChangeDA 0.8565; ChangeRWKV-B 0.7546 | −0.366 to −0.468 | **A** |
| Multi-label land cover | BigEarthNet-19 (**SatQuery: 30k-patch subset, partition shard**) | macro mAP | **0.2854** | **0.7998** (ResNet50, BigEarthNet-19 benchmark); SeCo AP 0.8262 | **−0.514** | **B** |
| Captioning | RSICD test (n=1,093) | BLEU-4 | **0.2446** *sentence-mean, smoothed* | **0.6574** (RSGPT, corpus BLEU-4) | not subtractable | **B** |
| Change captioning | LEVIR-CC test (n=1,929) | BLEU-4 | **0.3063** *1 ref* / 0.5686 aggregate | ≈**0.6550** (SAGE-CC), 0.6530 (KCFI), CIDEr 140.23 (SAT-Cap) | not subtractable | **C** |
| Referring grounding | DIOR-RSVG (**SatQuery: self-made split**) | Acc@0.5 | **0.0762** | **0.9441** (Earth-OneVision); 0.9287 (LHRS-Bot-Nova); 0.749 (TinyRS-R1, 2B) | not subtractable, but the order of magnitude is unambiguous | **C** |
| Opt–SAR land cover | WHU-OPT-SAR | **SatQuery: tile mAP** vs **theirs: mIoU/OA** | 0.7778 optical / 0.7714 fused | PAD mIoU 56.26, OA 84.56, mF1 70.14 | not comparable | **C** |

The LEVIR-CD and BigEarthNet rows are the two places where SatQuery's own components are measured on essentially standard footing and are far behind — but both are **capacity-and-budget** gaps, not method gaps: a 49,543-parameter change detector and a 30,000-patch, 3-epoch, dim-64 land-cover encoder are not attempting to compete with the cited work.

### 6.4 Reliability, where the comparison runs the other way

| Property | SatQuery | Any Top-5 model | Nearest published analogue |
|---|---|---|---|
| Calibrated confidence with reported ECE | **0.0668 → 0.0034** (change mask, affine fit; temperature scaling was tried and rejected) | **not reported by any** | — |
| Explicit refusal metric with a lexical-shortcut control | recall 0.4118, false-refusal 0.0077, probe 0.1667 | **not reported by any** | VHM's HnstD deceptive-question set (AAAI 2025) is the nearest idea; different construction and metric |
| Entailment gate on generated prose | built, benchmarked | **not reported by any** | — |
| Provable illegal-plan rate | **0 / 600** | **not reported by any** | — |
| Negative results published | complementarity −0.0064; refusal 16.7% image-conditional; land-cover head worse than always-negative at τ=0.5; CDVQA below constant on 3 of 8 types | rarely | — |

This is a real and defensible advantage, and it is also the smallest kind: it says SatQuery measures things others do not, not that SatQuery answers questions better.
---

## 7. Local SatQuery evaluation (measured during this audit)

### 7.1 What was run, and how

The repository's **existing** evaluation code was used. No metric was invented, no training or inference code was modified, no checkpoint was written, moved or deleted.

- **Generation path:** `evaluation/track_b_eval.py:Adapter` — the same 4-bit NF4 quantisation, the same `SYSTEM_PROMPT` from `satquery/tools/rs_vqa.py`, the same chat template and the same greedy decode (`do_sample=False`, `max_new_tokens=48`) the deployed tool uses.
- **Scoring:** `evaluation/track_b_eval.py:score`, `token_f1` and `evaluation/refusal.py`, plus `evaluation/metrics/vqa.py:normalise_answer` for the per-type re-scoring.
- **The only addition** was a `BaseOnly` handle that runs the identical path with **no PEFT adapter attached**, because `track_b_eval.py`'s command line cannot express "score the base model". It lives in the scratchpad, not in the repository.
- **Data:** `data/instruct_mix/val.jsonl` — the same 534-item held-out split every published Track B number uses (322 `whu_opt_sar`, 207 `rsvqa_lr`, 5 `synthetic_refusal`; 517 answerable + 17 refusal items). `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
- **Hardware:** RTX 4050 Laptop, 6,141 MiB, torch 2.13.0+cu126, 2.6–2.8 GiB peak VRAM.

Two runs: the **full 534-item split** for every checkpoint, and a **207-item RSVQA-LR-only run with per-example predictions dumped**, so the slice could be re-scored under the question-type convention the literature uses (§5.2).

### 7.2 Full held-out split, 534 items — all four checkpoints

| | **base** (no adapter) | **`track_b_v2`** deployed | **`track_b_v3_probe`** | **`track_b_v3`** |
|---|---|---|---|---|
| Trainable params | 0 | 37,152,768 | 82,726,912 | 82,726,912 |
| Optimiser steps | — | 300 | 200 | 2,000 |
| Vision params adapted | 0 | **0** | 22,859,776 | 22,859,776 |
| **Overall exact match** | 0.0793 | 0.3791 | 0.3946 | **0.4603** |
| **Overall token F1** | 0.2027 | 0.7927 | 0.8077 | **0.8550** |
| **`rsvqa_lr` exact match** (n=207) | 0.1981 | 0.6473 | 0.6715 | **0.7874** |
| **`whu_opt_sar` exact match** (n=322) | 0.0000 | 0.2000 | 0.2097 | **0.2419** |
| `whu_opt_sar` token F1 | 0.1771 | 0.8898 | 0.8987 | **0.9002** |
| **Refusal recall** (n=17) | 0.0000 | **0.4118** | 0.2941 | 0.3529 |
| False-refusal rate | 0.0000 | 0.0077 | **0.0019** | 0.0039 |
| Lexical-shortcut probe | 0.0000 | **0.1667** | 0.0000 | 0.0833 |
| — image-conditional refusal (`not_in_image`, n=12) | 0/12 | **2/12** | 0/12 | 1/12 |
| — lexical refusal (`out_of_scope`, `single_image_temporal`, `synthetic_refusal`, n=10) | 0/10 | 10/10 | 10/10 | 10/10 |
| Generation time (s/example) | 1.52 | — | 1.25 | 1.26 |

**All four columns were measured during this audit.** `v2`'s column also **reproduces the published run in `docs/assets/refusal/track_b_v2_fullval.json` exactly on all eight metrics** — §7.7.

**Three things this table settles.**

1. **The un-adapted base model refuses nothing.** 0/17 across every refusal category, lexical and image-conditional alike. It answers every impossible question. Refusal is created entirely by the adaptation.
2. **The base model scores zero exact match on the optical/SAR rows**, and 0.1771 token F1 — it produces prose, not answers.
3. **`track_b_v3` is the strongest checkpoint on every accuracy row and the weakest-but-one on refusal.** It beats the deployed v2 on overall exact match (+0.0812), token F1 (+0.0623), RSVQA-LR (+0.1401) and `whu_opt_sar` (+0.0419), while **losing** refusal recall (0.3529 vs 0.4118) and halving the image-conditional refusal count (1/12 vs 2/12). **Limitation L3 is not fixed by v3; it is slightly worse.**

### 7.3 How much of the base → adapter gain is answer *format*?

This has to be asked, because the base model is verbose and exact match punishes verbosity. Measured on the 207 RSVQA-LR items:

| | base | v2 | v3_probe | v3 |
|---|---|---|---|---|
| Mean answer length (tokens) | **8.74** | 1.00 | 1.00 | 1.00 |
| Exact match | 0.1981 | 0.6473 | 0.6715 | 0.7874 |
| **Lenient**: gold answer appears anywhere in the reply | **0.4831** | 0.6473 | 0.6715 | 0.7874 |

Under the lenient metric the base scores **0.4831**, so the base → v2 gain shrinks from **+0.4492 to +0.1642**: **roughly 63% of the headline exact-match gain is the model learning to answer in the dataset's format, not learning to see better.** The remaining ~37% is real, and the lenient metric is generous to the base — an 8.7-token reply has 8.7 chances to contain the word "yes" — so the true content gain lies between those two figures. Against v3 the lenient gain is +0.3043, of which format is the smaller share.

Format compliance is not worthless: the deployed pipeline needs short, parseable answers and the base model does not produce them. It simply must not be reported as perception.

### 7.4 The held-out RSVQA-LR slice, re-scored the way the literature scores it

207 questions, re-typed from RSVQA's own templated wording: **presence 68 · comparison 81 · count 57 · rural-urban 1**. 72.0% of gold answers are yes/no. **Zero train/val image overlap.**

**The honest constant baseline first** — the most common **training** answer per question type, applied to val, which is the same discipline the CDVQA work used:

| Type | n | Constant | Constant accuracy on val |
|---|---|---|---|
| presence | 68 | "yes" | 0.8088 |
| comparison | 81 | "no" | 0.7284 |
| count | 57 | "0" | 0.3509 |
| rural-urban | 1 | "urban" | 0.0000 |
| **overall** | **207** | — | **0.6473** |

**The train-fitted per-type constant scores 134/207 = 0.6473 — numerically identical to `track_b_v2`'s published headline.**

| Checkpoint | EM, all types | 95% CI | **Published convention** micro, presence+comparison+rural-urban (n=150) | presence (n=68) | comparison (n=81) | count (n=57) | vs. the constant |
|---|---|---|---|---|---|---|---|
| **base** | 0.1981 | 0.150–0.258 | 0.2733 | 0.1324 | 0.3951 | 0.0000 | 18 model-only / **111 constant-only** — far worse |
| **`track_b_v2`** | 0.6473 | 0.580–0.709 | 0.8133 | 0.8824 | 0.7531 | 0.2105 | **17 / 17 — no measurable difference** |
| **`track_b_v3_probe`** | 0.6715 | 0.605–0.732 | 0.8333 | 0.7500 | 0.9136 | 0.2456 | 30 / 25 — not significant |
| **`track_b_v3`** | **0.7874** | 0.727–0.838 | **0.9533** | **0.9559** | **0.9506** | 0.3509 | **35 / 6 — significant**, McNemar χ²≈20.5, **p<0.001** |
| *constant* | *0.6473* | — | *0.7600* | *0.8088* | *0.7284* | *0.3509* | — |

**Three findings, in order of importance.**

**1. `track_b_v2`'s headline number is statistically indistinguishable from a constant.** On the all-types exact-match metric the project quotes, v2 and the train-fitted per-type constant both score 134/207 and disagree symmetrically — 17 items each way. There is **no evidence in that number** that v2 beats answering "yes" to presence, "no" to comparison and "0" to count.

This does **not** mean v2 is a constant. §7.5 shows it is not, and under the per-type convention it scores 0.8133 against the constant's 0.7600 and beats the constant on presence (0.8824 vs 0.8088) and comparison (0.7531 vs 0.7284). It means **the aggregate metric is the wrong instrument** — precisely the failure mode this project already diagnosed for CDVQA and wrote up rather than deleted.

**2. `track_b_v3` is the best checkpoint and beats the constant decisively — which was not expected.** It was trained under the label-masking defect and its loss sat at ~6.8 for 1,950 steps, so the reasonable prior was that it had learned nothing. The likely explanation is that the defect cut the supervised span **early**, so it supervised *more* than intended: every answer token was still supervised, merely swamped by ~89% `<|image_pad|>` targets. Flat loss reflected the placeholders, not the absence of learning. **The v3 run is not a null run, and the team should not discard it.**

**3. Counting is not learned by any checkpoint.** v3's count accuracy is **0.3509 — exactly the constant's**. Of its 20 correct count answers, **19 are the gold "0"s**. What has been learned is *"is this class absent?"*, not *how many*. Every checkpoint fails the same way, and count is the type the published literature usually removes from the benchmark — which is why §5.2 matters.

### 7.5 Are these checkpoints just matching the answer prior?

Checked, because a model that reproduces the marginal distribution can look good on a skewed slice.

| | gold | base | v2 | v3_probe | **v3** |
|---|---|---|---|---|---|
| presence, yes / no | 55 / 13 | mostly prose | 55 / 13 | 40 / 28 | **54 / 14** |
| comparison, no / yes | 59 / 22 | 18 / 37 | **73 / 8** | 64 / 17 | **57 / 24** |
| presence accuracy | — | 0.1324 | 0.8824 | 0.7500 | **0.9559** |
| comparison accuracy | — | 0.3951 | 0.7531 | 0.9136 | **0.9506** |

- **v2's comparison answers are close to a constant**: 73 "no" out of 81 against a gold split of 59/22, for 0.7531 — barely above the 0.7284 all-"no" constant. On comparison, v2 is mostly saying "no".
- **v3 is genuinely discriminating.** Its marginals track gold closely (54/14 vs 55/13; 57/24 vs 59/22) *and* its per-item accuracy is 0.95+. Matching the marginal alone caps accuracy near 0.72–0.81; both facts must hold together, and both do.
- **v3_probe traded presence for comparison** — comparison rose to 0.9136 while presence fell to 0.7500 and its presence marginal drifted to 40/28. At 200 steps the prior has moved but the discrimination has not yet arrived.

### 7.6 What this says about the 82.7M-parameter visual adaptation

| | v2 | v3 | Difference |
|---|---|---|---|
| Trainable LoRA parameters | 37,152,768 | **82,726,912** | +45,574,144 |
| **Vision-tower parameters adapted** | **0** | **22,859,776** (3.42% of the tower) | — |
| LoRA rank / α | 16 / 32 | 32 / 64 | — |
| Target modules | 7 (language only) | 11 (language + `attn.qkv`, `attn.proj`, `merger.mlp.0/2`) | — |
| Optimiser steps | 300 | 2,000 | ×6.7 |
| Held-out RSVQA-LR, all types | 0.6473 | **0.7874** | **+0.1401** |
| Published-convention micro (n=150) | 0.8133 | **0.9533** | **+0.1400** |
| Full-val overall exact match | 0.3791 | **0.4603** | **+0.0812** |
| `whu_opt_sar` exact match | 0.2000 | **0.2419** | **+0.0419** |
| Refusal recall | **0.4118** | 0.3529 | **−0.0589** |
| Beats the per-type constant? | **no** | **yes, p<0.001** | — |

**This is the evidence the brief asks for, and it is positive — with three caveats that must travel with it.**

1. **The comparison confounds three changes at once**: LoRA rank (16→32), vision-tower targeting (0→22.9M parameters), and training length (300→2,000 steps). **Nothing here isolates the vision-tower contribution.** The 200-step probe with the *identical* 82.7M configuration scores 0.6715 — barely above v2 — which suggests **steps, not vision parameters, are doing much of the work.** The isolating ablation, a rank-32 **language-only** arm at 2,000 steps, has not been run. §12 P2.
2. **n=207 on a validation slice**, and the underlying run was trained under a defect. The +0.1401 gain exceeds the ±0.065 interval, so it is real *on this slice*; it is not a test-split result.
3. **Refusal got worse.** v3 loses 0.0589 refusal recall and halves image-conditional refusals. Whatever v3 bought on accuracy, it did not buy on the reliability property the project cares most about.

### 7.7 Reproduction of the published `track_b_v2` numbers — exact

v2's full 534-item run was re-executed during this audit with the same code, the same split and the same greedy decode. **All eight published metrics reproduce to twelve decimal places.**

| Metric | Published 2026-09-01 | Reproduced 2026-09-03 | Δ |
|---|---|---|---|
| Overall exact match | 0.379110251451 | 0.379110251451 | **0** |
| Overall token F1 | 0.792742410226 | 0.792742410226 | **0** |
| `rsvqa_lr` exact match | 0.647342995169 | 0.647342995169 | **0** |
| `whu_opt_sar` exact match | 0.200000000000 | 0.200000000000 | **0** |
| `whu_opt_sar` token F1 | 0.889831697055 | 0.889831697055 | **0** |
| Refusal recall | 0.411764705882 | 0.411764705882 | **0** |
| False-refusal rate | 0.007736943907 | 0.007736943907 | **0** |
| Lexical-shortcut probe | 0.166666666667 | 0.166666666667 | **0** |

Generation took 697.17 s (**1.31 s/example**), load 24.01 s.

**Why this matters beyond bookkeeping.** It establishes that the deployed inference path is deterministic end to end — quantisation, chat template, greedy decode and metric code alike — so the differences reported in §7.2 and §7.4 between base, v2, v3_probe and v3 are attributable to the checkpoints and to nothing else. Every number in §7 is now *(measured here)*; the distinction the brief asks for between published and reproduced collapses for this table, because they are identical.

### 7.8 Could any Top-5 model be evaluated here?

Assessed, not attempted. Nothing was downloaded.

| Model | Weights | Fits 6 GiB at 4-bit? | Verdict |
|---|---|---|---|
| Earth-OneVision | none released | — | **No** |
| RingMo-Agent | not stated | — | **No** |
| EarthMind | code public, weights unstated | 4B — yes | **Blocked** on weight availability |
| **EarthDial** | **public, CC BY 4.0** | 4B (Phi-3-mini + InternViT-300M) — **yes** | **Feasible.** Needs a multi-GB download plus its evaluation harness; not made |
| **EarthGPT** | **public** | 7B + dual vision encoders — marginal | **Possible but tight** |
| *ref:* TinyRS-R1 | **public** | 2B, publishes 4.6 GB — **yes** | **The easiest external baseline to obtain**, though optical-only |

**What would be needed before any of this is attempted:** disk for the weights, each project's own evaluation harness and prompt format, and — critically — the *evaluation datasets* those models report on (RSVQA-LR's official test split, VRSBench imagery from DOTA/DIOR). Downloading a model without its benchmark buys nothing.
---

## 8. Apples-to-apples: what may and may not be claimed

### A. Directly comparable — same dataset, split, task, metric, protocol

**There are exactly two, and SatQuery loses both.**

| # | Comparison | SatQuery | External | Verdict |
|---|---|---|---|---|
| A1 | **CDVQA test1** — 39,686 questions / 968 pairs, overall accuracy | **0.5380** | Qwen3.5-2B **0.7474** · VisTA 0.7310 · SOBA 0.6920 · CDVQA baseline 0.6590 | **Substantially behind.** −0.2094 vs SOTA, −0.1210 vs the 2021 baseline. Loses on all 8 question types |
| A2 | **LEVIR-CD** — official split, standard 256px tiling, change-class F1 | **0.5597** | PhyUnfold-Net ≈**0.9227** · ChangeRWKV-B 0.8601 · ConvFormer-CD/48 0.8530 | **Substantially behind.** −0.363 F1. Comparing a 49,543-parameter screening detector to full change-detection networks |

Two further comparisons are *internally* Category A — both arms ours, identical split, decode path and metric code — and both are worth stating for the record:

| # | Comparison | Verdict |
|---|---|---|
| A3 | **Base `Qwen2.5-VL-3B-Instruct` vs the adapters**, identical 534-item val split | **The adaptation wins decisively.** RSVQA-LR 0.1981 → 0.6473 (v2) → 0.7874 (v3); refusal recall 0.0000 → 0.4118 / 0.3529; `whu_opt_sar` 0.0000 → 0.2419. §7.2 |
| A4 | **`track_b_v2` vs a train-fitted per-type constant** on the 207-question RSVQA-LR slice | **No measurable difference** — both 134/207, discordant 17 vs 17. `track_b_v3` *does* beat the constant, 35 vs 6, p<0.001. §7.4 |

### B. Partially comparable — clearly labelled, never used to claim victory

| # | Comparison | Mismatch | What may be said |
|---|---|---|---|
| B1 | RSVQA-LR, **as SatQuery publishes it**: 0.6473 (v2) / 0.7874 (v3) vs published 89.61–92.91 | split (val vs test), n (207 vs ~10k), count included vs excluded, all-types exact match vs per-type accuracy | **Nothing about relative quality.** The only count-inclusive published anchor is the RSVQA baseline's OA 79.08; v2 sits 14.4 points below it and v3 0.3 points below it, on a different split |
| B1b | RSVQA-LR, **re-scored under the published convention** (§7.4): 0.8133 (v2) / 0.9533 (v3), n=150 | split (validation vs official test), n (150 vs ~10k) | *"SatQuery's VQA component is plausibly in the same band as published RS-VLMs."* **Not** that it matches or beats any of them — a 150-question validation slice cannot establish that |
| B2 | RSICD captioning: 0.2446 vs RSGPT 65.74 | sentence-mean smoothed BLEU vs corpus BLEU (same official test split, same 5 references) | Split and reference count match; **the metric does not**. The repository's own code says not to make this comparison |
| B3 | BigEarthNet-19: mAP 0.2854 vs ResNet50 0.7998 | 30k patches / 3 epochs / dim 64 vs full-corpus training; partition shard vs recommended split | A large gap that is explained by budget, not by method |
| B4 | Latency: 2,670 ms vs TinyRS-R1 689 ms / TinyRS 90 ms | different hardware, token budget, quantisation, unmerged adapter | Order-of-magnitude context only |
| B5 | Vision-adaptation share: v3 adapts 3.42% of the vision tower vs Top-5 models that train or replace theirs | different adaptation strategies | Structural observation, not a score |

### C. Not comparable — must not be used to claim anything

| # | Comparison | Why not |
|---|---|---|
| C1 | Grounding 0.0762 vs 0.9441 | SatQuery's split is self-made; `run_metadata` says so. The *magnitude* is nonetheless not in doubt |
| C2 | LEVIR-CC 0.3063 vs ≈65.5 | one reference vs five; the scale difference is mostly the protocol |
| C3 | WHU-OPT-SAR 0.7778 mAP vs PAD 56.26 mIoU | different tasks — tile-level multi-label vs pixel-level segmentation |
| C4 | SatQuery full-val 0.3791 vs anything | the benchmark exists only in this repository |
| C5 | Refusal recall 0.4118 vs VHM honesty | different construction, different metric |
| C6 | ECE 0.0034, illegal-plan rate 0/600, complementarity gain, lexical-shortcut probe, entailment gate | **no comparable model reports any of them** |
| C7 | "SatQuery" as a system vs any Top-5 model | different objects. SatQuery is an agent over specialists with a verifier and a trace; the Top 5 are single end-to-end VLMs. §10 separates the two |

### The rule this audit applied

> A number is a comparison only when the dataset, the split, the task definition, the metric implementation and the decoding protocol all match. Two of the roughly twenty candidate comparisons met that bar.
---

## 9. Parameter and resource comparison

SatQuery's row is **measured on this machine during this audit**; every other row is as published. Blank cells are genuinely unreported — no paper in the Top 5 states VRAM, and only one states latency.

| Model | Total params | Trainable params | Vision params | Image resolution / context | VRAM | Inference cost | Public weights |
|---|---|---|---|---|---|---|---|
| **SatQuery** (v2 deployed) | **4,031,034,874** system¹ | **46,412,791** (1.15%)² | **668,684,288** base tower, **0 adapted** | Qwen2.5-VL dynamic; measured on 256×256 and 512×512 inputs; 128k text context | **2.6–2.8 GiB measured** (4-bit NF4, RTX 4050 Laptop 6 GiB) | **2,670 ms** for 14 tokens @256×256 (trace `run_53198df6bbd4`); **1.52 s/example** measured over 534 mixed items, base model | **No** — licence undecided; one head licence-blocked |
| **SatQuery** (v3 experimental) | 4,031,034,874 | **91,986,935** (2.28%)³ | **22,859,776 adapted** (3.42% of the tower) | as above | as above | slower than v2 (LoRA also fires inside the vision tower) | No |
| SatQuery base only | 3,754,622,976 | 0 | 668,684,288 (17.8%) | as above | 2.6 GiB | 1.52 s/example measured | Yes (Qwen2.5-VL-3B-Instruct) |
| **Earth-OneVision** | 2B | not stated (full fine-tune, 8×H100) | SigLIP-2 0.3B | native dynamic resolution; 5,400-token max sequence | not reported | not reported | **No** |
| **EarthDial** | 4B | not stated | InternViT-300M | adaptive high-resolution block | not reported | not reported | **Yes** (CC BY 4.0) |
| **RingMo-Agent** | 3B | LoRA rank 64 (count not stated) | SigLIP-SO400M-384 | 384 base | not reported | not reported | not stated |
| **EarthMind** | 4B | not stated | InternVL2 multi-scale + SAM2 + GPT4RoI | multi-scale patches | not reported | not reported | code yes, weights not stated |
| **EarthGPT** | LLaMA-2 class (7B) | not stated | DINOv2 ViT-L/14 + CLIP ConvNeXt-L | dual-encoder | not reported | not reported | **Yes** |
| *ref:* **TinyRS** (2B) | 2B | not stated | Qwen2-VL-2B tower | — | **4.4 GB** | **90 ms** | **Yes** |
| *ref:* **TinyRS-R1** (2B) | 2B | not stated | Qwen2-VL-2B tower | — | **4.6 GB** | **689 ms** | **Yes** |
| *ref:* published 7B RS-VLM (TinyRS-R1 paper's own measurement) | 7B | — | — | — | **16.6 / 16.8 GB** | **216 / 1,990 ms** | — |
| *ref:* **GeoChat** (7B) | 7B | LoRA | CLIP ViT-L/14-336 | 504×504 | not reported | not reported | **Yes** |
| *ref:* **LHRS-Bot-Nova** (8B) | 8B | not stated | SigLIP-L/14 @336 | 336×336 | not reported | not reported | **Yes** |

¹ base 3,754,622,976 + v2 adapter 37,152,768 + 7 specialist heads 9,260,023 + frozen DeBERTa entailment gate 184,424,963.
² v2 adapter 37,152,768 + specialists 9,260,023.
³ v3 adapter 82,726,912 + specialists 9,260,023.

### 9.1 The trade-off, stated plainly

**A larger model is not automatically better, and this audit contains a clean demonstration.** Earth-OneVision at **2B** reports RSVQA-LR 92.91 and DIOR-RSVG P@0.5 94.41 — beating 7B–72B models on the paper's own account. TinyRS at **2B** reports a VQA average of 83.5, equal to GeoChat at **7B**, on one third of the memory and latency. Parameter count is not the axis these results move along.

The axes that do move them, in the order this audit's evidence supports:

1. **Instruction-data scale and breadth.** The single largest visible difference between SatQuery and the Top 5 is not architecture. It is **4,806 training examples against 11.11M (EarthDial), ~34M QA pairs (Earth-OneVision) and >3M pairs (RingMo-Agent)** — three to four orders of magnitude.
2. **Training budget.** SatQuery's v2 adapter is **300 optimiser steps, 6 h 26 m on one 6 GiB laptop GPU**. Earth-OneVision used 8×H100-80GB for two 4-epoch stages over 34M pairs. These are not the same experiment and should not be described as though they were.
3. **Whether the vision tower is adapted at all.** v2 adapts **zero** vision parameters; every Top-5 model adapts or replaces its visual encoder. v3 is SatQuery's first attempt at that (22.9M vision-side parameters) and is not yet validated.
4. **Parameter count** — last, and the weakest of the four.

**What SatQuery buys for its 2.6 GiB.** It runs the whole pipeline — VQA, seven specialists, an NLI entailment gate, a physics verifier and a georeferenced evidence exporter — inside a 6 GiB consumer laptop GPU, offline, with an end-to-end trace. No Top-5 model claims that envelope, and three of the five cannot be run at all because nothing has been released. That is a genuine deployment-envelope advantage and it is not an accuracy claim.

**What it costs.** 2,670 ms for a 14-token answer at 256×256 is roughly **30× TinyRS's 90 ms** and **4× TinyRS-R1's 689 ms** on unstated but presumably datacentre hardware. On a laptop 4050 with 4-bit weights and an unmerged LoRA, that is expected; merging the adapter into the base weights and serving at fp16 on a larger card would remove most of it. It is a latency figure to state, not to hide.
---

## 10. Competitive-position assessment

### 10.1 Final ranking of the Top 5

| Rank | Model | Params | Why it ranks here |
|---|---|---|---|
| **1** | **EarthDial** | 4B | Highest composite score (22/24). It is the only Top-5 model that combines the full capability set SatQuery needs — VQA, captioning, grounding-as-referred-detection, bi-temporal change, multispectral land cover, SAR — with **public weights under an explicit licence**, published numbers across all of them, and a size that could actually be run on this project's hardware. Reproducibility is what puts it above Earth-OneVision. |
| **2** | **Earth-OneVision** | 2B | **The strongest model on published capability, and it would be #1 on the numbers alone**: six sensor modalities, RSVQA-LR 92.91, DIOR-RSVG 94.41, SARLANG-Bench 80.68, EarthMind-Bench fusion 81.94 — all at 2B, smaller than SatQuery's base. It drops one place because **nothing has been released** — no code, no weights, no dataset, one preprint version — so no claim in it can be checked. |
| **3** | **RingMo-Agent** | 3B | Same parameter class as SatQuery and the same LoRA-on-a-frozen-VLM strategy, spanning optical + SAR + infrared across satellite and UAV platforms. Highest published RSVQA-LR *presence* figure found (93.10) and a real SAR detection number (SARDet-100k mAP@50 53.84). Ranked below Earth-OneVision on modality breadth; release status is not stated. |
| **4** | **EarthMind** | 4B | The most *architecturally* relevant model here: built on **Qwen2.5-3B**, the same LLM family as SatQuery, and it directly tests SatQuery's M6 hypothesis — its RGB+SAR fusion beats both single modalities (70.6 vs 69.0 / 67.5) where SatQuery's does not. Public code. Lower published numbers than the top three. |
| **5** | **EarthGPT** | ~7B | The historical anchor for multi-sensor RS-VLMs and still the most-cited. Public code and dataset, strong captioning (NWPU-Captions BLEU-4 65.5, CIDEr 192.6) and DIOR-RSVG mIoU 69.34. Ranked last on recency: 2024, and its VQA numbers have been overtaken. |

**Reference rows used throughout, not ranked:** GeoChat (7B, the universal baseline), LHRS-Bot-Nova (8B, best published optical-only VQA + grounding), TinyRS-R1 (2B, SatQuery's compute-class twin and the only model publishing VRAM and latency).

### 10.2 SatQuery's position, task by task

🟢 competitive/strong · 🟡 behind but plausible · 🔴 substantially behind · ⚪ cannot determine

| Capability | Verdict | The evidence, and its class |
|---|---|---|
| **VQA** | 🟡 **behind but plausible — and closer than the headline number suggests** | Cleanest statement, both arms ours: the adaptation lifts held-out RSVQA-LR exact match **0.1981 (base) → 0.6473 (v2) → 0.7874 (v3)**. Re-scored under the literature's own convention (presence + comparison + rural-urban, count excluded) the same slice gives **0.8133 (v2)** and **0.9533 (v3)** against published 89.61–92.91. That is Category **B** — 150 questions from the *validation* split against test splits ~60× larger — so it supports "plausibly in the same band", **not** a claim of parity or of beating anyone. Two caveats keep this at 🟡 rather than 🟢: v2's all-types number is statistically indistinguishable from a per-type constant (§7.4), and **count accuracy is exactly a constant for every checkpoint** |
| **Captioning** | ⚪ **cannot determine** on quality; 🔴 on diversity | RSICD BLEU-4 0.2446 is sentence-mean smoothed BLEU on the official test split; the corpus-BLEU literature (RSGPT 65.74) is a different metric, and the repository's own code forbids the comparison. What *is* determinable and bad: **146 unique captions across 1,093 images (13.4%)**. A model emitting 146 distinct sentences is not describing scenes individually, and no comparable model has that failure mode |
| **Grounding** | 🔴 **substantially behind** | Acc@0.5 **0.0762** against 0.9441 (Earth-OneVision), 0.9287 (LHRS-Bot-Nova) and **0.749 from a 2B model on 4.6 GB of VRAM**. The split is self-made so the subtraction is Category C, but nothing about the order of magnitude is in doubt. The PS's only grounding query routes here and the answer is usually wrong |
| **Land-cover / RS understanding** | 🔴 **substantially behind** | BigEarthNet-19 mAP **0.2854** against ResNet50's 0.7998. Category B — 30k patches, 3 epochs, dim 64 — but the head is also, by the project's own measurement, worse than always predicting negative at threshold 0.5, which is why it asserts on 0.25% of decisions. WHU-OPT-SAR transfer (Stage A2, mAP 0.7759 fine-tuned vs 0.7206 frozen probe) is the one healthy signal in this row |
| **Change understanding** | 🔴 **substantially behind** — and this is the best-evidenced verdict in the audit | Both Category-A comparisons live here. CDVQA test1 **0.5380** vs 0.7474 (identical split, identical metric), losing on all eight question types and beating a constant by only 3.0 points. LEVIR-CD F1 **0.5597** vs ≈0.9227. Change captioning is ⚪ (one reference vs five). The **oracle at 0.9975** says the design is right and the segmenter is the problem |
| **Multimodal / SAR** | 🔴 / ⚪ **split verdict** | ⚪ on capability: SatQuery has **no SAR VQA or SAR captioning result at all** — it has never been evaluated on SARLANG-Bench or any SAR benchmark, where Earth-OneVision reports 80.68. 🔴 on the specific claim M6 makes: SatQuery's complementarity gain is **−0.0064**, while EarthMind (+1.6 MCQ points) and Earth-OneVision (+1.24) both report fusion helping. SatQuery is the only one of the three whose fusion result is negative — though the metrics differ so completely that only the *sign* is comparable |
| **Calibration / reliability** | 🟢 **strong, and uncontested** | Change-mask ECE **0.0668 → 0.0034** with the transform choice recorded (affine accepted, temperature rejected); a three-component confidence with a named limiting component; an entailment gate on generated prose; refusal measured with a lexical-shortcut control; illegal-plan rate **0/600**. **No model in the Top 5 or the reference rows reports a single one of these.** The honest reading: SatQuery is uncontested here because nobody else competes, not because it won a contest |

### 10.3 Model-level versus system-level — kept separate, as the brief requires

| | SatQuery **model** (`rs_vqa_v1` = Qwen2.5-VL-3B + LoRA) | SatQuery **system** (agent + 7 specialists + verifier + trace) |
|---|---|---|
| What the Top 5 are | directly comparable in kind | **not comparable** — no Top-5 model is an agent, none has a physics verifier, none exports georeferenced evidence, none reports an illegal-plan rate |
| Where it stands | 🟡 on VQA, and the **v3** checkpoint is the reason; unevaluated on captioning, grounding and SAR — those run through specialists, not the VLM | 🔴 on task accuracy where measurable; 🟢 on auditability, calibration and deployment envelope |
| Which number belongs to which | 0.1981 / 0.6473 / 0.7874 (RSVQA-LR), refusal recall, token F1 | caption BLEU 0.2446, grounding 0.0762, CDVQA 0.5380, LEVIR-CD 0.5597, fusion −0.0064, ECE, illegal-plan rate |
| The trap to avoid | Quoting a specialist's number (caption BLEU 0.2446, grounding 0.0762, CDVQA 0.5380) as though it were the VLM's | Quoting the system's engineering properties as though they were accuracy |

**No single aggregate "SatQuery accuracy" is given anywhere in this document, and none should be.** No legitimate aggregate benchmark spans the five task families SatQuery covers; the closest candidates (VRSBench, GEOBench-VLM, XLRS-Bench) have not been run.
---

## 11. The largest measurable gaps

Ordered by how *defensible* the measurement is, not by how large the number looks. A gap on a Category-C row is not evidence.

### G1 — Change VQA. **−0.2094 overall accuracy. Category A.**

| | |
|---|---|
| SatQuery | **0.5380** OA, CDVQA test1, 39,686 questions / 968 pairs, 100% coverage |
| Best comparable external | **0.7474** OA — Qwen3.5-2B change-VQA (Bazi et al., Apr 2026), identical split |
| Also ahead of SatQuery | VisTA 0.7310 · SOBA 0.6920 · the 2021 CDVQA baseline itself 0.6590 |
| Difference | **−0.2094** vs SOTA; **−0.1210** vs a five-year-old baseline; **+0.0296** vs a constant |
| Diagnosis | **Not the answer layer.** SatQuery's own oracle over ground-truth change maps scores **0.9975**, so the deterministic arithmetic that turns a change map into an answer is essentially exact. The whole deficit is the **6.26M-parameter ImageNet-ResNet-18 semantic-change segmenter** at change-class mIoU **0.2636**, with per-class IoU of 0.068 (water), 0.071 (playgrounds), 0.098 (trees) — precisely the rare classes that `smallest_change`, `largest_change` and `change_to_what` ask about |
| Category of cause | **Data + training budget, with an architectural component.** 1,600 training pairs; training loss still falling at epoch 40 while val mIoU flattened. Every published method on SECOND starts from a stronger pretrained backbone |

**This is the most important row in the audit.** It is the only gap measured on identical data with an identical metric, it is on the benchmark the problem statement itself prescribes for change VQA, and 93% of its headroom sits in one well-posed segmentation problem with an existing literature.

### G2 — Binary change detection. **−0.363 F1. Category A.**

| | |
|---|---|
| SatQuery | **F1 0.5597**, IoU 0.3886 — LEVIR-CD official split, standard 256px tiling (7,120 / 1,024 / 2,048), change-class F1 |
| Best comparable external | **F1 ≈0.9227** (PhyUnfold-Net, 2026); ChangeRWKV-B 0.8601; ConvFormer-CD/48 0.8530; ChangeDA IoU 0.8565 |
| Difference | **−0.363 F1, −0.468 IoU** |
| Diagnosis | The model is **49,543 parameters trained for 4 epochs**. It was chosen as a TinyCD-style screening detector and it behaves like one: precision 0.4426 against recall 0.7613 — deliberately over-calling change |
| Category of cause | **Architectural + training budget**, and deliberately so. This is a design choice meeting its consequences, not a bug |

### G3 — Referring grounding. **0.0762 vs 0.9441. Category C, but unambiguous in magnitude.**

| | |
|---|---|
| SatQuery | Acc@0.5 **0.0762**, Acc@0.7 0.0088, mIoU 0.1405 — DIOR-RSVG, **self-made split**, 6,359 training examples, **backbone from scratch** |
| Best comparable external | **0.9441** P@0.5 (Earth-OneVision) · 0.9287 (LHRS-Bot-Nova) · **0.749 (TinyRS-R1, a 2B model)** |
| Difference | Not subtractable across splits, but roughly nine in ten referring expressions are not localised |
| Diagnosis | A text-conditioned box regressor with a randomly-initialised visual backbone cannot learn localisation from 6,359 examples. Every comparable model uses a pretrained detector or a pretrained ViT |
| Category of cause | **Architectural.** More data or more epochs on the same design will not close this |

The comparison that matters here is **TinyRS-R1 at 74.9% on 2B parameters and 4.6 GB of VRAM** — this is not a gap that requires a datacentre to close.

### G4 — Multi-label land cover. **−0.514 mAP. Category B.**

| | |
|---|---|
| SatQuery | **mAP 0.2854** all-band / 0.2573 Cartosat 4-band, BigEarthNet-19 |
| Best comparable external | **mAP 0.7998** (ResNet50 on the BigEarthNet-19 benchmark); SeCo AP 0.8262 |
| Difference | **−0.514** — but on 30,000 training patches, 3 epochs, dim 64, against full-corpus training |
| Diagnosis | The model card already records that at threshold 0.5 this head is **worse than always predicting negative** (0.2064 vs 0.1834), which is why `landcover_v1` asserts on ~0.25% of decisions |
| Category of cause | **Training budget.** The project's own ablation arms reached 0.4171 and 0.4310 on the same data, so 0.2854 is not even this design's ceiling |

### G5 — Instruction-data scale for the VLM adaptation. **Three to four orders of magnitude. Category B (an input, not a metric).**

| | |
|---|---|
| SatQuery | **4,806 instruction examples**, 300 optimiser steps, 6 h 26 m on one 6 GiB laptop GPU |
| Comparable external | EarthDial **11.11M** instruction pairs · Earth-OneVision **~34M** QA pairs, 8×H100-80GB · RingMo-Agent **>3M** pairs · EarthMind ~3.2M |
| Difference | **~2,300× to ~7,000×** |
| Diagnosis | This is the single clearest structural difference between SatQuery's VLM and the Top 5, and it is upstream of every VQA, captioning and grounding gap |
| Category of cause | **Data + compute.** It is not architectural: the base model, the LoRA method and the 4-bit recipe are the same family the Top 5 use |

### G6 — Cross-modal fusion direction. **Opposite sign. Category C (direction only).**

| | |
|---|---|
| SatQuery | complementarity gain **−0.0064** — fusion does **not** beat optical alone (WHU-OPT-SAR, tile-level mAP) |
| Comparable external | EarthMind: fusion **70.6** vs RGB 69.0 / SAR 67.5 (EarthMind-Bench MCQ) · Earth-OneVision: fusion **81.94** vs optical 80.70 / SAR 76.10 |
| Diagnosis | **Unknown, and this is the honest answer.** Three live possibilities, none tested: (a) the metric — tile-level multi-label mAP on 1,548 tiles may be too saturated to show a fusion effect; (b) the fusion module — a shared-encoder cross-attention head with 132,030 parameters against EarthMind's dedicated Modality Alignment and Modality Mutual Attention; (c) the dataset — WHU-OPT-SAR at 5 m may genuinely offer little SAR complementarity for the classes involved |
| Category of cause | **Unknown** — it is the only gap in this list that is not diagnosed |

---

## 12. Recommended improvement priorities

**All five items below appear on `docs/code-freeze.md` §"Explicitly out of scope after freeze".** Nothing here should be started without an explicit unfreeze decision by the team lead, recorded the way Unfreeze 1 was for the v2 retrain. This section says *what the evidence recommends*, not *what may be done today*.

### P1 — Give the CDVQA semantic-change segmenter a proper backbone and budget

**Expected gain: the largest and best-evidenced in this document.** SatQuery already proved the ceiling (oracle 0.9975) and localised the cause (change-class mIoU 0.2636, rare classes at IoU 0.07–0.10). The project's own ablation already showed a **+56% relative** jump in change-class mIoU from switching a from-scratch encoder to an ImageNet ResNet-18. A change-detection-pretrained encoder, full-resolution crops, and more than 40 epochs on 1,600 pairs are the three moves in ascending cost order.

- **Why it is the highest-impact item:** it is the only place with (i) a Category-A external comparison, (ii) a PS-prescribed benchmark, (iii) a proven ceiling, (iv) a diagnosed single cause, and (v) a few-GPU-hour fix.
- **Risk:** it changes a published number. Under `docs/code-freeze.md` that means a new dated section, not an edit.
- **Cost:** low — single GPU, hours not days.

### P2 — Commit the label-masking fix, re-run Track B, and run the one ablation that is missing

**The starting point changed during this audit.** `track_b_v3` — 2,000 steps, 82.7M trainable parameters, vision tower included — was assumed dead because its loss sat at ~6.8 for 1,950 steps under a label-masking defect. **Measured, it is the best checkpoint the project has** (§7.6): held-out RSVQA-LR 0.7874 against v2's 0.6473, 0.9533 against 0.8133 under the published convention, and it beats a per-type constant at p<0.001 where v2 does not. The defect cut the supervised span *early*, so it supervised more than intended rather than less; the flat loss was placeholder prediction, not absent learning.

Three things follow, in order:

1. **Commit the fix and its regression test** (`training/track_b_vlm_qlora.py` + `tests/test_vlm_label_masking.py`, currently uncommitted in the main checkout) so any future run is reproducible. The same diff adds held-out validation every N steps and writes `val_history.json`, which removes the "when do we stop?" gamble that made a 2,000-step run risky.
2. **Run the ablation that decides the brief's question.** v3 changes *three* things at once — LoRA rank 16→32, vision-tower targeting 0→22.9M parameters, and steps 300→2,000. The 200-step probe with the identical 82.7M configuration scores only 0.6715, which points at **steps** rather than vision parameters. **A rank-32, language-only, 2,000-step arm** separates them, and nothing else does. Until it is run, "the 82.7M visual adaptation is worthwhile" is not a supported claim; "82.7M parameters at 2,000 steps is worth +0.14 on this slice" is.
3. **Re-run v3's recipe with the corrected mask** and compare against the defective run on the same split. If the corrected run is better, v3 becomes the deployment candidate; if it is not, that is itself a finding worth publishing.

- **Cost:** one overnight run per arm on the existing 4050 (v2's 300-step run took 6 h 26 m; 2,000 steps is proportionally longer), plus ~12 minutes of eval each.
- **Note:** v3 is **worse** than v2 on refusal recall (0.3529 vs 0.4118) and on image-conditional refusals (1/12 vs 2/12). A checkpoint swap trades reliability for accuracy, and that trade should be made deliberately rather than by picking the higher headline.

### P3 — Close the evaluation gap before closing any more capability gaps

Seven of this audit's rows are ⚪ **only because SatQuery has not been evaluated on the benchmark the literature uses.** Two cheap changes convert opinions into numbers:

1. **Evaluate on the official RSVQA-LR *test* split**, reporting per-type accuracy (presence / comparison / rural-urban / count) *and* an overall accuracy that includes count. Today's 207-question validation slice cannot be compared to anything, and at ±0.065 it cannot resolve the differences the project has been reading from it.
2. **Evaluate on VRSBench.** It is the one public benchmark that scores captioning, grounding and VQA under a single protocol, four of the five Top-5 models report it, GeoChat's numbers are published for both zero-shot and fine-tuned settings, and the PS itself prescribes it. The blocker is a DOTA/DIOR imagery download, not modelling work.

This is the **cheapest** item on the list and arguably should precede P1 and P2, because without it the project cannot tell whether P1 or P2 worked in terms anyone outside the project can read.

### P4 — Replace the grounding head rather than retraining it

Acc@0.5 0.0762 from a from-scratch box regressor is an architecture verdict, not a tuning one. TinyRS-R1 reaches 74.9% at 2B parameters and 4.6 GB. The realistic options are a pretrained detector backbone, or routing grounding through the VLM itself with box tokens the way every Top-5 model does. Either is a rebuild.

### P5 — Diagnose the fusion result before trying to fix it

G6 is the only undiagnosed gap. Before changing the fusion module, separate the three hypotheses: re-score the existing triad under a **segmentation** metric (mIoU) to test whether tile mAP is hiding the effect; and re-split geographically rather than randomly by tile, since the current split is acknowledged as optimistic. Both are evaluation-only changes on data already on disk. **A negative result that survives a proper metric is a finding worth keeping**; the current one is a finding whose metric has not been challenged.

### What is explicitly *not* recommended

- **Do not scale parameters.** Earth-OneVision at 2B beats 7B–72B models on its own account, and TinyRS at 2B equals GeoChat at 7B. Nothing in this audit supports a bigger base model as the next move.
- **Do not chase the RSVQA-LR headline as currently defined.** It is an all-types exact match on a 207-question validation slice, and §7.4 shows a per-type constant achieves the same 0.6473 that v2 does. Fix the metric and the split (P3) before optimising against it.
- **Do not present `track_b_v3` as a validated result** — its ablation is confounded and its training run was defective — **and equally, do not discard it.** It is the best-measuring checkpoint the project has.
- **Do not swap the deployed checkpoint to v3 on the accuracy numbers alone.** It loses refusal recall.
---

## 13. Limitations and protocol mismatches

### 13.1 What this audit could not verify

| # | Limitation | Consequence |
|---|---|---|
| E1 | **No external model was run.** Three of the Top 5 (Earth-OneVision, RingMo-Agent, and EarthMind's weights) have released nothing runnable; EarthDial and EarthGPT publish weights but the multi-GB downloads were not made. | **Every external number is published and unreproduced.** If a paper's protocol differs from its description, this audit inherits the error. |
| E2 | **Earth-OneVision has no release at all** and only a v1 preprint (9 June 2026). Its numbers — the strongest in the table — cannot be checked by anyone. | Treat its column as a claim, not a measurement. |
| E3 | Some extracted tables carry values outside a metric's normal range: EarthDial's captioning "METEOR" column (up to 80.61), Earth-OneVision's LEVIR-MCI "METEOR 76.45" and xBD "ROUGE-1 92.33". | Those cells are **excluded** from this audit rather than quoted. Verify against the published PDFs before using them. |
| E4 | RingMo-Agent's parameter count, weight availability and full per-type RSVQA-LR breakdown could not all be read from the sources fetched. | Its row is partial and says so. |
| E5 | **VRSBench was not evaluated for SatQuery**, so the one benchmark that would give a single-protocol captioning + grounding + VQA comparison against four of five Top-5 models is missing. | The captioning and grounding verdicts stay ⚪. |
| E6 | The published CDVQA comparators (SOBA, VisTA, Qwen3.5-2B) are quoted from the Table III of Bazi et al. (Apr 2026), not from each original paper. | Numbers are second-hand within a peer venue; the split identity (39,686) was cross-checked and matches. |
| E7 | RSVQA-LR published figures come from six different papers with at least three different type-averaging conventions (presence+comparison; presence+comparison+rural-urban; overall accuracy). | Even the *external* RSVQA-LR column is not internally consistent. Do not rank Top-5 models against each other on it. |

### 13.2 Protocol mismatches, consolidated

Every one of these was verified in code or data during this audit, not assumed.

1. **RSVQA-LR**: validation subset vs official test; n=207 vs ~10k; count included (27.5%) vs excluded; whole-string exact match vs per-type accuracy. `docs/00` §3.5's "official split" wording should be corrected.
2. **RSICD captioning**: sentence-mean add-one-smoothed BLEU vs corpus BLEU. The repository's own code comment forbids the comparison.
3. **LEVIR-CC change captioning**: one reference vs five.
4. **DIOR-RSVG grounding**: self-made image-grouped split (`split_note: NO published split in this mirror`) vs the official test split; backbone trained from scratch vs pretrained.
5. **BigEarthNet**: 30,000-patch / 3-epoch / dim-64 training and a partition shard for evaluation vs full-corpus training on the recommended split; and SatQuery reports mAP where the Top-5 report accuracy or recall.
6. **WHU-OPT-SAR**: tile-level multi-label mAP vs pixel-level semantic segmentation mIoU/OA — different tasks. Also `split_method: deterministic random by tile; NOT geographic`, so absolute values are optimistic; the three-arm *comparison* is unaffected.
7. **whu_opt_sar VQA rows**: 22 of 322 validation images (6.8%) also appear in training. The `rsvqa_lr` rows have **zero** image overlap.
8. **Latency**: SatQuery measured on an RTX 4050 Laptop 6 GiB at 4-bit with an unmerged adapter; TinyRS-R1's 90/689 ms is on unstated hardware.
9. **Statistical power**: at n=207 the 95% interval around 0.6473 is about ±0.065. Differences smaller than ~6 points on that row are not resolvable, which is why the v1→v2 delta of +0.0048 was correctly published as reproduction rather than improvement.
10. **`track_b_v3` was trained under a defect, and still measures best.** The supervised span included ~89% `<|image_pad|>` tokens; the fix exists as an uncommitted change to `training/track_b_vlm_qlora.py` in the main checkout, with a regression test (`tests/test_vlm_label_masking.py`). Every v3 number in this document is therefore a measurement of a **defective run** and is labelled as such - which cuts both ways: it should not be quoted as a validated result, and it should not be dismissed as a null run either. Its advantage over v2 additionally confounds LoRA rank, vision-tower targeting and step count; the isolating ablation has not been run (§12 P2).

### 13.3 Things this audit deliberately did **not** do

- No architecture change, no training run, no hyper-parameter change.
- No checkpoint written, moved, renamed or deleted. `checkpoints/track_b_v1`'s corrupted adapters were left in place as evidence.
- No dataset downloaded and no external model downloaded.
- `make report` was **not** run; no file under `docs/assets/` was modified.
- No number in `docs/model-cards.md`, `docs/phase1-status.md` or `docs/00` was edited. Where this audit contradicts one of them (the RSVQA-LR "official split" wording), it says so here rather than editing there — consistent with `docs/code-freeze.md`.

---

## 14. Sources

### 14.1 Top-5 models

| Model | Primary source | Code / weights |
|---|---|---|
| EarthDial | [arXiv:2412.15190](https://arxiv.org/abs/2412.15190) — *EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues* | [github.com/hiyamdebary/EarthDial](https://github.com/hiyamdebary/EarthDial) |
| Earth-OneVision | [arXiv:2606.10819](https://arxiv.org/abs/2606.10819) — *Extending Remote Sensing MLLMs to More Sensor Modalities and Tasks* ([HTML](https://arxiv.org/html/2606.10819)) | none released |
| RingMo-Agent | [arXiv:2507.20776](https://arxiv.org/abs/2507.20776) — *A Unified Remote Sensing Foundation Model for Multi-Platform and Multi-Modal Reasoning* | not stated |
| EarthMind | [arXiv:2506.01667](https://arxiv.org/abs/2506.01667) — *Multi-Granular and Multi-Sensor Earth Observation with LMMs* | [github.com/shuyansy/EarthMind](https://github.com/shuyansy/EarthMind) |
| EarthGPT | [arXiv:2401.16822](https://arxiv.org/abs/2401.16822) — *A Universal Multimodal LLM for Multisensor Image Comprehension in Remote Sensing* (IEEE TGRS) | [github.com/wivizhang/EarthGPT](https://github.com/wivizhang/EarthGPT) |

### 14.2 Reference models

| Model | Source | Weights |
|---|---|---|
| GeoChat | [arXiv:2311.15826](https://arxiv.org/abs/2311.15826) · [CVPR 2024 open access](https://openaccess.thecvf.com/content/CVPR2024/html/Kuckreja_GeoChat_Grounded_Large_Vision-Language_Model_for_Remote_Sensing_CVPR_2024_paper.html) | [huggingface.co/MBZUAI/geochat-7B](https://huggingface.co/MBZUAI/geochat-7B) · [github.com/mbzuai-oryx/GeoChat](https://github.com/mbzuai-oryx/GeoChat) |
| LHRS-Bot-Nova | [arXiv:2411.09301](https://arxiv.org/abs/2411.09301) | [github.com/NJU-LHRS/LHRS-Bot](https://github.com/NJU-LHRS/LHRS-Bot) |
| TinyRS / TinyRS-R1 | [arXiv:2505.12099](https://arxiv.org/abs/2505.12099) | [github.com/aybora/TinyRS](https://github.com/aybora/TinyRS) |
| VHM | [arXiv:2403.20213](https://arxiv.org/abs/2403.20213) · [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32683) | [github.com/opendatalab/VHM](https://github.com/opendatalab/VHM) |

### 14.3 Benchmarks and datasets

| Benchmark | Source |
|---|---|
| RSVQA (LR/HR), original baseline numbers | [arXiv:2003.07333](https://arxiv.org/abs/2003.07333) · [rsvqa.sylvainlobry.com](https://rsvqa.sylvainlobry.com/#dataset) |
| RSVQA-LR-2k (the subset actually on disk) | [huggingface.co/datasets/dmarsili/RSVQA-LR-2k](https://huggingface.co/datasets/dmarsili/RSVQA-LR-2k) |
| VRSBench | [arXiv:2406.12384](https://arxiv.org/abs/2406.12384) · [vrsbench.github.io](https://vrsbench.github.io/) · [github.com/lx709/VRSBench](https://github.com/lx709/VRSBench) |
| CDVQA | [arXiv:2112.06343](https://arxiv.org/abs/2112.06343) · [github.com/YZHJessica/CDVQA](https://github.com/YZHJessica/CDVQA) |
| CDVQA comparators (CDVQA / SOBA / VisTA / Qwen3.5-2B, test1 & test2 OA/AA) | [arXiv:2604.18429](https://arxiv.org/abs/2604.18429) — *Revisiting Change VQA in Remote Sensing with Structured and Native Multimodal Qwen Models* |
| VisTA / CDQAG | [arXiv:2410.23828](https://arxiv.org/abs/2410.23828) · [like413.github.io/CDQAG](https://like413.github.io/CDQAG/) · [github.com/like413/VisTA](https://github.com/like413/VisTA) |
| LEVIR-CD | [justchenhao.github.io/LEVIR](https://justchenhao.github.io/LEVIR/) |
| LEVIR-CD SOTA (PhyUnfold-Net) | [arXiv:2603.19566](https://arxiv.org/abs/2603.19566) |
| LEVIR-CD SOTA (ChangeRWKV) | [arXiv:2603.19606](https://arxiv.org/abs/2603.19606) |
| LEVIR-CC SOTA (SAT-Cap) | [arXiv:2501.08114](https://arxiv.org/abs/2501.08114) |
| RSICD captioning SOTA (RSGPT) | [arXiv:2307.15266](https://arxiv.org/abs/2307.15266) |
| BigEarthNet / BigEarthNet-19 nomenclature | [arXiv:1902.06148](https://arxiv.org/abs/1902.06148) · [arXiv:2001.06372](https://arxiv.org/abs/2001.06372) · [bigearth.net](https://bigearth.net/) |
| BigEarthNet-MM and reBEN | [arXiv:2105.07921](https://arxiv.org/abs/2105.07921) · [arXiv:2407.03653](https://arxiv.org/abs/2407.03653) |
| BigEarthNet-19 CNN mAP baselines | [arXiv:2207.07189](https://arxiv.org/abs/2207.07189) — *An Open-source Benchmark Arena for Image Classification* |
| WHU-OPT-SAR | [github.com/AmberHen/WHU-OPT-SAR-dataset](https://github.com/AmberHen/WHU-OPT-SAR-dataset) |
| WHU-OPT-SAR SOTA (PAD, ASANet) | [arXiv:2504.19136](https://arxiv.org/abs/2504.19136) · [arXiv:2412.02044](https://arxiv.org/abs/2412.02044) |
| SARLANG-1M / SARLANG-Bench | [arXiv:2504.03254](https://arxiv.org/abs/2504.03254) |
| GEOBench-VLM | [arXiv:2411.19325](https://arxiv.org/abs/2411.19325) |
| XLRS-Bench | [arXiv:2503.23771](https://arxiv.org/abs/2503.23771) |
| DIOR-RSVG / RSVG | [ieeexplore.ieee.org/document/10056343](https://ieeexplore.ieee.org/document/10056343/) |
| Base VLM | [huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |

### 14.4 Considered and excluded (recorded for completeness)

[GeoGround](https://arxiv.org/abs/2411.11904) · [RSGround-R1](https://arxiv.org/abs/2601.21634) · [GeoEyes](https://arxiv.org/abs/2602.14201) · [ZoomEarth](https://arxiv.org/abs/2511.12267) · [GeoVista](https://arxiv.org/abs/2605.14475) · [RSUniVLM](https://arxiv.org/abs/2412.05679) · [SkyMoE](https://arxiv.org/abs/2512.02517) · [UniRS](https://arxiv.org/abs/2412.20742) · [SkyEyeGPT](https://www.sciencedirect.com/science/article/pii/S0924271625000206) · [MLRS "More with Less"](https://arxiv.org/abs/2607.15942) · [GeoVLM-R1](https://arxiv.org/abs/2509.25026) · [RSVLM-QA](https://arxiv.org/abs/2508.07918) · [TCSSM / BrightVQA](https://arxiv.org/abs/2508.08974) · [Vision-Language Modeling Meets Remote Sensing (survey)](https://arxiv.org/abs/2505.14361)

### 14.5 Internal sources

`docs/model-cards.md` · `docs/00-README-and-Requirement-Traceability.md` §3.1–3.6 · `docs/phase1-status.md` (CDVQA sections, 2026-08-29 → 08-30) · `docs/assets/refusal/track_b_v2_fullval.json` · `checkpoints/*/metrics.json` and `run_metadata.json` (all re-read and confirmed unchanged during this audit) · `evaluation/metrics/all_tasks.py`, `evaluation/metrics/vqa.py`, `evaluation/track_b_eval.py`, `training/prepare/instruction_mix.py`, `training/train_change_mask.py`, `training/train_caption.py`, `training/train_change_caption.py`, `training/train_grounding.py`, `training/train_optsar_fusion.py`, `training/track_a_full.py` · `trace_gpu_vqa2.json` (latency) · uncommitted diff to `training/track_b_vlm_qlora.py` in the main checkout (the label-masking fix).
---

## 15. Summary table and the six questions

### 15.1 The summary table

**Cells are not on a common scale and must not be read as a league table.** Each cell gives the strongest *defensible* figure for that model on that capability, with its dataset and metric, because no two of these models report the same thing. SatQuery's cells are its own measurements; every other cell is published and unreproduced.

| Model | VQA | Caption | Grounding | RS understanding | Multimodal / SAR | Overall evidence |
|---|---|---|---|---|---|---|
| **SatQuery** (3.75B base + 82.7M) | 🟡 **0.9533** micro, **0.7874** all-types — RSVQA-LR held-out slice, n=150/207, **validation** split *(measured)*. v2: 0.8133 / 0.6473 | ⚪ RSICD **0.2446** BLEU-4 *sentence-mean, smoothed* — metric not comparable; **13.4% unique captions** *(measured)* | 🔴 DIOR-RSVG **0.0762** Acc@0.5, **self-made split** *(measured)* | 🔴 BigEarthNet-19 **0.2854** mAP; WHU-OPT-SAR transfer 0.7759 *(measured)* | 🔴/⚪ **no SAR benchmark result exists**; opt–SAR complementarity **−0.0064** *(measured)* | **2 Category-A comparisons, both lost** (CDVQA −0.209, LEVIR-CD −0.363). Everything else B or C. 🟢 uncontested on calibration, refusal metrics, entailment and 0/600 illegal plans |
| **1. EarthDial** (4B) | RSVQA-LR **92.70** (P 92.58 / C 92.75), official test; RSVQA-HR 72.45 | RSICD ROUGE-1 33.77 / ROUGE-L 27.61; NWPU ROUGE-1 45.84; UCM 40.0 | referred detection only — NWPU VHR-10 mAP@0.5 11.4–39.1 | BigEarthNet RGB 68.82 / MS 69.94; AID 88.76; fMoW 70.03; xBD recall 96.37 | SAR ship mAP@0.5 6.06 / 26.02; multispectral, NIR, hyperspectral, bi-temporal | Broadest capability set with **public weights (CC BY 4.0)**; 11.11M training pairs |
| **2. Earth-OneVision** (2B) | RSVQA-LR **92.91**; RSVQA-HR 86.36; VRSBench-VQA 80.32 | RSICD METEOR 33.98; SARLANG complex captioning CIDEr 110.24 | DIOR-RSVG **94.41** P@0.5; OPT-RSVG 87.52; VRSBench-VG 90.77 | BigEarthNet-MS recall 75.74; UCMerced 91.83; WHU-RS19 97.31 | **SARLANG-Bench VQA 80.68**; EarthMind-Bench fusion **81.94** > optical 80.70 | **Strongest published numbers anywhere in this audit, at 2B** — but **no code, no weights, no dataset**, one preprint version |
| **3. RingMo-Agent** (3B) | RSVQA-LR **90.30** fine-tuned (P 93.10 / C 87.50); RSVQA-HR 79.58 zero-shot | UCM-Captions BLEU-4 **77.63**, CIDEr 373.68 | not reported | not reported | **SARDet-100k mAP@50 53.84**; IR-DET 59.88; satellite + UAV | Strong and recent (v3, Aug 2026); release status **not stated** |
| **4. EarthMind** (4B, Qwen2.5-3B) | VRSBench-VQA 78.9; RSVQA-HRBEN 74.0; EarthMind-Bench MCQ 69.0 | EarthMind-Bench captioning GPT-4 score 3.35/5; DIOR-RSVG region caption CIDEr 428.2 | VRSBench-VG 55.6 Acc@0.5; RRSIS-D 82.2 mIoU; RefSegRS 62.6 mIoU | BigEarthNet 70.4; AID 97.2; UC-Merced 95.0; SoSAT-LCZ42 58.3 | **fusion 70.6 > RGB 69.0 > SAR 67.5** — the published counter-example to SatQuery's M6 result; SAR ship mAP 13.6–36.8 | Same LLM family as SatQuery; **code public**, weights unstated |
| **5. EarthGPT** (~7B) | CRSVQA 82.00 supervised; RSVQA-HR **72.05** zero-shot | **NWPU-Captions BLEU-4 65.5**, METEOR 44.5, ROUGE-L 78.2, CIDEr 192.6 | DIOR-RSVG mIoU **69.34**, cIoU 81.54 | NWPU-RESISC45 93.84; CLRS 77.37 / NaSC-TG2 74.72 zero-shot | optical + SAR + infrared across 34 datasets (MMRS-1M); MAR20 AP@40 90.47 | Public code and dataset; the multi-sensor historical anchor, now 2024-vintage |
| *ref:* TinyRS / TinyRS-R1 (2B) | RSVQA-LR P 90.4 / C 89.9 / rural 92.0; VQA avg **83.5** | not a headline capability | DIOR-RSVG **69.4 / 74.9** P@0.5 | classification avg 81.0 / 85.6 | optical only | **The compute-class twin**: public weights, **4.4–4.6 GB VRAM, 90–689 ms** |
| *ref:* GeoChat (7B) | RSVQA-LR 90.70 fine-tuned; VRSBench-VQA 60.6 | VRSBench BLEU-4 13.8 (ft) / 1.4 (zero-shot) | VRSBench Acc@0.5 39.6 (ft) / 12.9 (zero-shot) | AID 73.5; NWPU 89.4 | optical only | The universal baseline; public weights |

### 15.2 The six questions, answered

#### 1. Where does SatQuery currently stand versus the strongest comparable remote-sensing VLMs?

**Behind where the comparison is valid, unmeasured where it is not, and ahead only on properties nobody else reports.**

Concretely: on the two benchmarks where protocol genuinely matches — **CDVQA test1** and **LEVIR-CD** — SatQuery is **20.9 and 36.3 points behind** the current state of the art, and 12.1 points behind a 2021 baseline on CDVQA. On the tasks the Top 5 actually compete on (RSVQA-LR, VRSBench, DIOR-RSVG, SARLANG-Bench), SatQuery has either an incomparable protocol or **no result at all** — it has never been evaluated on VRSBench, RSVQA-HR, or any SAR benchmark.

The one place SatQuery is clearly ahead is **measurement discipline**: calibrated confidence with a reported ECE, a refusal metric with a lexical-shortcut control, an entailment gate, a provable illegal-plan rate, and negative results published rather than buried. **None of the Top 5 reports any of these.** That is a real advantage and a narrow one — it says SatQuery measures things others do not, not that it answers better.

There is also a scope difference that no table captures: SatQuery is an **agent with a verifier and an auditable trace running in 2.6 GiB on a consumer laptop**, offline. The Top 5 are single end-to-end VLMs, three of which cannot be run by anyone outside their labs.

#### 2. Which metrics are already competitive?

Four, with their qualifications stated:

| Metric | SatQuery | Comparable external | Status |
|---|---|---|---|
| **RSVQA-LR presence** | **0.9559** (v3), 0.8824 (v2) — n=68, val slice | 87.46 (RSVQA baseline) – 93.10 (RingMo-Agent) | 🟡 **inside the published band**, but Category B |
| **RSVQA-LR comparison** | **0.9506** (v3) — n=81, val slice | 81.50 – 92.75 | 🟡 **inside the band** for v3; v2's 0.7531 is not |
| **Calibration (ECE)** | 0.0668 → **0.0034** | **nobody reports it** | 🟢 uncontested |
| **Illegal-plan rate** | **0 / 600** | **nobody reports it** | 🟢 uncontested |

Also worth naming, though not a competitive metric: the **Stage A2 transfer result** (WHU-OPT-SAR fine-tuned mAP 0.7759 vs frozen probe 0.7206) is healthy evidence that Track A's encoder learned transferable features, and the **band-dropout retention 0.9015** is a genuine engineering result for the Cartosat 4-band constraint that no comparable model addresses.

**Nothing here supports a claim that SatQuery beats a named model on a named benchmark.** The presence and comparison rows are on a 68- and 81-question validation slice against test-split figures 60× larger.

#### 3. Which metrics are significantly behind?

In descending order of how defensible the measurement is:

| Metric | SatQuery | Best comparable | Gap | Class |
|---|---|---|---|---|
| **LEVIR-CD change-class F1** | 0.5597 | 0.9227 | **−0.363** | **A** |
| **CDVQA test1 overall accuracy** | 0.5380 | 0.7474 | **−0.209** | **A** |
| **BigEarthNet-19 mAP** | 0.2854 | 0.7998 | **−0.514** | B |
| **DIOR-RSVG Acc@0.5** | 0.0762 | 0.9441 (0.749 at 2B) | order of magnitude | C |
| **RSVQA-LR count** | 0.3509 — exactly a constant | 0.6701 (RSVQA baseline) | **−0.319** | B |
| **Caption diversity** | 146 unique / 1,093 images (13.4%) | not reported, but no comparable model degenerates this way | — | — |
| **Image-conditional refusal** | 2/12 (v2), 1/12 (v3) | not reported | — | — |

#### 4. What is the strongest evidence that our 82.7M-parameter visual adaptation is worthwhile?

**Two pieces of evidence, in order of strength.**

**(a) The adaptation as a whole is unambiguously worthwhile**, and this is the cleanest Category-A comparison in the audit because both arms are ours, on the identical split, through the identical decode path and the identical metric code:

| | base `Qwen2.5-VL-3B-Instruct` | `track_b_v3` (82.7M) |
|---|---|---|
| Held-out RSVQA-LR exact match | **0.1981** | **0.7874** |
| `whu_opt_sar` exact match | **0.0000** | **0.2419** |
| Refusal recall | **0.0000** (0/17) | 0.3529 |
| Overall token F1 | 0.2027 | 0.8550 |

The base model **cannot refuse at all** and scores **zero** on the optical/SAR rows. Both capabilities exist only because of the adaptation.

**(b) The specific 82.7M configuration beats the 37.2M one, and beats a constant where 37.2M does not.** v3 reaches 0.7874 all-types and **0.9533** under the published convention, against v2's 0.6473 / 0.8133, and it beats the train-fitted per-type constant on 35 items against 6 (**McNemar χ²≈20.5, p<0.001**) where v2's 17-vs-17 shows no difference at all.

**Three things that must be said in the same breath.**

- **Roughly 63% of the raw base→v2 exact-match gain is answer-format compliance**, measured in §7.3, not perception.
- **The v3 comparison confounds rank, vision-tower targeting and step count.** The 200-step probe with the *same* 82.7M configuration scores only 0.6715, which points at **steps** rather than vision parameters. The isolating ablation has not been run. So the honest claim is: *"82.7M trainable parameters trained for 2,000 steps is worth +0.14 on the held-out RSVQA slice"* — **not** *"adapting the vision tower is worth +0.14."*
- **v3 was trained under a label-masking defect and still won.** That makes it a promising run, not a validated one.

#### 5. What benchmark should we optimize against next?

**VRSBench** — and it is not close.

1. It is **prescribed by the problem statement** and is the one prescribed benchmark still unevaluated (`docs/00` L11).
2. It is the **only public benchmark that scores captioning, grounding and VQA under a single protocol**, which is exactly the three-way hole in this audit's crosswalk.
3. **Four of the five Top-5 models report it**, and GeoChat's numbers exist for both zero-shot (caption BLEU-4 1.4 / grounding 12.9 / VQA 40.8) and fine-tuned (13.8 / 39.6 / 60.6) settings, so there is a graded ladder to place ourselves on rather than a single unreachable number.
4. Its test split is 9,350 images / 37,408 VQA pairs — large enough that the ±6.5-point uncertainty crippling the current 207-question slice disappears.
5. The blocker is a **DOTA/DIOR imagery download**, not modelling work.

**Second priority: the official RSVQA-LR *test* split, scored per type.** Today's headline number is on a validation subsample, includes count, and equals a constant. Fixing the evaluation is cheaper than fixing the model, and until it is fixed the project cannot tell whether any model change helped.

**Do not** optimise against the current 207-question slice. At n=207 it cannot resolve anything smaller than about 6 points, and its aggregate metric is the wrong instrument.

#### 6. What is the single highest-impact improvement to pursue next?

**Give the CDVQA semantic-change segmenter a pretrained backbone and a real training budget.**

The reasoning, and why it beats the alternatives:

- It is the **only place with a Category-A gap on a PS-prescribed benchmark** — 0.5380 against 0.7474 on identical data with an identical metric.
- **The ceiling is proven and enormous.** SatQuery's own oracle over ground-truth change maps scores **0.9975**, so the deterministic answer layer contributes no measurable error. **93% of the headroom is one well-posed segmentation problem.**
- **The cause is diagnosed, not suspected**: change-class mIoU 0.2636, with per-class IoU of 0.068 (water), 0.071 (playgrounds), 0.098 (trees) — exactly the rare classes that `smallest_change`, `largest_change` and `change_to_what` ask about, and exactly the three types SatQuery loses hardest on.
- **The fix has precedent inside this project.** Switching that encoder from scratch-initialised to ImageNet ResNet-18 already bought **+56% relative** change-class mIoU. A change-detection-pretrained backbone, full-resolution crops and more than 40 epochs are the next three moves in ascending cost order.
- **Cost is hours on one GPU**, not days, and no architecture is redesigned.
- It simultaneously lifts a **mandatory** PS capability (M4) that is currently only 3.0 points above a constant.

**The runner-up, and why it loses.** Committing the label-masking fix and re-running Track B (§12 P2) is the item that would *answer* the 82.7M question properly, and it should be done. But its expected accuracy gain is unknown, while CDVQA's is bounded below by a proven 44-point ceiling on a benchmark the evaluators will actually run.

**Both are on `docs/code-freeze.md`'s explicitly-out-of-scope list.** Neither should start without an unfreeze decision recorded the way Unfreeze 1 was.
