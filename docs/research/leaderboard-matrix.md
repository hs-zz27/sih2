# Competitor matrix — Phase 3

**Written 2026-09-03.** Every SatQuery figure is re-anchored to the **Phase-2 immutable baseline** (`docs/research/phase2-baseline.md`, commit `pre-import:P0-primary`). Every external figure is **published and unreproduced by this project**.

> **CDVQA is 0.6061, not 0.5380.** The stale figure is superseded by `docs/research/cdvqa-baseline-correction-2026-09-03.md`, proven by A/B. Any table elsewhere in the repository still showing 0.5380 has not yet been updated — see that document §5.

---

## 0. The separation rule

The directive requires two categories that are never merged. They are kept in physically separate tables throughout.

| Category | Meaning | Where |
|---|---|---|
| **MEASURED BY SATQUERY** | Produced by this repository's own evaluation code, reproduced under Gate D at commit `pre-import:P0-primary` | §1 |
| **PUBLISHED ELSEWHERE** | Quoted from a paper. **Not reproduced by us.** No external model was ever downloaded or run | §2 |

A third table (§4) places them side by side **only** where the protocol genuinely matches, and every row there carries a comparability class.

---

## 1. MEASURED BY SATQUERY

All values from `docs/research/phase2-baseline.md` §9.3. Reproduced 2026-09-03 with worst deviation 0.002 points.

| Capability | Benchmark | Split | n | Metric | **Value** | Reproduced? |
|---|---|---|---|---|---|---|
| **VQA** | **RSVQA-LR OFFICIAL TEST**, `track_b_v3` | **official test** | **7,057** | per-type micro, count excluded | **0.8923** [88.49, 89.93] | Phase 4 |
| **VQA** | **RSVQA-LR OFFICIAL TEST**, `track_b_v2` | **official test** | 7,057 | per-type micro, count excluded | **0.7731** | Phase 4 |
| **VQA** | **RSVQA-LR OFFICIAL TEST**, all types | **official test** | **10,004** | exact match | `v3` **0.6966** · `v2` 0.5985 · base 0.2593 | Phase 4 |
| **Counting** | **RSVQA-LR OFFICIAL TEST** | **official test** | **2,947** | exact match, count type | `v3` **0.2280** — **below the 0.2555 constant** | Phase 4 |
| VQA | RSVQA-LR slice, `track_b_v3` *(superseded)* | validation subsample | 150 | per-type micro | 0.9533 — **optimistic by 6.1 pts** | superseded by the row above |
| VQA | RSVQA-LR slice, `track_b_v3` | validation subsample | 207 | exact match, all types | **0.7874** | exact |
| VQA | RSVQA-LR slice, `track_b_v2` (deployed) | validation subsample | 150 / 207 | per-type micro / all types | **0.8133** / **0.6473** | exact |
| VQA | RSVQA-LR slice, base model (no adapter) | validation subsample | 207 | exact match | **0.1981** | exact |
| Counting | RSVQA-LR slice, `v3` | validation subsample | 57 | exact match, count type | **0.3509** | exact |
| Captioning | RSICD | **official test** | 1,093 | sentence-mean smoothed BLEU-4, 5 refs | **0.2446** | exact (17 s.f.) |
| Caption diversity | RSICD | official test | 1,093 | unique captions | **146 (13.4%)** | exact |
| Grounding | DIOR-RSVG | **self-made** | 1,141 | Acc@0.5 / Acc@0.7 / mIoU | **0.0762** / 0.0088 / 0.1405 | exact |
| Land cover | BigEarthNet-19 | partition shard | — | macro mAP (all-band / 4-band) | **0.2854** / 0.2573 | not re-run this phase |
| Change detection | LEVIR-CD | **official**, 256px tiling | 2,048 tiles | change-class F1 / IoU / P / R | **0.5597** / 0.3886 / 0.4426 / 0.7613 | exact |
| **Change VQA** | **CDVQA test1** | **official** | 39,686 q / 968 pairs | overall accuracy | **0.6061** | **corrected, A/B proven** |
| Change VQA ceiling | CDVQA test1 | official | 39,686 | oracle over GT change maps | **0.99748** | exact |
| Change captioning | LEVIR-CC | official test | 1,929 | sentence-mean BLEU-4, **1 ref**, aggregate | **0.5686** | exact |
| Change captioning | LEVIR-CC | official test | 964 | BLEU-4, changed pairs | **BLOCKED (D2)** | unreproducible from code |
| Opt–SAR | WHU-OPT-SAR | random-by-tile | 1,548 | tile mAP: optical / SAR / fused | 0.7778 / 0.7410 / 0.7714 | exact |
| Opt–SAR | WHU-OPT-SAR | random-by-tile | — | **complementarity gain** | **−0.0064** | exact |
| Refusal | project val split, `v2` | held-out | 17 | recall / false-refusal / lexical probe | **0.4118** / 0.0077 / 0.1667 | exact |
| Calibration | LEVIR-CD | official test, split by tile | 2,048 | ECE before → after (affine) | **0.0668 → 0.0034** | exact |
| Orchestration | adversarial suite | 200 q × 3 configs | 600 | illegal-plan rate | **0 / 600** | not re-run this phase |
| Routing | CLEAN_HOLDOUT | never-tuned | 29 | accuracy | **0.5862** | not re-run this phase |
| Resource | — | — | — | VRAM, VLM inference | **2.6–2.9 GB** of 6.0 | measured |
| Resource | — | — | — | latency | **not reproducible at single-run resolution** | see §6.3 of Phase 2 |

**BLOCKED — unreproducible, neither estimated nor substituted:**

| Item | Reason |
|---|---|
| v0 → v1 VQA, `rsvqa_lr` 0.4510 → 0.6425 | v0 adapter is 100% NUL; `.pt` files hold no `model_state_dict` (**W15**) |
| Change captioning `bleu4_changed` 0.3063 | Never existed in any tracked commit (**D2**) |

---

## 2. PUBLISHED ELSEWHERE

**No external model was downloaded or run by this project. Every number below is quoted.**

### 2.1 Comparable remote-sensing VLMs — the Top 5

| Model | Backbone / vision encoder | Params | Training scale | Modalities | Code | Weights | Date |
|---|---|---|---|---|---|---|---|
| **EarthDial** | Phi-3-mini / InternViT-300M | 4B | **11.11M** instruction pairs | RGB, SAR, MS, NIR, hyperspectral, bi-temporal | [public](https://github.com/hiyamdebary/EarthDial) | **public, CC BY 4.0** | 2025-04 |
| **Earth-OneVision** | Qwen3-2B / SigLIP-2 (0.3B) | **2B** | **~34M** QA pairs, 8×H100 | optical, SAR, IR, MS, temporal, video, fusion | none | **none** | **2026-06** |
| **RingMo-Agent** | DeepSeek-VL2 3B / SigLIP-SO400M-384 | 3B | **>3M** pairs | optical, SAR, IR; satellite + UAV | not stated | not stated | 2026-08 |
| **EarthMind** | **Qwen2.5-3B** / InternVL2 + SAM2 + GPT4RoI | 4B | ~3.2M | optical, SAR, MS, **RGB–SAR fusion** | [public](https://github.com/shuyansy/EarthMind) | not stated | 2025-06 |
| **EarthGPT** | LLaMA-2 / DINOv2 ViT-L/14 + CLIP ConvNeXt-L | ~7B | MMRS-1M, 34 datasets | optical, SAR, IR | [public](https://github.com/wivizhang/EarthGPT) | public | 2024-03 |

Reference rows (not Top-5): **GeoChat** 7B LLaVA-1.5, optical, public weights, CVPR 2024 · **LHRS-Bot-Nova** 8B LLaMA-3 + SigLIP-L/14, optical only, public · **TinyRS-R1** 2B Qwen2-VL-2B, optical only, public, publishes VRAM/latency · **VHM** 7B Vicuna, optical, public.

### 2.2 Published scores — VLMs

| Benchmark | Split | Metric | EarthDial | Earth-OneVision | RingMo-Agent | EarthMind | EarthGPT | GeoChat | LHRS-Bot-Nova | TinyRS / -R1 |
|---|---|---|---|---|---|---|---|---|---|---|
| RSVQA-LR | official test | per-type accuracy | **92.70** | **92.91** | **90.30** | N/R | N/R | 90.70 | 89.61 | 83.5 / 76.0 |
| RSVQA-HR | official test | avg accuracy | 72.45 | 86.36 | 79.58 zs | 74.0 | 72.05 zs | 70.82 zs | 92.06 | — |
| VRSBench-VQA | official test | accuracy | N/R | **80.32** | N/R | 78.9 | N/R | 60.6 ft / 40.8 zs | N/R | N/R |
| VRSBench-Cap | official test | BLEU-4 | N/R | N/R | N/R | N/R | N/R | 13.8 ft / 1.4 zs | N/R | N/R |
| VRSBench-VG | official test | Acc@0.5 | N/R | **90.77** | N/R | 55.6 | N/R | 39.6 ft / 12.9 zs | N/R | N/R |
| DIOR-RSVG | official test | P@0.5 | N/R | **94.41** | N/R | N/R | mIoU 69.34 | N/R | **92.87** | 69.4 / 74.9 |
| RSICD | official test | corpus BLEU-4 / METEOR | ROUGE-1 33.77 | METEOR 33.98 | N/R | N/R | N/R | N/R | N/R | N/R |
| NWPU-Captions | official test | corpus BLEU-4 | ROUGE-1 45.84 | N/R | N/R | N/R | **65.5** | N/R | N/R | N/R |
| UCM-Captions | official test | corpus BLEU-4 | ROUGE-1 40.0 | N/R | **77.63** | N/R | N/R | N/R | N/R | N/R |
| BigEarthNet | official | accuracy / recall | 69.94 acc | 75.74 recall | N/R | 70.4 acc | N/R | N/R | N/R | N/R |
| SARLANG-Bench | official | VQA accuracy | N/R | **80.68** | N/R | N/R | N/R | N/R | N/R | N/R |
| SAR detection | various | mAP@0.5 | 6.06 / 26.02 | N/R | **53.84** | 13.6–36.8 | N/R | N/R | N/R | N/R |
| **Calibration ECE** | — | — | **none** | **none** | **none** | **none** | **none** | **none** | **none** | **none** |
| **Refusal metric** | — | — | **none** | **none** | **none** | **none** | **none** | **none** | **none** | **none** |
| **Illegal-plan rate** | — | — | **none** | **none** | **none** | **none** | **none** | **none** | **none** | **none** |
| VRAM / latency | — | — | N/R | N/R | N/R | N/R | N/R | N/R | N/R | **4.4–4.6 GB / 90–689 ms** |

`zs` = zero-shot, `ft` = fine-tuned, `N/R` = not reported by that paper.

### 2.3 Published scores — task specialists

The Top-5 VLMs do not report LEVIR-CD, CDVQA, BigEarthNet mAP or LEVIR-CC, so these are the honest comparators for four of SatQuery's rows.

| Benchmark | Metric | Best published | Runners-up | Source |
|---|---|---|---|---|
| **CDVQA test1** | overall accuracy | **0.7474** Qwen3.5-2B (2026) | VisTA 0.7310 · SOBA 0.6920 · CDVQA baseline 0.6590 | arXiv 2604.18429 Table III |
| **LEVIR-CD** | change-class F1 | **≈0.9227** PhyUnfold-Net (2026) | ChangeRWKV-B 0.8601 · ConvFormer-CD/48 0.8530 | arXiv 2603.19566 / 2603.19606 |
| LEVIR-CD | IoU | **0.8565** ChangeDA | ChangeRWKV-B 0.7546 | — |
| **BigEarthNet-19** | macro mAP | **0.7998** ResNet50 | ResNet152 0.7978 · SeCo AP 0.8262 | arXiv 2207.07189 |
| **RSICD** | corpus BLEU-4 | **0.6574** RSGPT | — | arXiv 2307.15266 |
| **LEVIR-CC** | BLEU-4 (**5 refs**) | **≈0.6550** SAGE-CC | KCFI 0.6530 · SAT-Cap CIDEr 140.23 | arXiv 2501.08114 |
| WHU-OPT-SAR | mIoU / OA | PAD 56.26 / 84.56 | ASANet | arXiv 2504.19136 |

---

## 3. Comparability classification

Before any side-by-side, each SatQuery row is classified. **A** same dataset/split/task/metric/protocol · **B** same task, protocol differs · **C** not comparable.

| SatQuery row | Class | Why |
|---|---|---|
| CDVQA test1 overall accuracy | **A** | identical dataset, identical official test1 split (39,686 q / 968 pairs), identical OA metric |
| LEVIR-CD change-class F1 | **A** | official split, standard 256px tiling, standard change-class F1 |
| **RSVQA-LR official test** | **A on protocol, NOT on training regime** | same official test split, same question types, same metric, **leakage-checked**. But SatQuery's adapters saw 1,793 questions from the RSVQA-LR *validation* subset, not the official 57,223-question train split — **few-shot, where the published figures are fine-tuned**. Both halves of this phrase must travel with the number |
| RSVQA-LR slice, re-typed *(superseded)* | B | validation subsample n=150; replaced by the official-test row above |
| RSVQA-LR, all types (project headline) | **C** | count included where the literature excludes it; whole-string EM vs per-type accuracy |
| BigEarthNet-19 mAP | **B** | 30k-patch / 3-epoch training; partition shard vs recommended split |
| RSICD BLEU-4 | **B** | official test split, but sentence-mean smoothed BLEU vs corpus BLEU |
| LEVIR-CC BLEU-4 | **C** | **1 reference vs 5** |
| DIOR-RSVG Acc@0.5 | **C** | self-made split (`run_metadata`: "NO published split in this mirror") |
| WHU-OPT-SAR | **C** | tile multi-label mAP vs pixel segmentation mIoU — different tasks |
| Refusal / ECE / illegal-plan rate | **—** | **no comparator exists**; nobody publishes these |

---

## 4. Side-by-side — only where the protocol matches

### 4.1 Category A — the two defensible comparisons

**CDVQA test1** — identical split, identical metric. **Re-anchored to the corrected 0.6061.**

| System | Year | Accuracy | Gap vs SatQuery |
|---|---|---|---|
| Qwen3.5-2B change-VQA | 2026 | 0.7474 | **−14.1** |
| VisTA | 2024 | 0.7310 | **−12.5** |
| SOBA | 2024 | 0.6920 | **−8.6** |
| CDVQA baseline | 2021 | 0.6590 | **−5.3** |
| **SatQuery** *(measured, corrected)* | 2026 | **0.6061** | — |
| per-type majority constant | — | 0.5084 | **+9.8** |
| *SatQuery oracle ceiling* | — | *0.99748* | *(headroom 0.391)* |

**LEVIR-CD** — official split, standard tiling, change-class F1.

| System | Year | F1 | Gap vs SatQuery |
|---|---|---|---|
| PhyUnfold-Net | 2026 | ≈0.9227 | **−36.3** |
| ChangeRWKV-B | 2025 | 0.8601 | −30.0 |
| ConvFormer-CD/48 | 2025 | 0.8530 | −29.3 |
| **SatQuery** *(measured)* | — | **0.5597** | — |

SatQuery's detector is **49,543 parameters trained for 4 epochs**; the comparators are full change-detection networks. The gap is capacity and budget, and was a deliberate design choice.

**RSVQA-LR official test** — 10,004 questions / 100 images, published convention (count excluded, n=7,057).

| System | Params | Accuracy | vs SatQuery `v3` |
|---|---|---|---|
| Earth-OneVision | 2B | 92.91 | −3.68 |
| EarthDial | 4B | 92.70 | −3.47 |
| GeoChat | 7B | 90.70 | −1.47 |
| RingMo-Agent | 3B | 90.30 | −1.07 |
| **LHRS-Bot-Nova** | **8B** | **89.61** | **−0.38 — inside `v3`'s CI** |
| **SatQuery `track_b_v3`** *(measured)* | 3.75B + 82.7M | **89.23** [88.49, 89.93] | — |
| `track_b_v2` deployed *(measured)* | 3.75B + 37.2M | 77.31 | — |
| *train-fitted constant* | — | *70.06* | — |

> **This comparison is Category-A on protocol, but NOT on training regime.** Same official test split, same question types, same metric, verified free of leakage — but SatQuery's adapters were trained on **1,793 questions from the validation subset**, where every published figure comes from **fine-tuning on the official 57,223-question train split**. SatQuery's setting is closer to **few-shot**. That arguably makes 89.23% more impressive rather than less, but it is not apples-to-apples in the way "same test split" alone implies. **Never quote 89.23% without this sentence.**

### 4.2 Category B — labelled, never used to claim a win

| Comparison | SatQuery | Best published | What may be said |
|---|---|---|---|
| RSVQA-LR, published convention | **0.9533** (`v3`, n=150 **val**) · 0.8133 (`v2`) | 92.91 Earth-OneVision | *"plausibly in the same band."* **Not** parity — 150 validation questions against ~10k test questions |
| RSVQA-LR, count-inclusive | 0.7874 (`v3`) | 0.7908 OA, RSVQA baseline | v3 sits **0.3 points below** the only count-inclusive published anchor, on a different split |
| BigEarthNet-19 mAP | 0.2854 | 0.7998 ResNet50 | −0.514, explained by 30k-patch training rather than method |
| RSICD BLEU-4 | 0.2446 | 0.6574 RSGPT | **not subtractable** — different BLEU implementations |

### 4.3 Category C — no claim permitted

DIOR-RSVG (0.0762 vs 0.9441, self-made split) · LEVIR-CC (0.5686 aggregate, 1 ref vs 5) · WHU-OPT-SAR (different task) · the project's all-types RSVQA headline.

### 4.4 Uncontested — SatQuery measures what nobody else reports

| Property | SatQuery | Any comparator |
|---|---|---|
| Calibration ECE | **0.0668 → 0.0034** | **none reports ECE** |
| Refusal recall + lexical-shortcut control | **0.4118** / probe 0.1667 | **none** |
| Illegal-plan rate | **0 / 600** | **none** |
| Entailment gate on generated prose | built, benchmarked | **none** |
| Negative results published | fusion −0.0064; refusal 2/12; counting = constant | rare |

This is a genuine advantage of a narrow kind: SatQuery measures things others do not. It is not evidence that it answers better.

---

## 5. What changed versus the pre-Phase-2 matrix

| Row | Before | After Phase 2 |
|---|---|---|
| **CDVQA** | 0.5380, −20.9 vs SOTA, +3.0 vs constant, "loses on all 8 types" | **0.6061**, **−14.1** vs SOTA, **+9.8** vs constant, **beats constant on 7 of 8 types** |
| CDVQA headroom | 0.4595 | **0.3914** |
| Change captioning headline | 0.3063 quoted as the figure | **BLOCKED** — unreproducible from code |
| v0→v1 VQA gain | quoted as 0.4510 → 0.6425 | **BLOCKED** — unreproducible from disk |
| Everything else | — | unchanged, reproduced exactly |

**The direction of every conclusion in `docs/external_benchmark_audit.md` survives.** CDVQA and LEVIR-CD remain the two Category-A gaps; the segmenter remains the dominant CDVQA bottleneck; grounding remains near-floor. What changes is magnitude: the CDVQA gap is materially smaller than the project believed, and two headline figures turn out to be unsupported.

---

## 6. Local evaluability of comparators — unchanged

| Model | Weights | Fits 6 GiB at 4-bit? | Verdict |
|---|---|---|---|
| Earth-OneVision | none released | — | **impossible** |
| RingMo-Agent | not stated | — | **impossible** |
| EarthMind | code public, weights unstated | 4B — yes | **blocked** on availability |
| **EarthDial** | **public, CC BY 4.0** | 4B — **yes** | **feasible**, needs a multi-GB download |
| EarthGPT | public | 7B + dual encoders | marginal |
| **TinyRS-R1** | **public** | 2B, publishes 4.6 GB | **easiest external baseline**, optical-only |

**Nothing has been downloaded.** Three of five Top-5 models cannot be run by anyone outside their labs, which is itself a finding: the field's strongest published claims are largely uncheckable.
