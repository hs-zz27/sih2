# Phase 4 — official benchmarks: results

**Written 2026-09-04.** Measurement only. Two downloads made under individual approval; **zero existing repository files modified**; two new read-only evaluators added.

> **Status: COMPLETE.** RSVQA-LR, CDVQA and VRSBench (both arms) all measured.
>
> **All three PS-prescribed benchmarks now have a measured SatQuery result for the first time.**

---

## 1. Scope compliance

| | |
|---|---|
| Commit | `pre-import:P0-primary` |
| Files modified | **zero** |
| Files added | `evaluation/rsvqa_official_eval.py`, `evaluation/vrsbench_eval.py` — both write only to `--out` |
| Write-target audit | Both greped for `write_text` / `open(...,"w")` / `save_checkpoint` / `np.save` / `savefig` / `to_json` before running, per the Phase-2 standing caution. Only `args.out` and its parent |
| Outputs | `artifacts/phase4_rsvqa/`, `artifacts/phase4_vrsbench/` |
| Not touched | `docs/assets/`, `configs/`, any `metrics.json`, any checkpoint |
| Not downloaded | **`Images_train.zip` (8.359 GB)** — outside the approved scope and deliberately not fetched |

---

## 2. CDVQA — satisfied by Phase 2, cross-referenced

No new work. Data was already on disk and the official protocol was already used.

| | |
|---|---|
| Split | **official test1**, 968 pairs / 39,686 questions, 99.82% coverage |
| **Result** | **0.6061** *(corrected from 0.5380; A/B proven)* |
| Oracle ceiling | **0.99748** |
| Comparability | **Category A** |
| Record | `docs/research/phase2-baseline.md` §4 · `docs/research/cdvqa-baseline-correction-2026-09-03.md` |

---

## 3. RSVQA-LR — official test split. **COMPLETE.**

### 3.1 Protocol

| | |
|---|---|
| Source | Zenodo `10.5281/zenodo.6344334`, CC-BY-4.0 |
| Split | **official test** — 10,004 questions over **100 images** |
| Types | comp 4,002 · presence 2,955 · count 2,947 · rural_urban 100 |
| Published convention | presence + comparison + rural-urban, **count excluded** → **7,057 questions** |
| Decode | mirrors `satquery/tools/rs_vqa.py` — 4-bit NF4, same system prompt, same chat template, greedy |
| **Leakage check** | **PASSED.** Perceptual hash of all 2,000 images in the training subset against all 100 official test images: **zero overlap** |

### 3.2 Results

| arm | all types | 95% CI | **published convention** | 95% CI | presence | comp | count | rural |
|---|---|---|---|---|---|---|---|---|
| **`track_b_v3`** | **0.6966** | [0.6875, 0.7056] | **0.8923** | [0.8849, 0.8993] | 0.8897 | 0.8946 | **0.2280** | 0.8800 |
| `track_b_v3_probe` | 0.6244 | [0.6148, 0.6338] | 0.8039 | [0.7945, 0.8130] | 0.7570 | 0.8403 | 0.1944 | 0.7300 |
| `track_b_v2` (deployed) | 0.5985 | [0.5888, 0.6080] | 0.7731 | [0.7632, 0.7828] | 0.8474 | 0.7216 | 0.1802 | 0.6400 |
| base (no adapter) | 0.2593 | [0.2508, 0.2680] | 0.3676 | [0.3564, 0.3789] | 0.2017 | 0.4940 | 0.0000 | 0.2100 |
| **train-fitted constant** | **0.5695** | — | **0.7006** | — | 0.7503 | 0.6674 | **0.2555** | 0.5600 |

Against the constant (McNemar, continuity-corrected):

| arm | model-only | constant-only | χ² | significant |
|---|---|---|---|---|
| `v3` | **1,675** | 403 | **777.4** | yes |
| `v3_probe` | 1,589 | 1,040 | 114.2 | yes |
| `v2` | 1,022 | 732 | 47.6 | yes |
| base | 1,126 | **4,229** | 1,796.9 | yes — **in the wrong direction** |

### 3.3 Category-A comparison against published models

Same official test split, same question types, same accuracy metric. **This is the project's first Category-A VQA comparison.**

| System | Params | RSVQA-LR test | vs SatQuery `v3` |
|---|---|---|---|
| Earth-OneVision | 2B | 92.91 | −3.68 |
| EarthDial | 4B | 92.70 | −3.47 |
| GeoChat | 7B | 90.70 | −1.47 |
| RingMo-Agent | 3B | 90.30 | −1.07 |
| **LHRS-Bot-Nova** | **8B** | **89.61** | **−0.38 — inside `v3`'s CI** |
| **SatQuery `track_b_v3`** | **3.75B + 82.7M** | **89.23** [88.49, 89.93] | — |
| *train-fitted constant* | — | *70.06* | *+19.2* |

**`track_b_v3` is statistically indistinguishable from LHRS-Bot-Nova (8B)** and within 3.7 points of the strongest published 2B model — trained on 4,806 examples on a 6 GiB laptop GPU.

**Caveats that travel with this claim.** SatQuery's adapters were trained on 1,793 questions drawn from the RSVQA-LR *validation* subset, not the official 57,223-question train split — so this is neither a standard fine-tuned nor a standard zero-shot setting, and it is closer to few-shot than to the fine-tuned regime the published numbers come from. The leakage check confirms no test contamination. The comparison is Category A on *protocol*; the *training* regimes differ and that should be stated whenever the number is quoted.

### 3.4 Three findings

**1. The validation slice was optimistic by 4–6 points.** Every arm scores lower on the official test split:

| arm | slice (n=150) | official (n=7,057) | Δ |
|---|---|---|---|
| `v3` | 0.9533 | **0.8923** | **−6.10** |
| `v3_probe` | 0.8333 | 0.8039 | −2.94 |
| `v2` | 0.8133 | 0.7731 | −4.02 |
| base | 0.2733 | 0.3676 | +9.43 |

The 95% interval tightened from **±6.5 to ±0.7**. Both the correction and the precision were the point of the download.

**2. Counting is negative-value — resolved at n=2,947.** Every arm scores **below** the train-fitted constant (`v3` 0.2280 vs 0.2555). Answering "0" to every count question beats the deployed model *and* the best model. See **W17** in `system-audit.md` for the framing Phase 10 must inherit: this is not a metric to improve from 0.23, it is a capability to redesign or route away from.

**3. The official split separates `v2` from `v3`; the slice could not.** On the 207-question slice `v2` versus a constant was 17–17 — no measurable difference. Here it is 1,022–732, χ²=47.6: significant, but far narrower than `v3`'s 1,675–403, χ²=777.4. The benchmark now has the resolution to rank checkpoints.

---

## 4. VRSBench — VQA ⏳ IN PROGRESS

### 4.1 A correction to `docs/00` L11

L11 records VRSBench as unevaluable because *"its 142k rows reference images that live in the separate DOTA and DIOR datasets"*. **That is false.** The VRSBench HuggingFace repo hosts the imagery directly:

```
xiang709/VRSBench   Images_val.zip   3,976,656,690 bytes
                    Images_train.zip 8.359 GB      [NOT downloaded]
```

**No DOTA or DIOR download is required.** Flagged for the stale-figure update pass in `cdvqa-baseline-correction-2026-09-03.md` §5.1.

### 4.2 Protocol

| | |
|---|---|
| Split | VRSBench eval — **37,409 questions over 9,349 images**, all present |
| Types | 12: object existence 7,789 · quantity 6,374 · position 5,828 · category 5,434 · color 3,550 · scene type 3,197 · shape 1,423 · image 1,129 · size 1,011 · reasoning 902 · direction 477 · rural/urban 295 |
| **Setting** | **ZERO-SHOT.** SatQuery has never trained on VRSBench |
| Comparator | **GeoChat 40.8 zero-shot** — *not* its 60.6 fine-tuned. Any later summary citing 60.6 against this number is comparing different settings |
| Baselines | train-fitted **global** constant (`"yes"`, honest floor) and test-fitted **per-type** constant (**optimistic upper bound**, peeks at the eval set) |

### 4.3 Results — `track_b_v3`

Measured on a **7,999-question stratified subsample** (proportional by type, seed
`20260904`), not the full 37,409. See §4.4 for why, and for the independent
confirmation that the subsample did not move the number.

| | |
|---|---|
| **Strict accuracy** | **0.2968** [0.2869, 0.3069] |
| Lenient (gold appears in prediction) | 0.3548 |
| Decode cost | 1.975 s/question |

**Against both baselines:**

| | accuracy |
|---|---|
| GeoChat — **zero-shot** | **40.8** |
| *test-fitted per-type constant — OPTIMISTIC, peeks at the eval set* | *34.63* |
| **SatQuery `track_b_v3` — zero-shot** | **29.68** |
| *train-fitted global constant `"yes"` — honest floor* | *24.00* |

**`v3` clears the honest floor and loses to the optimistic constant**, and the
loss is not noise: model-only correct **810**, constant-only correct **1,206**,
McNemar χ² (continuity-corrected) **77.4**.

**This is a negative result and it is reported as one.** It is the opposite
shape from RSVQA-LR (§3), where the same checkpoint reached 89.23 and was
statistically indistinguishable from an 8B model. The two together are the
honest characterisation: **strong in-domain, weak out-of-domain.**

#### Per-type — the aggregate hides a split

Beats the constant on **6 of 12** types:

| type | n | `v3` | 95% CI | constant |
|---|---|---|---|---|
| rural or urban | 63 | **0.6984** | [0.5764, 0.7976] | 0.4444 |
| reasoning | 193 | **0.4974** | [0.4276, 0.5673] | 0.4352 |
| object size | 216 | **0.3194** | [0.2609, 0.3843] | 0.2454 |
| object position | 1,246 | **0.2978** | [0.2730, 0.3237] | 0.1942 |
| object direction | 102 | **0.2353** | [0.1635, 0.3263] | 0.2059 |
| scene type | 684 | **0.2135** | [0.1844, 0.2457] | 0.1111 |

Loses on the other **6 — including the three largest**:

| type | n | `v3` | 95% CI | constant |
|---|---|---|---|---|
| object existence | 1,666 | 0.6321 | [0.6086, 0.6549] | **0.8169** |
| object quantity | 1,363 | 0.1753 | [0.1561, 0.1964] | **0.3258** |
| object category | 1,162 | **0.0361** | [0.0269, 0.0485] | 0.0861 |
| image | 241 | 0.3983 | [0.3386, 0.4613] | **0.6183** |
| object color | 759 | 0.1858 | [0.1597, 0.2150] | 0.2082 |
| object shape | 304 | 0.1743 | [0.1358, 0.2210] | 0.1776 |

Three findings:

1. **The three largest types carry the aggregate** — 4,191 of 7,999 questions,
   and `v3` is below the constant on all three.
2. **`object existence` is the clean diagnostic.** Always answering "yes" scores
   0.8169; `v3` scores 0.6321. It is not failing to answer — it is answering and
   being wrong where silence would win.
3. **`object quantity` 0.1753 vs 0.3258 reproduces W17 on a second, independent
   benchmark.** Counting was already below a constant on RSVQA-LR at n=2,947.
   It is below a constant here too, at n=1,363, on different imagery and a
   different answer distribution. **W17 is no longer a single-benchmark finding.**

`object category` at **0.0361** is near-total failure and has no counterpart in
any benchmark this project has run before.

### 4.4 Why a subsample, and what it cost

The full-set run reached **32,955 / 37,409 (88%)** and was killed at 20:57 on
2026-09-04 when its harness task was orphaned. **The script wrote only on arm
completion, so 10.8 hours of compute produced no artifact.** Only the stdout log
survived.

Two changes followed, both confined to `evaluation/vrsbench_eval.py`:

* **mid-arm checkpointing** every 250 questions, written to a temp file and
  atomically renamed, with mid-arm resume;
* the run **detached from the harness** (`Start-Process`) so a task-registry
  reset cannot orphan it again.

**The subsample did not move the number.** The killed full run's last flushed
line reported **0.2920 at n=32,000**; the independent stratified draw returned
**0.2968 at n=7,999**. The two agree well inside the interval.

**Cost of subsampling, stated plainly:** precision only. ±2.4 to ±3.7 points on
the six largest types (86% of questions), ±1.1 overall, but **±9.5 on
`object direction` and ±12.0 on `rural or urban`** — those two rows should not
be read as equal-confidence with the rest. The task, prompt, decode and metric
are unchanged, and the draw was made without reference to any score.

An unanticipated second cost: the stratified draw touches 5,681 distinct images
for 7,999 questions, where sequential order reused each image about four times.
The lost image-read locality roughly doubled decode cost, 1.12 → 1.975 s/q.

### 4.5 `track_b_v2` — paired comparison

Same 7,999 rows, same seed, so the arms are paired.

| arm | strict | 95% CI | lenient | vs constant (χ²) |
|---|---|---|---|---|
| **`track_b_v3`** | **0.2968** | [0.2869, 0.3069] | 0.3548 | 77.4 — below |
| `track_b_v2` (deployed) | 0.2045 | [0.1958, 0.2135] | 0.3128 | 543.0 — far below |

**`v3` beats `v2` by 9.23 points out-of-domain, and beats it on all 12 types
without exception:**

| type | n | `v3` | `v2` | Δ | constant |
|---|---|---|---|---|---|
| object existence | 1,666 | 0.6321 | 0.4340 | **+0.1981** | 0.8169 |
| object quantity | 1,363 | 0.1753 | 0.1658 | +0.0095 | 0.3258 |
| object position | 1,246 | 0.2978 | 0.2464 | +0.0514 | 0.1942 |
| object category | 1,162 | 0.0361 | 0.0112 | +0.0250 | 0.0861 |
| object color | 759 | 0.1858 | 0.0527 | **+0.1331** | 0.2082 |
| scene type | 684 | 0.2135 | 0.1725 | +0.0409 | 0.1111 |
| object shape | 304 | 0.1743 | 0.0691 | **+0.1053** | 0.1776 |
| image | 241 | 0.3983 | 0.1369 | **+0.2614** | 0.6183 |
| object size | 216 | 0.3194 | 0.1898 | **+0.1296** | 0.2454 |
| reasoning | 193 | 0.4974 | 0.2953 | **+0.2021** | 0.4352 |
| object direction | 102 | 0.2353 | 0.2157 | +0.0196 | 0.2059 |
| rural or urban | 63 | 0.6984 | 0.5556 | **+0.1429** | 0.4444 |

**This answers the question the `v3` arm alone could not.** `v3`'s RSVQA-LR gain
over `v2` (89.23 vs 77.31) could have been in-domain overfitting to the
adaptation set. It was not: the same checkpoint is better on a benchmark neither
arm has ever trained on, uniformly across every question type. **The vision-tower
adaptation in `v3` improved genuine transfer, not just in-domain fit.**

The caveat stays attached: **both arms are still below the optimistic constant
(0.3463) and below GeoChat's 40.8 zero-shot.** `v3` is a real improvement over
`v2` on a benchmark where SatQuery is not yet competitive.

`v2`'s χ² of 543.0 against the constant is the starkest single number Phase 4
produced — the deployed model loses to a majority-class answer on 1,749
questions while winning only 615.

#### The counting result survives the paired test

`object quantity` is the *smallest* gain of the twelve (+0.0095, well inside the
interval). Both arms sit far below the constant (0.3258). Vision-tower adaptation
lifted every other capability and left counting where it was — which is direct
evidence for **W17**'s framing: counting is not a metric that responds to more or
better adaptation. It needs a different mechanism.

---

## 5. Download record

| Dataset | Approved | Actual | Verification |
|---|---|---|---|
| RSVQA-LR official | ~99.7 MB | **~136 MB** (7 files) | **7/7 MD5 verified** |
| VRSBench `Images_val.zip` | 3.977 GB | **3,976,656,690 bytes** | **SHA-256 verified** — on the second attempt |

### 5.1 Two honest notes

**I under-scoped the RSVQA-LR request by 36 MB.** My request listed the `LR_split_test_*` files as sufficient. They are **active-masks** (`{"id": 0, "active": false}`), not content; the question text and answers live in `all_questions.json` / `all_answers.json`, and fitting an honest constant needs `LR_split_train_questions.json`. Total 136 MB against the ~99.7 MB quoted — still inside the 150.5 MB full record, but the request table was wrong.

**The VRSBench download failed its checksum on the first attempt.**

| | bytes |
|---|---|
| Expected | 3,976,656,690 |
| First attempt | 3,940,208,795 |
| **Short by** | **36,447,895 (0.92%)** |

The truncated file had a **valid ZIP header and a readable `Images_val/` entry**. It would have extracted a subset of the images and produced a plausible VRSBench score on incomplete data, with nothing to signal the problem. It was caught only by comparing exact byte counts and SHA-256 against the source manifest — not by any property of the file itself.

This is the same failure that left the `.incomplete` fragments removed in Phase 1 (`docs/storage-audit.md` §6 A2), which are now explained rather than merely tidied away. **Resumed with `curl -C -` and verified before extraction.** Nothing was extracted or evaluated while unverified.

**Standing addition to the Phase-2 caution:** *a file that opens is not a file that is complete.* Verify size and checksum against the source manifest before extracting any archive.

---

## 6. What Phase 4 changes about the project's position

| Claim | Before Phase 4 | After |
|---|---|---|
| RSVQA-LR comparability | **Category C** — 207-question validation slice, count included, ±6.5 | **Category A** — official test split, 10,004 questions, ±0.7 |
| `v3` VQA headline | 0.9533 (optimistic) | **0.8923**, CI [88.49, 89.93] |
| Best comparable gap | unmeasurable | **−0.38 vs LHRS-Bot-Nova — statistically indistinguishable** |
| Counting | "equals a constant" (n=57) | **below a constant, resolved (n=2,947)** — W17 |
| VRSBench | unevaluated, believed to need DOTA/DIOR | **both arms measured: `v3` 29.68, `v2` 20.45 zero-shot — below GeoChat's 40.8 and below a constant** |
| Did `v3`'s gain generalise? | unknown — could have been in-domain overfitting | **yes — `v3` beats `v2` on all 12 out-of-domain types** |
| Out-of-domain generalisation | never measured | **measured and negative** — the counterweight to 89.23 |
| W17 (counting) | one benchmark | **reproduced on a second** — RSVQA-LR *and* VRSBench, both below a constant |
| Prescribed benchmarks evaluated | RSVQA (slice) + CDVQA | **RSVQA (official) + CDVQA + VRSBench** — all three |
