# CDVQA baseline correction — 2026-09-03

**New dated section. Nothing was edited.** Under `docs/code-freeze.md`, a changed number is published as a new dated entry, never as an edit to an existing one. Every document listed in §5 still contains the old figure and is flagged for a separate, explicit update pass.

---

## 1. The correction

| | |
|---|---|
| **Old published number** | **CDVQA test1 overall accuracy = 0.5380** |
| **New number** | **0.606133** |
| **Change** | **+6.81 points** |
| Cause | Commit **`pre-import:png-channels-fix`**, "Read the channels a PNG declares instead of guessing them" |
| Status | **Proven by A/B**, not inferred |
| Split | CDVQA test1 — 968 image pairs, 39,686 questions. **Identical in both runs** |
| Checkpoint | `checkpoints/change_vqa/best.pt` — **identical in both runs** |
| Data | `data/cdvqa` + `data/second` — **identical in both runs** |

**The old number was not wrong as a measurement. It was a correct measurement of a system that was being fed channel-reversed images.**

## 2. The A/B that proves it

The only variable changed between the two runs was the **code commit**. Same checkpoint, same data, same environment, same command, same machine, same session.

```
git checkout pre-import:P0-worktree          # pre-pre-import:png-channels-fix; grep photometric_bands -> 0 occurrences
SATQUERY_CHANGE_VQA=checkpoints/change_vqa/best.pt \
  python -m evaluation.cdvqa_predict --split Test --out <A>.json

git checkout <branch>         # pre-import:P0-primary; grep photometric_bands -> 2 occurrences
SATQUERY_CHANGE_VQA=checkpoints/change_vqa/best.pt \
  python -m evaluation.cdvqa_predict --split Test --out <B>.json
```

### 2.1 Result

| question type | n | **`pre-import:P0-worktree` (pre-fix)** | recorded 2026-08-30 | **`pre-import:P0-primary` (post-fix)** | Δ fix |
|---|---|---|---|---|---|
| `change_or_not` | 13,882 | **0.6772** | 0.6772 ✓ | 0.709336 | +3.21 |
| `change_ratio` | 1,936 | **0.1952** | 0.1952 ✓ | 0.187500 | −0.77 |
| `change_ratio_types` | 5,811 | **0.4791** | 0.4791 ✓ | 0.556359 | +7.73 |
| `change_to_what` | 2,991 | **0.3714** | 0.3714 ✓ | 0.521899 | **+15.05** |
| `decrease_or_not` | 4,658 | **0.6496** | 0.6496 ✓ | 0.754830 | +10.52 |
| `increase_or_not` | 4,600 | **0.6437** | 0.6437 ✓ | 0.717826 | +7.41 |
| `largest_change` | 2,904 | **0.4497** | 0.4497 ✓ | 0.616736 | **+16.70** |
| `smallest_change` | 2,904 | **0.1319** | 0.1319 ✓ | 0.152204 | +2.03 |
| **OVERALL** | **39,686** | **0.5380** | **0.5380 ✓** | **0.606133** | **+6.81** |

**The pre-fix run reproduces the recorded number exactly, on all eight question types and overall.** That closes the question: the recorded 0.5380 is reproducible, the code changed, and the code change is the entire difference.

## 3. Root cause

`evaluation/cdvqa_predict.py` line 47 imports `satquery.ingest`; line 70 calls `ingest([second/"im1"/name, second/"im2"/name])`. CDVQA therefore reads its imagery through the ingest reader.

Commit `pre-import:png-channels-fix` added `photometric_bands()` to `satquery/ingest/reader.py`. Its docstring names the defect it fixes:

> **Red and blue swapped.** GDAL reports band 1 of a PNG as `red`, but with no descriptions the fallback assumed the GeoTIFF convention `[BLUE, GREEN, RED]`. `to_rgb_preview` then looked "RED" up at index 3 and handed the model a channel-reversed image.

The SECOND imagery CDVQA reads is exactly that case — verified: `data/second/im1/00003.png` is **PNG, mode RGB, 512×512, bands (R, G, B)**.

The semantic-change head is an **ImageNet-pretrained ResNet-18** (`checkpoints/change_vqa/best.pt`, 6,257,001 parameters). ImageNet pretraining is RGB-specific, so a channel reversal degrades its features substantially. Before `pre-import:png-channels-fix`, every CDVQA measurement this project ever took was on BGR input.

**The per-type pattern corroborates the mechanism.** The two largest gains — `largest_change` **+16.70** and `change_to_what` **+15.05** — are the two question types that require discriminating land-cover classes **by colour**, which is precisely what a red/blue swap destroys. The type that barely moved, `smallest_change` (+2.03), is the one the oracle itself only reaches 0.9773 on, because "smallest" is ambiguous between close rare classes.

## 4. What the corrected number changes

### 4.1 External comparison, same test1 split

| System | Accuracy | Gap at 0.5380 | **Gap at 0.6061** |
|---|---|---|---|
| Qwen3.5-2B change-VQA (2026) | 0.7474 | −20.9 | **−14.1** |
| VisTA (2024) | 0.7310 | −19.3 | **−12.5** |
| SOBA (2024) | 0.6920 | −15.4 | **−8.6** |
| CDVQA baseline (2021) | 0.6590 | −12.1 | **−5.3** |
| **SatQuery** | **0.6061** | — | — |
| per-type majority constant | 0.5084 | +3.0 | **+9.8** |

* The deficit against the 2021 baseline falls from 12.1 points to **5.3**.
* The margin over a per-type constant **triples**, +3.0 → **+9.8**.
* The system now beats its per-type constant on **seven of eight** question types, up from five. The three that crossed over are `change_to_what`, `decrease_or_not` and `increase_or_not`.

### 4.2 Headroom

Oracle ceiling reproduced this session at **0.99748**. The gap narrows from 0.4595 to **0.3914**. The semantic-change segmenter is still the dominant bottleneck and the Phase-5/6 case is unchanged in direction — but it is a smaller gap than the project believed.

### 4.3 A claim that must also be corrected

`docs/judge-qa.md` asks: *"Your CDVQA score is 0.5380. A constant scores 0.5084. Why should I be impressed?"* That framing no longer matches the measurement.

## 5. Documents that quote the stale 0.5380 — flagged, NOT edited

**None of these has been touched.** Each needs a pointer to this correction in a separate, explicit update pass.

| Document | Where |
|---|---|
| `docs/00-README-and-Requirement-Traceability.md` | §3.1 (M4 row), §3.5, §3.6 **L10** |
| `docs/model-cards.md` | `change_vqa_v1` card, "system" row |
| `docs/phase1-status.md` | CDVQA sections dated 2026-08-29 / 08-30, and the per-type table |
| `docs/deck.md` | line ~128 |
| `docs/judge-qa.md` | §3 Q&A (see §4.3) |
| `docs/external_benchmark_audit.md` | §1.1, §6.1, §6.2, §8 (A1), §10.2, §11 G1, §15 — **and the derived gap figures in every one of them** |
| `docs/external_benchmark_results.json` | `satquery_published_component_metrics.change_vqa_v1`, `category_a_comparisons[0]`, `largest_gaps[G1]`, `competitive_position.change_understanding` |

### 5.1 A second known-wrong statement for the same update pass

Bundle this with the CDVQA correction — it is a different error in the same document:

| Document | Stale claim | Correction |
|---|---|---|
| **`docs/00-README-and-Requirement-Traceability.md` §3.6 L11** | VRSBench is unevaluable because *"its 142k rows reference images that live in the separate DOTA and DIOR datasets"* | **False.** The VRSBench HuggingFace repo (`xiang709/VRSBench`) hosts the imagery directly: `Images_val.zip` 3.977 GB, `Images_train.zip` 8.359 GB. **No DOTA or DIOR download is required.** Verified 2026-09-03 by metadata query; see `docs/research/phase4-requests.md` §2.1 |

**`docs/00` L11 is therefore named explicitly on the stale-document list above**, for the same reason and in the same pass as the 0.5380 → 0.6061 correction.

## 6. Secondary finding — deferrals. Logged, not investigated.

The A/B revealed a second change that the fix also caused, in the opposite direction to my first reading of it:

| | deferred | coverage |
|---|---|---|
| Documented claim (2026-08-30) | 0 | **100%** |
| `pre-import:P0-worktree` measured | **122** | 99.69% |
| `pre-import:P0-primary` measured | **73** | 99.82% |

Two things follow:

1. **The fix improved coverage** (122 → 73 deferrals), it did not regress it. My initial Phase-2 note said coverage "regressed 100% → 99.82%"; that was wrong, and this supersedes it.
2. **The documented "100% coverage, 39,686 / 39,686" was never reproducible at either commit.** At the commit where 0.5380 was measured, the same run defers 122 questions. That discrepancy is unexplained.

All deferrals at both commits fall in `change_to_what`.

**Decision 2026-09-03: note and move on.** This is a secondary behaviour change, not a Gate D violation in itself, and chasing it now would drift into Phase 5 / Phase 8 territory (deferral and refusal behaviour). It is logged here and in `docs/research/system-audit.md` §12 so it is not lost, and it may matter for **Phase 8, the reliability gate**, where coverage-aware scoring is the subject.

## 7. Provenance

| | |
|---|---|
| A/B run A | commit `pre-import:P0-worktree`, `artifacts/phase2_baseline/cdvqa_head_test_AB_a93982d.json`, 258.7 s |
| A/B run B | commit `pre-import:P0-primary`, `artifacts/phase2_baseline/cdvqa_head_test.json`, 271.3 s |
| Oracle | commit `pre-import:P0-primary`, `artifacts/phase2_baseline/cdvqa_oracle_test.json`, 0.99748 |
| Worktree state after | restored to `pre-import:P0-primary`, branch `agent-worktree`, identical to the primary checkout; `photometric_bands` present; all untracked research docs intact |
| Published numbers changed | **none** |
