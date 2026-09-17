# Presentation deck — speaker script and slide content

**Plan task 4.6.** This is the deck's *content*, slide by slide, with the words
to say and the evidence behind each claim. Rendering it into slides is a
formatting step; the substance is here so it can be reviewed against the
measurements rather than against a designer's draft.

**The recorded backup video is NOT produced** — see §"What is missing" at the
end. That half of task 4.6 is open.

Every number below is in the repository. If a slide claims it, the citation is
beside it.

---

## Slide 1 — The problem, in one sentence

> An analyst has an optical image, a SAR image, and a question. Today that
> takes three tools and an expert. Watch.

No numbers. Thirty seconds. Do not explain the architecture yet.

---

## Slide 2 — Open on a refusal

Live: upload a PNG in operational mode. The system refuses and quotes the
rule back.

> The problem statement says PNG is admissible **only for the prescribed
> public benchmark datasets**. This is operational mode, so the answer is no —
> and the trace names the check that failed: `crs_present`.

Then the second refusal: a pair whose footprints do not overlap.

> `footprint_overlap 0% — the images do not cover the same area, so no joint
> answer is meaningful.`

**Why open here.** Everything after a refusal is credible. A system that
accepts anything is a system whose successes mean nothing.

*Evidence: demo bundle beats `png_operational` and `incompatible_pair`, both
verified in the browser.*

---

## Slide 3 — Cross-modal, the flagship

Live: optical + SAR, the PS's own query —
*"Use the optical and SAR images together to identify built-up and
water-covered regions."*

⏱ **Warm this beat costs ~4 s; cold it costs 70–83 s (slot: 70 s).**
This is the beat to warm the API with — POST it once before the session and it
needs no filler. If it was not warmed, the "say the fusion result honestly"
paragraph below is what to say *while it runs*, not after. Full cue in
`docs/rehearsal.md`.

Point at the trace as it fills: `index_engine_v1` → `optsar_fusion_v1` (triad)
→ narrative. Then at the line reading *"NDBI unavailable on this 4-band
product; built-up via SAR σ⁰ + optical texture."*

> Cartosat-2S has no SWIR band, so two of the four classical indices do not
> exist on it. The system knows that, substitutes a documented proxy, and
> says so in the trace.

**Say the fusion result honestly:**

> We measured whether fusing beats the better single modality. It does not —
> optical alone scores 0.7778, fused 0.7714, a gain of **−0.0064**. We report
> the triad so you can see that rather than a single fused number that hides
> it.

*Evidence: `checkpoints/optsar_fusion/metrics.json`.*

---

## Slide 4 — Bi-temporal change

Live: the PS's query *"What changed between these two dates, and where did the
change occur?"*

The answer names what changed; the map shows where, in red, over the basemap.

> Two questions, two answers. "What" is a caption conditioned on the change
> mask. "Where" is a georeferenced raster you can open in QGIS — 50.2% of this
> tile is flagged changed, and the unchanged half is transparent so you can
> see the ground underneath.

Then the PS's fifth query — *"Has the built-up area increased, decreased, or
remained unchanged?"*

> Increased. From 82.2% to 83.6% of the scene, about 0.02 km², measured by
> NDBI. That number comes from arithmetic on the index rasters — we never ask
> a language model to compute an area.

*Evidence: run on LEVIR-CD test_000271, 42.1% labelled change.*

---

## Slide 5 — Close the live portion on an abstention

Live: the heavily clouded optical.

> Status: **Abstained**. Not an error — a decision. The three confidence
> components are shown, and input quality is the one that dropped.

> **A system that knows what it cannot see is the one you can actually
> deploy.**

> ⚠️ **Do not present this slide on the CPU/stub fallback.** Measured
> 2026-09-07: this beat abstains on the GPU configuration and **answers** on
> the CPU one, because the learned tools are stubs there and the plan falls
> back to the deterministic index engine. Demonstrating "the system abstains"
> on a config where it does not abstain gives exactly the wrong impression -
> and it would be the one slide a judge remembers. On the CPU fallback, close
> the live portion on the `incompatible_pair` rejection instead, which
> abstains in both configurations. See `docs/00` §3.6 **L36**.

---

## Slide 6 — The engineering

Three artifacts on screen: `capability_matrix.yaml`, the model registry page,
the metric table.

> Orchestration is a constrained planner over a version-controlled capability
> matrix, not free-form tool-calling. The legal task set is computed from the
> **images**, never from the query text, so no phrasing can widen it.

> Measured: the same classifier, ungated, selects an impossible task on
> **148 of 600 plans — 24.7%**. Gated: **0 of 600**.

*Evidence: `docs/assets/adversarial/report.json`, `docs/assets/ablations/`.*

---

## Slide 7 — The result we are most proud of, which is a correction

> The problem statement names CDVQA as the benchmark for change VQA. Our first
> measurement was **0.0000**. Our second was **0.4439** — which is *below* what
> a constant answer scores, 0.5084. We published that as a failure. The third
> is **0.6061**.

> The oracle over ground-truth change maps is **0.9975**. That tells us the
> answer layer is not the problem — nearly all of the remaining gap is one
> segmentation model. A vague "improve the VQA" became a well-posed problem.

> We also found a 20-point gap between calling the tool and running the
> pipeline, because only 67.4% of questions reached the tool that answers
> them. It is now 1.000, including on 151 phrasings the router never saw.

*Evidence: `docs/phase1-status.md`, three dated sections; `artifacts/cdvqa/`.*

---

## Slide 8 — What does not work

Do not skip this slide.

| | |
|---|---|
| Grounding Acc@0.5 | **0.0762** — near floor |
| Optical–SAR fusion gain | **−0.0064** — does not help |
| Tier-1 routing, never-tuned holdout | **0.5862** |
| Image-conditional refusal | **2/12** learned |
| VRSBench, zero-shot | **0.2968** — below GeoChat's 0.408 and below a test-fitted constant |
| Two-track ablation | **not comparable** — reasoned, not demonstrated |

> Twenty limitations are written down with evidence and consequence. These six
> are the ones we would want to know about if we were judging.

---

## Slide 9 — Close

> Two-track adaptation, because the training data is 10 m and your data is
> 1.6 m. A constrained planner, because you grade the trace and not the
> reasoning. Physics verifies neural, because a confident wrong answer is
> worse than an abstention.

> 1,247 tests. Zero illegal plans in 600. It runs offline, on a laptop.

---

## Backup slides — for questions

Keep these unshown unless asked; the answers are in `docs/judge-qa.md`.

* Calibration: change-mask ECE 0.0668 → 0.0034, **affine** not temperature.
* Selective prediction: E-AURC router 0.0405 vs land-cover 0.0966 — raw AURC
  makes them look equal, excess-over-optimal separates them 2.4×.
* Entailment gate: 96% on the clean suite, +1.9 ms; the NLI backend costs
  +2,625 ms, 22× the pipeline.
* Soak: +0.0239 MB/query over 120 iterations with warm-up excluded. At the
  plan's 20 it reads +0.2445 — a false alarm.
* Sensor: EOS-04 measured at 5.40 GHz C-band, 0.09% from Sentinel-1; every
  accessible high-res SAR source is X-band at 9.69 GHz.

---

## What is missing from task 4.6

**The recorded backup video does not exist.** It is insurance against a live
failure at the venue and it cannot be produced by the build: it needs a screen
recording of a human driving the demo end to end, with narration, on the
machine that will be used.

**What exists to make recording cheap:** the demo bundle builds in one command
and verifies all nine beats; `scripts/rehearse.py` runs the whole sequence ten
times and reports per-beat timings. The recording is a session, not a
construction task.

**One timing fact to plan around — in-container GPU, median of four runs
(2026-09-07).** Exactly **one** beat overruns its slot consistently:

| beat | slot | median (n=4) | over slot in |
|---|---|---|---|
| **2:20 single optical, real Cartosat** | 50 s | **63.0 s** | **4 of 4** |
| 1:10 cross-modal flagship | 70 s | 36.9 s | 1 of 4 — *4.2 s warm, 83 s cold* |
| 5:40 the large scene | 60 s | 57.4 s | 1 of 4 — median fits |

**Warm the API before you present — POST the cross-modal pair once.**
Measured, not assumed: the same beat through `/runs` costs **70.0 s, then
3.8 s, then 3.6 s**. The API is one long-lived process, so warming it warms
what serves the demo. Warm with the *cross-modal* input specifically — a
caption query does not load the fusion head.

**A CLI warm-up does not work** and was measured not working (219.9 s, a cold
run): `docker compose exec` starts a new process each time and the warm state
dies with it.

Warmed, total system time is **~131 s** — 31% of the 420 s slot, leaving
**~4:49 for narration** — and the cross-modal and large-scene beats come off
the board. The one thing that survives every run is the single-optical beat
at 13 s over its slot.

Either pre-warm the two real-product runs and show stored permalinks, or use
the **filler cues in `docs/rehearsal.md` §"Filler cues"** — the Cartosat cue
is needed every run; the other two are cold-start contingency.
