# SatQuery AI — Innovation & Extra Features

**PS 26167 · ISRO / Department of Space · SIH 2026**
Document 5 of 6 · Written 2026-08-27

> This is the separately-requested extra-features document. Everything here is **beyond the PS's mandatory scope**. Each item carries an honest cost estimate and a blunt verdict, because an unbuilt feature on a slide is worth nothing and a half-built one is worth less than nothing. The tiers are ordered by **marks-per-hour**, not by how impressive they sound.
>
> **Read §5 first if you are short on time.** It is the "do these five, skip the rest" summary.

---

## Tier 0 — Already in the core architecture (listed so you can *name* them as innovations)

These are in documents `01`–`04` already. They are listed here because in a pitch you must be able to say "here are our five novel contributions" without hunting for them.

### 0.1 Two-track resolution-bridged adaptation

The observation that the mandated training data (BigEarthNet, 10 m) and the evaluation data (Cartosat-2S, ~1.6 m) are 10–20× apart in GSD, and that this makes single-track adaptation structurally unable to answer object-level queries. The bridge — band-presence masking, random band dropout, GSD-conditioned scale augmentation, and a 10 m → 5 m → 1.6 m ladder — is the technical core of the submission.

**Why it is a real contribution:** most teams will fine-tune on BigEarthNet, test on BigEarthNet, and be surprised by the private set. The ablation table in doc `03` §5.1 makes this argument empirically rather than rhetorically.

### 0.2 Constrained planner over a version-controlled capability matrix

Not free-form LLM tool-calling. The PS grades the observable trace, so determinism plus a provable **illegal-plan rate of zero** beats a cleverer but unverifiable agent.

### 0.3 Physics-in-the-loop verification with sensor-aware fallbacks

Classical remote sensing indices as an independent check on neural outputs — including the **SWIR-free built-up path** required by Cartosat-2S's 4-band VNIR composition, and **adaptive σ⁰ thresholding** required by the C-band/X-band RISAT ambiguity.

### 0.4 Three-component calibrated confidence with measured abstention

Model confidence, agreement confidence and input-quality confidence, reported separately, temperature-scaled, with ECE and a risk–coverage curve. Nobody else will bring a reliability diagram.

### 0.5 Georeferenced evidence packs

Masks as GeoTIFF, boxes as GeoJSON in the source CRS, areas in hectares, centroids in lat/lon, opened live in QGIS. Output an analyst can ingest.

---

## Tier 1 — Build these. Cheap, high impact, low risk.

### 1.1 Cross-modal complementarity score

**Cost:** ~1 day (the triad plumbing is already in the fusion tool).

The PS asks for *complementary* extraction from optical + SAR. A good fused number does not prove complementarity. Run three passes — optical-only, SAR-only, fused — and compute a **per-query** number:

```
gain        = metric(fused) − max(metric(optical), metric(SAR))
agreement   = IoU(mask_optical, mask_SAR)
attribution = per class, which modality drove the fused decision
```

Surfaced as *"SAR contributed +14 % IoU on built-up; optical contributed +9 % on water; the modalities agreed on 71 % of pixels."*

**Why build it:** it is simultaneously a runtime feature, the ablation the evaluators want, and the strongest twenty seconds of the demo. Three uses from one day of work. **Highest marks-per-hour item in this entire document.**

### 1.2 Deterministic quantitative path

**Cost:** ~1 day.

Never let a generative model produce a number. Counts come from detections after NMS; areas come from mask pixel counts times the affine transform; increase/decrease/unchanged comes from a signed area difference against an explicit significance threshold. Answers are filled into templates.

**Why build it:** two of the PS's five representative queries are quantitative (*"how many aircraft"*, *"has built-up increased, decreased or unchanged"*). A subtraction is right essentially always; a VLM asked the same question gets the direction wrong some fraction of the time. This converts two representative queries from "usually right" to "reliably right," and reliability on the PS's own examples is what gets checked first.

### 1.3 Entailment gate over generated narrative

**Cost:** ~1.5 days. Model is ~150 MB and runs on CPU.

Split generated prose into sentences; check each for entailment against the structured `ToolResult.payload` facts with a small NLI model; drop or flag unsupported sentences; record `{sentences, retained, flagged}` in the trace.

**Why build it:** it converts "we reduce hallucination" from an assertion into a **measurable** mechanism with a number attached, and it produces the verifier-on/off ablation. Judges ask "how do you know it isn't hallucinating?" — this is the answer that survives follow-up questions.

### 1.4 Hot-swappable LoRA adapters on a shared base

**Cost:** ~1 day (mostly integration with the VRAM manager).

`rs_vqa_v1` and `caption_v1` differ only by adapter. Load the base once, swap adapters in ~100 ms instead of reloading a model in several seconds.

**Why build it:** it is what makes multi-tool plans feel fast rather than merely functional, and it is the difference between a 6-second and a 20-second demo response. Perceived responsiveness carries more weight in a live demo than most teams expect.

### 1.5 Adversarial routing suite (200 queries) with a published pass rate

**Cost:** ~1.5 days, mostly writing queries.

Wrong-config, out-of-scope, contradictory, compound, Hinglish, misspelled, empty, absurdly long. Assert: zero illegal plans, zero unhandled exceptions, every rejection carries a named reason.

**Why build it:** it turns robustness into a table. "We tested 200 adversarial queries; illegal-plan rate 0; every rejection explained" is a claim with evidence, and it is exactly the kind of rigour an ISRO reviewer recognises.

### 1.6 Model registry page in the UI

**Cost:** ~half a day.

A page listing every tool: name, version, base model, adapter, training datasets, weights sha256, VRAM budget, measured latency, and its headline benchmark metric.

**Why build it:** the PS says "selects the appropriate model or tool from a **predefined registry**." This page *is* that registry, visible. It answers a whole class of judge questions before they are asked, and it costs an afternoon.

### 1.7 Cloud-aware modality arbitration

**Cost:** ~half a day (cloud fraction is already computed in Layer 0).

Optical cloud fraction becomes a **routing signal**, not just a warning. Above threshold, the fusion weight shifts to SAR, the confidence penalty applies via the `degraded_if` rule, and the answer says *"optical member 63 % cloud-obscured; analysis is SAR-weighted."*

**Why build it:** this is the textbook operational justification for having SAR at all, and demonstrating that the system acts on it — automatically, with the decision logged — is a genuinely strong operational-realism signal. Cheap because the plumbing already exists.

---

## Tier 2 — Build if Phase 2 finishes on schedule. Real value, real cost.

### 2.1 Box → mask upgrade via SAM 2 / Lang-SAM

**Cost:** 2–3 days. Adds ~2–3 GB VRAM when loaded (manage with LRU eviction).

Grounding produces boxes; feed each box as a prompt to a promptable segmentation model to obtain a pixel-accurate mask.

**Why:** upgrades every grounding answer from a rectangle to a precise polygon, improves the evidence pack, and produces better area figures. Also feeds the optional change-map task.

**Honest caveat:** SAM-family models are trained on natural images and are noticeably weaker on overhead SAR and on small objects in satellite imagery. Expect good results on water bodies and building footprints, mediocre results on vehicles and aircraft. Ship it as an *optional* refinement toggle with the box always available as the fallback, and evaluate before trusting it.

### 2.2 Change-magnitude heat map instead of a binary mask

**Cost:** ~2 days.

Output a continuous change-intensity raster (from feature-space distance or index differencing) alongside the thresholded binary mask, rendered as a colour ramp with an adjustable threshold slider in the UI.

**Why:** binary masks hide the model's uncertainty at boundaries. A heat map with a live threshold slider is visually striking and lets a judge explore the model's behaviour themselves — interactive exploration is far more convincing than a static result. Also makes the `change_threshold` permitted parameter tangible.

### 2.3 Multi-turn conversational context

**Cost:** 2–3 days.

*"How many aircraft?" → "8." → "Where exactly?"* → the follow-up resolves against the previous turn's task and artifacts rather than starting fresh.

Implement as a bounded session store keyed by `run_id`, carrying the previous task, the previous artifacts, and resolved entity references. Critically: **keep the router constrained** — a follow-up still produces a fully validated plan, it just inherits resolved context. Cap history at 3–5 turns.

**Why:** it makes the system feel like an assistant rather than a form. The PS does say "interactive."

**Risk:** this is where agentic systems get unpredictable. Bound it hard, and if it destabilises routing, cut it without regret.

### 2.4 Query-conditioned tile retrieval as a visible planner step

**Cost:** ~2 days on top of the tile pyramid (which is core work in Phase 2).

Embed tiles once, retrieve the top-k relevant to the query, and **show the retrieved tiles highlighted on the scene overview** while the trace logs why they were selected.

**Why:** it makes large-scene handling *visible* rather than merely implemented, and it reframes retrieval as part of the orchestration story. Watching the system decide where to look in a 10,000-pixel-wide scene is a memorable demo moment.

### 2.5 Uncertainty map, not just an uncertainty number

**Cost:** ~2 days.

Render a per-pixel confidence raster: prediction entropy for segmentation heads, plus disagreement between the optical-only and SAR-only masks from the triad, plus verifier disagreement per region.

**Why:** *"the model is uncertain about this specific area, here is where"* is qualitatively better than a scalar. Pairs naturally with the confidence card and gives the abstention message a spatial argument.

### 2.6 Batch / area-monitoring mode in the UI

**Cost:** ~2 days (the eval CLI already does the work; this is a frontend surface on it).

Upload a folder of dated scenes; the system runs the change pipeline pairwise across the sequence and produces a timeline of change magnitude with the mask for each interval.

**Why:** this is what an operational monitoring workflow actually looks like, and it reuses the batch runner you must build anyway (Axiom 4). Turning a compliance requirement into a visible feature is efficient.

### 2.7 Bhoonidhi-sourced Indian-context showcase

**Cost:** ~1 day beyond the week-1 download.

Curate three genuinely interesting Indian scenes — an urban expansion pair (any fast-growing tier-2 city), a water-body seasonal change (a reservoir between pre- and post-monsoon), and a coastal or agricultural scene — with a short narrative for each.

**Why:** you are pitching to ISRO. Real Cartosat/RISAT imagery of Indian terrain, analysed by a system that never trained on it, is a materially stronger demo than Rotterdam from SpaceNet. Very high impact for one day of work — and the download is already a week-1 action for verification reasons.

---

## Tier 3 — Only if you are ahead of schedule. Genuinely novel, genuinely risky.

### 3.1 Counterfactual "why" explanations

**Cost:** 3–4 days.

For a classification or detection decision, report which evidence was decisive: *"classified as built-up primarily because SAR σ⁰ = −6.2 dB with local variance 0.31; optical texture was consistent; NDVI = 0.11 ruled out vegetation. Had σ⁰ been below −14 dB, the classification would have been water."*

Implementable without gradient-based attribution: ablate each verifier signal and observe the decision flip. **Because you have an independent physics verifier, you get a genuine causal explanation rather than a saliency heat map** — and saliency maps are widely and correctly regarded as weak explanations.

**Why it is novel:** this is a real capability that the physics-verifier architecture uniquely enables. No competing design reviewed for this project can produce it.

**Why Tier 3:** it needs the verifier to be solid first, and it is easy to sink days into presentation polish.

### 3.2 Physics-guided self-correction loop

**Cost:** 3–4 days. **Higher risk than it looks.**

When the verifier disagrees strongly with a neural output, re-run the tool with adjusted parameters (different threshold method, higher target GSD, modality reweighting) and take the more consistent result. Every iteration is a logged plan step, so it remains auditable.

**Why:** it is the most defensible form of "agentic" behaviour — a closed loop with an independent, non-neural referee, which is more principled than an LLM critiquing itself.

**Why Tier 3, with a warning:** loops introduce unbounded latency and oscillation. If you build it, **cap at one retry**, require the retry to be a distinct parameter set, and make the loop's exit condition explicit. A demo query that takes 40 seconds because the system is arguing with itself is worse than a slightly wrong fast answer.

### 3.3 Sensor-transfer report card

**Cost:** ~2 days.

An automatically generated page reporting how performance varies across GSD bands, band configurations, and sensors — the *same* model evaluated at 10 m, 5 m and ~1.6 m, on 12-band, 4-band and PAN inputs, on C-band and X-band SAR.

**Why:** it is the empirical proof of the two-track thesis, presented as a product feature rather than buried in an appendix. It is also exactly the analysis an operational agency would demand before deploying anything.

### 3.4 Active-learning queue from abstentions

**Cost:** ~2 days.

Every abstention and every low-confidence answer is logged with its inputs into a review queue, ranked by information value, exportable as an annotation task.

**Why:** it closes the operational loop — the system tells you what to label next. Strong for the "future work / deployment path" slide, and it is a real product idea rather than a hackathon flourish.

### 3.5 Spectral-signature explanation panel

**Cost:** ~1.5 days.

Click a pixel or region; see its per-band spectral profile plotted against reference signatures for water, vegetation, built-up and bare soil, with the SAR backscatter value alongside.

**Why:** it makes the physics visible and interactive, and it is the kind of tool a real analyst would actually use. Cheap, and it strengthens the domain-competence impression more than its cost suggests.

### 3.6 Natural-language report generation for a non-technical reader

**Cost:** ~1.5 days.

A second output register: the same evidence rendered as a short administrative brief — *"Between March and November 2025, built-up area within the analysed footprint increased by 4.7 hectares (18 %), concentrated in the south-eastern quadrant. Vegetation cover declined by 3.1 hectares over the same period."*

**Why:** it addresses the actual user of a space-agency product, who is frequently not a remote-sensing specialist. Runs through the same entailment gate, so it stays grounded.

---

## Tier 4 — Explicitly declined, with reasons

Being able to say *why you did not build something* is a stronger signal than a longer feature list. Each of these appeared in at least one competing design pass.

| Feature | Why declined |
|---|---|
| **Fine-tuning a 7B RS-VLM** (GeoChat, EarthGPT, RS-LLaVA) | Does not fit free-tier T4 at useful visual resolution (doc `03` §1). Used as zero-shot baselines instead — which is a *better* result to report. |
| **Free-form LLM agent as the primary router** | Cannot guarantee legal plans; the PS grades the trace (doc `02` §1). |
| **RAG over remote-sensing literature** | Sounds sophisticated, adds no capability the PS asks for, and adds a retrieval failure mode. The PS wants image analysis, not document QA. |
| **Knowledge graph of detected entities** | Genuinely interesting, entirely out of scope, and a multi-week sink. |
| **Real-time satellite tasking / live data ingestion** | Not in scope; would require API access you do not have and cannot demo offline. |
| **Mobile app** | Zero marginal marks. The evaluators will use a laptop. |
| **User authentication, multi-tenancy, RBAC** | Zero marks. Do not spend an hour on it. |
| **Kubernetes deployment** | Docker Compose is sufficient and far more likely to boot on finale night. |
| **Training a foundation model from scratch** | Requires thousands of GPU-hours. Not a real option; adapting a pretrained one is the correct engineering answer and should be stated as such if asked. |
| **3D reconstruction / DEM generation from stereo Cartosat** | Fascinating, completely out of scope, would consume the entire runway. |
| **ESA SNAP / snappy integration** | Heavy fragile JVM bridge for capability you do not need, since inputs arrive calibrated and georeferenced. |
| **WebSockets for the trace** | SSE is one-directional, reconnects cleanly, and is simpler. No benefit to the upgrade. |

---

## §5 — If you only build five things from this document

Ranked strictly by marks-per-hour. Total cost roughly **five days across two people**, and it lifts the submission more than any single model improvement available in that time.

| Rank | Feature | Cost | Why it wins |
|---|---|---|---|
| **1** | **Complementarity score** (§1.1) | 1 day | Runtime feature + required ablation + best demo moment, from one day of work |
| **2** | **Deterministic quantitative path** (§1.2) | 1 day | Makes two of the PS's own five representative queries reliably correct |
| **3** | **Bhoonidhi Indian-context showcase** (§2.7) | 1 day | Real Cartosat/RISAT imagery, out-of-distribution, in front of ISRO |
| **4** | **Entailment gate** (§1.3) | 1.5 days | Turns the anti-hallucination claim into a measured number |
| **5** | **Model registry page** (§1.6) | 0.5 day | Makes "predefined registry" visible; pre-answers a class of judge questions |

Then, if there is room: cloud-aware modality arbitration (§1.7, half a day), the adversarial routing suite (§1.5, 1.5 days), and the change-magnitude heat map with a threshold slider (§2.2, 2 days).

Everything in Tier 3 is a bonus. **Do not start any Tier 3 item before the W9 checkpoint passes with zero gaps in the mandatory areas.** A team with all five mandatory capabilities working and five Tier-1 extras will beat a team with a counterfactual explanation engine and no change detection — every time, and by a wide margin, because of how normalised score combination works.

---

*Continues in `00-README-and-Requirement-Traceability.md`.*
