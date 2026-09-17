# Recording the demo video — without a GPU

For whoever records the SatQuery AI demo on an ordinary 16 GB laptop. It uses
the beats `scripts/make_demo_bundle.py --verify` checks, and only numbers from
[`docs/as-built-status.md`](as-built-status.md).

**How this works without a GPU.** The `cpu` profile runs every learned model on
the processor. The question-answering model loads unquantised (bfloat16, about
7.5 GB of RAM) instead of the 4-bit GPU build, so answers are **slower, not
worse in kind**. Each such answer carries a warning that the benchmark figures
were measured on the GPU build. The small models (caption, grounding, change
detection, land cover, fusion) are light on a CPU.

---

## 1. What you need

| Need | Amount |
|---|---|
| RAM | **16 GB** — close other apps while recording |
| Free disk | **~20 GB**: checkpoints 8.56 GB, base models 7.36 GB, PyTorch + web UI build ~2–3 GB |
| Software | Python 3.12, Node.js 20+, QGIS (for the evidence beat) |
| Weights | the team's Phase 5 checkpoints, plus the public Qwen2.5-VL-3B base model |

No GPU and no Docker are needed.

---

## 2. One-time setup

```bash
python3.12 -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -e ".[cpu,report]"
# Linux only, before the line above, to skip ~2 GB of CUDA libraries:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python scripts/fetch_models.py --dest models --only qwen25_vl_3b   # ~7 GB download
python scripts/demo_assets.py                                     # what is still missing
```

Copy the team checkpoints into `checkpoints/` until `demo_assets.py` shows
nothing missing. It prints the exact folder each one goes in.

## 3. Start it

```bash
python scripts/run_demo_cpu.py        # or: make demo-cpu
```

It starts the API and the web UI (the first run installs and builds the UI),
then prints a pre-flight report. Open http://localhost:3000.

**Do not record unless:**
- the pre-flight ends in **`GO - ready to record`** (not "layout rehearsal only"), and
- the header shows a **green `MODELS 8/8 LIVE`** chip. An amber `STUBS` chip means
  some answers will be placeholder text; hover it to see which models and why.

**Warm up.** The first question to each model loads its weights, and the VQA
model is by far the slowest to load on a CPU. Ask one throwaway question of
each kind (a description, a change question, a cross-modal question) before
the real take. CPU answer times have not been measured yet: time one warm VQA
answer before planning the recording.

---

## 4. Recording on a CPU

- **Record continuously, cut the waiting in editing.** Keep the spinner and
  the live trace panel on screen for a second or two, then cut to the answer.
- **Say that waits were shortened.** A small caption such as "processing time
  shortened" keeps the video honest. Do not describe it as real time.
- Plug in the charger and close browser tabs, Docker Desktop and IDEs: the
  VQA model needs the RAM.
- Screen: 1920×1080, browser zoom 110–125%, notifications off, one tab.

---

## 5. Shot list (about 6 minutes after editing)

| Time | Beat | Input (demo bundle) | Query | Show and say |
|---|---|---|---|---|
| 0:00–0:25 | **Hook** | — | — | "An analyst has an optical image, a SAR image and a question. Today that takes several tools and an expert." |
| 0:25–1:05 | **The rejection** | `incompatible_pair` | *Use the optical and SAR images together to identify built-up and water-covered regions.* | Refused before any model runs: footprint overlap below the 70% gate. "Everything after this is trustworthy because it refuses first." |
| 1:05–2:05 | **Single-image VQA** | `single_optical` | *Describe the land-cover and major objects visible in this image.* then *Is there water in this scene?* | Answer, the three-part confidence card, the live trace. "0.8947 on the official RSVQA-LR test; the base model alone scores 0.37." |
| 2:05–3:05 | **What changed, and where** | `change_what_and_where`, then `bitemporal_pair` | *What changed between these two dates, and where did the change occur?* then *Has the built-up area increased, decreased, or remained unchanged?* | Prose **and** the change mask on the map; swipe comparator. "Change mask F1 0.855 on LEVIR-CD." |
| 3:05–3:40 | **Cross-modal, honestly** | `crossmodal_pair` | the rejection beat's query | The answer, then the PDF report's complementarity line. "Fusion does not beat optical alone on the only open paired dataset, and the system reports that rather than hiding it." |
| 3:40–4:20 | **Evidence in QGIS** | the change run | — | On the run page, **Download the evidence pack**; open its GeoTIFF in QGIS over the source scene. "A georeferenced product, not a screenshot." |
| 4:20–5:05 | **Knowing its limits** | `clouded_optical`, then any image | *Describe the land-cover…*, then *What is the weather forecast for this location?* | Cloudy scene abstains and names cloud cover. The weather question is declined as out of scope. "A system that knows what it cannot see is one you can deploy." |
| 5:05–5:45 | **The engineering** | — | — | The `MODELS 8/8 LIVE` chip; the Models page (versions, SHA-256 of each checkpoint); `configs/capability_matrix.yaml`; a run's execution warnings. "0 illegal plans out of 600 adversarial ones; over 1,300 automated tests." |
| 5:45–6:05 | **Close** | — | — | "A constrained planner because the trace is what gets graded. Physics checks because a confident wrong answer is worse than a refusal. And it runs on an ordinary laptop." |

Optional: `png_operational` (a PNG accepted for a visual question, with the
georeferencing loss stated) and `large_scene` (the tiling path).

Build the inputs once with
`python scripts/make_demo_bundle.py --out data/demo_bundle --verify`.

---

## 6. Numbers you can say

Measured on the GPU build; the demo runs the same models on a CPU.

| Claim | Figure |
|---|---|
| Single-image VQA, official RSVQA-LR test | 0.8947 (base model 0.3717) |
| Change mask, LEVIR-CD | F1 0.8550 |
| Captioning, RSICD | BLEU-4 0.2658 |
| Change VQA, CDVQA | 0.6061 vs majority 0.5084 |
| Orchestration | 0 illegal plans / 600 |

## 7. Do not say or show

Lines from the original plan (`docs/04` §10) that are **not true of the built system**:

- ~~"SAR contributed +14% IoU"~~ — fusion is negative (fused 0.7420 vs optical 0.7722).
- ~~"Eight aircraft, eight boxes on the map"~~ — grounding is Acc@0.5 0.1604; don't feature box-drawing.
- ~~"Built-up grew 4.7 hectares (+18%)"~~ — an invented example; read what the real run says.
- ~~"A PNG in operational mode is rejected"~~ — PNGs are accepted for visual questions, with a warning.
- ~~"Runs on a T4" / "real time"~~ — not measured; on this setup it runs on a CPU, with waits shortened in editing.
- Any answer containing `[STUB` — re-take the shot.
- "just tell me what I'm looking at" — this phrasing gets a rephrase request; use the queries above.

## 8. If the checkpoints are not available

Without them, these beats are still real and can be recorded honestly: the
**rejection**, the **cloud abstention**, the **out-of-scope refusal**, the
**routing decision and trace**, and **index maths on a GeoTIFF** (NDVI,
water, built-up). Leave the amber `STUBS` chip visible, and never show a
stubbed answer as a model's.
