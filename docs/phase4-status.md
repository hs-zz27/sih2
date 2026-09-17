# Phase 4 (W13–W14) — status

Audit of plan tasks 4.1–4.8. Every DONE carries the artifact that proves it;
every PARTIAL names precisely what is missing; every BLOCKED names what blocks
it and who can unblock it.

**Three of the eight cannot be closed by the build.** They need a person in a
room, a licence decision, or a screen recorder. Marking them DONE would be the
kind of claim this project has spent four phases not making.

---

## Summary

| # | Task | Status |
|---|---|---|
| 4.1 | Curated demo bundle | **DONE** |
| 4.2 | Ten 7-minute rehearsals, incl. offline venue laptop | **PARTIAL** |
| 4.3 | Technical report | **DONE** |
| 4.4 | PS requirement traceability matrix | **DONE** |
| 4.5 | Model cards + published weights | **PARTIAL** (cards done; weights BLOCKED) |
| 4.6 | Deck + recorded backup video | **PARTIAL** (deck done; video BLOCKED) |
| 4.7 | Code freeze | **DONE** |
| 4.8 | Ten hardest judge Q&As | **DONE** |

**5 done, 3 partial, 0 wholly blocked.** Of the three partials, the missing
halves are: ten *narrated* run-throughs on the venue laptop, a licence
decision on weights, and a screen recording.

> **Read §"Post-freeze audit" at the end of this file before quoting anything
> above.** The task statuses are unchanged, but a deep audit on 2026-08-30
> closed eight defects under the freeze's bug-fix bar, and the trained
> checkpoints were destroyed in an accident during that audit and could not
> be recovered. 4.5's weights are now unavailable as well as unlicensed.

---

## 4.1 — Curated demo bundle · **DONE**

**Evidence:** `scripts/make_demo_bundle.py`, `data/demo_bundle/manifest.json`,
`tests/test_demo_bundle.py` (8 tests).

Nine inputs covering every beat, **three on real Bhoonidhi products**
(Cartosat-2E MX, EOS-04 FRS-1 HH, and the 7687×7640 Cartosat scene). Verified
**9 / 9 beats behave as scripted** via `--verify`, which runs each input
through the real controller and asserts the beat it exists to produce.

The plan asked for 8; there are 9, because the scripted opening rejection
turned out to need two inputs — the overlap rejection *and* the PNG rejection
— for reasons recorded under 4.2 below.

The bundle found three defects that testing had not: the PS's built-up query
abstaining, the opaque mask overlay, and a bi-temporal fixture containing no
change. All three are fixed.

## 4.2 — Ten rehearsals, including offline on the venue laptop · **PARTIAL**

**Evidence:** `scripts/rehearse.py`, `docs/assets/rehearsal/online.json`,
`docs/assets/rehearsal/offline.json`, `docs/rehearsal.md`.

**Done:** twenty automated rehearsals — ten online, ten with the socket layer
blocked. **All beats behaved in all twenty runs.** Offline median 116.6 s
against online 118.5 s, so the system does not need the network, measured.

**Missing, and it is the half the plan actually means:** narrated run-throughs
timed against the spoken script, on the venue laptop. A person is required.

**One finding the team must plan around:** the two real-Cartosat beats take
**≈56 s each** — essentially their whole slot. Every other beat finishes in
under 3 seconds. `docs/rehearsal.md` gives three options in preference order.

## 4.3 — Technical report · **DONE**

**Evidence:** `docs/technical-report.md`.

Architecture, all metrics, four ablations (two measured, one measured-negative,
one `not_comparable`), calibration, AURC/E-AURC, abstention, soak, and
fourteen limitations. Every number cited to the `metrics.json` or
`docs/assets/` artifact that produced it.

It leads on the CDVQA correction — 0.0000 → 0.4439 (below baseline, reported
as failure) → 0.5380 against a 0.9975 oracle — because a corrected measurement
is the strongest evidence of method the project has.

## 4.4 — PS requirement traceability matrix · **DONE**

**Evidence:** `docs/00-README-and-Requirement-Traceability.md` §3;
`docs/ps-26167.md` as the in-repo source of truth.

Refreshed with measured status per requirement, then **checked clause-by-clause
against the authoritative PS text**, which found four defects in it: two
"representative queries" that were not the PS's and one missing entirely; six
claimed deliverables where the PS states two; a controller clause with no row;
and I6 presented as a PS requirement when the PS says nothing about scene size.
All corrected. A test now asserts the PS's five queries still appear verbatim
in `ps-26167.md`, so drift becomes a test failure.

## 4.5 — Model cards + published weights · **PARTIAL**

**Evidence:** `docs/model-cards.md`.

**Done:** a card per trained component — Track A (+ Stage A2/A3), Track B,
caption, grounding, change mask, change caption, semantic change head,
optical–SAR fusion — each with training provenance, metrics, and its weakest
number stated rather than omitted.

**BLOCKED — weights are not published.** The semantic change head is trained on
**SECOND, which states no licence at all** (not a restrictive one — none), so
redistributing weights derived from it is an unresolved question. The rest
await a licence decision the team must make. The PS deliverable is *"codes and
models"*; code and tests are in the repository, weights are not.

**Unblocked by:** an email to the SECOND authors (addresses in
`docs/verification.md`) and a team decision on our own licence.

## 4.6 — Deck + recorded backup video · **PARTIAL**

**Evidence:** `docs/deck.md`.

**Done:** the deck's full content — nine slides plus backup slides, with the
words to say and the citation behind every claim. Slide 8 is "What does not
work" and lists six weaknesses with numbers.

**BLOCKED — the recorded backup video does not exist.** It needs a screen
recording of a human driving the demo with narration, on the machine that will
be used. It is insurance against a live failure and cannot be produced by the
build.

**Unblocked by:** one recording session. The bundle builds in one command and
`rehearse.py` reports per-beat timings, so the recording is a session rather
than a construction task.

## 4.7 — Code freeze · **DONE**

**Evidence:** `docs/code-freeze.md`, and the SHA below.

Defines what may still change (bug fixes, evidence, demo material), the
four-point bar a bug fix must clear, and an explicit out-of-scope list — the
CDVQA segmenter, grounding, refusal, VRSBench, `max_coreg_shift_px`, and the
router — each with why it is tempting and why not now.

## 4.8 — Ten hardest judge Q&As · **DONE**

**Evidence:** `docs/judge-qa.md`.

Ten questions with honest answers, chosen as the ones a careful reviewer
actually asks: the 7.6% grounding accuracy, the negative fusion gain, the
narrow CDVQA margin, whether the agentic layer earns itself, the BigEarthNet.txt
substitution, the confidence-above-abstention contradiction, the
co-registration claim, the RISAT question, the missing third benchmark, and
"what is most likely wrong in what you have shown me".

Plus two questions we want to be asked, where the honest answer is the
strongest one.

---

## Freeze commit

The authoritative pointer is the **annotated tag `phase-4-freeze`**, not a SHA
written into this file. A commit cannot contain its own hash, so any SHA
recorded here would name the commit *before* the one that records it — which
is exactly the mistake this line replaces.

```bash
git rev-list -n1 phase-4-freeze     # the freeze commit
git show phase-4-freeze             # what it marks
```

| | |
|---|---|
| Branch | `phase-0-closeout` |
| Tag | `phase-4-freeze` |
| Date | 2026-08-30 |

## Verification at freeze

| gate | result |
|---|---|
| Full test suite | **855 passed** |
| No-torch CI simulation | **730 passed, 18 skipped, 0 failed** |
| Illegal-plan rate | **0 / 600** |
| Matrix validation | successful |
| Demo bundle | **9 / 9 beats** |
| Rehearsals | **20 / 20 clean** (10 online, 10 offline) |
| Frontend | typechecks and builds |
| Docker | three images build; API container serves a real query |
| `pip-audit` | no known vulnerabilities |

## What Phase 4 changed about the plan's own claims

Three items in the plan turned out to be wrong when checked, and are corrected
rather than carried forward:

1. **The PS lists two deliverables, not six.** The technical report, model
   cards and this document set are team-chosen and worth having; they are not
   PS requirements. The *demonstration* is, and it did not exist until 4.1.
2. **Two of the plan's five "PS representative queries" were not the PS's**,
   and the PS's only grounding query was missing from the matrix entirely.
3. **The 7-minute script cites a fusion gain of "+14% IoU on built-up".** The
   measured gain is **−0.0064** — fusion does not beat optical alone. That
   line must not be spoken; `docs/deck.md` replaces it with the real number.


---

## Post-freeze audit — 2026-08-30

A full-repository audit was run after the freeze: inspection, fixes, regression
tests and verification. Everything it changed clears the four-point bar in
`docs/code-freeze.md` §"The bug-fix bar" - each item was measured to be wrong,
each fix is scoped to that defect, the regression set was re-run, and each is
recorded in `docs/00` §3.6 as **L21-L28**. No capability was added, nothing was
retrained, and no published number was re-derived.

### What was closed

| # | Defect | Evidence |
|---|---|---|
| L21 | `Trace.weights_hashes` was always `{}` while real checkpoints were loading | `satquery/tools/provenance.py`; digest verified against `sha256sum`; 20 tests |
| L22 | Router state leaked between concurrent runs - **97/800** contaminated reads pre-fix, 0 after | `Router.decide()`; `tests/test_concurrent_runs.py` |
| L23 | A `worker` compose service that printed one line and exited | `docs/adr/002-no-async-worker.md`; 13 packaging tests |
| L24 | Frontend image ran `npm run dev`; no page linked to `/models` or `/benchmarks` | multi-stage image, shared `Nav`, run permalink |
| L25 | `artifacts/` unbounded - 23 GB across 1,133 directories at audit time | `satquery/controller/retention.py`; 24 tests |
| L26 | **Checkpoint loss** - see below | this section |
| L27 | The resume harness deleted `checkpoints/` unconditionally | `training/run_checkpoint_test.py`; entry-point tests |
| L28 | `evaluation/cdvqa_predict.py` could not be run as documented | `sys.path` fix; 44 scripts exercised |

Three things were **deliberately not** implemented, and the reasons are
scientific rather than schedule-driven:

* **Runtime calibration stays inactive.** `CALIBRATABLE_CONFIDENCE_METHODS` is
  empty because no tool reports a probability of correctness - `sharpness`
  measures decisiveness, `logprob` measures fluency, `mean_asserted_probability`
  is an aggregate over a threshold-selected subset, and `segmentation_derived`
  is a stated constant. The fitted parameters and their ECE tables are the
  deliverable of task 3.3; the path activates by itself when a tool reports a
  real per-head P(correct). Calibrating a stub's hardcoded 0.8 with the
  land-cover transform would put a fabricated "calibrated" number in front of a
  judge.
* **Confidence weights stay equal.** Fitting them needs labelled
  (components -> was the answer correct) pairs. No such set exists, so a fitted
  weight would be fitted to nothing.
* **The Tier-2 LLM tiebreak stays unbuilt.** `docs/ps-26167.md` does not ask for
  one; it says the controller *may* perform internal task planning and that only
  the observable trace is evaluated. `llm_tiebreak_invoked` is an honest flag on
  an unbuilt feature (L9).

### Checkpoint recovery — 2026-08-31

**RECOVERED.** The checkpoints deleted on 2026-08-30 were restored in full
from a Windows volume shadow copy taken at 13:23 that day, about six hours
forty minutes before the deletion.

| | |
|---|---|
| Restored | **4.542 GB, 136 files, 18 directories** |
| Source | shadow copy `{a76216eb-4a3e-4cc2-9fe2-c45fd07349ba}`, created 2026-08-30 13:23 |
| Proof | `change_mask/ckpt_step_1780.pt` → `sha256:02b060ff…4c168`, **the digest recorded from the live file before the deletion** |
| Weights | **61 / 61 `.pt` files load** under `safe_torch_load` |
| Metrics | match the published numbers exactly — Track A mAP 0.285365, grounding Acc@0.5 0.076249, fusion gain −0.006376, CDVQA change-class mIoU 0.263639 |
| Manifest | `checkpoints_restored/RECOVERY_MANIFEST.json`, beside an untouched preserved copy |

**Not everything came back.** Twelve small JSON sidecars — 42,104 bytes,
0.00093% of the tree — restored as their correct size in NUL bytes, because
their data was still in the write cache when the snapshot froze the volume.
Three of them matter: `caption/vocab.json` and `grounding/vocab.json`, so
`caption_v1` and `grounding_v1` fall back to their stubs, and
`track_a_full_multires/band_stats.json`, so that variant cannot be normalised
(`track_a_full_base` is unaffected). Full account: `docs/00` §3.6 **L29**.

The incident also exposed **L30**: `is_available()` checked that a sidecar
*existed*, not that it could be *read*, so `caption_v1` reported ready and
then raised inside the loader. `satquery/tools/sidecars.py` now parses the
file, and the same rule gates the tests.

**Sidecar repair, 2026-08-31.** Three of the twelve were repaired, each
validated by reproducing a published metric rather than by inspection:

| Sidecar | Validation |
|---|---|
| `caption/vocab.json` | BLEU-4 **0.24460787515482577** against the published **0.24460787515482577**; `n` 1093, `unique_captions` 146 — exact |
| `grounding/vocab.json` | Acc@0.5 and Acc@0.7 bit-exact; mIoU to 4.2e-9 |
| `track_a_full_multires/band_stats.json` | multires mAP identical to 17 digits at 10/20/30/40 m |

~~All eight learned tools now load.~~ **CORRECTED 2026-09-01: seven of eight load, not eight.** The Track B QLoRA adapter is destroyed - `adapter_model.safetensors` is 148,712,776 bytes of which the first 148,701,184 are NUL (99.9922%), and the same is true of all eleven adapter files, 1.636 GB in total. The earlier claim came from a verification that loaded the 61 `.pt` files and only *hashed* the safetensors, so a whole model's weights were reported as recovered without ever being opened. Re-verified by loading every weight file: **64 load (10.784 GB), 11 fail (1.636 GB)**, the failures being exactly the adapters. See `docs/00` section 3.6 **L32**.

The recovered NUL files are kept as
`*.zeroed-2026-08-30`; all 61 `.pt` digests were re-verified unchanged.
Nine sidecars remain zeroed and are reporting files only.

**Index substitution defect, 2026-08-31 (L31).** The index engine claimed
*"MNDWI unavailable (no SWIR1); NDWI used as the water index"* on inputs where
NDWI was not computable either - a 3-band RGB raster reported a water-index
substitution while computing no water index at all, and the executor forwarded
the false claim to the verifier as a conflict. Fixed by making the
substitution conditional on NDWI having actually run, and by reporting the
absence explicitly when neither water index is computable. Panchromatic inputs
were affected too. The SWIR-free VNIR path the Cartosat beats depend on is
unchanged.


---

### The checkpoint incident

**Status: DATA LOSS - NOT RECOVERED.**

`training/run_checkpoint_test.py` hardcoded `ckpt_dir = "checkpoints"`, called
`shutil.rmtree` on it unconditionally, and had no argument parser - so passing
`--help` to check that it ran did not print help, it ran the program. Every
trained checkpoint was destroyed.

* **Minimum measured loss: 2.75 GB** across 10 entries, plus six directories
  whose sizes were never measured - `track_a_full_base`,
  `track_a_full_multires` (with `band_stats.json`), `track_b_v1/adapter_final`,
  `stage_a2`, `stage_a2_frozen`, `stage_a3`. The full inventory is L26.
* **Recovery was attempted and failed.** `checkpoints_backup/` holds only 244 KB
  of this same harness's scratch tensors. `checkpoints/` is gitignored, so
  nothing was ever committed, and there are no LFS pointers. A profile-wide and
  C:-wide search for `*.pt`, `*.pth`, `*.safetensors`, `*.ckpt` found no trace.
  Desktop is not OneDrive-redirected; there is no File History; the Recycle Bin
  is empty because `shutil.rmtree` unlinks directly. **Volume Shadow Copy is the
  one source that could not be checked** - `vssadmin list shadows` requires an
  elevated prompt.
* **What survives:** every published number. `docs/assets/` (calibration,
  abstention, adversarial, entailment, ablations, confidence, soak, rehearsal),
  `docs/phase1-status.md`, `docs/model-cards.md`, `docs/technical-report.md`,
  `configs/calibration.json`, `configs/thresholds.yaml`,
  `configs/model_lock.json`. The third-party base models under `models/`
  (Qwen2.5-VL-3B, DeBERTa-MNLI) were untouched - only the fine-tuned adapter
  that pairs with Qwen was in `checkpoints/`.
* **Effect on the system:** none on the test suite or the demo path, both of
  which run on stubs and the deterministic index engine. The `/models` registry
  page renders empty, because it reads `metrics.json` and `run_metadata.json`
  from inside the deleted directories. No learned tool can be switched on.
* **Retraining is deferred as a separate decision.** It costs GPU-hours and it
  would change published numbers, which the freeze forbids; and a retrained head
  cannot reproduce a previous run exactly, so whatever comes back belongs in a
  new dated section rather than in an edit.

### Verification after the audit — 2026-08-30

Measured on the final tree. The freeze-time figures in §"Verification at
freeze" above stand as recorded and are **not comparable**: the suite has since
grown by 115 tests, and the trained checkpoints were deleted.

| gate | result |
|---|---|
| Full test suite | **1070 passed, 0 failed, 0 skipped**, 456.8 s (2026-09-01, after the stub-confidence cap) |
| No-torch CI simulation | **851 passed, 32 skipped, 0 failed**, 153 s — `docs/assets/ci/no_torch.json` |
| Illegal-plan rate | **0 / 600** — `docs/assets/adversarial/report.json` |
| Matrix validation | successful |
| Frontend | typechecks (`tsc --noEmit`) and builds (`next build`, 5 routes) |
| Docker | `docker compose config` valid; both images build; API container served a real query as `uid=1000(satquery)`; healthcheck reported healthy |
| Security sweep | no credential patterns in tracked files; no `eval`/`exec`/`os.system`/`shell=True`/`yaml.load`/`pickle.load` in shipped code; `torch.load(weights_only=True)`; CORS defaults to localhost, never `*` |
| Artifact retention | `satquery prune --dry-run`: 1,129 disposable directories / 12.29 GB, **76 named evidence directories protected** |

**No skips.** Every checkpoint-gated test now runs: the suite skipped nine
and failed one immediately after the loss, three after the recovery, and none
after the sidecar repair. The run is longer (455 s against 235 s) because the
real models now load and infer instead of skipping.

### Task statuses after the audit

Unchanged: **5 DONE, 3 PARTIAL**. No Phase-4 task moved in either direction.
The audit did not touch the three partials, because none of them is a code
defect - they are a person in a room, a licence decision, and a screen
recorder. 4.5 gains a second obstacle: the weights it would publish no longer
exist.
