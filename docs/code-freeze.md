# Code freeze — W13

**Plan task 4.7: "Freeze code. Only bug fixes after W13."**

## What freeze means here

From the freeze commit onward, the only changes that go in are:

* **bug fixes** — something measured to be wrong is made right;
* **evidence** — a measurement, a document, a model card, a rehearsal record;
* **demo material** — inputs and scripts for the demonstration.

Not permitted: new capabilities, new tools, retraining that changes a
published number, refactors, dependency bumps that are not security fixes.

**A number in `docs/` may only change if the run that produced it is
re-executed and the new value is recorded with its date.** `phase1-status.md`
is append-only in spirit: later sections correct earlier ones, and nothing is
deleted when it turns out to be wrong. That property is the reason the CDVQA
history — 0.0000, then 0.4439, then 0.5380 — is still readable.

## State at freeze

| | |
|---|---|
| Branch | `phase-0-closeout` |
| Tests | **855 passing** |
| No-torch CI simulation | 730 passed, 18 skipped, 0 failed |
| Illegal-plan rate | **0 / 600** |
| Matrix validation | successful |
| Frontend | typechecks and builds |
| Docker | three images build; API container serves a real query |
| Dependency audit | `pip-audit`: no known vulnerabilities |
| Demo bundle | 9 / 9 beats behave as scripted |
| Working tree | clean |

Freeze commit: the annotated tag **`phase-4-freeze`**. Resolve it with
`git rev-list -n1 phase-4-freeze`; a SHA written into the file it commits
would necessarily name the previous commit.

## The bug-fix bar

A change qualifies as a bug fix if all four hold:

1. Something is **measured** to be wrong — a failing test, a wrong number, a
   defect reproduced in the browser or against the live API.
2. The fix is **scoped to that defect** and does not add capability.
3. The **full regression set** is re-run: `pytest` (**1,247** as of 2026-09-07;
   it was 855 at the freeze — the suite grew with the fixes), the no-torch CI
   simulation, `evaluation/adversarial.py` for the 0/600 guarantee, matrix
   validation, and `make_demo_bundle.py --verify` if any beat could be
   affected.
4. The defect and its fix are **recorded** in `docs/00` §3.6.

Every Phase-4 fix so far met that bar. Three were found by rehearsing rather
than by testing — the PS's built-up query abstaining, the opaque mask overlay,
and the bi-temporal fixture with no change in it — which is the argument for
rehearsing at all.

## Explicitly out of scope after freeze

These are known, documented, and **must not** be started now. Each is real
work with a real regression surface, and the risk of breaking a working demo
exceeds the gain.

| Item | Why it is tempting | Why not now |
|---|---|---|
| CDVQA segmenter (0.2636 mIoU, 0.9975 ceiling) | 93% of the headroom | GPU-hours, and it changes a published number |
| Grounding (Acc@0.5 0.0762) | Weakest component | A retrain, not a fix |
| Image-conditional refusal (2/12) | An open negative result | Needs a designed ablation |
| VRSBench | Closes the third prescribed benchmark | Needs the DOTA download |
| `max_coreg_shift_px` enforcement | Completes the input gate | The estimator is unvalidated — see L16 |
| Tier-1 routing (0.5862) | Weakest measured number | Touching the router risks the 0/600 guarantee |

## The one thing that may still need code

The **recorded backup video** (task 4.6) is not produced. If recording it
exposes a defect, fixing that defect is in scope under the bar above. Nothing
else about the video requires code.

## What has changed since the freeze — 2026-08-30

A full-repository audit ran after the freeze commit. Eight defects were found,
fixed, tested and recorded; the list with evidence is `docs/00` §3.6 **L21-L28**
and `docs/phase4-status.md` §"Post-freeze audit". Every one of them cleared the
four-point bar above:

1. **measured** — each was reproduced before it was touched. The router race
   was replayed against the pre-fix code and contaminated **97 of 800** reads;
   the empty `weights_hashes` was observed in a live trace; the disposable
   artifact backlog was measured at 1,129 directories and 12.29 GB; the two
   unrunnable scripts failed with the command their own docstrings document.
2. **scoped** — no capability was added. The one removal (the stub `worker`
   service) deletes a component that did not exist, and is argued in
   `docs/adr/002-no-async-worker.md`.
3. **regression set re-run** — full `pytest`, the no-torch simulation, the
   0/600 adversarial gate, matrix validation, the frontend production build
   and both Docker images. Results are in the final section of
   `docs/phase4-status.md`.
4. **recorded** — in `docs/00` §3.6, as the bar requires.

Three tempting fixes were **not** made, and the reasons are in
`docs/phase4-status.md`: runtime calibration stays inactive because no tool
reports a probability of correctness, the confidence weights stay equal
because no labelled set exists to fit them against, and the Tier-2 LLM
tiebreak stays unbuilt because the PS does not ask for one.

The six items in §"Explicitly out of scope" were **not** started. That
remains true.

### The freeze's own verification became reproducible

The bar names the no-torch CI simulation, and until this audit there was **no
script for it**. It was run by hand and its result was quoted in four
documents that nobody could reproduce. `scripts/ci_no_torch_sim.py` now blocks
torch, peft, transformers, bitsandbytes, accelerate and datasets at import
time, in a subprocess, and writes its parsed result to
`docs/assets/ci/no_torch.json`. The historical figure of 730 passed / 18
skipped / 0 failed stands as recorded; the new measurements are not comparable
to it, because the suite has since grown, and they are reported with their own
dates rather than replacing it.

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

### One thing the freeze did not anticipate

**The trained checkpoints were destroyed during the audit and could not be
recovered** (`docs/00` §3.6 L26). The freeze protects published numbers from
being *re-derived*; it had nothing to say about the weights behind them being
*deleted*. They were gitignored, and nothing backed them up.

Retraining is a decision for the team lead, and it is not a bug fix: it costs
GPU-hours, it cannot reproduce a previous run exactly, and every number it
produced would have to be published as a new dated section rather than as an
edit. Until that decision is made, the system runs as it does in CI — on stubs
and the deterministic index engine — which is the configuration the whole test
suite exercises and the demo bundle was verified under.

---

## Unfreezing

Only the team lead, and only with the reason written into this file. If a
change cannot be justified in one sentence here, it does not go in before the
finale.

### Unfreeze 1 — Track B retrain, authorised 2026-09-01

**Reason, in one sentence:** the Track B QLoRA adapter is unrecoverable
(`docs/00` §3.6 L32), so the PS's mandatory single-image VQA has no model
behind it, and retraining is the only way to restore a capability the
submission is required to demonstrate.

**Authorised by:** the team lead, 2026-09-01, after the search for an intact
copy was exhausted - 22 candidates across two trees, git history, LFS, the
remote, the HF cache, Docker images and archives, **0 loadable**.

**Scope of the unfreeze:** retraining Track B on the *existing* recipe only -
same base model, same 4,806-example mix, same 300 steps, same LoRA
r=16/α=32/dropout=0.05 and the same seven target modules, all read from the
surviving `checkpoints/track_b_v1/run_metadata.json`. No architecture,
dataset or hyperparameter change. Output goes to a **new** directory,
`checkpoints/track_b_v2`; the eleven corrupted adapters stay on disk as
evidence.

**What this does not license:** the historical `rsvqa_lr` exact match of
**0.6425** is not overwritten. It stands as the v1 result, and whatever v2
measures is published beside it with its own date, because a retrain cannot
reproduce a previous run and pretending otherwise is the failure mode this
document exists to prevent.
