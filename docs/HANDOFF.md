# Session handoff — 2026-08-30 (post-freeze audit, and a data loss)

> **Newest — 2026-09-12: Phase 5 has run.** All nine tools retrained on the
> AI Lab L40S; results in `docs/phase1-status.md` §"Phase 5 — 2026-09-12"
> and `docs/model-cards.md` §"Phase 5 cards". Headline: `rs_vqa_v1` scores
> **0.8947** on the official RSVQA-LR test (published convention, in the
> literature's 89–93% range); `change_mask` F1 0.56 → 0.86; `optsar_fusion`
> still shows no SAR complementarity under a second architecture.
>
> **DEPLOYED, later that day.** The default compose files point at the Phase 5
> checkpoints and the refitted v2 registries; `docker-compose.v1.yml` reverts
> with one flag. 8/8 tools load, demo bundle 9/9. One tool stays on v1:
> `change_caption`, whose v2 regressed on the changed half (0.1641 vs 0.3063)
> once the evaluator's split was reinstated - the aggregate had hidden it.
> Deploying also surfaced and fixed four pre-existing defects, including that
> the pipeline had never looked at cloud cover. `docs/phase5-full-training.md`
> §7 "Deployed".
>
> **Earlier — 2026-09-09: Phase 5 exists.** `docs/phase5-full-training.md` is a
> harness and a plan for retraining all nine tools on college cluster GPU,
> because the constraint every model here was sized for — "free-tier Colab and
> Kaggle only, T4 16 GB" (`docs/03` §1) — no longer holds.
>
> **Nothing in this file or in any other document has changed as a result, and
> no training run has been executed.** The freeze is honoured structurally: v2
> architectures live in `training/v2/`, `--arch` defaults to `v1` everywhere, a
> checkpoint with no `arch` field rebuilds as v1, and every new `ckpt_dir` is
> under `checkpoints/v2/`. Every number below is still the number it was.
>
> Tests are **1331 passed, 0 failed** (was 1274); no-torch simulation **1171
> passed, 71 skipped, 0 failed**.

> **Newer than this file — 2026-09-07.** This handoff is dated 2026-08-30 and
> its numbers are the numbers of that day. Kept as written; superseded here.
>
> | | 2026-08-30 (below) | **2026-09-07** |
> |---|---|---|
> | Tests | 1,070 | **1,274 passed, 0 failed** |
> | No-torch CI simulation | 851 passed, 32 skipped | **1,121 passed, 35 skipped, 0 failed** |
> | Illegal plans | 0 / 600 | **0 / 600** (re-verified) |
> | Demo bundle | 9 / 9 | **9 / 9** (was 8/9 for six days - see `docs/00` **L34**) |
> | CDVQA | 0.5380 | **0.6061** - the earlier run was fed BGR images (`docs/research/cdvqa-baseline-correction-2026-09-03.md`) |
> | Prescribed benchmarks | RSVQA-LR + CDVQA | **all three** - VRSBench evaluated zero-shot; `docs/00` **L11** closed |
>
> Also new: **`docs/checkpoint-decision.md`** - whether to deploy `track_b_v3`
> over the deployed `track_b_v2`, what it would cost, and the four
> preconditions. It is a decision document, not a decision; `v2` stays
> deployed until someone acts on it. And **W16 is closed** - the six
> specialist heads now have `--eval-only`, so re-measuring one no longer
> overwrites the number being checked (`docs/00` **L35**).

**Start here.** The project is past the Phase-4 code freeze and has since had a
full-repository audit. Read `docs/phase4-status.md` first — its last section,
§"Post-freeze audit", is the newest state — then `docs/code-freeze.md` for what
may still change. `docs/phase1-status.md` carries every measured number in
dated sections, and later sections correct earlier ones.

| | |
|---|---|
| Branch | `phase-0-closeout`, **uncommitted audit changes in the working tree** |
| Freeze tag | **`phase-4-freeze`** — resolve with `git rev-list -n1 phase-4-freeze` |
| Tests | **1070 passed, 0 failed, 0 skipped**, 456.8 s (2026-09-01, after the stub-confidence cap) |
| No-torch CI simulation | **851 passed, 32 skipped, 0 failed** — `docs/assets/ci/no_torch.json` |
| Illegal plans | **0 / 600**, re-verified after the router change |
| Demo bundle | **9 / 9 beats**; rehearsals **20 / 20** (10 online, 10 offline) |
| Trained checkpoints | **RECOVERED 2026-08-31** — 4.542 GB, 136 files, bit-exact; three zeroed sidecars repaired and validated. All 8 tools load |
| Open PR | none — pre-import review history is not carried into this repository |

**The freeze still holds: bug fixes, evidence and demo material only.** The
six items in `docs/code-freeze.md` §"Explicitly out of scope" — the CDVQA
segmenter, grounding, refusal, VRSBench, `max_coreg_shift_px`, the router —
must not be started.

---

## 1. Read this first: the checkpoints were lost, then recovered

`training/run_checkpoint_test.py` hardcoded `ckpt_dir = "checkpoints"`, called
`shutil.rmtree` on it unconditionally, and had **no argument parser** — so
passing `--help` to check whether it ran did not print help, it ran the
program. Every trained checkpoint was destroyed on 2026-08-30 during the audit.
`make test-resume` reached the same code.

**Recovered on 2026-08-31** from a Windows volume shadow copy taken at 13:23
that day, about six hours forty minutes before the deletion:

* **4.542 GB, 136 files, 18 directories** restored, and **bit-exact**:
  `change_mask/ckpt_step_1780.pt` came back as
  `sha256:02b060ff…4c168`, the digest recorded from the live file before the
  deletion during the L21 provenance work.
* **All 61 `.pt` files load**, and every `metrics.json` matches the published
  numbers (Track A mAP 0.285365, grounding Acc@0.5 0.076249, fusion gain
  −0.006376, CDVQA change-class mIoU 0.263639). **No number was ever
  re-derived or adjusted.**
* A manifest of every file, size and SHA-256 is at
  `checkpoints_restored/RECOVERY_MANIFEST.json`, beside a **preserved copy
  that must not be modified**. `C:\shadow_ro` is the read-only mount of the
  snapshot; it can be removed once you are satisfied.
* **Twelve small JSON sidecars did not survive** — 42,104 bytes, restored as
  their correct size in NUL bytes because the data was still in the write
  cache when the snapshot froze the volume. **The three that mattered were
  repaired on 2026-08-31**, each validated by reproducing a published metric:
  the caption vocabulary (BLEU-4 exact to 17 digits), the grounding vocabulary
  (Acc@0.5 and Acc@0.7 bit-exact) and the multires band statistics (mAP
  identical at all four GSD levels). ~~All eight learned tools load.~~ **RESTORED 2026-09-01: all eight load again** after the Track B retrain (`checkpoints/track_b_v2`, rsvqa_lr 0.6473 against v1's 0.6425 on the identical split).~~ **CORRECTED 2026-09-01: seven of eight load, not eight.** The Track B QLoRA adapter is destroyed - `adapter_model.safetensors` is 148,712,776 bytes of which the first 148,701,184 are NUL (99.9922%), and the same is true of all eleven adapter files, 1.636 GB in total. The earlier claim came from a verification that loaded the 61 `.pt` files and only *hashed* the safetensors, so a whole model's weights were reported as recovered without ever being opened. Re-verified by loading every weight file: **64 load (10.784 GB), 11 fail (1.636 GB)**, the failures being exactly the adapters. See `docs/00` section 3.6 **L32**. Nine
  reporting-only sidecars remain zeroed. `docs/00` §3.6 **L29**.
* **The hole is closed**: the harness now defaults to a scratch path under
  `artifacts/`, refuses `checkpoints/` by name, refuses any directory holding
  files it did not write, and has a parser. `tests/test_script_entrypoints.py`
  fails if **any** `__main__` script lacks an argument parser (L27). And
  `is_available()` now parses its sidecars rather than checking they exist,
  which is what let a zeroed vocabulary report "ready" (L30).

**Back up `checkpoints/` off this volume.** It is gitignored by design and had
no backup at all; the only reason 4.5 GB came back is that System Protection
happened to have taken a snapshot that morning. That is luck, not a strategy.

---

## 2. What the audit changed

Eight defects, each measured before it was touched, each with a regression
test, all recorded as `docs/00` §3.6 **L21–L28**:

| | Defect | Fix |
|---|---|---|
| L21 | `Trace.weights_hashes` always `{}` while real checkpoints loaded | `satquery/tools/provenance.py` — SHA-256 of the artifact each tool loaded. **Stubs get no hash** |
| L22 | Router state leaked between concurrent runs — **97/800** contaminated reads pre-fix | `Router.decide()` returns a `RouteDecision`; the controller carries it |
| L23 | A `worker` compose service that printed one line and exited | Removed. `docs/adr/002-no-async-worker.md` |
| L24 | Frontend image ran `npm run dev`; no page linked to `/models` or `/benchmarks` | Multi-stage production image (non-root), shared `Nav`, run permalink |
| L25 | `artifacts/` unbounded — 23 GB / 1,133 directories at audit time | `satquery/controller/retention.py`, `satquery prune` |
| L26 | **The checkpoint loss** | Not recoverable — see above |
| L27 | The resume harness deleted `checkpoints/` unconditionally | Scratch default, two refusals, an argument parser |
| L28 | `evaluation/cdvqa_predict.py` could not be run as its docstring documents | `sys.path` pattern from `evaluation/adversarial.py` |

Also new: `scripts/ci_no_torch_sim.py`, which makes the freeze's own no-torch
verification reproducible for the first time — it was quoted in four documents
and had no script.

**Three things were deliberately not built**, and the reasons are scientific:
runtime calibration stays inactive because no tool reports a probability of
correctness; the confidence weights stay equal because no labelled set exists
to fit them against; the Tier-2 LLM tiebreak stays unbuilt because the PS does
not ask for one (L9).

---

## 3. What the next session should actually do

Nothing in the codebase is blocking. **The four remaining items need a person
or a decision, not a build:**

1. **Ten narrated rehearsals on the venue laptop, one recorded** (tasks 4.2 and
   4.6). `python scripts/rehearse.py --runs 10 --offline` checks the system's
   half and exits non-zero if any beat misbehaves or overruns — run it on the
   venue machine first. **Plan around this:** the two real-Cartosat beats take
   ≈56 s each, essentially their whole slot; every other beat is under 3 s.
   `docs/rehearsal.md` recommends pre-warming them and showing the stored
   `/runs/{id}` permalinks — which the query page now links to directly.
3. **A licence decision on publishing weights** (task 4.5). The semantic change
   head is blocked outright — SECOND states *no licence at all*. The weights
   exist again, so this decision is live rather than moot.
4. **The SIH deadline and submission format** — open since W0 and the only
   thing that decides whether anything must be cut.

If asked to improve a number instead, point at the freeze.

**The working tree is not committed.** `git status` shows the audit's changes;
review and commit them deliberately.

---

## 4. Environment

- Local RTX 4050 with CUDA, torch 2.13+cu126, bitsandbytes, peft, accelerate.
- `gh` is installed but not on the Git Bash PATH: call it as
  `"/c/Program Files/GitHub CLI/gh.exe"`.
- Gitignored but on disk: BigEarthNet shards, WHU-OPT-SAR, LEVIR-CD/MCI, RSICD,
  DIOR-RSVG, the Bhoonidhi products, CDVQA (`data/cdvqa/`), `models/`
  (`qwen25_vl_3b` and `nli_deberta_mnli`, digests in
  `configs/model_lock.json`), and `checkpoints/` — **restored 2026-08-31**,
  4.542 GB, with the three exceptions in L29.
- Learned tools are all **opt-in by environment variable** and fall back to
  stubs, which is what keeps CI green and what kept the system working while
  no checkpoint existed: `SATQUERY_CAPTION`, `SATQUERY_GROUNDING`,
  `SATQUERY_CHANGE_CAPTION`, `SATQUERY_LANDCOVER`, `SATQUERY_CHANGE_MASK`,
  `SATQUERY_FUSION`, `SATQUERY_NLI`, `SATQUERY_VQA_BASE`/`_ADAPTER`.
- `make report` regenerates every evaluation artifact under `docs/assets/`.
  **Do not run it casually** — it overwrites published reports, and any number
  it changes belongs in a new dated section rather than in an edit.
- **CI has no torch.** Verify with `python scripts/ci_no_torch_sim.py` before
  claiming a green CI — it caught a module-scope import that made the whole
  package unimportable, which the normal suite could not.
- `satquery prune --dry-run` reports the reclaimable artifact backlog; at the
  audit it was 1,129 directories and 12.29 GB, and nothing was deleted.

---

## 5. Longer history

Everything before this session — the Phase 1–3 results, the CDVQA correction
from 0.0000 to 0.4439 to 0.5380, the 20-point routing gap that was found and
closed, the RISAT narrowing, the twenty limitations catalogued with evidence —
is in `docs/phase1-status.md`, `docs/verification.md` and `docs/00` §3.6, in
dated sections. Nothing there has been deleted or rewritten; later sections
correct earlier ones, which is why the CDVQA history is still readable.
