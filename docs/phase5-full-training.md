# Phase 5 — full training of all nine tools on cluster GPU

**Written 2026-09-09; results appended 2026-09-12 in §7.** §§1–6 are the
plan as written before any run existed and are kept unedited; §7 is what
happened, including which of the plan's premises turned out to be false. The
measured numbers live in a dated section of `docs/phase1-status.md` and in
new cards in `docs/model-cards.md` — **never as an edit to a v1 number**, for
the reason in §1.

---

## 1. This is post-freeze work, and how it stays legitimate

`docs/code-freeze.md` permits bug fixes, evidence and demo material, and
explicitly forbids "new capabilities, new tools, **retraining that changes a
published number**, refactors".

A full retrain of all nine tools is squarely the forbidden clause. It is being
done anyway, as a deliberate decision, because the compute constraint the whole
plan was built around no longer holds: `docs/03` §1 opens with "the team has
free-tier Colab and Kaggle only — T4 (16 GB) and occasionally P100", and every
model in the registry is small because of that sentence. College cluster GPU
access removes it.

**The freeze is honoured structurally rather than by exemption.** Four rules,
each enforced by something other than good intentions:

| rule | enforced by |
|---|---|
| No v1 checkpoint directory is written to | every `ckpt_dir` in `configs/campaign.yaml` is under `checkpoints/v2/`, asserted in `tests/test_campaign.py` |
| No v1 architecture changes | v2 lives in `training/v2/architectures.py`; `--arch` defaults to `v1`, asserted per-trainer |
| No published number is edited | v2 results are published as a new dated section; the v1 number is the baseline the v2 number is *compared against*, so overwriting it destroys the comparison |
| v1 stays loadable | checkpoints carry `arch` in `extra`; absent means v1, so all seven loadable v1 checkpoints load unchanged |

The one thing this cannot preserve is the claim that `phase-4-freeze` is the
last commit that touched training code. It is not, and this document is the
record of that.

---

## 2. Why new architectures rather than longer training

Retraining the existing models for longer would not have worked, and the
project's own measurements say why.

| tool | v1 measured | published range | cause |
|---|---|---|---|
| `landcover_v1` | mAP **0.2854** | ~0.65–0.85 | dim-64 4-layer CNN on 30k patches |
| `grounding_v1` | Acc@0.5 **0.0762** | ~0.70–0.80 | **global-average-pools before regressing the box** |
| `change_vqa_v1` (scratch) | mIoU 0.1691 | — | 1,600 training pairs, no pretraining |

The grounding row is the clearest case. `docs/model-cards.md` already names it:
pooling the feature map to a vector discards every pixel coordinate, so the
model "can only learn an average box". That is not a training-budget problem.
Ten thousand GPU-hours on the same architecture would produce the same average
box.

So each v2 model changes what the measurement identified and nothing else. The
data loaders, losses, metrics and `--eval-only` guards are v1 code, unmodified.
**That is what makes a v2 number and a v1 number measurements of the same
thing.**

| tool | v1 params | v2 params | what changed |
|---|---|---|---|
| `landcover_v1` | 0.42 M | 5.40 M | residual trunk; per-band features survive to depth; FiLM at every stage, not once |
| `grounding_v1` | 0.55 M | 18.98 M | **no pooling**: phrase cross-attends over the feature map, box read from where attention landed; auxiliary heatmap |
| `change_mask_v1` | 0.05 M | 0.83 M | differences at every scale, decoder with concat skips |
| `change_caption_v1` | 0.32 M | 25.39 M | transformer decoder with cross-attention over the difference map |
| `change_vqa_v1` | 1.02 M | 3.79 M | multi-scale skips into two unshared per-date decoders |
| `optsar_fusion_v1` | 0.13 M | 6.26 M | cross-attention **before** pooling, both directions |
| `caption_v1` | 1.56 M | 26.17 M | transformer decoder with cross-attention |
| `rs_vqa_v1` | — | — | **unchanged architecture**; it was always real QLoRA on Qwen2.5-VL-3B |
| `index_engine_v1` | — | — | deterministic NumPy; no training run, by design |

### What is deliberately not adopted

* **No `trust_remote_code`.** `training/train_grounding.py` records the decision
  to build a fallback grounder rather than execute Python fetched from a model
  repo. A bigger GPU is not a reason to reverse a security decision, so
  Florence-2 is still out and the v2 grounder is still built from parts.
* **No `timm` / `torchgeo` pretrained weights.** A cluster notebook may have no
  outbound network, and a run that dies at `from_pretrained` after 63 GB is
  staged is the worst available failure. Stated as a trade, not as a free
  choice: on the two small corpora it is probably the wrong one, which is why
  `change_vqa` keeps `--pretrained` and the v2 arm is an ablation.

---

## 3. The environment this is built for

JupyterHub, notebook-only, on a shared cluster. That constraint drives the
whole harness, because a notebook session gets killed — idle timeout, wall
clock, closed tab, node reclaimed.

**The notebook holds no state.** `notebooks/satquery_campaign.ipynb` calls
`campaign.run_all()` and nothing else; state lives in `runs/campaign_state.json`
and in checkpoint directories that already survive a kill.

| module | what it decides |
|---|---|
| `training/cluster/env_probe.py` | precision, attention impl, batch shape — from the card actually allocated |
| `training/cluster/stage_data.py` | is the data here and intact, and which runs does that unblock |
| `training/cluster/campaign.py` | dependency order, crash recovery, session budgets |

Three design points that are not obvious, each written because the naive
version was wrong:

**Effective batch is invariant across cards.** `micro_batch` scales with VRAM
and `grad_accum` is derived from the target, so an A100 gives 4×4 and a T4
gives 1×16 — both effective 16. Without this, two runs of the same config on
two nodes would be solving different optimisation problems and their metrics
would not be comparable, which a campaign spread over whatever node is free
cannot afford.

**Budgets are advisory.** The first version refused to start a run longer than
the remaining session. Sessions are 4–12 hours and the longest run here is
estimated at 20, so it would have reported "nothing ready to run" indefinitely
and trained nothing. Every trainer checkpoints; partial progress is kept;
`strict=True` restores the refusing behaviour for a node that must be handed
back on time.

**Stale runs are reclaimed, live ones are not.** A killed session leaves a run
marked `running` forever. A heartbeat plus an owner record distinguishes "the
session died" from "it is running on another node" — the second must never be
restarted, because two processes writing one checkpoint directory corrupts it.

---

## 4. The runs

Ten runs, eight of them producing a deployed checkpoint, two of them ablation
arms. `index_engine_v1` has no run.

| run | tool | data | est. h | note |
|---|---|---|---|---|
| `track_a` | `landcover_v1` | BigEarthNet full | 14 | the encoder two tools share |
| `track_a_nodropout` | — | BigEarthNet full | 14 | **ablation**: the band-dropout claim was single-seed |
| `optsar_fusion` | `optsar_fusion_v1` | WHU-OPT-SAR | 6 | PS-mandatory |
| `track_b_vqa` | `rs_vqa_v1` | instruction mix | 20 | **recovery** — the v1 adapter is destroyed |
| `caption` | `caption_v1` | RSICD | 4 | the model the tool actually loads — see below |
| `grounding` | `grounding_v1` | DIOR-RSVG | 8 | largest expected gain |
| `change_mask` | `change_mask_v1` | LEVIR-CD | 7 | cheapest real run; good first test |
| `change_caption` | `change_caption_v1` | LEVIR-MCI | 8 | |
| `change_vqa` | `change_vqa_v1` | CDVQA/SECOND | 5 | `--pretrained`, deliberately |
| `change_vqa_scratch_v2` | — | CDVQA/SECOND | 5 | **ablation**: does v2 capacity beat a pretrained stem on 1,600 pairs? |

**~91 GPU-hours estimated** for one complete pass. `docs/03` §1.2 budgeted
55–95 for the v1 pass and 150–220 with realistic iteration; the same multiplier
applies here.

### `caption_v1` is not a Qwen adapter, whatever the plan said

`docs/03` §3 lists `caption_v1` as a second LoRA adapter on the shared
Qwen2.5-VL base — "two LoRA adapters, one base" — and the first version of
this campaign scheduled exactly that, for 10 GPU-hours.

The implementation diverged from the plan and the plan is the thing that is
out of date. `satquery/tools/caption.py` loads a `training/train_caption.py`
checkpoint from `SATQUERY_CAPTION`, and **nothing in the registry loads a
caption LoRA at all**. Training one would have produced 10 GPU-hours of
weights no tool can read. The run in the table is the one that improves the
tool that is actually deployed.

`rs_vqa_v1` is the opposite case and was checked the same way: it genuinely
does load a Qwen adapter (`SATQUERY_VQA_ADAPTER`), which is why `track_b_vqa`
stays.

`track_b_vqa` is the only run that is *recovery* rather than improvement.
`docs/model-cards.md` records its adapter as 99.99% NUL bytes — 148,701,184 of
148,712,776 — so `rs_vqa_v1` currently cannot be loaded or published at all.

### Two results that would be honest failures

Worth stating in advance, so that neither gets quietly reinterpreted afterwards:

1. **`optsar_fusion`'s fused head not beating `max(optical, sar)`.** The
   three-output shape exists to make that falsifiable. If the gain is not
   there, the cross-attention is decoration and the report says so.
2. **`change_vqa_scratch_v2` losing to `--pretrained`.** Expected, on 1,600
   pairs. It is run because the alternative is assuming it a second time.

---

## 5. Running it

```bash
# on the machine that has prepared data (63.18 GB across nine datasets)
python training/cluster/stage_data.py build --root data
rsync -a --info=progress2 data/ cluster:/scratch/$USER/data/

# on the cluster, in the notebook or a terminal
python training/cluster/env_probe.py
python training/cluster/stage_data.py verify --root /scratch/$USER/data
python training/cluster/campaign.py status
python training/cluster/campaign.py --budget-minutes 480 all
```

Then re-run the same commands after every session death. The queue resumes.

**On transferring 63 GB.** Three of the nine directories are ~88,000 small
files (`levircd`, `levir_mci`, `bigearthnet_14k`), and rsync of many small
files is dominated by per-file round trips. A tar stream is usually much
faster:

```bash
tar cf - data/levircd data/levir_mci | ssh cluster 'tar xf - -C /scratch/$USER/'
```

Verify with digests afterwards either way. `stage_data.py verify` opens and
hashes every file rather than trusting size and mtime, because a zero-filled
file keeps its size — the exact failure that produced twelve NUL-byte sidecars
in the 2026-08-30 incident and survived a verification that hashed without
opening.

---

## 6. What is not done

Stated plainly rather than left to be discovered:

* **No run has been executed.** There is no GPU here big enough and the cluster
  is yours. Every number in this repository is still a v1 number.
* **Wall-clock estimates are estimates**, extrapolated from v1 run times and
  parameter counts. They are not measured on your hardware, and `est_hours`
  only affects scheduling preference, never correctness.
* **The inference-side tools have not been re-pointed at v2 checkpoints.** They
  read `arch` from `extra` and dispatch correctly, and the plumbing is tested,
  but no `SATQUERY_*` variable points at a v2 directory because no v2 directory
  has contents. That switch is a deployment decision to make per tool once
  numbers exist — the shape of the decision is in `docs/checkpoint-decision.md`.
* **Licensing is unchanged.** `change_vqa_v1` weights remain unpublishable:
  SECOND states no licence at all. Retraining does not change that.

---

## 7. What happened — 2026-09-12

§6 above was written before any run existed and is kept as written. Three of
its four bullets are now out of date; this section supersedes them.

**Every run executed.** Ten campaign runs, 16.63 GPU-h measured, on the AI
Lab's NVIDIA L40S (`compute01`, RHEL 9, shared with other students). Then a
40-epoch extension of both Track A arms, an early-stopping rerun of Track B,
the official RSVQA-LR evaluation, and pretrained-backbone arms for grounding
and caption — roughly 25 GPU-h in all. Results and the per-tool discussion
are in `docs/phase1-status.md` §"Phase 5 — 2026-09-12" and the cards in
`docs/model-cards.md` §"Phase 5 cards". The numbers themselves are in
`docs/assets/phase5/`.

**The estimates were wrong by 3–5×, in the safe direction.** 91 GPU-h
estimated, 16.6 measured. They were laptop extrapolations and the L40S is
simply faster; `est_hours` in `configs/campaign.yaml` should be revised
downward before the config is reused.

**The environment was not what §3 planned for.** The lab gives real SSH, not
notebook-only access, so the campaign ran as a `systemd --user` service with
lingering enabled rather than from the notebook. That turned out to be
necessary and not merely nicer: bare `nohup`/`setsid` processes were killed
with the SSH session on this box. The notebook still works and was not
needed.

### What the hardware found that the plan did not

Fourteen distinct problems surfaced by actually running, each fixed and
tested. The ones that changed a design decision:

* **The shared GPU had 10.8 GB free of 47.7.** `env_probe` sized batches from
  total VRAM and would have OOMed on the first step. It now sizes from
  `mem_get_info`, and records the shared-card condition in the run metadata.
* **The probe's batch size was never wired in.** `campaign.yaml` pinned
  `--batch-size 64`; Track A's stem reshapes to `batch × 12 bands` and OOMed
  after loading 43 GB. The driver now injects the measured size.
* **Two runs trained fully and died in the evaluator** because the v2
  captioners had no `generate()`. Forward contracts were verified; auxiliary
  methods were not. Fixed, with a test for the class of bug.
* **Checkpoint pruning deleted the best model.** `keep_last=3` removed the
  step the trainer had itself recorded as best. Pruning now protects it, and
  the best adapter is saved the moment it is found.
* **A driver killed mid-run left its trainer orphaned**, and the queue —
  tracking only the driver's pid — would have started a second trainer on the
  same checkpoint directory. The trainer's pid is now tracked and verified
  against `/proc`.
* **Three trainers need `pyarrow`, Track B needs `torchvision`**, neither in
  `requirements.txt`. `bootstrap.sh` checks both.

### Two premises of this document that turned out to be false

**§2 "No `timm` / `torchgeo` pretrained weights", on the grounds that the
cluster may have no network.** `compute01` reaches pypi, github and
huggingface, and the whole environment was pip-installed over that network.
Pretrained ImageNet backbones were added for grounding and caption and both
beat their from-scratch arms (Acc@0.5 0.1262 → 0.1604; BLEU-4 0.2255 →
0.2658), with the internal evidence — pretrained vs scratch on SECOND, 0.2933
vs 0.1730 — having predicted it. The no-`trust_remote_code` decision stands;
this one does not.

**§4's reading of Track A as the run "with the most headroom".** 28 extra
epochs moved test mAP by 0.0023 while training loss fell tenfold. Track A is
data-limited — 65,867 patches against ~549k — not schedule-limited, and the
published 0.65–0.85 was never a like-for-like target.

### What §6 said that is still true

* **Licensing is unchanged.** `change_vqa` weights remain unpublishable.

### Deployed — 2026-09-12, later the same day

The two preconditions were met and the default was switched. Doing so found
four more defects, all pre-existing, all masked until the learned tools were
actually run through the pipeline:

* **A train/serve mismatch in calibration.** The fit used raw logits; the
  runtime clamped probabilities and capped the recovered logit at 13.8. The
  v2 land-cover head emits logits to +71, and after the affine every one of
  its top 33,620 decisions served as p ≤ 0.365 - a tool that could assert
  nothing, with a raw ranking 89% precise in its top 56.
  `CalibrationEntry.apply_logit` closes it; identical for every v1 head.
* **A silent head vetoed answers.** landcover asserting nothing reported 0.0,
  the executor took the minimum, and four of nine demo beats abstained while
  caption had answered. v1 passed only by asserting at 0.98 on imagery it had
  never seen. `no_assertion` is now its own confidence method.
* **Cloud had never been looked at.** `cloud_pct` was "Phase 2 work" since
  Phase 1; the abstention beat's expectation could not fail; the 63% clouded
  scene was captioned at 0.88. ingest now estimates it and ≥50% blocks.
* **`change_caption`'s split was gone.** Reinstated; v2 scores 0.1641 on the
  changed half against v1's 0.3063, so that one tool stays on v1.

The default compose files now point at the Phase 5 checkpoints and the v2
registries, with `docker-compose.v1.yml` as the one-flag revert. Verified:
8/8 tools load through the production loaders, and the demo bundle is 9/9
with the clouded scene abstaining for the actual reason for the first time.
Golden traces changed in exactly one field (`ingest.checks`, the new check).

### What remains

1. **A pretrained arm for `change_caption`** - the only tool Phase 5 made
   worse, and the only one whose architecture changed without one.
2. **Full BigEarthNet** for Track A - now provably the only lever left there.
   ~380 GB; a decision, not a default.
3. **`optsar_fusion`**: state the negative result. Two designs on the one
   paired corpus have not shown SAR complementarity.
4. The executor's `warnings` list is collected and never written into the
   trace - every append is write-only. Noted, not fixed: it predates Phase 5
   and touches the trace contract.
