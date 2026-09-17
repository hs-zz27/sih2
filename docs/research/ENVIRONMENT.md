# Environment facts

**Filled 2026-09-03 by measurement, not by assumption.** Every value below was read
from this machine during the session that wrote this file. Values that could not be
measured are marked `UNVERIFIED (source: <file>)` per `CLAUDE.md` rule 1, not estimated.

Two of the four **Budgets** are policy decisions rather than facts; they carry a
working default and are flagged `CONFIRM`.

---

## Hardware

| | |
|---|---|
| GPU | **NVIDIA GeForce RTX 4050 Laptop GPU**, count **1** |
| VRAM per GPU | **6,141 MiB (6.0 GB)** — driver 616.56 |
| System RAM | **15.7 GB** |
| CPU | **13th Gen Intel Core i5-13450HX** — 10 physical cores / 16 logical |
| Disk total / free | **C: 474.7 GB total / 49.8 GB free** (single volume; no second drive) |
| Scratch path | `C:\Users\<dev>\AppData\Local\Temp` — same volume, so **49.8 GB is shared** between scratch, datasets, checkpoints and any download |

**Read the free-space number before planning anything.** 49.8 GB is the entire
budget. `data/` alone is already 79.5 GB and `checkpoints/` 9.19 GB on this volume.
A single 4B competitor model in fp16 is ~8 GB; VRSBench needs DOTA + DIOR imagery,
which is tens of GB. **Disk, not GPU, is the binding constraint on this machine.**

Measured directory sizes (2026-09-03):

| path | size |
|---|---|
| `data/` | **79.50 GB** (of which `data/ben_full` ≈ 46 GB, `data/whu_opt_sar` ≈ 6.85 GB) |
| `checkpoints/` | **9.19 GB** |
| `models/` | **7.36 GB** (`qwen25_vl_3b` 7.51 GB on disk, `nli_deberta_mnli` the rest) |
| `artifacts/` | **4.68 GB** |
| `frontend/` | 0.24 GB |
| `checkpoints_backup/` | **~0.00 GB — the name is misleading.** It holds only small `ckpt_step_*.pt` scratch files from the resume test. **It contains no model weights and is not a backup of anything** |

---

## Software

| | |
|---|---|
| OS | **Windows 11 Home Single Language**, version 10.0.26200, build 26200 |
| Python env + activation | **No virtualenv exists.** System interpreter: `C:\Users\<dev>\AppData\Local\Programs\Python\Python314\python.exe`, **Python 3.14.2**. Activation command: **none — `python` on PATH is already the environment** |
| CUDA / torch version | **torch 2.13.0+cu126**, `torch.cuda.is_available() == True`, bf16 supported |
| Package manager | **pip** (`.../Python314/Scripts/pip`). `uv` and `conda` are **not installed** |

Key package versions, read from the live interpreter:

| package | version |
|---|---|
| torch | 2.13.0+cu126 |
| transformers | 5.15.1 |
| peft | 0.20.0 |
| bitsandbytes | 0.50.2 |
| accelerate | 1.14.0 |
| datasets | 5.0.1 |
| rasterio | 1.5.1 |
| numpy | 2.4.1 |
| pytest | 9.1.1 |

**Two warnings the agent will otherwise hit and misdiagnose:**

- `triton not found; flop counting will not work for triton kernels` — printed by torch on **every** import. Harmless. Not a failure.
- `inner dimension (3420) is not aligned for fast kernel with blocksize=64, falling back to slower implementation` — printed by bitsandbytes on **every** Qwen2.5-VL 4-bit forward pass (3420 is the vision MLP width). Harmless. Not a failure.

Shell available to the agent is **Git Bash (POSIX sh)** and **PowerShell 7+**. There is
no `nvidia-smi` alias problem; `nvidia-smi`, `docker`, `git` and `curl` are all on PATH.

---

## Network

| | |
|---|---|
| Internet from agent shell | **YES** — verified by `curl` |
| HuggingFace reachable | **YES** — `https://huggingface.co/api/models/...` returned HTTP 200 in 3.4 s |
| Model download allowed | **YES technically, ASK FIRST by policy.** `CLAUDE.md` puts weight/dataset downloads in the ASK FIRST list, and disk (49.8 GB free) is the real limiter |
| Web search available | **YES** — a competitor matrix was built this way on 2026-09-03 (`docs/external_benchmark_audit.md` §14) |
| PyPI reachable | **YES** — HTTP 200 |
| GitHub reachable | **YES** — HTTP 200 |

**But every evaluation script in this repo is run offline on purpose.** The working
convention is `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, and `models/` is loaded with
`local_files_only=True`. Do not remove those flags to "fix" a load error — a load error
under them means a local file is missing, which is the thing you want to know.

---

## Paths

| | |
|---|---|
| Repo root (**primary checkout**) | `C:\Users\<dev>\Desktop\sih2` |
| Repo root (agent worktree) | `C:\Users\<dev>\Desktop\sih2\.claude\worktrees\<name>` |
| Checkpoints | `C:\Users\<dev>\Desktop\sih2\checkpoints` |
| Datasets | `C:\Users\<dev>\Desktop\sih2\data` |
| Base models | `C:\Users\<dev>\Desktop\sih2\models` |
| Eval outputs | `C:\Users\<dev>\Desktop\sih2\artifacts` (run dirs) and `docs/assets/**` (published reports) |
| HF cache | **Does not exist at the default `~/.cache/huggingface`.** Every dataset was fetched with `snapshot_download(local_dir=...)`, so its cache lives inside the dataset folder as `data/<name>/.cache` |
| Docker in use | **Installed (v29.6.1) and defined** (`docker-compose.yml`, `docker-compose.gpu.yml`, `docker/api.Dockerfile`, `frontend/Dockerfile`), **but not required for evaluation.** Every command in this file runs on the host |

### The single most important path fact

**`checkpoints/`, `data/` and `models/` are gitignored and exist ONLY in the primary
checkout.** A git worktree under `.claude/worktrees/` has the code but **none** of them.

Any session running in a worktree must pass absolute paths into the primary checkout,
for example:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python evaluation/track_b_eval.py \
  --base   "C:/Users/<dev>/Desktop/sih2/models/qwen25_vl_3b" \
  --data   "C:/Users/<dev>/Desktop/sih2/data/instruct_mix" \
  --adapters v2="C:/Users/<dev>/Desktop/sih2/checkpoints/track_b_v2/adapter_final"
```

Do **not** `cd` into the primary checkout to work around this — the worktree isolation
is deliberate. Pass absolute paths instead.

---

## Commands

Verified present in the repo. `<REPO>` = the primary checkout path above.

| purpose | command |
|---|---|
| Run tests | `python -m pytest tests/ -q` (Makefile `test`). Offline variant: `make offline-test`. **Read the warning in "Traps" §T1 before running this in the primary checkout.** |
| No-torch CI simulation | `python scripts/ci_no_torch_sim.py` — the only way to reproduce the freeze's CI claim |
| **VQA eval** (Track B adapters) | `python evaluation/track_b_eval.py --base models/qwen25_vl_3b --data data/instruct_mix --split val --limit 534 --adapters v2=checkpoints/track_b_v2/adapter_final v3=checkpoints/track_b_v3/adapter_final --out <path>.json` |
| **Counting eval** | **No dedicated script.** Counting is a question *type* inside the VQA eval; it must be split out from `data/instruct_mix/val.jsonl` by question wording. See §T3 |
| **Captioning eval** | **No eval-only entry point.** `python training/train_caption.py --data data/rsicd --ckpt-dir checkpoints/caption --resume --epochs 8` reaches the eval block **but mutates the checkpoint directory** — see §T2 |
| **Grounding eval** | **No eval-only entry point.** Same pattern: `python training/train_grounding.py --data data/dior_rsvg --ckpt-dir checkpoints/grounding --resume --epochs 5`. Same mutation warning |
| **Change-mask eval** | Same pattern: `python training/train_change_mask.py --index data/levircd/index.json --ckpt-dir checkpoints/change_mask --resume --epochs 4` |
| **Land-cover eval** | Same pattern: `python training/track_a_full.py --data data/ben_full --ckpt-dir checkpoints/track_a_full_base --resume --epochs 3`. Needs the 46 GB corpus |
| **Opt–SAR fusion eval** | Same pattern: `python training/train_optsar_fusion.py --index data/whu_opt_sar/index.json --ckpt-dir checkpoints/optsar_fusion --resume --epochs 5` |
| **CDVQA eval — tool only** | `SATQUERY_CHANGE_VQA=checkpoints/change_vqa/best.pt python -m evaluation.cdvqa_predict --split Test --out artifacts/cdvqa/head_test.json` |
| **CDVQA eval — end to end** | `python -m satquery eval --benchmark CDVQA --manifest data/cdvqa/cdvqa_test.json --root data/cdvqa --out artifacts/cdvqa/test.json` |
| **CDVQA oracle ceiling** | `python evaluation/cdvqa_oracle.py --split Test --out artifacts/cdvqa/oracle_test.json` |
| **CDVQA error stratification** | `python evaluation/cdvqa_diagnosis.py --predictions <preds>.json --manifest <manifest>.json --out <out>.json` |
| **Refusal eval** | `python evaluation/refusal.py --adapter checkpoints/track_b_v2/adapter_final` |
| **Calibration / ECE** | `python evaluation/calibrate.py --heads landcover intent change_mask` (Makefile `calibrate`) |
| Selective risk / AURC | `python evaluation/selective.py` |
| Adversarial + illegal-plan rate | `python evaluation/adversarial.py` |
| Entailment benchmark | `python evaluation/entailment_bench.py` (`--compare` needs `SATQUERY_NLI`) |
| Ablations | `python evaluation/run_ablations.py` |
| Soak / memory | `python evaluation/soak.py --iterations 120 --warmup 20` |
| Capability-matrix validation | `python -m satquery matrix --validate` |
| **Regenerate all published reports** | `make report` — **do not run casually**, it overwrites `docs/assets/**` |

### Load production model

The learned tools are **opt-in by environment variable** and fall back to stubs when
unset. This is why CI stays green with no GPU. To load the real VQA model:

```bash
export SATQUERY_VQA_BASE="<REPO>/models/qwen25_vl_3b"
export SATQUERY_VQA_ADAPTER="<REPO>/checkpoints/track_b_v2/adapter_final"
```

Other tools: `SATQUERY_CAPTION`, `SATQUERY_GROUNDING`, `SATQUERY_CHANGE_CAPTION`,
`SATQUERY_LANDCOVER`, `SATQUERY_CHANGE_MASK`, `SATQUERY_FUSION`, `SATQUERY_CHANGE_VQA`,
`SATQUERY_NLI`.

**If a tool is unset the system answers from a stub and the run still succeeds.** A
plausible-looking evaluation with every metric present can therefore be a stub run.
Check `Trace.weights_hashes` — stubs get no hash — before trusting any number.

---

## Budgets

| | |
|---|---|
| Max wall-clock per training run | **10 hours** (one overnight). `CONFIRM.` Calibrated on measurement: the 300-step Track B v2 run took **6 h 26 m** on this GPU; the 2,000-step v3 run took longer still. Anything projected beyond 10 h must be escalated per `CLAUDE.md` |
| Max single-session tool calls | **150.** `CONFIRM.` Paired with the `CLAUDE.md` rule that the agent must write to disk at least every 20 calls |
| Max disk the agent may consume | **10 GB.** Hard fact behind it: only **49.8 GB** is free and it is shared with scratch. Any plan needing more must be approved *and* paired with a cleanup proposal |
| Human review checkpoints | **After every session**, per the Part C design |

---

## Known-good state

| | |
|---|---|
| Production checkpoint | `checkpoints/track_b_v2/adapter_final` — sha256 `10f4830141237846a439f9166acc21eef0be050c5580381e2e66256cf7041174`, 696 tensors, **37,152,768** trainable params |
| Best-measuring checkpoint (**not** production) | `checkpoints/track_b_v3/adapter_final` — 828 tensors, **82,726,912** params. Higher accuracy, **lower refusal recall**; trained under a label-masking defect. Do not promote without §T5 |
| Git sha (primary checkout) | **`pre-import:P0-primary`** (`pre-import:P0-primary`), branch `phase-0-closeout`, dated 2026-09-03 01:57 +0530, *"Add 8-bit Adam so vision adaptation fits in 6 GB"* |
| Uncommitted in the primary checkout | ` M training/track_b_vlm_qlora.py` (the label-masking fix + validation loop) and `?? tests/test_vlm_label_masking.py`. **Load-bearing and not yet committed** |
| Last date all tests passed | **2026-09-01 — 1,070 passed, 0 failed, 0 skipped, 456.8 s.** `UNVERIFIED (source: docs/HANDOFF.md)`. Not re-run since, and the tree has changed twice since; treat as a claim, not a state |

**Verified-by-measurement baseline, 2026-09-03**, on `data/instruct_mix/val.jsonl`
(n=534), all four arms, identical decode path. Full detail in
`docs/external_benchmark_audit.md` §7 and `docs/external_benchmark_results.json`:

| | base (no adapter) | `track_b_v2` | `track_b_v3_probe` | `track_b_v3` |
|---|---|---|---|---|
| overall exact match | 0.079304 | 0.379110 | 0.394584 | 0.460348 |
| overall token F1 | 0.202717 | 0.792742 | 0.807712 | 0.855027 |
| `rsvqa_lr` exact match | 0.198068 | 0.647343 | 0.671498 | 0.787440 |
| `whu_opt_sar` exact match | 0.000000 | 0.200000 | 0.209677 | 0.241935 |
| refusal recall | 0.000000 | 0.411765 | 0.294118 | 0.352941 |
| s/example | 1.52 | 1.31 | 1.25 | 1.26 |

`track_b_v2`'s row **reproduces the published 2026-09-01 run to twelve decimal places
on all eight metrics**, which is the evidence that this inference path is deterministic
end to end.

---

## Traps — facts the template does not ask for, and the agent will get wrong without them

### T1 — A script in this repo has already destroyed every checkpoint once

On 2026-08-30 `training/run_checkpoint_test.py` called `shutil.rmtree("checkpoints")`
unconditionally and had **no argument parser**, so passing `--help` to inspect it ran
it. 4.542 GB was lost and recovered only by luck, from a Windows volume shadow copy.
`make test-resume` reached the same code. The hole is closed (scratch default under
`artifacts/`, two refusals, a parser, and `tests/test_script_entrypoints.py` fails if
any `__main__` script lacks a parser) — but:

- **`checkpoints/` still has no backup.** `checkpoints_backup/` is effectively empty.
- **Never run an unfamiliar script in the primary checkout to find out what it does.**
  Read it first. `--help` is not safe on code you have not read.

### T2 — There is no read-only evaluator for five of the seven specialist heads

`train_caption.py`, `train_grounding.py`, `train_change_mask.py`, `track_a_full.py` and
`train_optsar_fusion.py` evaluate *after* their training loop. Passing
`--resume --epochs <the count already trained>` makes `for epoch in range(state.epoch,
args.epochs)` empty, so it goes straight to eval — **but it still calls
`save_checkpoint()` and overwrites `metrics.json`, `run_metadata.json` and
`vocab.json` in the checkpoint directory.**

Consequence: **re-measuring captioning or grounding overwrites the published number in
place.** Copy the checkpoint directory to `experiments/` and point `--ckpt-dir` there,
or accept that the artifact of record is being rewritten. This is the single most
likely way a session silently destroys evidence.

### T3 — Counting and question-type accuracy are not measured by any script

`evaluation/metrics/vqa.py` reports one exact-match number over all question types at
once. RSVQA-LR's four types (presence / comparison / count / rural-urban) must be
recovered from the templated question wording. A working classifier and the finding it
produced — **a train-fitted per-type constant scores 0.6473, identical to `track_b_v2`'s
headline** — are in `docs/external_benchmark_audit.md` §7.4.

**Always report a per-type constant baseline alongside any RSVQA number.** Without it
the aggregate is uninterpretable.

### T4 — The RSVQA data on disk is not the official test split

`data/rsvqa_lr_2k` is `dmarsili/RSVQA-LR-2k`, a **2,000-question subset of the RSVQA-LR
_validation_ split**, further cut 90/10 by `training/prepare/instruction_mix.py`
(seed 42) into 1,793 train / **207** val. `docs/00` §3.5 describes this as the "official
split"; it is not. The official split is 572/100/100 images and ~77,232 questions.

At n=207 the 95% interval is about **±0.065**. Do not read differences under ~6 points
as signal on this slice.

### T5 — `track_b_v3` is better on accuracy and worse on reliability

v3 wins every accuracy row and loses refusal recall (0.3529 vs 0.4118) and
image-conditional refusal (1/12 vs 2/12). Under `CLAUDE.md` rule 7 that combination is
**a regression, not a promotion candidate**. Its advantage also confounds three
variables at once (LoRA rank 16→32, vision-tower targeting 0→22.9M params, steps
300→2,000); the isolating ablation — rank-32, **language-only**, 2,000 steps — has not
been run.

### T6 — Six areas are frozen and three of the obvious next moves are among them

`docs/code-freeze.md` §"Explicitly out of scope after freeze" freezes: **the CDVQA
segmenter, grounding, image-conditional refusal, VRSBench, `max_coreg_shift_px`
enforcement, and the Tier-1 router.** The highest-value work identified by the
2026-09-03 benchmark audit lands on three of those six. Emit the `CLAUDE.md` unfreeze
request; do not route around it.

### T7 — Two benchmark protocols in this repo are not comparable to published numbers

- **Captioning BLEU** is *sentence-mean with add-one smoothing*
  (`evaluation/metrics/all_tasks.py:bleu`). The function's own docstring says: compare
  models to each other, **not** to a paper's corpus BLEU.
- **LEVIR-CC change captioning** is scored against **one** reference; the published
  protocol uses **five**.

Two protocols *are* comparable and should be used as the honest external anchors:
**CDVQA test1** (39,686 questions, identical split, overall accuracy) and **LEVIR-CD**
(official split, standard 256px tiling, change-class F1).

### T8 — Timing on this GPU is bimodal, and it is not a hang

Short-answer items (RSVQA, 256×256) run at ~1.3 s/example. Long-generation items
(`whu_opt_sar`, 512×512, generating to the 48-token cap) can run **5–10× slower** and a
534-item mixed sweep has taken anywhere from 12 minutes to over 90. A quiet process at
~30% GPU utilisation with rising CPU time is working, not stuck. Budget accordingly and
prefer `run_in_background`.

---

## BLOCKED

| item | what is missing |
|---|---|
| Verified "all tests pass" state | The suite has not been run since 2026-09-01 and the tree has changed. Needs one `python -m pytest tests/ -q` — but see §T1 before running it in the primary checkout |
| VRSBench evaluation | DOTA + DIOR imagery is not on disk (`docs/00` L11). Network is available; **disk is the constraint** (49.8 GB free) |
| Official RSVQA-LR test split | Only the 2k validation subset is on disk (§T4) |
| Any SAR VQA / captioning benchmark | SARLANG-Bench and equivalents are not on disk. SatQuery has **no SAR benchmark result of any kind** |
| Local evaluation of any competitor model | Nothing downloaded. Earth-OneVision and RingMo-Agent have released no weights at all; EarthMind's are unconfirmed; EarthDial (public, CC BY 4.0) and TinyRS-R1 (public, 2B) are the two feasible targets |
| `band_stats.json` for `landcover_v1` on a fresh clone | Gitignored; regenerating it needs the 46 GB `data/ben_full` (`docs/00` L2) |
| Publishing any trained weights | Licence undecided; `change_vqa_v1` is blocked outright because SECOND states no licence (`docs/verification.md`) |
