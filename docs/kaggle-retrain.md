# Retraining lost checkpoints on Kaggle's free GPU

**Situation (2026-09-17):** all Phase 5 checkpoints are lost — never copied off
the college lab server, and the older v1 protective copy was on a machine the
team no longer has access to. The team has no GPU, only a 16 GB CPU laptop.
This retrains the three models the demo video needs, using Kaggle's free GPU
quota (~30 h/week per account, T4 x2), unattended.

**What this is not.** It does not reproduce the Phase 5 numbers
(`docs/as-built-status.md`). Different data mix, fewer epochs, a T4 instead of
an L40S. Treat every retrained model as new and re-measure it before quoting
any accuracy for it. Each run's manifest records exactly how it differs from
Phase 5 (see `deviations` below).

## What gets retrained, and why these three

| Model | Deploys to | Dataset (verified against the HuggingFace API) | Rows |
|---|---|---|---|
| `rs_vqa_v1` (VQA) | `checkpoints/v2/track_b_vqa/adapter_final` | `dmarsili/RSVQA-LR-2k` | 2,000 |
| `change_mask_v1` | `checkpoints/v2/change_mask` | `ericyu/LEVIRCD_Cropped256` | 7,120 / 1,024 / 2,048 (matches the official split recorded in `docs/storage-audit.md`) |
| `caption_v1` | `checkpoints/v2/caption_pre` | `arampacha/rsicd` | 8,734 / 1,094 / 1,093 |

These are the demo's headline beats: single-image VQA, change detection, and
the cheap-to-retrain caption fallback. Grounding, fusion and the semantic
change-VQA head are not included — they need WHU-OPT-SAR (~10 GB, not on
HuggingFace) or the unlicensed SECOND dataset; add them later the same way if
there is spare quota.

**Verified before writing the plan**, not assumed: `dmarsili/RSVQA-LR-2k`'s
schema (`image`, `question`, `answer`) was downloaded and run through
`training/prepare/rsvqa.py --src ... --out ...` end to end on 2026-09-17 — all
2,000 rows converted, images resolve, `training/prepare/instruction_mix.py`
built a working instruction mix from them. `ericyu/LEVIRCD_Cropped256`'s
column names (`imageA`, `imageB`, `label`) and split sizes were confirmed
against the HuggingFace datasets-server API. `arampacha/rsicd`'s columns
(`filename`, `captions`, `image`) match what `training/train_caption.py`
expects.

## Running it — no local involvement needed after this

For each model:

1. Open **kaggle.com/code → New Notebook → File → Import Notebook**, upload
   `notebooks/kaggle/retrain_vqa.ipynb` (or `retrain_change_mask.ipynb` /
   `retrain_caption.ipynb`) from this repository.
2. **Settings → Accelerator → GPU T4 x2**, **Internet → On** (needs a
   phone-verified Kaggle account — verify this first if you have not).
3. **Save Version → Save & Run All (Commit)**. This is the step that lets it
   run unattended and survive the browser closing.
4. Wait. VQA is the long one (8–10 h estimated; the driver stops itself at
   10.5 h regardless, so it never loses work to Kaggle's 12 h hard limit).
   Change mask and caption are much shorter.
5. When the version finishes, open it and check the last cell's output:
   `STATUS: complete` (or `stopped by budget - packaged last good weights`,
   which is also usable) means it worked. `STATUS: failed at: <step>` means
   read that step's log.
6. **Output tab → Download** the `retrained/` folder (or its zip).

Run the three notebooks in **separate** Kaggle sessions (one GPU quota each);
they do not depend on each other and can run in parallel across accounts if
more than one is available.

## Installing what comes back

```bash
python scripts/install_retrained.py ~/Downloads/output.zip
python scripts/demo_assets.py        # confirm what's now present
```

`install_retrained.py` copies the packaged weights into `checkpoints/`,
prints each run's status, its metrics if training finished, and its
deviations from Phase 5. It never overwrites an existing checkpoint folder
unless `--force` is given.

Then follow `docs/demo-video.md`: `python scripts/run_demo_cpu.py`, and do not
record until pre-flight says **GO** with **MODELS n/8 LIVE**.

## If a run fails or is cut short

- **`retrain_manifest_<model>.json`** (in the Kaggle output) always exists
  once the driver starts, and always says which step it reached and why.
- **A step failing** (exit code != 0) stops the driver; the log for that step
  is in `logs/<model>.log` in the same output.
- **The budget stop is not a failure.** Training is killed at the hour limit,
  the last checkpoint is evaluated and packaged, and the manifest says
  `stopped by budget - packaged last good weights`. Quote its metrics as
  "partial run," and re-run with a longer `--budget-hours` (edit the
  notebook's driver cell) if there is more quota later.
- **Resuming across sessions:** re-run the same notebook. Every trainer here
  resumes from its own checkpoint directory when one exists (`--resume` is
  passed by the driver); a fresh Kaggle session starts a clean `/kaggle/tmp`,
  so resuming across *sessions* needs the previous session's `retrained/`
  output placed back under `--work` first — out of scope for a first pass,
  and not needed if a run finishes inside its budget.

## Known deviations from Phase 5, and why

| Model | Deviation | Why |
|---|---|---|
| VQA | No WHU-OPT-SAR — no SAR examples, no `not_in_image` refusals | The dataset (~10 GB) is not on HuggingFace; downloading it needs the GitHub mirror and manual staging, out of scope for an unattended run |
| VQA | 1,500 steps with `--patience 3`, not the 6,000-step run | Matches the Phase 5 **early-stopping** arm (`track_b_vqa_es`), whose best checkpoint (step 1,500) was the one actually recommended for deployment |
| VQA | fp16, effective batch 16 (2×8) | A T4 has no bf16 and 16 GB VRAM against the L40S's 47.7 GB |
| Change mask | 30 epochs, not 60 | Fits one T4 session; re-run with more epochs if quota allows |
| Caption | pretrained ResNet-50 downloaded at train start | Matches the deployed `caption_pre` recipe; needs internet on, which the notebook already requires |
| All three | No official benchmark re-measurement | RSVQA-LR official split, LEVIR-CD F1, RSICD BLEU-4 all need the dedicated evaluators (`evaluation/rsvqa_official_eval.py`, etc.), which are a separate, deliberate next step — do not quote Phase 5 numbers for these weights until that is run |

## Files

- `training/kaggle/retrain.py` — the driver (also runnable directly, e.g. to
  test locally with `--dry-run`, which prints the plan without downloading or
  training anything).
- `notebooks/kaggle/retrain_{vqa,change_mask,caption}.ipynb` — one notebook
  per model, each just clones the repo and calls the driver.
- `scripts/install_retrained.py` — copies a finished run's output into
  `checkpoints/`.
- `tests/test_kaggle_retrain.py` — the driver's plans, budget-stop behaviour,
  and packaging logic, tested without a GPU or any download.
