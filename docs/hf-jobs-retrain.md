# Retraining on Hugging Face Jobs (using a friend's credits)

Runs the same retrain as `docs/kaggle-retrain.md`, but as a **Hugging Face
Job**: one command starts it, it runs on HF's servers with no browser open
and no 12-hour session limit, and it uploads its result to a private Hub repo
when it finishes. Billed per second of actual use.

Split used by the team (2026-09-18): **caption on Kaggle**, **vqa and
change_mask on HF Jobs**.

## What it costs

`t4-small` (1x T4, 16 GB, $0.40/h), the default:

| Job | Expected | Timeout (maximum it can bill) | Worst case |
|---|---|---|---|
| vqa smoke pass | ~30 min | 14 h, but ends in minutes | ~$0.20 |
| change_mask smoke pass | ~20 min | 9 h, but ends in minutes | ~$0.15 |
| **vqa** (training + official scoring) | 8–11 h | 14 h | $5.60 |
| **change_mask** | 4–8 h | 9 h | $3.60 |
| **Total** | | | **~$9.55** |

That fits inside $11 even in the worst case. You pay for the seconds a job
actually runs, not its timeout. The time estimates are extrapolated, not
measured on a T4. The smoke passes show the real speed.

## Whose credits get used

**Whoever is logged in on the machine that runs `launch.py` gets billed.**
The job is not "sent" to anyone: it runs under the logged-in account. The
result repo is also created under that account, as a private repo.

So to use your friend's credits, the job has to be started with **his**
login. Two ways, safest first:

### Option A — your friend runs it himself (safest)

He shares nothing. On his own computer:

```bash
git clone https://github.com/hs-zz27/sih2.git && cd sih2
pip install "huggingface_hub>=1.0"
hf auth login                      # with his own token
python training/hf_jobs/launch.py --model vqa --smoke
```

He then runs the commands in "The run" below, and at the end sends you the
results (see "Getting the result").

### Option B — he gives you a token, you run it, and it is revoked afterwards

1. He opens <https://huggingface.co/settings/tokens> → **Create new token**.
   - If the **Fine-grained** type offers a Jobs permission, choose it: tick
     starting Jobs and write access to his own repos, and nothing else.
   - Otherwise choose **Write**.
   - Give it a name he will recognise, e.g. `sih-satquery-temp`.
2. He sends it to you privately (not in a group chat or email thread).
3. You run `hf auth login`, paste it, and run the commands below.
4. **When the jobs have finished and you have downloaded the results:**
   - you run `hf auth logout`
   - he deletes the token on that same settings page

A write token can do anything his account can do on the Hub, which is why
it is deleted the moment it is no longer needed.

## The run

Every command prints **"Account that will be billed"** first. Check it is
your friend's account before a real run.

```bash
# 0. Preview: prints the exact job, submits nothing, needs no login
python training/hf_jobs/launch.py --model vqa --dry-run

# 1. Smoke passes: the whole pipeline with tiny limits, minutes each
python training/hf_jobs/launch.py --model vqa --smoke
python training/hf_jobs/launch.py --model change_mask --smoke
```

Each prints a job URL and id. Follow a job with `hf jobs logs <id>`, or open
its URL. A smoke pass has worked when its log shows `STATUS: smoke complete`
and ends with `PUSHED TO: ...`. Only then:

```bash
# 2. The real runs - start both, they run in parallel
python training/hf_jobs/launch.py --model vqa
python training/hf_jobs/launch.py --model change_mask
```

The vqa job trains, then scores the adapter on the official RSVQA-LR test
split (10,004 questions), in the same job. That score is the only VQA
accuracy you may quote for these weights.

## Getting the result

When a job's status is **COMPLETED**, run this while logged in as the
account that ran it:

```bash
python scripts/install_retrained.py --hf-repo <friend>/satquery-retrain-vqa
python scripts/install_retrained.py --hf-repo <friend>/satquery-retrain-change_mask
python scripts/demo_assets.py      # what is still missing
```

With **Option A**, your friend can either:
- run those two commands and send you the resulting `checkpoints/` folders, or
- download each repo from its page on huggingface.co as a zip and send it to
  you, which you then install with `python scripts/install_retrained.py <zip>`.

The results are 100–260 MB in total.

## If something goes wrong

- **Smoke pass fails:** the log names the failing step, and the manifest
  (pushed to the `-smoke` repo even on failure) says why. Fix before
  spending money on the real run.
- **`non-finite loss` in the vqa smoke pass:** re-run it with
  `--fp32-compute`. That is slower, but it does not overflow on a T4.
- **The job is killed for memory (OOM) while loading the model:** `t4-small`
  has 15 GB of system RAM. Re-run with `--flavor t4-medium` (30 GB RAM,
  $0.60/h).
- **Scoring fails after training succeeds:** the trained adapter is still
  pushed. The job uploads whatever exists before it exits.
- **Training fails:** the manifest and logs are still pushed, and the job
  shows ERROR.

## Files

- `training/hf_jobs/launch.py` builds and submits the job. It reuses
  `training/kaggle/retrain.py` unchanged, pointed at generic paths instead of
  Kaggle's.
- `scripts/install_retrained.py --hf-repo` downloads a job's pushed result
  and installs it.
- `tests/test_hf_jobs.py` covers the generated script (valid bash; the push
  always runs; the exit code is kept; scoring happens after training), the
  token going as an encrypted secret and never printed, and the Hub install
  path.
