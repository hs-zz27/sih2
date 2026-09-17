# Storage audit — Phase 1

**Measured 2026-09-03.** §§1–8 were written read-only, before anything was touched. **§9 records the deletion that was subsequently approved and executed** — Tier 1 + Tier 2 only. Tier 3 was deferred and remains on disk.

§6 was the Gate A proposal; §9 is what actually happened.

Two standing decisions are already recorded and are honoured throughout:

* **`data/ben_full` train shards p1–p3 — DO NOT TOUCH pending verification. Not a delete candidate.** They do not appear in §6.
* The **checkpoint protective copy** made earlier today (8.561 GB, verified bit-exact) is **KEEP** until a genuine off-site backup exists. See `docs/research/checkpoint-protective-copy-2026-09-03.md`.

---

## 1. Headline

| | |
|---|---|
| Volume | **C: only** — one physical disk (SKHynix 476.9 GB SSD), no removable, no network |
| Free now | **44.11 GB** of 474.71 GB |
| SatQuery repo tree | **100.34 GB** (`data` 79.50 · `checkpoints` 8.56 · `models` 7.36 · `artifacts` 4.68 · rest 0.24) |
| Protective copy (new, outside repo) | 8.56 GB |
| **Largest single consumer on this machine** | **`docker_data.vhdx` — 65.85 GB**, and the Docker daemon is **not running** |

**The finding that reframes Phase 1:** the biggest reclaimable object on this disk is not in the repository at all. Docker's virtual disk is **65.85 GB — larger than `data/`, `checkpoints/` and `models/` combined** — while `docker system df` cannot even connect because the daemon is stopped. Nothing in the SatQuery evaluation path needs Docker (`docs/research/ENVIRONMENT.md`: "not required for evaluation"; every command runs on the host).

That is **not** a SatQuery decision and is not proposed in §6. It is raised in §7 because a Phase-4 VRSBench download needs tens of GB and this is where that space is.

---

## 2. Full survey

### 2.1 Inside the repository

| Path | Size | Class | Verified how |
|---|---|---|---|
| `data/ben_full` | 44.13 GB | **DO NOT TOUCH** | Read by `training/track_a_full.py`, `evaluation/calibrate.py` (`--ben-data` default), `evaluation/splits/multires.py`, and the L2 `band_stats.json` regeneration path |
| `data/whu_opt_sar` | 9.22 GB | mixed — see §6 | `index.json` resolves **only** to `prepared/` |
| `data/bigearthnet_14k` | 8.08 GB | mixed — see §6 | `index.json` resolves **into `extracted/BEN_14k/`** |
| `checkpoints/` | 8.56 GB | mixed — see §3 | protective copy verified today |
| `models/` | 7.36 GB | **KEEP** | `qwen25_vl_3b` (7.51 GB on disk) is the base VLM; `nli_deberta_mnli` is the entailment gate. Digests in `configs/model_lock.json` |
| `data/bhoonidhi` | 6.58 GB | **DO NOT TOUCH** | Real Cartosat-2E MX + EOS-04 products. Never trained on. **Not re-downloadable without a Bhoonidhi account.** 43 tests depend on them |
| `data/levir_mci` | 5.28 GB | **KEEP** | LEVIR-CC / LEVIR-MCI, `train_change_caption.py` |
| `artifacts/` | 4.68 GB | mixed — see §4 | `satquery prune --dry-run` |
| `data/second` | 2.25 GB | **KEEP** | `change_vqa` training data; CDVQA imagery resolves through it |
| `data/dior_rsvg` | 1.87 GB | **KEEP** | `train_grounding.py`; also the source for the L29 vocabulary regeneration |
| `data/levircd` | 0.65 GB | **KEEP** | Official LEVIR-CD split, 7,120/1,024/2,048 tiles |
| `data/rsicd` | 0.49 GB | **KEEP** | RSICD official test split; L29 vocabulary regeneration source |
| `data/rsvqa_lr_2k` | 0.32 GB | **KEEP** | The `instruct_mix` source |
| `data/vrsbench` | 0.31 GB | mixed — see §6 | annotations **keep**; aborted download fragments are waste |
| `frontend/node_modules` | 0.24 GB | **KEEP** | Needed to build the frontend; `frontend/.next` is absent (never built here) |
| `data/whu_opt_sar_lbl` | 0.19 GB | **KEEP** | fusion labels |
| `data/cdvqa` | 0.12 GB | **KEEP** | Q/A/image indices for the one Category-A benchmark |
| `data/demo_bundle` | 0.01 GB | **KEEP** | demo evidence |
| `.git` | 0.016 GB | **KEEP** | trivial |
| `data/instruct_mix` | 0.001 GB | **KEEP** | the entire Track B training set |
| `__pycache__` (all), `.pytest_cache`, `satquery.egg-info` | 0.006 GB | REGENERABLE | Too small to be worth the risk budget |
| `checkpoints_backup/` | 0.000 GB | REGENERABLE | **The name is misleading** — it holds only resume-test scratch and is not a backup of anything |

### 2.2 Outside the repository, same volume

**None of these are SatQuery's to delete.** They are listed because they share the 44.11 GB and several belong to other projects.

| Path | Size | Note |
|---|---|---|
| `AppData\Local\Docker\wsl\disk\docker_data.vhdx` | **65.85 GB** | Daemon stopped. See §7 |
| `AppData\Local\Claude-3p\vm_bundles\` | 8.38 GB | Agent VM images (`rootfs.vhdx` 7.84 GB) |
| `Python314\Lib\site-packages` | 5.60 GB | The live interpreter's packages — **KEEP** |
| `AppData\Local\npm-cache` | 3.29 GB | npm's global cache; regenerable, not SatQuery-specific |
| `~\.cache\huggingface` | 2.99 GB | **See §5 — mixed, and partly another project's** |
| `~\.claude` | 1.42 GB | Agent session data |
| `~\.cache\codex-runtimes` | 1.30 GB | A different tool's runtime cache |
| `AppData\Local\Temp` | 0.64 GB | mostly the agent's own scratch (0.044 GB) |
| `~\.cache\torch` | 0.08 GB | torch hub cache |
| `AppData\Local\pip\Cache` | 0.01 GB | trivial |

---

## 3. Checkpoints — and a finding that changes the classification

The Phase-0 audit assumed `checkpoints/killtest` was usable as the **v0 baseline arm**, because `evaluation/track_b_eval.py` names it in two places:

```
--adapters v0=checkpoints/killtest/adapter_final
```

**It is not usable, and this was verified rather than assumed.** Reading every `.safetensors` header under the three affected directories:

| Directory | `.pt` files | `.safetensors` | Corrupt |
|---|---|---|---|
| `killtest` | 3 (**0.894 GB, all load**) | 4 (0.595 GB) | **4 of 4** |
| `smoke` | 2 (**0.596 GB, all load**) | 3 (0.446 GB) | **3 of 3** |
| `track_b_v1` | 3 (**0.894 GB, all load**) | 4 (0.595 GB) | **4 of 4** |

Eleven corrupt adapters in total, matching `docs/00` **L32** exactly.

**The `.pt` files cannot rescue them.** For a PEFT run this repository's `save_checkpoint()` writes:

```
top-level keys: ['step', 'is_peft', 'adapter_dir', 'optimizer_state_dict',
                 'scheduler_state_dict', 'training_state', 'rng_state', 'extra']
```

There is **no `model_state_dict`** — the weights live in the sibling `adapter_dir`, which is the destroyed part. So **2.384 GB of optimiser, scheduler and RNG state now points at weights that no longer exist.** Optimiser state without its weights cannot reconstruct, resume or evaluate anything.

Consequences:

* **The v0 baseline arm in `track_b_eval.py` is dead.** Any future run of that command with `v0=checkpoints/killtest/adapter_final` will fail to load, not silently produce a number. Worth knowing before Phase 2 or 7 depends on it.
* The **evidence** L32 rests on is the eleven **corrupt safetensors** (1.636 GB) plus the L32 write-up — not the orphaned optimiser state.

Everything else in `checkpoints/` is either production, an ablation arm cited in `docs/model-cards.md`, or the deliberately-retained L32 evidence. All of it is **KEEP**.

---

## 4. Artifacts

`satquery prune` is whitelist-shaped by design: it deletes **only** directories matching the generated run-id pattern `run_<hex>` and leaves anything a human named. Measured independently of the tool, then cross-checked against it:

| | Size |
|---|---|
| Generated `run_*` directories (343) | **3.907 GB** |
| Named / evidence directories (16) | **1.116 GB** — never pruned |
| Loose files at root | 0.000 GB |

`satquery prune --dry-run` reports **323 deletable, 20 kept, 16 protected, 3.35 GB**.

The 1.116 GB of protected directories is dominated by two demo runs — `demo_large_scene` and `demo_single_optical` at **550.8 MB each** — with `calibration/` (8.1 MB, the cached logits behind the published thresholds) and `cdvqa/` (2.2 MB, the artifacts behind the 0.5380 correction) making up almost all the rest. The two demo directories are the only place in this audit where a large protected object could plausibly be re-classified, and doing so needs a human who knows whether the demo will be re-run.

---

## 5. The HuggingFace cache — corrected, and it contains something useful

`docs/research/ENVIRONMENT.md` records: *"HF cache — Does not exist at the default `~/.cache/huggingface`."* **That is no longer true.** It exists and holds **2.99 GB**.

What is in it matters more than the size:

| Cached repo | Size | Whose |
|---|---|---|
| `models--openai--clip-vit-large-patch14` | 1.59 GB | plausibly SatQuery-adjacent |
| `models--dima806--deepfake_vs_real_image_detection` | 0.32 GB | **another project** |
| `models--hamzenium--ViT-Deepfake-Classifier` | 0.32 GB | **another project** |
| `models--facebook--dinov2-base` | 0.32 GB | **relevant to SatQuery** |
| `models--sentence-transformers--all-MiniLM-L6-v2` | 0.17 GB | unclear |
| `models--facebook--detr-resnet-50` | 0.16 GB | **relevant to SatQuery** |
| `models--microsoft--resnet-50` | 0.10 GB | **relevant to SatQuery** |
| `models--Arko007--…`, `models--Dharshaneshwaran--MultimodalDeepfakeDetector` | ~0 | **another project** |
| every `datasets--*` entry (RSICD, DIOR-RSVG, RSVQA-LR-2k, LEVIR-CD, LEVIR-MCI, VRSBench, WHU-OPT-SAR, BigEarthNet-14K) | **0.00 GB each** | pointers only — the data was fetched with `local_dir=` |

**Two conclusions.**

1. **This cache is shared across the user's projects.** Deepfake detectors are not SatQuery's. **Nothing here is proposed for deletion** — it is not this project's to remove.
2. **A pretrained DETR-ResNet-50 detector, DINOv2-base and ResNet-50 are already on this machine.** Those are directly relevant to the two largest modelling bottlenecks: the Phase-9 grounding rebuild (a pretrained detector is exactly what the from-scratch box regressor lacks) and the Phase-6 change-detector replacement. **This is an asset discovered by the storage audit, not waste.**

---

## 6. GATE A — dry-run proposal

**Nothing below has been deleted. This table is a request for approval.**

Every candidate was verified against: git tracking · code references (`grep` across `.py/.md/.yaml/.json/.sh`) · runtime dependency · test dependency · benchmark dependency · reproducibility value. The "verified by" column names the specific check, not a judgement.

### Tier 1 — regenerable by the project's own tooling, zero reproducibility loss

| # | Path | Size | Class | Reason | Verified by |
|---|---|---|---|---|---|
| A1 | `artifacts/run_<hex>/` beyond the newest 20 (323 dirs) | **3.35 GB** | REGENERABLE | One query's scratch output, reproducible by re-running the query. The repo's own `satquery prune` deletes only generated run-ids and protects all 16 named directories by construction | `satquery prune --dry-run`; independent recount (3.907 GB generated vs 1.116 GB named); `satquery/controller/retention.py` docstring enumerates the protected set |
| A2 | `data/vrsbench/.cache/huggingface/download/*.incomplete` (2 files) + `.lock` | **0.19 GB** | SAFE TO DELETE | Fragments of an **aborted** VRSBench imagery download (128 MB + 64 MB `.incomplete`, plus `Images_train.zip.lock`). Not data; a partial transfer that `huggingface_hub` re-fetches from scratch | Filenames end `.incomplete`; the annotations themselves (`VRSBench_EVAL_*.json`) live outside `.cache` and are **KEEP** |
| | **Tier 1 total** | **3.54 GB** | | | |

### Tier 2 — verified-unreferenced archives whose extracted form is present and live

| # | Path | Size | Class | Reason | Verified by |
|---|---|---|---|---|---|
| A3 | `data/whu_opt_sar/whu-opt-sar-512.zip` | **6.38 GB** | REGENERABLE | Post-extraction archive. `prepared/` (2.84 GB) is what everything reads | `grep` for the filename across the repo returns **only this audit**; `data/whu_opt_sar/index.json` resolves every id to `data\whu_opt_sar\prepared\{opt,sar,lbl}\*.tif`; re-downloadable via `scripts/fetch_datasets.py` (`WHU-OPT-SAR`, sha256-locked) |
| A4 | `data/bigearthnet_14k/BigEarthNet_14K.zip` | **2.92 GB** | REGENERABLE | Post-extraction archive | `grep` returns **only this audit**; re-downloadable via `fetch_datasets.py`. ⚠ **`extracted/` (5.16 GB) must be KEPT** — `index.json` resolves into `extracted/BEN_14k/BigEarthNet-S2/…`, and `evaluation/cross_sensor.py` + `training/track_a_encoder.py` read that index |
| A5 | `data/ben_full/bigearthnet_test_p8.hdf5.gz` | **1.69 GB** | REGENERABLE | The compressed form of `test_p8.hdf5` (3.78 GB), which is present and is what `track_a_full.py` globs (`*test*.hdf5`) | `grep` for the `.gz` name returns **only this audit**. **This is the only `ben_full` item proposed, and it is not a train shard — p1–p3 are excluded per the standing decision** |
| | **Tier 2 total** | **10.99 GB** | | | |

### Tier 3 — orphaned optimiser state, requires a judgement call

| # | Path | Size | Class | Reason | Verified by |
|---|---|---|---|---|---|
| A6 | `checkpoints/{killtest,smoke,track_b_v1}/ckpt_step_*.pt` (8 files) | **2.38 GB** | **ARCHIVE — needs your call** | Optimiser / scheduler / RNG state for runs whose weights are destroyed. Contains **no `model_state_dict`** — cannot resume, reconstruct or evaluate anything | Loaded a representative file: top-level keys are `['step','is_peft','adapter_dir','optimizer_state_dict','scheduler_state_dict','training_state','rng_state','extra']`. The `adapter_dir` it points at is 100% NUL (L32) |

**Why A6 is ARCHIVE and not SAFE TO DELETE.** These files sit inside directories that `docs/00` L32 describes as deliberately-retained evidence ("the eleven corrupted adapters remain on disk as evidence"). The *evidence* is the 1.636 GB of corrupt `.safetensors`, which **stays** under every option. But the directories are a documented forensic exhibit, and thinning an exhibit is a decision for whoever owns the L32 record, not for this audit. **Recommendation: defer A6 to a separate decision; take Tiers 1–2 now.**

### Explicitly excluded from Gate A

| Not proposed | Size | Why |
|---|---|---|
| `data/ben_full` train shards p1–p3 | ~31 GB | **Standing decision: DO NOT TOUCH pending verification** |
| `data/bigearthnet_14k/extracted/` | 5.16 GB | **LIVE** — `index.json` resolves into it |
| `data/bhoonidhi` | 6.58 GB | Irreplaceable without a Bhoonidhi account; 43 tests depend on it |
| The eleven corrupt `.safetensors` | 1.636 GB | L32 evidence |
| 16 named `artifacts/` directories | 1.116 GB | Whitelist-protected; includes the calibration logits and CDVQA artifacts behind published numbers |
| `checkpoints_protective_copy_2026-09-03` | 8.56 GB | Made today; the only protection `checkpoints/` has |
| Everything in §2.2 (Docker, npm, HF cache, `.claude`, site-packages) | ~89 GB | **Outside the repository. Not SatQuery's to delete.** See §7 |
| `__pycache__`, `.pytest_cache`, `egg-info` | 0.006 GB | Too small to justify any risk |

### If approved

| | Free space |
|---|---|
| Now | **44.11 GB** |
| After Tier 1 | 47.65 GB |
| After Tiers 1+2 | **55.10 GB** |
| After Tiers 1+2+3 | 57.48 GB |

Post-cleanup verification would be: `python -m satquery matrix --validate`, `python -c "import satquery"`, the full test suite, and a load-check of every remaining checkpoint — with the results recorded here before the phase closes.

---

## 7. The Docker question — raised, not proposed

`docker_data.vhdx` is **65.85 GB**, which is **1.5× the entire `data/` directory** and more than the whole repository. The daemon is stopped, so its contents could not be enumerated (`docker system df` cannot connect).

Why this belongs in a SatQuery storage audit even though it is out of scope:

* **Phase 4 (VRSBench) needs DOTA + DIOR imagery — tens of GB.** After Tiers 1–2 there would be 55.10 GB free, which may or may not be enough. The 65.85 GB in Docker is where the headroom actually is.
* A VHDX **grows and never shrinks on its own**. The 65.85 GB is high-water mark, not necessarily live data — the real occupancy could be far smaller and reclaimable by `docker system prune` plus a VHDX compact.
* The repository defines three images (`docker-compose.yml`, `docker-compose.gpu.yml`, `docker/api.Dockerfile`, `frontend/Dockerfile`), so **some** of that space is probably SatQuery's build cache. But it is shared with every other project on this machine, and the daemon must be started to tell them apart.

**Not proposed, because it is not mine to decide and cannot be measured without starting the daemon.**

### 7.1 Decision — deferred, 2026-09-03

**Not investigated as part of Phase 1, by explicit instruction.** It is outside the repository, shared with other projects, and starting the daemon is a separate action carrying its own risk.

**Recorded as a known future option, to be revisited only when disk pressure actually returns** — the realistic trigger being the **VRSBench imagery download (DOTA + DIOR)** in Phase 4. If that download does not fit in the free space available at the time, this is the first place to look, and the procedure would be: start the daemon → `docker system df -v` → identify which images and build cache are SatQuery's → propose a scoped prune under a fresh Gate. Until then, no action.

---

## 8. Corrections this audit makes to earlier documents

| Document | Claim | Correction |
|---|---|---|
| `docs/research/ENVIRONMENT.md` | "HF cache — does not exist at the default `~/.cache/huggingface`" | **It exists and holds 2.99 GB**, including a DETR detector, DINOv2 and ResNet-50 that are directly relevant to the grounding and change-detector work (§5) |
| `docs/research/system-audit.md` §11 | `killtest` classified "KEEP for now — the v0 baseline arm named in `track_b_eval.py`" | **The v0 arm is dead.** All four of its adapter safetensors are corrupt, and its `.pt` files hold no `model_state_dict` (§3) |
| `docs/research/system-audit.md` §11 | `data/bigearthnet_14k` treated as a zip/extracted duplicate pair | **Only the zip is redundant.** `extracted/` is a live dependency of `index.json` (§6, A4) |
| `docs/research/ENVIRONMENT.md` | `checkpoints/` 9.19 GB | **8.56 GB** measured today |

---

## 9. Execution record — Gate A approved, Tier 1 + Tier 2 only

**Approved 2026-09-03: Tier 1 + Tier 2 (14.53 GB). Tier 3 deferred. Docker not investigated.**

### 9.1 What was deleted

Every path was re-verified to exist, and every surviving form confirmed present, immediately before deletion.

| # | Path | Freed | Method |
|---|---|---|---|
| A1 | `artifacts/run_<hex>/` × 323 | **3.35 GB** | `python -m satquery prune` — the repository's own whitelist tool. Reported `kept: 20`, `deleted: 323`, `protected: 16` |
| A2 | 2 × `*.incomplete` (128 MB + 64 MB) + `Images_train.zip.lock` under `data/vrsbench/.cache/huggingface/download/` | **0.19 GB** | `Remove-Item` on the named files only |
| A3 | `data/whu_opt_sar/whu-opt-sar-512.zip` | **6.377 GB** | `Remove-Item` |
| A4 | `data/bigearthnet_14k/BigEarthNet_14K.zip` | **2.921 GB** | `Remove-Item` |
| A5 | `data/ben_full/bigearthnet_test_p8.hdf5.gz` | **1.686 GB** | `Remove-Item` |
| | **Total** | **14.52 GB** | matches the 14.53 GB approved |

### 9.2 Storage before and after

| | Before | After | Δ |
|---|---|---|---|
| **Free space** | 43.67 GB | **57.97 GB** | **+14.30 GB** |
| `data/` | 79.503 GB | **68.331 GB** | −11.17 |
| `artifacts/` | 4.678 GB | **1.555 GB** | −3.12 |
| `checkpoints/` | 8.561 GB | 8.561 GB | unchanged |
| `models/` | 7.358 GB | 7.358 GB | unchanged |

### 9.3 Survivors confirmed present after deletion

| Path | Size | Why it had to survive |
|---|---|---|
| `data/whu_opt_sar/prepared` | 2.838 GB | what `index.json` actually resolves to |
| `data/bigearthnet_14k/extracted` | 5.156 GB | **live** — `index.json` resolves into it |
| `data/ben_full/test_p8.hdf5` | 3.779 GB | the decompressed form `track_a_full.py` globs |
| `data/ben_full/bigearthnet_train_p0…p3.hdf5` | 9.663 GB each | **standing decision: DO NOT TOUCH.** All four intact |
| `data/vrsbench/VRSBench_EVAL_vqa.json` | 0.009 GB | the annotations, which are not the aborted download |
| `data/bhoonidhi` | 6.581 GB | irreplaceable real ISRO products |
| `artifacts/{calibration,cdvqa,demo_large_scene,reports}` | — | whitelist-protected evidence |
| `checkpoints/{killtest,smoke,track_b_v1}/*.pt` | 0.832 + 0.555 + 0.832 GB | **Tier 3, deferred — untouched** |

### 9.4 Verification

| Check | Result |
|---|---|
| `python -c "import satquery"` | **OK** |
| `python -m satquery matrix --validate` | **Matrix validation successful** |
| Full test suite | **1,125 passed, 44 skipped, 0 failed** (154.4 s, exit 0) — **identical to the pre-cleanup run** |
| Load-check, every weight file under `checkpoints/` + `models/` | **94 scanned · 83 load (15.435 GB) · 11 fail (1.636 GB)** |

**On the load-check method.** L32's lesson was that *a hash proves two files match; it does not prove either is loadable* — the 2026-08-31 verification hashed the safetensors without opening them and reported a destroyed model as recovered. So this check **opens every `.safetensors` with `safe_open` and reads a real tensor**, and `torch.load`s every `.pt`. Nothing was hashed.

**On the 11 failures.** They are exactly the eleven pre-existing corrupt adapters under `killtest/`, `smoke/` and `track_b_v1/` — four, three and four respectively — and they total **1.636 GB, matching `docs/00` L32's recorded figure to three decimal places**. Every one failed with `SafetensorError: invalid JSON in header`, the same signature L32 records. **No new failure was introduced by the cleanup**, and every production checkpoint loads.

### 9.5 Open items carried out of Phase 1

| Item | Status |
|---|---|
| **Tier 3** — 2.38 GB of orphaned optimiser state in `killtest/`, `smoke/`, `track_b_v1/` | **Deferred by decision. On disk, untouched.** Revisit only alongside the W15 decision below |
| **W15 — the v0 baseline arm is unloadable** | Logged in `docs/research/system-audit.md` §9 (weakness **W15**) and §12 (BLOCKED). **A decision is required before Phase 2 or Phase 7 runs**: either re-establish a real v0 by retraining under its recorded recipe (a freeze decision), or mark every v0 comparison `BLOCKED` per Rule 4. Letting the arm fail at runtime and dropping the row, or substituting another checkpoint while still calling it v0, is explicitly not acceptable |
| **Docker `docker_data.vhdx`, 65.85 GB** | **Not investigated, by decision.** Recorded in §7.1 as a known future option; revisit trigger is the Phase-4 VRSBench imagery download |
| **Checkpoint protective copy**, 8.56 GB | Retained. Still the only protection `checkpoints/` has, and still same-volume |

**Phase 1 is closed.** No further deletion is proposed.
