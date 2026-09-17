# Checkpoint protective copy and worktree rebase — 2026-09-03

Record of the two operations approved after the Phase-0 audit, in the order they were approved: **checkpoint backup first, then rebase and verify.** No cleanup, no deletion, and no Phase-2 work was performed.

---

## 1. Checkpoint protective copy

### What was asked for, and what is actually possible

The Phase-0 audit asked whether `checkpoints/` could be copied **off-volume** before any cleanup, because the directory is gitignored, has no backup, and was destroyed once already on 2026-08-30 by `training/run_checkpoint_test.py` calling `shutil.rmtree` unconditionally (`docs/00` §3.6 **L26/L27**).

**A literal off-volume copy is not possible on this machine.** Measured 2026-09-03:

```
Physical disks : 1  — SKHynix_HFS512GEJ4X112N, 476.9 GB SSD
Volumes        : 1  — C:  (Windows-SSD), 474.7 GB total
Removable      : none
Network drives : none
```

So the copy below lives **on the same SSD as the original**. That distinction is recorded in the manifest itself rather than left for a future reader to discover.

| | |
|---|---|
| **Protects against** | accidental deletion or overwrite of the `checkpoints/` path — the failure mode that actually occurred here |
| **Does NOT protect against** | disk failure, controller failure, or loss of the machine |
| **Still outstanding** | a true off-site copy needs external media or cloud storage, and is a decision for the team |

### What was done

| | |
|---|---|
| Source | `C:\Users\<dev>\Desktop\sih2\checkpoints` |
| Destination | `C:\Users\<dev>\Desktop\checkpoints_protective_copy_2026-09-03` — **outside the repository tree**, so no path inside the repo can reach it |
| Method | SHA-256 manifest of the source → `robocopy /E /COPY:DAT` → SHA-256 manifest of the copy → compare |
| Manifest | `MANIFEST_SHA256.json` inside the copy, one entry per file with digest and byte count |

### Verification

| Check | Result |
|---|---|
| File count, source / copy | **183 / 183** |
| Total bytes, source / copy | **8.561 GB / 8.561 GB** |
| Files missing from the copy | **0** |
| SHA-256 mismatches | **0** |
| Unexpected extra files | **0** (excluding the manifest itself) |
| Free space after | **44.33 GB** (was 53.2 GB) |

**Independent anchor.** The verification above uses the script's own manifest, so it was checked a second time against a digest recorded *outside* this session:

```
copy   track_b_v2/adapter_final/adapter_model.safetensors
       sha256 10f4830141237846a439f9166acc21eef0be050c5580381e2e66256cf7041174
docs/model-cards.md, published 2026-09-01
       sha256 10f4830141237846a439f9166acc21eef0be050c5580381e2e66256cf7041174   ✓ match
```

`track_b_v1/adapter_final/adapter_model.safetensors` — the deliberately-retained corrupted evidence for `docs/00` **L32** — is present in the copy at its exact original size of **148,712,776 bytes**. The copy preserves evidence, not only working weights.

### One thing to know about the exit code

The backup script exited **1**, which looks like a failure and is not one. `robocopy` returns 1 for "one or more files were copied successfully", and PowerShell propagated that ambient code past the script's own successful completion. The script's verification block reported `bit_exact = True`, and the independent digest check above confirms it. **Read the verification block, not the exit code.**

---

## 2. Worktree rebase

### Why

The Phase-0 audit found the agent worktree and the primary checkout had **diverged**: the worktree sat at `pre-import:P0-worktree` (the `main` review merge 2) while the primary checkout sat at `pre-import:P0-primary` on `phase-0-closeout`. Four commits — including the 8-bit Adam change that trained `track_b_v3` — existed only in the primary checkout. Taking a Phase-2 "immutable baseline" against code the project does not run would have made that baseline worthless.

### Safety analysis performed before touching git

| Check | Result |
|---|---|
| Merge base | `pre-import:merge-base` |
| Commits in `pre-import:P0-worktree` **not** in `pre-import:P0-primary` | **two merge commits only** — `pre-import:P0-worktree` (review merge 2) and `pre-import:review-merge-1` (review merge 1); their content is already in the shared ancestry |
| Files deleted going `pre-import:P0-worktree → pre-import:P0-primary` | **none** |
| Lines removed going `pre-import:P0-worktree → pre-import:P0-primary` | **5**, all deliberate replacements: band-name guessing → declared PNG channels (`satquery/ingest/reader.py`, 2), and `AdamW` → 8-bit Adam (`training/track_b_vlm_qlora.py`, 3) |

`pre-import:P0-primary` is therefore a **strict content superset** of `pre-import:P0-worktree`.

### What was done

```
git branch phase0/pre-rebase-pre-import:P0-worktree pre-import:P0-worktree     # safety marker, kept
git reset --hard pre-import:P0-primary
```

A `reset --hard` was used rather than a rebase because the worktree branch carried **zero commits of its own** — all work in it was untracked files, which `reset --hard` does not touch.

### Verification

| Check | Result |
|---|---|
| Worktree HEAD | `pre-import:P0-primary` — **byte-identical to the primary checkout's HEAD** |
| Safety marker | `phase0/pre-rebase-pre-import:P0-worktree` created and retained |
| Untracked work preserved | all four files intact: `docs/external_benchmark_audit.md`, `docs/external_benchmark_results.json`, `docs/research/ENVIRONMENT.md`, `docs/research/system-audit.md` |
| Gained code present | 8-bit Adam references in `training/track_b_vlm_qlora.py`; 23 geospatial references in `satquery/controller/executor.py` |
| `import satquery` | OK |
| `python -m satquery matrix --validate` | **Matrix validation successful** |
| No-torch CI simulation | **999 passed, 63 skipped, 0 failed** (147.3 s) |
| Full test suite | **1,125 passed, 44 skipped, 0 failed** (158.4 s, exit 0) |

**On the 44 skips — they are worktree geography, not a regression.** Every one was inspected with `pytest -rs`: 43 come from `tests/test_real_products.py` ("Cartosat sample not downloaded", "EOS-04 FRS-1 / MRS / SLC sample not downloaded") and 1 from `tests/test_report_pages.py` ("no checkpoint directories … the registry has nothing to list"). Both causes are the same fact: **`data/` and `checkpoints/` are gitignored and exist only in the primary checkout**, so a worktree cannot see them.

Two consequences worth carrying forward:

* The last recorded full-suite state was **1,070 passed / 0 skipped** on 2026-09-01 in the *primary* checkout (`docs/HANDOFF.md`). The suite has since grown by four commits, so **1,125 passed is not comparable to 1,070** — it is a larger suite, and the skip count differs because of where it ran. Neither number is a regression on the other.
* **The real-product ingest path cannot be verified from a worktree at all.** Any later phase that touches ingest, co-registration, SAR handling or geospatial metadata (directive Phases 13 and 14) must be validated in the primary checkout, or those 43 tests will silently skip rather than fail.

### What the rebase did *not* bring across

The primary checkout still carries **uncommitted** changes that a commit-level rebase cannot move:

```
 M training/track_b_vlm_qlora.py     the label-masking fix + validation loop
?? tests/test_vlm_label_masking.py   its regression test
```

So the worktree now has **8-bit Adam but not the label-masking fix**. Reproducing `track_b_v3`'s recipe exactly is now possible; reproducing it *with the defect corrected* still requires those two files to be committed. That remains the first item of Phase 7 / directive P2.

---

## 3. Decisions recorded

| Question from `system-audit.md` §13 | Decision | Status |
|---|---|---|
| May `checkpoints/` be copied off-volume first? | **Approved — step 1 only.** No off-volume target exists; a verified same-volume protective copy was made instead, with the limitation stated | **Done** |
| Are `data/ben_full` train shards p1–p3 (~31 GB) still needed? | **DO NOT TOUCH pending verification. Not a delete candidate.** | **Closed for Phase 1.** Must not appear on any Gate A dry-run list |
| Rebase the worktree onto `pre-import:P0-primary` before Phase 2? | **Approved** | **Done and verified** |

## 4. What was explicitly not done

- No file was deleted, archived or moved.
- No `artifacts/` pruning, despite the measured 3.35 GB / 323 directories available — that is Phase 1 proper and requires **Gate A**.
- No dataset was touched.
- No training, no `make report`, no checkpoint-directory mutation.
- The primary checkout's working tree was not modified.
