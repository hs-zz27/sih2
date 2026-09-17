#!/usr/bin/env bash
# Prepare the AI Lab server for the Phase 5 campaign. Run it ON the server,
# after `ssh <cluster-user>@<cluster-host>`, from inside the uploaded repo.
#
#   bash training/cluster/bootstrap.sh
#
# It is idempotent: run it again after a disconnect, or after the admin
# changes something, and it will report the state rather than redo the work.
#
# WHAT IT DOES NOT DO
#
# It does not install anything system-wide and it does not touch another
# user's files. The AI Lab manual asks that shared resources be used
# responsibly, so everything lands in a virtualenv under this checkout and
# every heavy step is opt-in.
#
# WHY A VIRTUALENV RATHER THAN THE SYSTEM PYTHON
#
# The server is shared. `pip install` into a shared interpreter changes the
# environment for every other student on the box, and a version bump that
# suits this project can break someone else's run hours later, invisibly.
# A venv is the difference between "my project has its dependencies" and
# "I altered the machine".

set -uo pipefail

BOLD=$'\033[1m'; RESET=$'\033[0m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
say()  { printf '%s\n' "$*"; }
head2() { printf '\n%s=== %s%s\n' "$BOLD" "$*" "$RESET"; }
ok()   { printf '  %sok%s   %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '  %swarn%s %s\n' "$YELLOW" "$RESET" "$*"; }
bad()  { printf '  %sFAIL%s %s\n' "$RED" "$RESET" "$*"; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
VENV="$REPO/.venv"

head2 "WHERE"
say "  repo   $REPO"
say "  host   $(hostname)"
say "  user   $(whoami)"

head2 "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
               --format=csv,noheader 2>/dev/null | sed 's/^/  /'
    # Other people's jobs are the reason to look before starting a 14-hour run.
    running=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
    if [ -n "$running" ]; then
        warn "GPU already has processes on it (shared box - plan around them):"
        printf '%s\n' "$running" | sed 's/^/       /'
    else
        ok "no other compute processes on the GPU right now"
    fi
else
    bad "nvidia-smi not found - this node may have no GPU. Ask the admin which"
    say "       node has one before staging 66 GB onto it."
fi

head2 "DISK"
df -h "$REPO" 2>/dev/null | sed 's/^/  /'
if command -v quota >/dev/null 2>&1; then
    q=$(quota -s 2>/dev/null)
    [ -n "$q" ] && { say "  quota:"; printf '%s\n' "$q" | sed 's/^/    /'; }
fi
say ""
say "  The full corpus is ~66 GB. If that does not fit, stage in tiers -"
say "  'python training/cluster/recon.py --only tiers' prints them, and the"
say "  first 0.69 GB is enough to train a real model."

head2 "PYTHON"
PY=""
for candidate in python3.12 python3.11 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        v=$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
        if [ -z "$v" ]; then
            # On the PATH but not runnable - a broken shim or a stale module.
            warn "$candidate is on PATH but did not run"
            continue
        fi
        # pyproject.toml requires >=3.12. `sort -V` compares versions properly,
        # so 3.9 sorts BELOW 3.12 rather than above it as string order would.
        if [ "$(printf '%s\n3.12\n' "$v" | sort -V | head -1)" = "3.12" ]; then
            PY="$candidate"; ok "$candidate is $v (>= 3.12 required)"; break
        fi
        warn "$candidate is $v - too old, this project needs >= 3.12"
    fi
done
if [ -z "$PY" ]; then
    bad "no Python >= 3.12 found. Check for a module system:"
    say "       module avail python     # then: module load python/3.12"
    say "       conda env list"
    exit 1
fi

head2 "VIRTUALENV"
if [ -d "$VENV" ]; then
    ok "already exists at $VENV"
else
    say "  creating $VENV ..."
    "$PY" -m venv "$VENV" || { bad "venv creation failed"; exit 1; }
    ok "created"
fi
# POSIX venvs put the activate script in bin/; Windows puts it in Scripts/.
# The server is Linux, but checking both means this script can be dry-run on a
# laptop before it is trusted with a 14-hour job.
ACTIVATE="$VENV/bin/activate"
[ -f "$ACTIVATE" ] || ACTIVATE="$VENV/Scripts/activate"
if [ ! -f "$ACTIVATE" ]; then
    bad "no activate script under $VENV"
    exit 1
fi
# shellcheck disable=SC1090
source "$ACTIVATE" || { bad "could not activate the venv"; exit 1; }
ok "active: $(python -V 2>&1), $(which python)"

head2 "PACKAGES"
missing=""
# pyarrow is not in requirements.txt but three trainers import it: RSICD,
# DIOR-RSVG and the instruction mix are stored as parquet. Measured on the AI
# Lab box - `caption` and `grounding` both died at `import pyarrow.parquet`
# seconds after starting, which is cheap to fix and annoying to discover
# twice.
for mod in torch numpy yaml PIL h5py pyarrow pandas; do
    if python -c "import $mod" >/dev/null 2>&1; then
        ok "$mod"
    else
        warn "$mod missing"; missing="$missing $mod"
    fi
done

if [ -n "$missing" ]; then
    say ""
    say "  Install them into the venv (NOT system-wide):"
    say ""
    say "    source .venv/bin/activate"
    say "    pip install -r requirements.txt"
    say "    pip install h5py pyarrow pandas       # shards + parquet datasets"
    say "    # torch: match the server's CUDA. Check with 'nvidia-smi' (top right),"
    say "    # then pick the matching wheel index, e.g. for CUDA 12.1:"
    say "    pip install torch --index-url https://download.pytorch.org/whl/cu121"
    say ""
    say "  track_b_vqa additionally needs:"
    say "    pip install peft bitsandbytes accelerate datasets transformers"
    say "    pip install torchvision --index-url https://download.pytorch.org/whl/cu124"
    say ""
    say "  torchvision is not optional there despite no video being involved:"
    say "  transformers 5.x resolves Qwen2.5-VL through a processor that pulls"
    say "  in Qwen2VLVideoProcessor, which hard-requires it. Every other run"
    say "  trains without any of these."
fi

head2 "DATA"
if [ -d "$REPO/data" ]; then
    du -sh "$REPO/data"/* 2>/dev/null | sed 's/^/  /'
else
    warn "no data/ directory yet - nothing has been transferred"
fi

head2 "NEXT"
say "  1. Full picture of this machine:"
say "       python training/cluster/recon.py"
say ""
say "  2. Verify whatever data has arrived, and see what it unblocks:"
say "       python training/cluster/stage_data.py plan --root data \\"
say "           --manifest configs/data_manifest.json"
say ""
say "  3. Start training DETACHED, so it survives your laptop disconnecting."
say "     This is the main advantage of having real SSH rather than a notebook:"
say ""
say "       nohup python training/cluster/campaign.py all > runs/campaign.out 2>&1 &"
say ""
say "     Then watch it, and close the terminal whenever you like:"
say "       tail -f runs/campaign.out"
say "       python training/cluster/campaign.py status"
say ""
say "     If tmux exists, it is nicer than nohup - you get the session back:"
command -v tmux >/dev/null 2>&1 \
    && say "       tmux new -s satquery      (detach: ctrl-b then d; return: tmux attach -t satquery)" \
    || say "       (tmux is not installed here; nohup above is the fallback)"
say ""
