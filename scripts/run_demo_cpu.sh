#!/usr/bin/env bash
# Run the full SatQuery demo on a machine with no GPU (16 GB RAM), natively.
#
# Why not Docker here: Docker Desktop runs containers in a VM with its own RAM
# ceiling, and the VQA model alone needs ~7.5 GB unquantised. Running natively
# lets the API use the machine's memory directly.
#
# What it does:
#   1. reports which checkpoints are present (scripts/demo_assets.py)
#   2. exports the SATQUERY_* paths, the cpu profile, offline mode
#   3. starts the API on :8000 and waits for /health
#   4. builds the web UI once if needed, starts it on :3000
#   5. runs the pre-flight check
# Ctrl-C stops both. Logs go to .demo/api.log and .demo/web.log.
#
# One-time setup (heavy - several GB of downloads):
#   python3.12 -m venv .venv
#   .venv/bin/pip install -e ".[cpu,report]"
#   (Linux: install torch from https://download.pytorch.org/whl/cpu first)
#   python scripts/fetch_models.py --dest models --only qwen25_vl_3b
#   copy the team checkpoints into ./checkpoints (see scripts/demo_assets.py)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
API_URL="http://127.0.0.1:8000"
WEB_URL="http://localhost:3000"

if [ ! -x "$PY" ]; then
  echo "No virtualenv at .venv. Set it up first:"
  echo "  python3.12 -m venv .venv && .venv/bin/pip install -e \".[cpu,report]\""
  exit 1
fi

echo "== assets =="
"$PY" scripts/demo_assets.py || echo "   (missing items will answer as stubs - see above)"

if ! "$PY" -c "import torch, transformers, peft" >/dev/null 2>&1; then
  echo
  echo "!! torch / transformers / peft are not installed: every learned tool will be a stub."
  echo "   Install them with: .venv/bin/pip install -e \".[cpu,report]\""
fi

eval "$("$PY" scripts/demo_assets.py --env)"
mkdir -p .demo

API_PID=""
WEB_PID=""
cleanup() {
  [ -n "$WEB_PID" ] && kill "$WEB_PID" 2>/dev/null || true
  [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for() {  # url, seconds, name
  local i=0
  until curl -sf -o /dev/null "$1"; do
    i=$((i + 1))
    if [ "$i" -ge "$2" ]; then echo "!! $3 did not come up in $2 s - see .demo/"; exit 1; fi
    sleep 1
  done
}

echo
echo "== starting API (profile: $SATQUERY_PROFILE) =="
"$PY" -m uvicorn satquery.api.main:app --host 127.0.0.1 --port 8000 > .demo/api.log 2>&1 &
API_PID=$!
wait_for "$API_URL/health" 120 "API"
echo "   API up at $API_URL"

if ! command -v npm >/dev/null 2>&1; then
  echo "!! npm not found - install Node.js 20+ to run the web UI. The API is running."
  wait "$API_PID"
  exit 0
fi

echo
echo "== starting web UI =="
export NEXT_PUBLIC_API_URL="http://localhost:8000"
if [ ! -d frontend/node_modules ]; then
  echo "   first run: installing frontend dependencies (npm ci)..."
  (cd frontend && npm ci --no-audit --no-fund)
fi
if [ ! -f frontend/.next/BUILD_ID ]; then
  echo "   first run: building the UI (next build)..."
  (cd frontend && npm run build)
fi
(cd frontend && npm start -- -p 3000) > .demo/web.log 2>&1 &
WEB_PID=$!
wait_for "$WEB_URL" 90 "web UI"
echo "   UI up at $WEB_URL"

echo
"$PY" scripts/demo_preflight.py --api "$API_URL" --web "$WEB_URL" --allow-stubs || true

cat <<MSG

Open $WEB_URL
The first question for each model loads its weights: expect 1-3 minutes for
the VQA model on CPU, then tens of seconds per answer. Ask one throwaway
question per tool before you start recording. Ctrl-C to stop.
MSG
wait "$API_PID"
