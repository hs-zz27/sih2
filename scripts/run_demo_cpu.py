"""Run the full SatQuery demo on a machine with no GPU (16 GB RAM), natively.

Works the same on Windows, macOS and Linux:

    python scripts/run_demo_cpu.py            (or: make demo-cpu)

Run it with the project's virtualenv Python. It:
  1. reports which checkpoints are present (scripts/demo_assets.py)
  2. sets the SATQUERY_* paths, the cpu profile and offline mode
  3. starts the API on :8000 and waits for /health
  4. installs and builds the web UI on first run, starts it on :3000
  5. runs the pre-flight check
Ctrl-C stops both. Logs go to .demo/api.log and .demo/web.log.

Why not Docker: Docker Desktop runs containers in a VM with its own RAM
ceiling, and the VQA model alone needs ~7.5 GB unquantised on a CPU.

One-time setup (several GB of downloads):
    python3.12 -m venv .venv
    Windows:      .venv\\Scripts\\pip install -e ".[cpu,report]"
    macOS/Linux:  .venv/bin/pip install -e ".[cpu,report]"
    (Linux: install torch from https://download.pytorch.org/whl/cpu first)
    python scripts/fetch_models.py --dest models --only qwen25_vl_3b
    copy the team checkpoints into ./checkpoints   (python scripts/demo_assets.py)
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from demo_assets import env_dict  # noqa: E402

API = "http://127.0.0.1:8000"
WEB = "http://localhost:3000"


def up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:  # noqa: S310 - localhost
            return r.status < 500
    except OSError:
        return False


def wait_for(url: str, seconds: int, name: str, proc: subprocess.Popen) -> None:
    for _ in range(seconds):
        if up(url):
            return
        if proc.poll() is not None:
            sys.exit(f"!! {name} exited early - see .demo/")
        time.sleep(1)
    sys.exit(f"!! {name} did not come up in {seconds} s - see .demo/")


def ml_stack_installed() -> bool:
    try:
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _stop(signum, _frame):
    # SIGTERM (kill) and SIGHUP (terminal closed) end the launcher without
    # running `finally`, which left the API holding port 8000 as an orphan.
    # Route them through the same path as Ctrl-C so children always stop.
    raise KeyboardInterrupt


def main() -> int:
    for name in ("SIGTERM", "SIGHUP", "SIGINT"):
        if hasattr(signal, name):  # SIGHUP does not exist on Windows
            signal.signal(getattr(signal, name), _stop)
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--api-only", action="store_true", help="do not start the web UI")
    args = p.parse_args()
    os.chdir(ROOT)

    print("== assets ==", flush=True)
    subprocess.call([sys.executable, str(ROOT / "scripts" / "demo_assets.py")])
    if not ml_stack_installed():
        print("\n!! torch / transformers / peft are not installed in this Python: "
              "every learned tool will be a stub.\n"
              '   Install them with: pip install -e ".[cpu,report]"')

    env = {**os.environ, **env_dict(ROOT)}
    logs = ROOT / ".demo"
    logs.mkdir(exist_ok=True)
    procs: list[subprocess.Popen] = []
    try:
        print(f"\n== starting API (profile: {env['SATQUERY_PROFILE']}) ==", flush=True)
        api = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "satquery.api.main:app",
             "--host", "127.0.0.1", "--port", "8000"],
            env=env, stdout=open(logs / "api.log", "w"), stderr=subprocess.STDOUT,
        )
        procs.append(api)
        wait_for(f"{API}/health", 120, "API", api)
        print(f"   API up at {API}")

        if not args.api_only:
            npm = shutil.which("npm")
            if npm is None:
                print("!! npm not found - install Node.js 20+ for the web UI. The API is running.")
            else:
                web_env = {**env, "NEXT_PUBLIC_API_URL": "http://localhost:8000"}
                frontend = ROOT / "frontend"
                print("\n== starting web UI ==", flush=True)
                if not (frontend / "node_modules").is_dir():
                    print("   first run: installing frontend dependencies (npm ci)...", flush=True)
                    subprocess.check_call([npm, "ci", "--no-audit", "--no-fund"], cwd=frontend, env=web_env)
                if not (frontend / ".next" / "BUILD_ID").is_file():
                    print("   first run: building the UI (next build)...", flush=True)
                    subprocess.check_call([npm, "run", "build"], cwd=frontend, env=web_env)
                web = subprocess.Popen(
                    [npm, "start", "--", "-p", "3000"], cwd=frontend, env=web_env,
                    stdout=open(logs / "web.log", "w"), stderr=subprocess.STDOUT,
                )
                procs.append(web)
                wait_for(WEB, 90, "web UI", web)
                print(f"   UI up at {WEB}")

        print(flush=True)
        subprocess.call([sys.executable, str(ROOT / "scripts" / "demo_preflight.py"),
                         "--api", API, "--web", "" if args.api_only else WEB, "--allow-stubs"])
        print(f"\nOpen {WEB}\n"
              "The first question for each model loads its weights - the VQA model takes\n"
              "longest on a CPU. Ask one throwaway question per tool before recording.\n"
              "Ctrl-C to stop.", flush=True)
        api.wait()
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        for proc in reversed(procs):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
