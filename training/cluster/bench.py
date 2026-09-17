"""How fast is this machine, really, on these models?

`configs/campaign.yaml` carries an `est_hours` for every run. Those numbers
were extrapolated from v1 GPU runs and parameter counts, and on a machine with
no GPU they are not merely imprecise - they are wrong by an order of magnitude
in an unknown direction. Planning a 91-hour campaign on them would be planning
on fiction.

This measures instead. It builds each v2 model at its real input shape, runs
forward and backward until the timing is stable, and reports samples per
second. Then it multiplies by the actual corpus size and epoch count from the
campaign config to project the hours each run would take *here*.

WHY FORWARD **AND** BACKWARD

A forward-only benchmark reports roughly three times the throughput of
training and is the most common way to produce an encouraging number that
predicts nothing. Backward is where most of the time goes, so it is timed.

WHY THE FIRST STEPS ARE DISCARDED

The first call allocates workspaces, picks convolution algorithms, and on CPU
lets oneDNN choose kernels. On CUDA it also runs asynchronously, so timing it
measures queue submission rather than compute. Warmup steps are run and
thrown away, and `torch.cuda.synchronize()` is called before every reading.

WHAT THIS DOES NOT MEASURE

Data loading. Every number here is compute on synthetic tensors already in
memory, so it is an **upper bound** on real throughput. On a 16-core CPU
feeding a model from 30,000 small PNGs, decode can easily become the limit -
which is worth knowing, and is why the projection is labelled optimistic
rather than expected.

Usage:
    python training/cluster/bench.py                    # every model, auto device
    python training/cluster/bench.py --only change_mask --batch 8
    python training/cluster/bench.py --device cpu --dtype bf16 --threads 16
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Corpus sizes, counted from the prepared data on 2026-09-09 rather than taken
# from a paper - what matters is how many samples a run actually iterates.
#
#   levircd      index.json splits: train=7,120  val=1,024  test=2,048
#   levir_mci    index.json splits: train=6,815  val=1,333  test=1,929
#   whu_opt_sar  index.json splits: train=1,548  validation=387
#   ben_full     4 x 15,000 train shards + 5,867 test = 60,000 train
#
# NOT the ~549k of full BigEarthNet v2: only 60,000 patches were prepared.
# The v1 run used 30,000 of them, so v2 sees 2x the data, not 20x.
#
# The two marked `approx` are HuggingFace parquet corpora whose row counts were
# not read directly; override with --samples if a run's projection matters.
CORPUS: dict[str, tuple[int, int, str]] = {
    # run id -> (train samples, epochs, provenance)
    "track_a": (60_000, 12, "counted: 4 x 15,000 hdf5 train shards"),
    "track_a_nodropout": (60_000, 12, "counted"),
    "change_mask": (7_120, 60, "counted: levircd index.json"),
    "change_caption": (6_815, 50, "counted: levir_mci index.json"),
    "optsar_fusion": (1_548, 40, "counted: whu_opt_sar index.json"),
    "change_vqa": (1_600, 60, "approx: SECOND train pairs"),
    "change_vqa_scratch_v2": (1_600, 60, "approx: SECOND train pairs"),
    "grounding": (20_000, 40, "approx: DIOR-RSVG, v1 --limit-train default"),
    "caption": (8_700, 60, "approx: RSICD train images"),
}

# model key -> (campaign run it stands for, builder kwargs, input factory)
# Input shapes are the ones the trainers actually feed, read off their call
# sites - 120px for BigEarthNet, 256px for the change models, 224px elsewhere.
SPECS: dict[str, dict] = {
    "track_a": {"run": "track_a", "kwargs": {"dim": 96}, "shape": "track_a"},
    "grounding": {"run": "grounding", "kwargs": {"vocab_size": 85, "dim": 128},
                  "shape": "grounding"},
    "change_mask": {"run": "change_mask", "kwargs": {"dim": 48}, "shape": "pair256"},
    "change_caption": {"run": "change_caption",
                       "kwargs": {"vocab_size": 500, "dim": 128},
                       "shape": "change_caption"},
    "change_vqa": {"run": "change_vqa_scratch_v2", "kwargs": {"dim": 48, "n_classes": 7},
                   "shape": "pair256"},
    "optsar_fusion": {"run": "optsar_fusion", "kwargs": {"dim": 48, "n_classes": 7},
                      "shape": "optsar"},
    "caption": {"run": "caption", "kwargs": {"vocab_size": 1781, "dim": 192},
                "shape": "caption"},
}


def cpu_features() -> dict:
    """AMX and AVX-512 flags, which decide whether bf16 on CPU is worth using."""
    flags: set[str] = set()
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("flags"):
                flags = set(line.split(":", 1)[1].split())
                break
    except OSError:
        pass
    return {
        "amx_bf16": "amx_bf16" in flags,
        "amx_tile": "amx_tile" in flags,
        "avx512_bf16": "avx512_bf16" in flags,
        "avx512f": "avx512f" in flags,
    }


def make_inputs(kind: str, batch: int, torch, device, dtype):
    """Real input shapes and dtypes, including the integer ones."""
    f = lambda *s: torch.randn(*s, device=device)          # noqa: E731
    if kind == "track_a":
        return (f(batch, 12, 120, 120),
                torch.ones(batch, 12, device=device),
                torch.full((batch,), 10.0, device=device))
    if kind == "grounding":
        return (f(batch, 3, 224, 224),
                torch.randint(1, 85, (batch, 16), device=device))
    if kind == "pair256":
        return (f(batch, 3, 256, 256), f(batch, 3, 256, 256))
    if kind == "change_caption":
        return (f(batch, 3, 256, 256), f(batch, 3, 256, 256),
                torch.rand(batch, 1, 256, 256, device=device),
                torch.randint(1, 500, (batch, 20), device=device))
    if kind == "optsar":
        return (f(batch, 4, 128, 128), f(batch, 1, 128, 128))
    if kind == "caption":
        return (f(batch, 3, 224, 224),
                torch.randint(1, 1781, (batch, 24), device=device))
    raise KeyError(kind)


def reduce_output(out, torch):
    """One scalar to call `.backward()` on, whatever the model returned."""
    if isinstance(out, (tuple, list)):
        return sum(o.float().mean() for o in out)
    return out.float().mean()


def bench_one(name: str, batch: int, steps: int, warmup: int,
              device: str, dtype_name: str, torch) -> dict:
    from training.v2 import architectures as v2

    spec = SPECS[name]
    model = v2.build(name, **spec["kwargs"]).to(device).train()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = make_inputs(spec["shape"], batch, torch, device, dtype_name)

    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                      "fp32": None}[dtype_name]

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                loss = reduce_output(model(*inputs), torch)
        else:
            loss = reduce_output(model(*inputs), torch)
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(steps):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    per_step = elapsed / steps
    return {
        "model": name,
        "run": spec["run"],
        "params_m": round(params, 2),
        "batch": batch,
        "s_per_step": round(per_step, 4),
        "samples_per_s": round(batch / per_step, 2),
    }


def project(result: dict) -> dict:
    """Hours for the campaign run this model stands for, at measured speed."""
    run = result["run"]
    if run not in CORPUS:
        return result
    samples, epochs, provenance = CORPUS[run]
    total = samples * epochs
    hours = total / result["samples_per_s"] / 3600
    result.update(train_samples=samples, epochs=epochs,
                  total_samples=total, projected_hours=round(hours, 1),
                  corpus_provenance=provenance)
    return result


def main() -> int:
    p = argparse.ArgumentParser(
        description="Measure real training throughput on this machine.")
    p.add_argument("--only", nargs="*", choices=sorted(SPECS),
                   help="benchmark just these models (default: all)")
    p.add_argument("--batch", type=int, help="fixed batch size for every model")
    p.add_argument("--steps", type=int, default=8, help="timed steps (default 8)")
    p.add_argument("--warmup", type=int, default=3,
                   help="untimed steps first (default 3)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--threads", type=int, help="torch CPU threads (default: physical cores)")
    p.add_argument("--out", type=Path, help="also write results as JSON")
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        print("torch is not installed; nothing to measure.", file=sys.stderr)
        return 2

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    features = cpu_features()
    dtype_name = args.dtype
    if dtype_name == "auto":
        if device == "cuda":
            major = torch.cuda.get_device_properties(0).major
            dtype_name = "bf16" if major >= 8 else "fp16"
        else:
            # bf16 on CPU is only a win with hardware for it. Without AMX or
            # AVX-512-BF16 it is emulated and usually SLOWER than fp32, so the
            # default follows the silicon rather than a preference.
            dtype_name = "bf16" if (features["amx_bf16"]
                                    or features["avx512_bf16"]) else "fp32"

    if device == "cpu":
        threads = args.threads or (torch.get_num_threads())
        torch.set_num_threads(threads)
        print(f"device    cpu, {threads} threads")
        print(f"cpu       {platform.processor() or platform.machine()}")
        flags = [k for k, v in features.items() if v]
        print(f"features  {', '.join(flags) if flags else 'no AMX / AVX-512 bf16'}")
        if dtype_name == "bf16":
            print("          AMX/AVX-512-BF16 present: bf16 autocast is a real speedup")
        else:
            print("          no bf16 hardware: running fp32, as bf16 would be emulated")
    else:
        props = torch.cuda.get_device_properties(0)
        print(f"device    cuda: {props.name}, {props.total_memory / 1e9:.1f} GB")
    print(f"dtype     {dtype_name}")
    print(f"timing    {args.warmup} warmup + {args.steps} timed steps, "
          f"forward AND backward\n")

    chosen = args.only or list(SPECS)
    # A batch that fits a 6 GB laptop and a 16-core CPU alike. Throughput is
    # reported per sample, so a modest batch still projects correctly - it
    # merely leaves some hardware idle, which understates a big machine.
    default_batch = 8

    header = (f"{'model':<16}{'params':>8}{'batch':>7}{'s/step':>9}"
              f"{'samp/s':>9}{'run hours':>11}")
    print(header)
    print("-" * len(header))

    results = []
    for name in chosen:
        try:
            result = project(bench_one(name, args.batch or default_batch,
                                       args.steps, args.warmup, device,
                                       dtype_name, torch))
        except Exception as exc:                       # noqa: BLE001
            print(f"{name:<16} FAILED: {type(exc).__name__}: {exc}")
            continue
        results.append(result)
        hours = result.get("projected_hours")
        print(f"{result['model']:<16}{result['params_m']:>7.2f}M"
              f"{result['batch']:>7}{result['s_per_step']:>9.3f}"
              f"{result['samples_per_s']:>9.1f}"
              f"{(f'{hours:.1f}' if hours else '-'):>11}")

    total = sum(r.get("projected_hours", 0) for r in results)
    print("-" * len(header))
    print(f"{'projected total':<16}{'':>7} {'':>7}{'':>9}{'':>9}{total:>11.1f} h")

    print("\nThese projections are an UPPER BOUND on speed and so a LOWER bound")
    print("on time: they measure compute on tensors already in memory and")
    print("exclude data loading entirely. On a CPU feeding a model from tens of")
    print("thousands of small PNGs, decode can become the real limit.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"device": device, "dtype": dtype_name, "cpu_features": features,
             "results": results, "projected_total_hours": round(total, 1)},
            indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
