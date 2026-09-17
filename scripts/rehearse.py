"""Rehearse the seven-minute demo, repeatedly and measurably (plan task 4.2).

The plan asks for ten rehearsals "including on the actual venue laptop with
networking off". That splits into a part a machine can do and a part it
cannot, and this script is deliberate about the boundary:

* **What this measures.** Every beat of `docs/04` §10, executed through the
  real controller in the scripted order, ten times, with per-beat and total
  wall-clock timings and the variance across runs. If a beat regresses, stops
  answering, or drifts past its time budget, this fails.
* **What this cannot do.** It does not speak, it does not click, and it is not
  the venue laptop. A human rehearsal - narration, timing against the script,
  recovering from a question mid-beat - is not automatable, and reporting this
  script's output as "ten rehearsals done" would be a false claim. The plan's
  item is only partly closed by it, and `docs/rehearsal.md` says so.

`--offline` blocks the socket layer for the duration, which is the venue
condition that matters: no network, everything served locally. It is the same
mechanism `tests/test_offline.py` uses rather than a second implementation.

Usage:
    python scripts/rehearse.py --runs 10
    python scripts/rehearse.py --runs 10 --offline --out docs/assets/rehearsal.json
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.common.paths import index_path  # noqa: E402

# The seven-minute script, beat by beat, as (beat, budget_seconds, bundle_key).
# Budgets come from docs/04 §10's own timings; they bound the *system* time,
# not the narration, so they are generous by design - the demo fails if a beat
# takes longer than its slot, not if it is fast.
BEATS: list[tuple[str, float, str]] = [
    ("0:30 the rejection - incompatible pair", 40.0, "incompatible_pair"),
    ("0:30 the rejection - PNG in operational mode", 40.0, "png_operational"),
    ("1:10 cross-modal flagship", 70.0, "crossmodal_pair"),
    ("2:20 single optical, real Cartosat", 50.0, "single_optical"),
    ("2:20 single SAR, real EOS-04", 50.0, "single_sar"),
    ("3:10 bi-temporal - what changed and where", 60.0, "change_what_and_where"),
    ("3:10 bi-temporal - increased or decreased", 60.0, "bitemporal_pair"),
    ("4:50 the abstention - clouded optical", 50.0, "clouded_optical"),
    ("5:40 the large scene", 60.0, "large_scene"),
]


def block_network() -> None:
    """Refuse every outbound connection, loopback included.

    Loopback is blocked too: a rehearsal that quietly reaches a local service
    the venue will not have is not an offline rehearsal.
    """
    def blocked(*args, **kwargs):  # noqa: ANN002, ANN003
        raise OSError("network disabled for the offline rehearsal")

    socket.socket.connect = blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = blocked  # type: ignore[method-assign]
    socket.create_connection = blocked  # type: ignore[assignment]
    socket.getaddrinfo = blocked  # type: ignore[assignment]


def load_bundle(bundle_dir: Path) -> dict[str, dict]:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"no demo bundle at {bundle_dir}. Build it first:\n"
            f"  python scripts/make_demo_bundle.py --out {bundle_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {item["key"]: item for item in manifest["inputs"]}


def one_rehearsal(controller, bundle: dict[str, dict], index: int) -> dict:
    """One full pass through the script, in order."""
    beats: list[dict] = []
    started = time.perf_counter()

    for beat, budget, key in BEATS:
        item = bundle.get(key)
        if item is None:
            beats.append({"beat": beat, "key": key, "ok": False,
                          "reason": "not in the demo bundle"})
            continue

        t0 = time.perf_counter()
        record: dict = {"beat": beat, "key": key, "budget_s": budget}
        try:
            trace = controller.run(
                [index_path(p) for p in item["images"]], item["query"],
                run_id=f"rehearsal_{index}_{key}",
            )
            record.update(
                seconds=round(time.perf_counter() - t0, 3),
                task=str(trace.routing.selected_task),
                abstained=bool(trace.abstained),
                answered=bool(trace.answer),
            )
        except Exception as exc:  # noqa: BLE001 - a rejection beat may raise
            record.update(
                seconds=round(time.perf_counter() - t0, 3),
                task="EXCEPTION", abstained=True, answered=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        expected = item["expect"]
        if expected == "rejected_or_abstained":
            record["ok"] = record["abstained"] or record["task"] in (
                "CLARIFY_OR_ABSTAIN", "EXCEPTION"
            )
        elif expected == "answered":
            record["ok"] = not record["abstained"]
        elif expected == "answered_or_abstained":
            record["ok"] = True
        else:
            record["ok"] = record["task"] == expected
        record["within_budget"] = record["seconds"] <= budget
        beats.append(record)

    return {
        "run": index,
        "total_seconds": round(time.perf_counter() - started, 3),
        "beats_ok": sum(1 for b in beats if b.get("ok")),
        "beats_total": len(beats),
        "over_budget": [b["beat"] for b in beats if not b.get("within_budget", True)],
        "beats": beats,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=Path("data/demo_bundle"))
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--offline", action="store_true",
                   help="block the socket layer - the venue condition")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    bundle = load_bundle(args.bundle)
    if args.offline:
        block_network()

    from satquery.controller.pipeline import Controller

    controller = Controller()
    runs = [one_rehearsal(controller, bundle, i + 1) for i in range(args.runs)]

    totals = [r["total_seconds"] for r in runs]
    per_beat: dict[str, list[float]] = {}
    for run in runs:
        for beat in run["beats"]:
            per_beat.setdefault(beat["beat"], []).append(beat.get("seconds", 0.0))

    report = {
        "runs": args.runs,
        "offline": bool(args.offline),
        "all_beats_ok": all(r["beats_ok"] == r["beats_total"] for r in runs),
        "total_seconds": {
            "median": round(statistics.median(totals), 3),
            "min": round(min(totals), 3),
            "max": round(max(totals), 3),
            "stdev": round(statistics.stdev(totals), 3) if len(totals) > 1 else 0.0,
        },
        "per_beat_median_seconds": {
            beat: round(statistics.median(times), 3)
            for beat, times in per_beat.items()
        },
        "beats_over_budget": sorted(
            {beat for run in runs for beat in run["over_budget"]}
        ),
        "detail": runs,
        "not_measured_here": (
            "Narration, timing against the spoken script, the venue laptop "
            "itself, and recovery from an interruption. This measures the "
            "system's half of the rehearsal only - see docs/rehearsal.md."
        ),
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    mode = "offline" if args.offline else "online"
    print(f"{args.runs} rehearsals, {mode}")
    print(f"{'beat':<46}{'median s':>10}{'budget':>9}")
    for beat, budget, _ in BEATS:
        median = report["per_beat_median_seconds"].get(beat)
        if median is not None:
            print(f"{beat:<46}{median:>10.2f}{budget:>9.0f}")
    t = report["total_seconds"]
    print(f"\ntotal per rehearsal: median {t['median']}s "
          f"(min {t['min']}, max {t['max']}, sd {t['stdev']})")
    print(f"all beats behaved in all {args.runs} runs: {report['all_beats_ok']}")
    if report["beats_over_budget"]:
        print(f"OVER BUDGET: {report['beats_over_budget']}")
    return 0 if report["all_beats_ok"] and not report["beats_over_budget"] else 1


if __name__ == "__main__":
    sys.exit(main())
