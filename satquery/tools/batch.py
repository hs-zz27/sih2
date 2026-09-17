"""Batch execution with failure isolation and throughput measurement (2.13).

Every tool already exposes `run_batch`, but the inherited implementation is a
plain list comprehension: one malformed manifest raises and the whole batch is
lost. Over a 200-item benchmark that turns a single bad file into a wasted run,
which is precisely the failure the eval harness must not have.

This wraps any tool with per-item isolation and records throughput, because
task 2.13's deliverable is a measured number rather than the mere existence of
a batch method.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from satquery.contracts.input_manifest import InputManifest
from satquery.contracts.tool_result import ToolResult
from satquery.tools.base import ToolProtocol


@dataclass
class BatchReport:
    tool: str
    n_items: int
    n_ok: int
    n_failed: int
    wall_seconds: float
    failures: list[dict] = field(default_factory=list)

    @property
    def items_per_second(self) -> float:
        return self.n_items / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def seconds_per_item(self) -> float:
        return self.wall_seconds / self.n_items if self.n_items else 0.0

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "n_items": self.n_items,
            "n_ok": self.n_ok,
            "n_failed": self.n_failed,
            "wall_seconds": round(self.wall_seconds, 4),
            "items_per_second": round(self.items_per_second, 4),
            "seconds_per_item": round(self.seconds_per_item, 4),
            "failures": self.failures[:20],
        }


def run_batch(
    tool: ToolProtocol | str,
    manifests: list[InputManifest],
    params: dict[str, Any] | None = None,
) -> tuple[list[ToolResult | None], BatchReport]:
    """Run `tool` over `manifests`, isolating per-item failures.

    Returns results positionally aligned with `manifests`, with None where an
    item failed. Alignment matters: a caller matching results back to item ids
    by index would silently mis-attribute every answer after a dropped item.
    """
    from satquery.tools.stubs import REGISTRY

    name = tool if isinstance(tool, str) else type(tool).__name__
    instance = REGISTRY[tool] if isinstance(tool, str) else tool
    params = params or {}

    results: list[ToolResult | None] = []
    failures: list[dict] = []
    started = time.perf_counter()

    for i, manifest in enumerate(manifests):
        try:
            results.append(instance.run(manifest, params))
        except Exception as exc:  # noqa: BLE001 - isolation is the point
            results.append(None)
            failures.append({
                "index": i,
                "run_id": manifest.run_id,
                "error": f"{type(exc).__name__}: {exc}",
            })

    wall = time.perf_counter() - started
    report = BatchReport(
        tool=name,
        n_items=len(manifests),
        n_ok=sum(1 for r in results if r is not None),
        n_failed=len(failures),
        wall_seconds=wall,
        failures=failures,
    )
    return results, report


def benchmark(
    tool_names: list[str],
    manifests: list[InputManifest],
    params: dict[str, Any] | None = None,
) -> dict:
    """Measure throughput for several tools over the same inputs."""
    reports = {}
    for name in tool_names:
        _, report = run_batch(name, manifests, params)
        reports[name] = report.to_dict()
    return {
        "n_items": len(manifests),
        "tools": reports,
        "note": (
            "Wall-clock on this machine, sequential. Batching a VLM on a 6 GB "
            "GPU is the fastest route to an OOM, so tools run one item at a "
            "time by design."
        ),
    }
