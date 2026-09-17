"""Runtime profiles: full and lite (plan task 3.10).

`configs/profiles/full.yaml` and `lite.yaml` existed from Phase 0 and were
both empty, so "the lite profile" was a name with nothing behind it. This
gives them content and a loader.

The requirement is **"every task answers in lite, degraded but never
failing"**, and the second half is the harder one. Degradation here is a
budget, not a switch: the router already sums each plan's estimated VRAM and
drops steps that exceed the profile's budget, so a lite run keeps the
deterministic index engine (0 MB, pure numpy) and sheds the learned tools.
The answer then comes from `synth/narrative.py`, which builds prose from
measured indices - so a lite answer is quantitatively grounded even though no
model ran.

That is why lite is a real fallback rather than an error path: the physics
half of this system never needed a GPU. What lite loses is the open-ended
language half.

A profile is deliberately not allowed to change what is *legal*. The
capability matrix governs legality; a profile governs resources. Letting a
profile widen the legal set would put a second, quieter authority in front of
the guarantee that task 3.8 measures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parents[2] / "configs" / "profiles"
ENV_PROFILE = "SATQUERY_PROFILE"
DEFAULT_PROFILE = "full"


@dataclass(frozen=True)
class Profile:
    name: str
    description: str = ""
    # None means "no budget" - every tool may load. A number in MB makes the
    # router drop steps whose estimated peak exceeds it.
    vram_budget_mb: int | None = None
    device: str = "auto"
    # The entailment gate's NLI backend is a 370 MB model. Lite runs the
    # deterministic backend only, which needs neither GPU nor download.
    enable_nli: bool = True
    verifier_enabled: bool = True
    max_tile_px: int = 512
    notes: list[str] = field(default_factory=list)

    @property
    def is_cpu_only(self) -> bool:
        return self.device == "cpu"


BUILTIN: dict[str, Profile] = {
    "full": Profile(
        name="full",
        description="Everything loads; assumes a CUDA GPU with >=6 GB VRAM.",
        vram_budget_mb=None,
        device="auto",
        enable_nli=True,
    ),
    "lite": Profile(
        name="lite",
        description=(
            "CPU-only. Learned tools are shed by the VRAM budget; the "
            "deterministic index engine and narrative still answer."
        ),
        # 0 MB keeps only tools that declare no VRAM cost - today that is
        # index_engine_v1 alone. This is a deliberate floor rather than a
        # small positive number: a budget of, say, 900 MB would admit the
        # land-cover head onto a CPU where it would run but take minutes, and
        # a demo that appears to hang is worse than one that says it degraded.
        vram_budget_mb=0,
        device="cpu",
        enable_nli=False,
        verifier_enabled=True,
        max_tile_px=256,
        notes=[
            "Answers come from measured indices via synth/narrative.py, so "
            "they are quantitatively grounded but not open-ended language.",
            "The entailment gate still runs, on its deterministic backend.",
        ],
    ),
}


def load_profile(name: str | None = None) -> Profile:
    """Load a profile by name, from YAML when present, else the builtin.

    The builtin is the source of truth for behaviour and the YAML overrides
    it. A missing or malformed profile file falls back to the builtin rather
    than raising: a demo laptop with a corrupted config should still boot in
    a known state.
    """
    name = (name or os.environ.get(ENV_PROFILE) or DEFAULT_PROFILE).lower()
    base = BUILTIN.get(name)
    if base is None:
        raise ValueError(
            f"unknown profile {name!r}; available: {sorted(BUILTIN)}"
        )

    path = PROFILE_DIR / f"{name}.yaml"
    if not path.exists():
        return base

    try:
        import yaml

        blob = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(blob, dict):
            return base
        budget = blob.get("vram_budget_mb", base.vram_budget_mb)
        return Profile(
            name=name,
            description=str(blob.get("description", base.description)),
            vram_budget_mb=None if budget is None else int(budget),
            device=str(blob.get("device", base.device)),
            enable_nli=bool(blob.get("enable_nli", base.enable_nli)),
            verifier_enabled=bool(
                blob.get("verifier_enabled", base.verifier_enabled)
            ),
            max_tile_px=int(blob.get("max_tile_px", base.max_tile_px)),
            notes=list(blob.get("notes", base.notes)),
        )
    except Exception:  # noqa: BLE001 - degradation, not a crash
        return base
