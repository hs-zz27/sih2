"""Model provenance: SHA-256 of the weights a run actually loaded.

`Trace.weights_hashes` has been in the contract since Phase 0 and the executor
emitted `{}` unconditionally, with a comment saying it would be populated
"when real checkpoints load". Real checkpoints have loaded since Phase 2, so
the field was empty for a reason that had stopped being true: an auditable
execution summary that names the tool but not the bytes it ran cannot answer
"which weights produced this answer", which is the question the field exists
for.

## What is recorded, and what is not

* **Recorded:** the checkpoint file (or adapter directory) each learned tool
  loaded, hashed from its bytes at load time. Every `_Handle` in
  `satquery/tools/` already resolves `find_latest_checkpoint(...)` to a single
  path before loading it; that exact path is what is hashed, so the digest
  describes the weights in memory rather than the directory they came from.

* **NOT recorded: stubs.** A stub loads nothing. Giving it a hash - of its
  source file, of a constant, of anything - would put a digest next to a
  fabricated answer and make the two look alike in the trace. A tool that ran
  from a stub is simply absent from the map, and its `model_card` in the step
  trace already says `stub_*`.

* **NOT recorded: the VQA base model.** `rs_vqa_v1` loads a multi-gigabyte
  third-party base plus our own small adapter. Only the adapter is hashed
  here: it is the part this project produced and the part that changes
  between runs. The base model's digest is recorded separately by
  `scripts/fetch_models.py` in `configs/model_lock.json`, and it is a
  trust-on-first-use digest rather than a publisher-provided one, which is
  stated there. Hashing 7 GB on every cold start to restate a number that is
  already on disk would cost the demo its warm-up budget and add nothing.

## Determinism

The digest is over file bytes, so it is reproducible on any machine holding
the same checkpoint. A directory is hashed as a sorted manifest of
`(relative POSIX path, size, file digest)`, so it does not depend on
readdir order, on the absolute path, or on mtimes.

Digests are cached per `(resolved path, size, mtime_ns)`, so a re-trained
checkpoint written to the same path is re-hashed rather than served stale,
and a warm process pays the cost once.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterable
from pathlib import Path

_CHUNK = 1024 * 1024

_lock = threading.Lock()
# tool id -> "sha256:<hex>"
_recorded: dict[str, str] = {}
# (path, size, mtime_ns) -> "sha256:<hex>"
_cache: dict[tuple[str, int, int], str] = {}


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_dir(root: Path) -> str:
    """Digest of a directory's contents, independent of traversal order.

    Each file contributes its POSIX-relative path, its size and its own
    digest, joined in sorted order. Path and size are included so that moving
    bytes between files - or truncating one to zero - changes the result.
    """
    lines = []
    for file in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = file.relative_to(root).as_posix()
        lines.append(f"{relative} {file.stat().st_size} {_digest_file(file)}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def sha256_of(path: str | Path) -> str:
    """`sha256:<hex>` for a file or a directory. Raises if it does not exist."""
    target = Path(path).resolve()
    if not target.exists():
        raise FileNotFoundError(target)

    stat = target.stat()
    key = (str(target), stat.st_size, stat.st_mtime_ns)
    with _lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached

    value = (
        f"sha256:{_digest_dir(target)}" if target.is_dir()
        else f"sha256:{_digest_file(target)}"
    )
    with _lock:
        _cache[key] = value
    return value


def record(tool_id: str, path: str | Path) -> str | None:
    """Record the digest of the artifact `tool_id` just loaded.

    Called from the tools' load paths, which run inside a double-checked
    singleton, so this happens once per process per tool.

    Never raises. A checkpoint that vanished between the availability check
    and the load is a real possibility on a shared machine, and a provenance
    record must not be the thing that takes a working answer down. The
    consequence of failure is an absent key, which reads as "not recorded" -
    the truth - rather than as a wrong hash.
    """
    try:
        value = sha256_of(path)
    except (OSError, ValueError):
        return None
    with _lock:
        _recorded[tool_id] = value
    return value


def recorded() -> dict[str, str]:
    """Everything recorded so far, as a copy."""
    with _lock:
        return dict(_recorded)


def hashes_for(tool_ids: Iterable[str]) -> dict[str, str]:
    """Digests for the tools that ran, omitting any that loaded no weights.

    Omission is deliberate and load-bearing: the map is a claim about bytes,
    and a tool with no entry made no such claim.
    """
    with _lock:
        return {
            tool_id: _recorded[tool_id]
            for tool_id in dict.fromkeys(tool_ids)
            if tool_id in _recorded
        }


def reset() -> None:
    """Drop every record and cached digest (tests, and a reload after training)."""
    with _lock:
        _recorded.clear()
        _cache.clear()
