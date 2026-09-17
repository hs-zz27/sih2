"""Readability checks for the JSON sidecars a checkpoint cannot load without.

`is_available()` in each learned tool checked that `vocab.json` or
`band_stats.json` **exists**. Existence is not readability, and the difference
stopped being theoretical on 2026-08-31: recovering `checkpoints/` from a
volume shadow copy returned twelve small JSON files as **entirely NUL bytes** -
their data was still in the write cache when VSS froze the volume, so the file
size was on disk and the contents were not.

Two of the twelve are load-bearing. With `caption/vocab.json` zeroed,
`caption_v1.is_available()` answered `(True, "ready")` and then died inside
`_Handle.__init__` with `JSONDecodeError: Expecting value: line 1 column 1`,
which reaches the user as a failed tool rather than as an unavailable one.
`grounding_v1` did the same. The registry's whole contract is that an
unavailable tool degrades to a stub *before* it is selected; a tool that
reports ready and then raises breaks that contract.

The check here is deliberately narrow: parse the file, and require the shape
the loader is about to assume. It does not validate the vocabulary against the
checkpoint's embedding size - that is a different check, and it needs torch.
"""

from __future__ import annotations

import json
from pathlib import Path


def readable_json(path: Path, *, expect: type | None = None) -> tuple[bool, str]:
    """Whether `path` parses as JSON, and what is wrong if it does not.

    `expect` optionally requires a top-level type (`dict` for a vocabulary or
    a band-statistics file). The reason string names the file and the failure
    so an operator can act on it, which is the difference between "the caption
    tool is unavailable" and a traceback.
    """
    if not path.exists():
        return False, f"{path.name} not found beside {path.parent}"

    raw = path.read_bytes()
    if not raw:
        return False, f"{path} is empty"
    if set(raw) == {0}:
        # Called out separately because the cause is specific and the fix is
        # different: this file was not written, it was reserved. Restoring it
        # from a backup or regenerating it is the remedy; re-running the
        # training is not.
        return False, (
            f"{path} is {len(raw)} bytes of NUL - the file's size reached disk "
            "but its contents did not. Restore or regenerate it"
        )

    try:
        blob = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"{path} is not readable JSON ({type(exc).__name__}: {exc})"

    if expect is not None and not isinstance(blob, expect):
        return False, (
            f"{path} parsed as {type(blob).__name__}, expected {expect.__name__}"
        )
    if expect is dict and not blob:
        return False, f"{path} is an empty object"

    return True, "ok"


def readable_safetensors(path: Path) -> tuple[bool, str]:
    """Whether `path` is a `.safetensors` file whose header can be parsed.

    The same lesson as `readable_json`, one file type later and far more
    expensive. The shadow-copy recovery on 2026-08-31 also returned **eleven
    QLoRA adapter weight files** - 1.636 GB - as essentially all NUL:
    `track_b_v1/adapter_final/adapter_model.safetensors` is 148,712,776 bytes
    of which the first 148,701,184 are zero, leaving 11,592 bytes of real
    data at the tail. The `.pt` checkpoints were verified by loading them and
    were bit-exact; the safetensors were inventoried and hashed but never
    loaded, which is exactly how a whole model's weights were reported as
    recovered when they were not.

    `is_available()` answered `(True, "ready")` for that destroyed adapter,
    because it checked the path existed. The tool was then selected, and died
    mid-run inside the loader with `SafetensorError: invalid JSON in header`.
    An unavailable tool must degrade to its stub *before* selection.

    Deliberately cheap: read the 8-byte little-endian header length and parse
    that many bytes of JSON. It does not read the tensor payload, so the cost
    is one small read regardless of file size, and it does not need torch.
    """
    if not path.exists():
        return False, f"{path} not found"

    if path.is_dir():
        # A PEFT adapter is a directory; the weights sit inside it under one
        # of two names depending on the version that wrote them.
        for name in ("adapter_model.safetensors", "model.safetensors"):
            candidate = path / name
            if candidate.exists():
                return readable_safetensors(candidate)
        shards = sorted(path.glob("*.safetensors"))
        if shards:
            # A sharded model (`model-00001-of-00002.safetensors`). Checking
            # the first shard's header catches the failure this exists for -
            # a file whose size landed and whose contents did not - without
            # reading gigabytes.
            return readable_safetensors(shards[0])
        if any(path.glob("*.bin")):
            # A .bin adapter is a torch pickle, not safetensors; this check
            # does not cover it and must not claim otherwise.
            return True, "ok (bin adapter, header not checked)"
        return False, f"{path} contains no .safetensors weights"

    size = path.stat().st_size
    if size < 8:
        return False, f"{path} is {size} bytes - too short to hold a header"

    with path.open("rb") as handle:
        prefix = handle.read(8)
        length = int.from_bytes(prefix, "little")
        if length == 0:
            return False, (
                f"{path} declares a zero-length header - the file's size "
                f"reached disk but its contents did not ({size:,} bytes). "
                "Restore it from a backup or retrain"
            )
        if 8 + length > size:
            return False, (
                f"{path} declares a {length:,}-byte header but the file is "
                f"only {size:,} bytes - truncated"
            )
        raw = handle.read(length)

    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"{path} has an unreadable header ({type(exc).__name__}: {exc})"

    if not isinstance(header, dict) or not (set(header) - {"__metadata__"}):
        return False, f"{path} header declares no tensors"

    return True, "ok"
