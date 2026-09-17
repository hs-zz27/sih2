"""Resolving file paths recorded inside dataset index files.

The four dataset indexes - `data/levircd/index.json`,
`data/bigearthnet_14k/index.json`, `data/levir_mci/index.json`,
`data/whu_opt_sar/index.json` - and `data/demo_bundle/manifest.json` were all
generated on Windows, so every path in them is stored with backslash
separators:

    "label": "data\\\\levircd\\\\tiles\\\\test\\\\label\\\\000000.png"

On Windows those resolve. On Linux a backslash is an ordinary filename
character, so the whole string is read as ONE filename and every open fails.
Measured 2026-09-07 inside the API container:

    FileNotFoundError: 'data\\\\levircd\\\\tiles\\\\test\\\\label\\\\000000.png'
      scripts/make_demo_bundle.py:205 -> build_change_pair()

which meant `make_demo_bundle.py --verify` could not run in a container at
all, and the demo's 9/9 guarantee was in practice a Windows-only guarantee.

**Why this normalises on read rather than rewriting the JSON.** The index
files are generated artifacts under a gitignored `data/`, rebuilt by
`training/prepare/*.py` on whatever machine runs them. Rewriting today's
copies would fix today's copies and the next regeneration on a Windows box
would put the backslashes straight back, with nothing to catch it. Reading
defensively is the fix that survives the next regeneration; it also costs
nothing on a machine where the paths were already POSIX.

**Why `PureWindowsPath` rather than `str.replace`.** `PureWindowsPath` treats
both `/` and `\\` as separators, so one code path handles indexes written on
either OS without the caller having to know which wrote it, and without
hand-rolling separator handling at each of the eleven call sites. Splitting
into `.parts` and rebuilding with `Path(*parts)` is ordinary pathlib
construction: the stored separator is discarded rather than trusted.

The one thing this deliberately does not do is rescue an **absolute** path
from the other OS. `C:\\data\\x.png` has no meaning on Linux and
`/mnt/data/x.png` has none on Windows; an absolute string is handed to `Path`
unchanged so it fails as itself rather than being silently mangled into a
relative path. Every path in the five index files is relative.
"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath


def index_path(stored: str | Path) -> Path:
    """A usable `Path` from a path string recorded in a dataset index.

    Accepts separators in either convention and returns a path native to the
    running OS. Absolute paths are passed through untouched - see the module
    docstring for why.
    """
    text = str(stored)
    if not text:
        return Path(text)

    # Absolute in either convention: a POSIX root, a UNC/backslash root, or a
    # drive letter. Re-splitting these would turn an unusable absolute path
    # into a plausible-looking relative one, which is worse than failing.
    if text[0] in "/\\" or (len(text) > 1 and text[1] == ":"):
        return Path(text)

    return Path(*PureWindowsPath(text).parts)
