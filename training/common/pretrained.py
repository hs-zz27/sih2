"""Skip ImageNet downloads when a trained checkpoint is about to replace them.

The pretrained caption, grounding and change-VQA models build a torchvision
ResNet with ImageNet weights, which downloads them on first use (~100 MB for
ResNet-50) and needs network access. That is correct when training starts
from them. At inference it is pure cost: the tool loads its own checkpoint
over every one of those weights immediately afterwards. On an offline demo
machine the download failed the load outright, and on a fresh one it made the
first query wait for a file the answer never used.

Loaders wrap model construction in `weights_from_checkpoint()`; builders ask
`imagenet_weights(...)` which weights to request. The architecture is
identical either way, so the checkpoint's keys match.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

_state = threading.local()


@contextmanager
def weights_from_checkpoint():
    """Within this block, pretrained builders skip their ImageNet download."""
    previous = getattr(_state, "skip", False)
    _state.skip = True
    try:
        yield
    finally:
        _state.skip = previous


def imagenet_weights(requested):
    """`requested` when training, None when a checkpoint will supply weights."""
    return None if getattr(_state, "skip", False) else requested
