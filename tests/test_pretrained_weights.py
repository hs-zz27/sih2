"""Loading a trained checkpoint never downloads ImageNet weights first."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.common.pretrained import imagenet_weights, weights_from_checkpoint

ROOT = Path(__file__).resolve().parent.parent


def test_training_gets_the_requested_weights():
    assert imagenet_weights("IMAGENET1K_V2") == "IMAGENET1K_V2"


def test_checkpoint_load_skips_them_and_restores_after():
    with weights_from_checkpoint():
        assert imagenet_weights("IMAGENET1K_V2") is None
        with weights_from_checkpoint():
            assert imagenet_weights("x") is None
        assert imagenet_weights("x") is None
    assert imagenet_weights("IMAGENET1K_V2") == "IMAGENET1K_V2"


def test_restored_even_when_the_load_raises():
    with pytest.raises(RuntimeError):
        with weights_from_checkpoint():
            raise RuntimeError("bad checkpoint")
    assert imagenet_weights("w") == "w"


@pytest.mark.parametrize("builder", [
    "training/v2/architectures.py", "training/train_change_vqa.py",
])
def test_builders_ask_before_requesting_imagenet(builder):
    assert "imagenet_weights(" in (ROOT / builder).read_text(encoding="utf-8")


@pytest.mark.parametrize("loader", [
    "satquery/tools/caption.py", "satquery/tools/grounding.py", "satquery/tools/change_vqa.py",
])
def test_loaders_build_inside_the_skip_block(loader):
    assert "with weights_from_checkpoint():" in (ROOT / loader).read_text(encoding="utf-8")
