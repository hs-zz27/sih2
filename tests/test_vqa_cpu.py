"""rs_vqa_v1 loads on a machine without CUDA instead of needing bitsandbytes.

bitsandbytes' 4-bit kernels are CUDA-only, so the loader used to fail on every
CPU machine even though the rest of the stack runs there. These tests pin the
two branches without torch or weights: a stand-in torch module is enough to
check which keyword arguments reach `from_pretrained`.
"""

from __future__ import annotations

import sys
import types

import pytest

from satquery.tools import rs_vqa


class FakeTorch:
    bfloat16 = "torch.bfloat16"
    float32 = "torch.float32"
    float16 = "torch.float16"

    class cuda:  # noqa: N801 - mirrors torch.cuda
        @staticmethod
        def is_bf16_supported():
            return True


def test_cpu_loads_unquantised_bfloat16_by_default(monkeypatch):
    monkeypatch.delenv(rs_vqa.ENV_CPU_DTYPE, raising=False)
    kwargs = rs_vqa.model_load_kwargs(FakeTorch, on_cpu=True)
    assert kwargs == {"torch_dtype": "torch.bfloat16", "low_cpu_mem_usage": True}
    assert "quantization_config" not in kwargs


def test_cpu_dtype_can_be_float32(monkeypatch):
    monkeypatch.setenv(rs_vqa.ENV_CPU_DTYPE, "float32")
    assert rs_vqa.model_load_kwargs(FakeTorch, on_cpu=True)["torch_dtype"] == "torch.float32"


def test_unknown_cpu_dtype_falls_back_to_bfloat16(monkeypatch):
    monkeypatch.setenv(rs_vqa.ENV_CPU_DTYPE, "int3")
    assert rs_vqa.load_precision(on_cpu=True) == "bfloat16"


def test_gpu_branch_is_unchanged_4bit(monkeypatch):
    captured = {}

    class BitsAndBytesConfig:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(BitsAndBytesConfig=BitsAndBytesConfig))
    kwargs = rs_vqa.model_load_kwargs(FakeTorch, on_cpu=False)
    assert kwargs["device_map"] == {"": 0}
    assert captured["load_in_4bit"] is True and captured["bnb_4bit_quant_type"] == "nf4"
    assert rs_vqa.load_precision(on_cpu=False) == "nf4-4bit"


@pytest.mark.parametrize("precision", ["bfloat16", "float32"])
def test_cpu_answers_carry_a_precision_warning(precision):
    text = rs_vqa.CPU_PRECISION_WARNING.format(precision=precision)
    assert precision in text and "4-bit GPU build" in text
