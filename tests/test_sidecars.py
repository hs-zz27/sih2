"""A checkpoint's JSON sidecars must be readable, not merely present.

Found by recovering `checkpoints/` from a volume shadow copy on 2026-08-31.
Twelve small JSON files came back as **entirely NUL bytes** - their size had
reached the volume and their contents had not, because the data was still in
the write cache when VSS froze it. Every `.pt` weight file was bit-perfect;
the sidecars were not.

Two of them are load-bearing, and the failure they produced is the specific
one the registry's stub fallback exists to prevent:

    >>> caption.is_available()
    (True, 'ready')
    >>> CaptionTool().run(manifest, params)
    JSONDecodeError: Expecting value: line 1 column 1 (char 0)

A tool that reports **ready** and then raises has broken the contract that
lets the controller plan against it. `is_available()` now parses the file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from satquery.tools import caption as caption_mod
from satquery.tools import change_caption as change_caption_mod
from satquery.tools import grounding as grounding_mod
from satquery.tools import landcover as landcover_mod
from satquery.tools.sidecars import readable_json, readable_safetensors


class TestReadableJson:
    def test_a_valid_object_is_accepted(self, tmp_path):
        target = tmp_path / "vocab.json"
        target.write_text(json.dumps({"a": 1}), encoding="utf-8")

        assert readable_json(target, expect=dict) == (True, "ok")

    def test_a_missing_file_names_itself(self, tmp_path):
        ok, reason = readable_json(tmp_path / "vocab.json")

        assert ok is False
        assert "vocab.json" in reason

    def test_an_all_nul_file_is_diagnosed_specifically(self, tmp_path):
        """The exact shape the shadow-copy restore produced.

        Diagnosed apart from "not valid JSON" because the cause and the fix
        differ: the file was reserved, not written, so it is restored or
        regenerated rather than retrained.
        """
        target = tmp_path / "vocab.json"
        target.write_bytes(b"\x00" * 512)

        ok, reason = readable_json(target, expect=dict)

        assert ok is False
        assert "512 bytes of NUL" in reason
        assert "contents did not" in reason

    def test_an_empty_file_is_rejected(self, tmp_path):
        target = tmp_path / "vocab.json"
        target.write_bytes(b"")

        ok, reason = readable_json(target)

        assert ok is False
        assert "empty" in reason

    def test_truncated_json_is_rejected_with_the_parser_error(self, tmp_path):
        target = tmp_path / "vocab.json"
        target.write_text('{"a": 1', encoding="utf-8")

        ok, reason = readable_json(target, expect=dict)

        assert ok is False
        assert "not readable JSON" in reason

    def test_the_wrong_top_level_type_is_rejected(self, tmp_path):
        """A vocabulary is an object. A list would load and then mis-decode."""
        target = tmp_path / "vocab.json"
        target.write_text("[1, 2, 3]", encoding="utf-8")

        ok, reason = readable_json(target, expect=dict)

        assert ok is False
        assert "expected dict" in reason

    def test_an_empty_object_is_rejected(self, tmp_path):
        """An empty vocabulary decodes every token id to nothing."""
        target = tmp_path / "vocab.json"
        target.write_text("{}", encoding="utf-8")

        ok, reason = readable_json(target, expect=dict)

        assert ok is False
        assert "empty object" in reason


class TestAvailabilityGates:
    """The four tools whose loader would raise on an unreadable sidecar."""

    CASES = [
        (caption_mod, "SATQUERY_CAPTION", "vocab.json"),
        (grounding_mod, "SATQUERY_GROUNDING", "vocab.json"),
        (change_caption_mod, "SATQUERY_CHANGE_CAPTION", "vocab.json"),
        (landcover_mod, "SATQUERY_LANDCOVER", "band_stats.json"),
    ]

    @pytest.mark.parametrize(
        "module,env,sidecar",
        CASES,
        ids=["caption", "grounding", "change_caption", "landcover"],
    )
    def test_a_nul_filled_sidecar_reports_unavailable(
        self, module, env, sidecar, monkeypatch, tmp_path
    ):
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()
        (checkpoint / sidecar).write_bytes(b"\x00" * 1156)
        monkeypatch.setenv(env, str(checkpoint))

        available, reason = module.is_available()

        assert available is False
        assert "NUL" in reason

    @pytest.mark.parametrize(
        "module,env,sidecar",
        CASES,
        ids=["caption", "grounding", "change_caption", "landcover"],
    )
    def test_a_present_but_unparseable_sidecar_reports_unavailable(
        self, module, env, sidecar, monkeypatch, tmp_path
    ):
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()
        (checkpoint / sidecar).write_text("not json at all", encoding="utf-8")
        monkeypatch.setenv(env, str(checkpoint))

        available, reason = module.is_available()

        assert available is False
        assert sidecar in reason

    @pytest.mark.parametrize(
        "module,env,sidecar",
        CASES,
        ids=["caption", "grounding", "change_caption", "landcover"],
    )
    def test_a_readable_sidecar_passes_this_gate(
        self, module, env, sidecar, monkeypatch, tmp_path
    ):
        """The gate must not become a blanket refusal.

        With a readable sidecar the only remaining reason to decline is a
        missing torch, so either outcome is acceptable - what must NOT happen
        is a refusal that names the sidecar.
        """
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()
        (checkpoint / sidecar).write_text(json.dumps({"a": 1}), encoding="utf-8")
        monkeypatch.setenv(env, str(checkpoint))

        _, reason = module.is_available()

        assert sidecar not in reason


class TestRegistryFallsBackRatherThanCrashing:
    def test_a_damaged_checkpoint_yields_the_stub(self, monkeypatch, tmp_path):
        """The contract: unavailable means the stub is selected, not a raise.

        This is what the incident broke - `caption_v1` reported ready with a
        NUL-filled vocabulary, so the registry chose the real tool and the
        run died inside the loader.
        """
        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()
        (checkpoint / "vocab.json").write_bytes(b"\x00" * 4096)
        monkeypatch.setenv("SATQUERY_CAPTION", str(checkpoint))

        from satquery.tools.stubs import CaptionStub, _caption_tool

        assert isinstance(_caption_tool(), CaptionStub)


def _write_safetensors(path, header, payload=bytes(32)):
    """Write a minimal valid .safetensors: 8-byte LE header length, JSON, data."""
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(len(raw).to_bytes(8, "little") + raw + payload)
    return path


TENSOR = {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}


class TestReadableSafetensors:
    """The `.pt` files were verified by loading them; the safetensors were not.

    That gap is how eleven QLoRA adapters - 1.636 GB - were reported as
    recovered while being 99.99% NUL. `is_available()` answered
    `(True, "ready")` for a destroyed adapter and the tool then died inside the
    loader with `SafetensorError: invalid JSON in header`.
    """

    def test_a_well_formed_file_is_readable(self, tmp_path):
        p = _write_safetensors(tmp_path / "adapter_model.safetensors", TENSOR)
        assert readable_safetensors(p) == (True, "ok")

    def test_a_zero_length_header_is_rejected(self, tmp_path):
        # Exactly the observed damage: the size landed, the contents did not.
        p = tmp_path / "adapter_model.safetensors"
        p.write_bytes(bytes(4096))
        ok, reason = readable_safetensors(p)
        assert ok is False
        assert "zero-length header" in reason
        assert "Restore it from a backup or retrain" in reason

    def test_a_truncated_file_is_rejected(self, tmp_path):
        p = tmp_path / "adapter_model.safetensors"
        p.write_bytes((10_000).to_bytes(8, "little") + b"{}")
        ok, reason = readable_safetensors(p)
        assert ok is False
        assert "truncated" in reason

    def test_a_corrupt_header_is_rejected(self, tmp_path):
        p = tmp_path / "adapter_model.safetensors"
        raw = b"{not json"
        p.write_bytes(len(raw).to_bytes(8, "little") + raw)
        ok, reason = readable_safetensors(p)
        assert ok is False
        assert "unreadable header" in reason

    def test_a_header_declaring_no_tensors_is_rejected(self, tmp_path):
        p = _write_safetensors(tmp_path / "a.safetensors", {"__metadata__": {}})
        ok, reason = readable_safetensors(p)
        assert ok is False
        assert "no tensors" in reason

    def test_a_missing_file_is_rejected(self, tmp_path):
        ok, reason = readable_safetensors(tmp_path / "absent.safetensors")
        assert ok is False
        assert "not found" in reason

    def test_an_adapter_directory_is_resolved(self, tmp_path):
        d = tmp_path / "adapter_final"
        d.mkdir()
        _write_safetensors(d / "adapter_model.safetensors", TENSOR)
        assert readable_safetensors(d)[0] is True

    def test_a_sharded_directory_is_resolved(self, tmp_path):
        d = tmp_path / "base"
        d.mkdir()
        _write_safetensors(d / "model-00001-of-00002.safetensors", TENSOR)
        assert readable_safetensors(d)[0] is True

    def test_the_check_does_not_read_the_payload(self, tmp_path):
        """Cost must not scale with file size - adapters are hundreds of MB."""
        p = _write_safetensors(
            tmp_path / "big.safetensors", TENSOR, payload=bytes(4 * 1024 * 1024)
        )
        assert readable_safetensors(p) == (True, "ok")


class TestVqaAvailabilityChecksReadability:
    def test_a_zeroed_adapter_is_reported_unavailable(self, tmp_path, monkeypatch):
        from satquery.tools import rs_vqa

        base = tmp_path / "base"
        base.mkdir()
        adapter = tmp_path / "adapter_final"
        adapter.mkdir()
        (adapter / "adapter_model.safetensors").write_bytes(bytes(8192))

        monkeypatch.setenv("SATQUERY_VQA_BASE", str(base))
        monkeypatch.setenv("SATQUERY_VQA_ADAPTER", str(adapter))
        ok, reason = rs_vqa.is_available()
        assert ok is False
        assert "zero-length header" in reason


STUB_CLASSES = ["rs_vqa_v1", "caption_v1", "change_caption_v1", "change_vqa_v1"]


def _stub(name):
    from satquery.tools import stubs

    return {
        "rs_vqa_v1": stubs.RSVQAStub,
        "caption_v1": stubs.CaptionStub,
        "change_caption_v1": stubs.ChangeCaptionStub,
        "change_vqa_v1": stubs.ChangeVQAStub,
    }[name]()


class TestStubsAnnounceThemselves:
    """A placeholder must not read like a result.

    `ChangeCaptionStub` returned "A new building was constructed in the
    center." - specific, plausible and fabricated - and that string was the
    recorded golden answer to the PS's own "what changed and where" query.
    """

    @pytest.mark.parametrize("name", STUB_CLASSES)
    def test_stub_text_is_marked(self, name, msi_6band, msi_6band_t2):
        from satquery.ingest import ingest
        from satquery.tools import stubs

        manifest = ingest([msi_6band, msi_6band_t2])
        result = _stub(name).run(manifest, {})
        text = " ".join(str(v) for v in result.payload.data.values())
        assert stubs.STUB_NOTICE in text, f"{name} emits unmarked text: {text!r}"

    @pytest.mark.parametrize("name", STUB_CLASSES)
    def test_stub_warns(self, name, msi_6band, msi_6band_t2):
        from satquery.ingest import ingest

        manifest = ingest([msi_6band, msi_6band_t2])
        result = _stub(name).run(manifest, {})
        assert any("placeholder" in w for w in result.warnings), name

    def test_no_stub_fabricates_a_specific_finding(self):
        """The two that mattered most asserted things that had not happened.

        Comments are excluded deliberately: the module documents the old
        strings so the reason for the change survives, and that record should
        not be what this test forbids.
        """
        source = Path("satquery/tools/stubs.py").read_text(encoding="utf-8")
        code = chr(10).join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        for fabricated in (
            "A new building was constructed in the center.",
            "Yes, there is a new road.",
            "fake answer string",
        ):
            assert fabricated not in code, fabricated
