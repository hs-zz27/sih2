"""Model provenance: the digests behind `Trace.weights_hashes`.

The field was declared in the contract from Phase 0 and emitted as `{}`
unconditionally, long after real checkpoints started loading. These tests
cover the three properties that make the replacement worth trusting:

* the digest is over the **actual bytes** and is deterministic;
* a **stub gets no digest at all**, so an empty map cannot be mistaken for a
  provenance claim about a fabricated answer;
* recording **never takes a run down**, whatever the filesystem does.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from satquery.tools import provenance


@pytest.fixture(autouse=True)
def clean_provenance():
    provenance.reset()
    yield
    provenance.reset()


class TestDigest:
    def test_a_file_digest_is_the_sha256_of_its_bytes(self, tmp_path):
        """Not a digest of the path, the size, or the mtime - the bytes."""
        blob = b"\x00weights\xff" * 1000
        target = tmp_path / "ckpt_step_10.pt"
        target.write_bytes(blob)

        assert provenance.sha256_of(target) == f"sha256:{hashlib.sha256(blob).hexdigest()}"

    def test_changing_one_byte_changes_the_digest(self, tmp_path):
        target = tmp_path / "ckpt.pt"
        target.write_bytes(b"a" * 100)
        before = provenance.sha256_of(target)

        target.write_bytes(b"a" * 99 + b"b")
        provenance.reset()  # defeat the mtime cache, which tmp_path can share
        after = provenance.sha256_of(target)

        assert before != after

    def test_a_directory_digest_is_stable_and_content_addressed(self, tmp_path):
        """An adapter is a directory. Two identical trees must agree."""
        for root in (tmp_path / "a", tmp_path / "b"):
            (root / "nested").mkdir(parents=True)
            (root / "adapter_model.safetensors").write_bytes(b"lora" * 64)
            (root / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
            (root / "nested" / "extra.bin").write_bytes(b"\x01\x02")

        assert provenance.sha256_of(tmp_path / "a") == provenance.sha256_of(tmp_path / "b")

    def test_a_renamed_file_inside_a_directory_changes_the_digest(self, tmp_path):
        """The manifest includes the relative path, so a rename is a change."""
        root = tmp_path / "adapter"
        root.mkdir()
        (root / "one.bin").write_bytes(b"same-bytes")
        before = provenance.sha256_of(root)

        (root / "one.bin").rename(root / "two.bin")
        provenance.reset()
        assert provenance.sha256_of(root) != before

    def test_a_missing_path_raises_rather_than_returning_a_hash(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            provenance.sha256_of(tmp_path / "absent.pt")


class TestRecording:
    def test_recording_makes_the_digest_available_by_tool_id(self, tmp_path):
        target = tmp_path / "ckpt.pt"
        target.write_bytes(b"weights")

        value = provenance.record("caption_v1", target)

        assert value is not None and value.startswith("sha256:")
        assert provenance.hashes_for(["caption_v1"]) == {"caption_v1": value}

    def test_a_tool_that_recorded_nothing_is_absent_not_empty_string(self, tmp_path):
        """A stub loads no bytes. It must not appear in the map at all.

        An entry with a placeholder value would put something that looks like
        provenance next to an answer that has none.
        """
        target = tmp_path / "ckpt.pt"
        target.write_bytes(b"weights")
        provenance.record("caption_v1", target)

        hashes = provenance.hashes_for(["caption_v1", "index_engine_v1", "rs_vqa_v1"])

        assert set(hashes) == {"caption_v1"}

    def test_recording_a_missing_checkpoint_records_nothing_and_does_not_raise(
        self, tmp_path
    ):
        """A checkpoint that vanishes mid-run must not take the answer down."""
        assert provenance.record("caption_v1", tmp_path / "gone.pt") is None
        assert provenance.hashes_for(["caption_v1"]) == {}

    def test_hashes_for_deduplicates_a_tool_that_ran_twice(self, tmp_path):
        target = tmp_path / "ckpt.pt"
        target.write_bytes(b"weights")
        provenance.record("change_mask_v1", target)

        hashes = provenance.hashes_for(["change_mask_v1", "change_mask_v1"])

        assert list(hashes) == ["change_mask_v1"]


class TestTraceIntegration:
    def test_a_stub_only_run_reports_no_weights_hashes(self, msi_4band):
        """The CI case. Every tool is a stub or deterministic; nothing loaded.

        This is also what keeps the golden traces valid: they were recorded
        with `weights_hashes: {}` and must stay that way under stubs.
        """
        from satquery.controller.pipeline import Controller

        trace = Controller().run([msi_4band], "Describe this image.")

        assert trace.weights_hashes == {}

    def test_a_recorded_digest_reaches_the_trace(self, msi_4band, tmp_path):
        """The wiring, exercised without needing torch or a real checkpoint.

        The tools record at load time; this records directly and asserts the
        executor carries the record for a tool that actually ran into the
        trace, and only for such a tool.
        """
        from satquery.controller.pipeline import Controller

        target = tmp_path / "ckpt.pt"
        target.write_bytes(b"pretend-weights")
        digest = provenance.record("index_engine_v1", target)
        # A tool that did NOT run in this plan, to prove the map is filtered
        # by what executed rather than by what the process has ever loaded.
        provenance.record("change_mask_v1", target)

        trace = Controller().run([msi_4band], "Describe this image.")

        ran = {step.tool for step in trace.execution}
        assert "index_engine_v1" in ran
        assert trace.weights_hashes == {"index_engine_v1": digest}
        assert "change_mask_v1" not in trace.weights_hashes


class TestToolsRecordOnLoad:
    """Every learned tool records; the stubs and template paths do not.

    Asserted against the source rather than by loading eight models, which
    needs torch and eight checkpoints. The claim being protected is narrow and
    structural: if a tool gains a load path with no `record(...)` beside it,
    its answers become unattributable and nothing else would notice.
    """

    LEARNED = {
        "caption.py": "caption_v1",
        "grounding.py": "grounding_v1",
        "change_caption.py": "change_caption_v1",
        "change_mask.py": "change_mask_v1",
        "optsar_fusion.py": "optsar_fusion_v1",
        "landcover.py": "landcover_v1",
        "rs_vqa.py": "rs_vqa_v1",
        "change_vqa.py": "change_vqa_v1",
    }

    @pytest.mark.parametrize("filename,tool_id", sorted(LEARNED.items()))
    def test_the_load_path_records_its_tool_id(self, filename, tool_id):
        source = (Path(__file__).resolve().parents[1] / "satquery" / "tools" / filename)
        assert f'record("{tool_id}"' in source.read_text(encoding="utf-8")

    def test_the_stub_module_records_nothing(self):
        source = (
            Path(__file__).resolve().parents[1] / "satquery" / "tools" / "stubs.py"
        )
        assert "record(" not in source.read_text(encoding="utf-8")
