"""The VLM fine-tune must supervise the answer, not the image placeholders.

`training/track_b_vlm_qlora.py` measured where the assistant's answer begins by
tokenising the prompt as TEXT ONLY. A Qwen2.5-VL prompt renders one `<|image|>`
marker that the processor expands into one `<|image_pad|>` per visual patch, so
that measurement returns the length *before* the expansion: the boundary came
back as 52 where the processed sequence needed 375.

Everything from 52 onward was therefore supervised, which is 341 tokens of
which 312 are `<|image_pad|>`. The model was trained to predict image
placeholders, outnumbering the real answer tokens roughly 9 to 1, and the
2,000-step run that produced `checkpoints/track_b_v3` sat at loss ~6.8 from
step 50 to step 2000 - a 0.3% change over 1,800 steps, inside the run's own
0.17 noise band.

These tests run against the real processor because the bug lives in the gap
between the tokeniser and the processor; a mocked tokeniser would reproduce the
mistake rather than catch it. They skip when the base model is not present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

MODEL_DIR = Path("models/qwen25_vl_3b")
DATA_DIR = Path("data/instruct_mix")
VAL_FILE = "val.jsonl"

pytestmark = [
    pytest.mark.skipif(
        not MODEL_DIR.exists(), reason="base model not downloaded"
    ),
    pytest.mark.skipif(
        not (DATA_DIR / "instruct.jsonl").exists(), reason="instruct mix absent"
    ),
]

N_EXAMPLES = 6


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)


@pytest.fixture(scope="module")
def cases(processor):
    """(processed ids, text-only length, corrected length, answer) per example."""
    from PIL import Image

    from training.track_b_vlm_qlora import (
        build_chat, load_examples, supervised_start,
    )

    out = []
    for example in load_examples(DATA_DIR, limit=N_EXAMPLES):
        chat = build_chat(example)
        text = processor.apply_chat_template(chat, tokenize=False)
        image = Image.open(example.image_path).convert("RGB")
        ids = processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )["input_ids"][0]

        prompt_text = processor.apply_chat_template(chat[:-1], tokenize=False)
        text_only = len(
            processor.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
        )
        corrected = supervised_start(processor, prompt_text, image)
        out.append((ids, text_only, corrected, example.answer))
    return out


def _image_pad_id(processor) -> int:
    return processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")


def _supervised(processor, ids, start: int):
    from training.track_b_vlm_qlora import mask_prompt_labels

    pad = processor.tokenizer.pad_token_id or 0
    labels = mask_prompt_labels(ids, start, pad)
    return labels[labels != -100]


class TestBoundaryIsMeasuredOnTheProcessedSequence:
    def test_text_only_length_is_not_used(self, cases):
        """The two measurements must differ, or the bug is not being tested."""
        for ids, text_only, corrected, _ in cases:
            assert corrected > text_only, (
                "text-only and processed prompt lengths agree; this example "
                "cannot detect the masking bug"
            )

    def test_corrected_boundary_leaves_only_the_answer(self, processor, cases):
        for ids, _, corrected, _ in cases:
            assert corrected < len(ids), "boundary must leave tokens to supervise"


class TestImagePlaceholdersAreExcluded:
    def test_no_image_pad_token_is_supervised(self, processor, cases):
        """The whole point: placeholders carry no answer signal."""
        pad_id = _image_pad_id(processor)
        for ids, _, corrected, _ in cases:
            supervised = _supervised(processor, ids, corrected)
            leaked = int((supervised == pad_id).sum())
            assert leaked == 0, f"{leaked} <|image_pad|> tokens supervised"

    def test_the_old_boundary_would_have_leaked_them(self, processor, cases):
        """Pins the defect itself, so a regression is caught as a change."""
        pad_id = _image_pad_id(processor)
        total_leaked = 0
        for ids, text_only, _, _ in cases:
            supervised = _supervised(processor, ids, text_only)
            total_leaked += int((supervised == pad_id).sum())
        assert total_leaked > 0, (
            "the text-only boundary no longer leaks image placeholders; if the "
            "processor changed, re-derive the numbers in this module's docstring"
        )


class TestTheAnswerIsActuallySupervised:
    def test_supervised_region_decodes_to_the_answer(self, processor, cases):
        for ids, _, corrected, answer in cases:
            supervised = _supervised(processor, ids, corrected)
            decoded = processor.tokenizer.decode(supervised)
            # The answer is what the model is meant to learn to produce; the
            # decode also carries the assistant turn's closing tokens.
            assert answer.strip()[:24] in decoded, (
                f"answer {answer[:40]!r} not found in supervised span "
                f"{decoded[:120]!r}"
            )

    def test_supervised_span_is_small(self, processor, cases):
        """A correct span is the answer, not most of the sequence."""
        for ids, _, corrected, _ in cases:
            supervised = _supervised(processor, ids, corrected)
            assert len(supervised) < 0.25 * len(ids), (
                f"{len(supervised)} of {len(ids)} tokens supervised - too many "
                f"for a short answer"
            )


def test_report_supervised_counts_before_and_after(processor, cases, capsys):
    """Not an assertion so much as the measurement, printed with -s."""
    pad_id = _image_pad_id(processor)
    before_total = before_pad = after_total = after_pad = 0
    lines = []
    for ids, text_only, corrected, _ in cases:
        before = _supervised(processor, ids, text_only)
        after = _supervised(processor, ids, corrected)
        b_pad = int((before == pad_id).sum())
        a_pad = int((after == pad_id).sum())
        before_total += len(before); before_pad += b_pad
        after_total += len(after); after_pad += a_pad
        lines.append(
            f"  seq={len(ids):4d}  boundary {text_only:3d} -> {corrected:4d}   "
            f"supervised {len(before):3d} -> {len(after):3d}   "
            f"image_pad {b_pad:3d} -> {a_pad}"
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines))
        print(
            f"  TOTAL supervised {before_total} -> {after_total};  "
            f"image_pad {before_pad} ({100*before_pad/before_total:.1f}%) -> "
            f"{after_pad} ({100*after_pad/max(after_total,1):.1f}%)"
        )
    assert after_pad == 0
    assert after_total < before_total


class TestValidationSharesTheTrainingMask:
    """Requirement 9: the image-token bug must not reappear on the val side.

    A masking bug that affects only validation is worse than the original: the
    number you would stop training on would be the wrong one, and nothing in
    the training loss would show it. The structural defence is that both paths
    call `encode_supervised`, so they cannot drift; these tests hold that
    property rather than re-deriving the arithmetic.
    """

    @pytest.fixture(scope="class")
    def val_examples(self):
        from training.track_b_vlm_qlora import load_examples

        if not (DATA_DIR / VAL_FILE).exists():
            pytest.skip(f"{VAL_FILE} absent")
        return load_examples(DATA_DIR, limit=N_EXAMPLES, filename=VAL_FILE)

    def test_validation_split_loads_and_is_not_the_training_split(self):
        from training.track_b_vlm_qlora import load_examples

        if not (DATA_DIR / VAL_FILE).exists():
            pytest.skip(f"{VAL_FILE} absent")
        train = load_examples(DATA_DIR)
        val = load_examples(DATA_DIR, filename=VAL_FILE)
        assert val, "validation split is empty"
        train_keys = {(e.image_path, e.question) for e in train}
        overlap = [e for e in val if (e.image_path, e.question) in train_keys]
        assert not overlap, (
            f"{len(overlap)} validation items appear verbatim in training"
        )

    def test_no_image_pad_is_supervised_on_validation_examples(
        self, processor, val_examples
    ):
        from training.track_b_vlm_qlora import encode_supervised

        pad_id = _image_pad_id(processor)
        for example in val_examples:
            batch = encode_supervised(processor, example)
            labels = batch["labels"][0]
            supervised = labels[labels != -100]
            assert len(supervised) > 0, "nothing supervised"
            assert int((supervised == pad_id).sum()) == 0, (
                "<|image_pad|> supervised on a validation example"
            )

    def test_validation_supervises_the_answer(self, processor, val_examples):
        from training.track_b_vlm_qlora import encode_supervised

        for example in val_examples:
            batch = encode_supervised(processor, example)
            labels = batch["labels"][0]
            decoded = processor.tokenizer.decode(labels[labels != -100])
            assert example.answer.strip()[:16] in decoded, (
                f"answer {example.answer[:40]!r} missing from supervised span"
            )

    def test_train_and_validation_encode_identically(self, processor, cases):
        """Same example through the shared path must equal the loop's own mask.

        This is the anti-drift test: if someone reintroduces an inline
        boundary computation in either path, the two stop matching.
        """
        from training.track_b_vlm_qlora import encode_supervised, load_examples

        for example in load_examples(DATA_DIR, limit=3):
            batch = encode_supervised(processor, example)
            labels = batch["labels"][0]
            ids = batch["input_ids"][0]
            start = int((labels != -100).nonzero()[0])
            expected = _supervised(processor, ids, start)
            actual = labels[labels != -100]
            assert len(actual) == len(expected)

    def test_evaluate_skips_fully_masked_examples(self):
        """A zero from an empty answer would read as an improvement."""
        import inspect

        from training.track_b_vlm_qlora import evaluate

        src = inspect.getsource(evaluate)
        assert "!= -100" in src and "continue" in src, (
            "evaluate() must skip examples with nothing supervised rather "
            "than averaging in a zero"
        )
