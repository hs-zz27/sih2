"""Mask-conditioned change captioning tests (plan task 2.5)."""

from __future__ import annotations

import numpy as np
import pytest

# torch is a TRAINING dependency, not a runtime one: the ingest pipeline,
# index engine, controller, API and evidence pack all work without it. CI
# therefore does not install it, and these tests skip there rather than
# adding an ~800 MB download to every run. They do execute locally, where
# the training environment exists.
pytest.importorskip("torch")
import torch

from training.train_change_caption import (
    BOS, EOS, MAX_LEN, PAD, UNK, build_model, build_vocab, decode, encode, tokenize,
)


class TestVocabulary:
    def test_specials_occupy_the_first_slots(self):
        v = build_vocab(["a road appeared", "a road appeared"], min_count=1)
        assert v["<pad>"] == PAD and v["<bos>"] == BOS
        assert v["<eos>"] == EOS and v["<unk>"] == UNK

    def test_rare_words_dropped(self):
        v = build_vocab(["common common", "common rare"], min_count=2)
        assert "common" in v and "rare" not in v

    def test_encode_wraps_with_bos_and_eos(self):
        v = build_vocab(["a road appeared"], min_count=1)
        ids = encode("a road appeared", v)
        assert ids[0] == BOS
        assert EOS in ids.tolist()

    def test_encode_pads_to_fixed_length(self):
        v = build_vocab(["short"], min_count=1)
        assert len(encode("short", v)) == MAX_LEN

    def test_long_caption_truncated_but_still_terminated(self):
        v = build_vocab([" ".join(["word"] * 100)], min_count=1)
        ids = encode(" ".join(["word"] * 100), v)
        assert len(ids) == MAX_LEN
        assert ids[-1] == EOS

    def test_unknown_word_maps_to_unk(self):
        v = build_vocab(["known known"], min_count=1)
        assert UNK in encode("mystery", v).tolist()

    def test_decode_round_trips(self):
        v = build_vocab(["a new road was built"], min_count=1)
        inverse = {i: w for w, i in v.items()}
        assert decode(encode("a new road was built", v), inverse) == "a new road was built"

    def test_decode_stops_at_eos(self):
        v = build_vocab(["one two"], min_count=1)
        inverse = {i: w for w, i in v.items()}
        ids = [BOS, v["one"], EOS, v["two"]]
        assert decode(ids, inverse) == "one"


class TestModel:
    @staticmethod
    def _batch(n=2, s=64):
        g = torch.Generator().manual_seed(4)
        return (
            torch.rand(n, 3, s, s, generator=g),
            torch.rand(n, 3, s, s, generator=g),
            torch.rand(n, 1, s, s, generator=g),
        )

    def test_forward_shape(self):
        model = build_model(vocab_size=30, dim=16).eval()
        a, b, m = self._batch()
        tokens = torch.zeros(2, 5, dtype=torch.long)
        assert model(a, b, m, tokens).shape == (2, 5, 30)

    def test_mask_changes_the_caption_features(self):
        """Mask conditioning must actually condition; an ignored mask would
        make 'mask-conditioned' a claim with nothing behind it."""
        model = build_model(vocab_size=30, dim=16).eval()
        a, b, m = self._batch()
        empty = torch.zeros_like(m)
        assert not torch.allclose(
            model.features(a, b, m), model.features(a, b, empty), atol=1e-5
        )

    def test_symmetric_in_date_order(self):
        """Absolute difference means swapping dates changes the sign of the
        change, not its magnitude."""
        model = build_model(vocab_size=30, dim=16).eval()
        a, b, m = self._batch()
        assert torch.allclose(model.features(a, b, m), model.features(b, a, m), atol=1e-5)

    def test_generate_produces_token_ids(self):
        model = build_model(vocab_size=30, dim=16).eval()
        a, b, m = self._batch()
        out = model.generate(a, b, m, max_len=7)
        assert out.shape == (2, 7)
        assert out.min() >= 0 and out.max() < 30

    def test_empty_mask_still_captions(self):
        """A pair with no detected change must still produce output rather
        than failing - "there is no difference" is a valid caption."""
        model = build_model(vocab_size=30, dim=16).eval()
        a, b, _ = self._batch()
        out = model.generate(a, b, torch.zeros(2, 1, 64, 64), max_len=5)
        assert out.shape == (2, 5)
