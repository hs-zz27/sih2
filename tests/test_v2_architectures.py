"""The v2 models must be drop-in: same forward signature, same output shapes.

That is the property the whole Phase 5 design rests on. `--arch v2` changes the
model and nothing else - the data loaders, the losses, the metrics and the
`--eval-only` guards are the v1 code, unmodified, so a v2 number and a v1
number are measurements of the same thing. If a shape here drifts, the trainer
fails somewhere deep in a loop after the data is loaded, on a cluster, hours in.

Every shape below was read off the v1 model's actual call site, not assumed.
Two of them were assumed first and were wrong: `change_vqa` returns two
per-date SEGMENTATION maps rather than classification logits, and
`optsar_fusion` takes 4-band optical and 1-band SAR rather than 3 and 2.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.v2 import architectures as v2  # noqa: E402

B = 2


def params_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


# --- Shapes match the v1 contracts ------------------------------------------


def test_track_a_forward_matches_v1():
    model = v2.build_track_a(dim=32)
    out = model(torch.randn(B, 12, 120, 120), torch.ones(B, 12),
                torch.full((B,), 10.0))
    assert out.shape == (B, 19)


def test_track_a_runs_on_a_four_band_input():
    """The Cartosat case: eight of twelve bands absent.

    This is the property `docs/phase1-status.md` identifies as the actual
    enabler - a fixed-channel convolution could not run on 4 bands at all.
    """
    model = v2.build_track_a(dim=32)
    mask = torch.zeros(B, 12)
    mask[:, [1, 2, 3, 7]] = 1.0        # B02, B03, B04, B08
    out = model(torch.randn(B, 12, 120, 120), mask, torch.full((B,), 1.6))
    assert out.shape == (B, 19)
    assert torch.isfinite(out).all()


def test_track_a_masked_bands_cannot_affect_the_output():
    """A band that is masked off must be genuinely excluded, not down-weighted.

    Zero-filling teaches the model that "absent" means "zero reflectance",
    which is actively wrong - `docs/03` section 2.1. The masked mean is what
    prevents it, and this asserts the mechanism rather than trusting it.
    """
    model = v2.build_track_a(dim=32).eval()
    mask = torch.zeros(1, 12)
    mask[:, [1, 2, 3, 7]] = 1.0
    x = torch.randn(1, 12, 120, 120)
    noise = x.clone()
    absent = [i for i in range(12) if mask[0, i] == 0]
    noise[:, absent] = torch.randn(1, len(absent), 120, 120) * 50

    with torch.no_grad():
        a = model(x, mask, torch.tensor([10.0]))
        b = model(noise, mask, torch.tensor([10.0]))
    assert torch.allclose(a, b, atol=1e-4)


def test_track_a_gsd_conditioning_changes_the_answer():
    """Conditioning that no output depends on is decoration, not conditioning."""
    model = v2.build_track_a(dim=32, gsd_conditioning=True).eval()
    # Zero-init FiLM makes an untrained model scale-invariant by design, so
    # perturb the conditioning weights to represent a trained one.
    with torch.no_grad():
        for film in model.films:
            film.mlp[-1].weight.normal_(0, 0.05)
            film.mlp[-1].bias.normal_(0, 0.05)
        x, mask = torch.randn(1, 12, 120, 120), torch.ones(1, 12)
        coarse = model(x, mask, torch.tensor([10.0]))
        fine = model(x, mask, torch.tensor([1.6]))
    assert not torch.allclose(coarse, fine)


def test_track_a_without_gsd_ignores_the_argument():
    model = v2.build_track_a(dim=32, gsd_conditioning=False).eval()
    with torch.no_grad():
        x, mask = torch.randn(1, 12, 120, 120), torch.ones(1, 12)
        assert torch.allclose(model(x, mask, torch.tensor([10.0])),
                              model(x, mask, torch.tensor([1.6])))


def test_grounding_forward_matches_v1():
    model = v2.build_grounding(vocab_size=85, dim=64)
    box = model(torch.randn(B, 3, 224, 224), torch.randint(0, 85, (B, 16)))
    assert box.shape == (B, 4)
    # Normalised cx,cy,w,h - the v1 convention, and what keeps the loss
    # independent of image size.
    assert ((box >= 0) & (box <= 1)).all()


def test_grounding_output_depends_on_where_the_object_is():
    """The defect v2 exists to fix, asserted directly.

    v1 global-average-pooled before regressing, so the box could not depend on
    position: Acc@0.5 came out at 0.0762 against a published 0.70-0.80. Move
    the content, and the prediction must move.
    """
    model = v2.build_grounding(vocab_size=85, dim=64).eval()
    tokens = torch.randint(1, 85, (1, 16))
    image = torch.zeros(1, 3, 224, 224)
    image[:, :, 20:60, 20:60] = 1.0
    moved = torch.zeros(1, 3, 224, 224)
    moved[:, :, 160:200, 160:200] = 1.0

    with torch.no_grad():
        left = model(image, tokens)
        right = model(moved, tokens)
    assert not torch.allclose(left, right, atol=1e-3)


def test_grounding_heatmap_is_optional_and_spatial():
    model = v2.build_grounding(vocab_size=85, dim=64)
    box, heat = model(torch.randn(B, 3, 224, 224), torch.randint(0, 85, (B, 16)),
                      return_heatmap=True)
    assert box.shape == (B, 4)
    assert heat.shape == (B, 14, 14)


def test_change_mask_returns_input_resolution_logits():
    model = v2.build_change_mask(dim=16)
    out = model(torch.randn(B, 3, 256, 256), torch.randn(B, 3, 256, 256))
    assert out.shape == (B, 1, 256, 256)


def test_change_mask_is_symmetric_in_the_two_dates():
    """Swapping the dates must change the sign of the change, not its magnitude.

    v1 got this right with an absolute difference and said so; v2 keeps it,
    and a shared encoder is what makes it true.
    """
    model = v2.build_change_mask(dim=16).eval()
    a, b = torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        assert torch.allclose(model(a, b), model(b, a), atol=1e-5)


def test_change_mask_on_identical_dates_is_uniform():
    """No change in, no structure out - |f(a) - f(a)| is exactly zero."""
    model = v2.build_change_mask(dim=16).eval()
    a = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(a, a)
    assert out.std() < 1e-4


def test_change_caption_forward_matches_v1():
    model = v2.build_change_caption(vocab_size=300, dim=64)
    out = model(torch.randn(B, 3, 256, 256), torch.randn(B, 3, 256, 256),
                torch.rand(B, 1, 256, 256), torch.randint(1, 300, (B, 20)))
    assert out.shape == (B, 20, 300)


def test_change_caption_accepts_a_mask_without_a_channel_axis():
    model = v2.build_change_caption(vocab_size=300, dim=64)
    out = model(torch.randn(B, 3, 256, 256), torch.randn(B, 3, 256, 256),
                torch.rand(B, 256, 256), torch.randint(1, 300, (B, 20)))
    assert out.shape == (B, 20, 300)


def test_change_caption_is_causal():
    """Without a causal mask each position sees the token it must predict.

    Training loss then collapses to zero and generation produces nothing -
    a failure that looks like success right up until inference.
    """
    model = v2.build_change_caption(vocab_size=300, dim=64).eval()
    a, b = torch.randn(1, 3, 128, 128), torch.randn(1, 3, 128, 128)
    mask = torch.rand(1, 1, 128, 128)
    tokens = torch.randint(1, 300, (1, 10))
    changed = tokens.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 300

    with torch.no_grad():
        first = model(a, b, mask, tokens)
        second = model(a, b, mask, changed)
    # Altering the LAST token must not move any earlier position's logits.
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-5)


def test_change_vqa_returns_two_per_date_segmentations():
    """Not classification. v1 returns two (B,C,H,W) maps and argmaxes dim 1."""
    model = v2.build_change_vqa(dim=16, n_classes=7)
    p1, p2 = model(torch.randn(B, 3, 256, 256), torch.randn(B, 3, 256, 256))
    assert p1.shape == (B, 7, 256, 256)
    assert p2.shape == (B, 7, 256, 256)


def test_change_vqa_decoders_are_not_shared():
    """The same pixel is 'buildings' at one date and 'trees' at the other.

    v1 reasoned this out and used two decoders. If they were shared, the two
    outputs would be identical whenever the two dates were.
    """
    model = v2.build_change_vqa(dim=16, n_classes=7).eval()
    a = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        p1, p2 = model(a, a)
    assert not torch.allclose(p1, p2, atol=1e-4)


def test_optsar_fusion_takes_four_band_optical_and_one_band_sar():
    model = v2.build_optsar_fusion(dim=16, n_classes=7)
    lo, ls, lf = model(torch.randn(B, 4, 128, 128), torch.randn(B, 1, 128, 128))
    assert lo.shape == ls.shape == lf.shape == (B, 7)


def test_optsar_per_stream_heads_see_only_their_own_stream():
    """The ablation the three outputs exist to support has to stay honest.

    If the optical head's answer moved when only the SAR input changed,
    'optical alone' would not mean optical alone and the complementarity gain
    the trainer reports would be meaningless.
    """
    model = v2.build_optsar_fusion(dim=16, n_classes=7).eval()
    optical = torch.randn(1, 4, 128, 128)
    with torch.no_grad():
        lo_a, _, lf_a = model(optical, torch.randn(1, 1, 128, 128))
        lo_b, _, lf_b = model(optical, torch.randn(1, 1, 128, 128))
    assert torch.allclose(lo_a, lo_b, atol=1e-5), "optical head saw the SAR stream"
    assert not torch.allclose(lf_a, lf_b, atol=1e-5), "fused head ignored the SAR stream"


def test_caption_forward_matches_v1():
    model = v2.build_caption(vocab_size=1781, dim=96)
    out = model(torch.randn(B, 3, 224, 224), torch.randint(1, 1781, (B, 24)))
    assert out.shape == (B, 24, 1781)


# --- The registry and the trainer wiring ------------------------------------


def test_every_builder_is_reachable_by_name():
    assert set(v2.BUILDERS) == {
        "track_a", "grounding", "change_mask", "change_caption",
        "change_vqa", "optsar_fusion", "caption",
    }


def test_unknown_task_names_the_ones_that_exist():
    with pytest.raises(KeyError, match="track_a"):
        v2.build("not_a_task")


@pytest.mark.parametrize("module_name,kwargs", [
    ("training.track_a_full", {"dim": 32}),
    ("training.train_grounding", {"vocab_size": 50, "dim": 64}),
    ("training.train_change_mask", {"dim": 16}),
    ("training.train_change_caption", {"vocab_size": 50, "dim": 64}),
    ("training.train_change_vqa", {"dim": 16}),
    ("training.train_optsar_fusion", {"dim": 16}),
    ("training.train_caption", {"vocab_size": 50, "dim": 64}),
])
def test_trainers_default_to_v1_and_can_select_v2(module_name, kwargs):
    """The default must stay v1.

    Every checkpoint on disk was written by a v1 model and is rebuilt by
    calling `build_model` with the recorded hyperparameters. A default flip
    would rebuild them all with the wrong architecture and load partially -
    which is how a randomly-initialised encoder gets reported as fine-tuned.
    """
    import importlib

    module = importlib.import_module(module_name)
    v1_model = module.build_model(**kwargs)
    v2_model = module.build_model(**kwargs, arch="v2")
    assert type(v1_model).__name__ != type(v2_model).__name__
    assert params_m(v2_model) > params_m(v1_model)


def test_v2_models_produce_finite_gradients():
    """A NaN on the first backward is cheap here and expensive on a cluster."""
    model = v2.build_change_mask(dim=16)
    model(torch.randn(B, 3, 128, 128), torch.randn(B, 3, 128, 128)).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


# --- The whole round trip ---------------------------------------------------


@pytest.mark.parametrize("arch,dim", [("v1", 16), ("v2", 24)])
def test_checkpoint_round_trips_through_the_real_tool_loader(tmp_path, arch, dim):
    """Save with the real saver, load with the real tool. Both architectures.

    This is the integration the rest of Phase 5 rests on. The trainer records
    `arch` in the checkpoint's `extra`; the tool reads it back and rebuilds the
    matching graph. Get it wrong in either direction and `load_state_dict`
    either raises on a key mismatch or - far worse - loads partially and leaves
    half the model randomly initialised, which is how an untrained encoder gets
    reported as fine-tuned.
    """
    from satquery.tools.change_mask import _Handle
    from training.common.checkpointing import save_checkpoint
    from training.train_change_mask import build_model

    model = build_model(dim, arch=arch)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    save_checkpoint(tmp_path, 10, model, optimizer,
                    extra={"arch": arch, "dim": dim})

    loaded = _Handle(tmp_path).model
    assert type(loaded).__name__ == type(model).__name__

    saved_state, loaded_state = model.state_dict(), loaded.state_dict()
    assert set(saved_state) == set(loaded_state)
    for key, value in saved_state.items():
        assert torch.allclose(value.float().cpu(), loaded_state[key].float().cpu()), key


def test_checkpoint_without_an_arch_field_loads_as_v1(tmp_path):
    """Every checkpoint written before Phase 5 has no `arch`. All must still load."""
    from satquery.tools.change_mask import _Handle
    from training.common.checkpointing import save_checkpoint
    from training.train_change_mask import build_model

    model = build_model(16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    save_checkpoint(tmp_path, 10, model, optimizer, extra={"dim": 16})

    assert type(_Handle(tmp_path).model).__name__ == type(model).__name__


# --- Generation: the method forward-shape tests do not reach ----------------


def test_caption_v2_can_generate():
    """The gap that cost two completed runs their evaluation.

    Both caption models trained their full schedule and then died in the
    evaluator with `'CaptionV2' object has no attribute 'generate'`. The
    forward contract was checked; the method the evaluator actually calls at
    the end was not. Shape and dtype here match what v1 returned.
    """
    model = v2.build_caption(vocab_size=200, dim=96, bos_id=1, max_len=24)
    ids = model.generate(torch.randn(B, 3, 224, 224))
    assert ids.shape == (B, 24)
    assert ids.dtype == torch.long
    assert (ids >= 0).all() and (ids < 200).all()


def test_change_caption_v2_can_generate():
    model = v2.build_change_caption(vocab_size=200, dim=64, bos_id=1, max_len=24)
    ids = model.generate(torch.randn(B, 3, 256, 256), torch.randn(B, 3, 256, 256),
                         torch.rand(B, 1, 256, 256))
    assert ids.shape == (B, 24)
    assert ids.dtype == torch.long


def test_generate_respects_max_len():
    model = v2.build_caption(vocab_size=200, dim=96, bos_id=1, max_len=24)
    assert model.generate(torch.randn(1, 3, 224, 224), max_len=7).shape == (1, 7)


def test_generate_does_not_return_the_seed_token():
    """v1 returned only produced tokens; `decode` assumes no leading BOS."""
    bos = 5
    model = v2.build_caption(vocab_size=200, dim=96, bos_id=bos, max_len=6).eval()
    ids = model.generate(torch.randn(4, 3, 224, 224))
    assert ids.shape == (4, 6)
    # A model that returned its seed would start every row with `bos`.
    assert not (ids[:, 0] == bos).all()


@pytest.mark.parametrize("task,builder", [
    ("caption", "build_caption"),
    ("change_caption", "build_change_caption"),
])
def test_v2_captioners_expose_every_method_v1_had(task, builder):
    """Guards the whole class of bug, not just this instance.

    A v2 model is a drop-in only if the trainer can call everything on it that
    it called on v1. `forward` was verified for all seven; `generate` was not,
    and only the two captioners have it.
    """
    model = getattr(v2, builder)(vocab_size=50, dim=64)
    for name in ("forward", "generate"):
        assert callable(getattr(model, name, None)), f"{task} lacks {name}()"


# --- Pretrained backbones ---------------------------------------------------


def test_pretrained_backbone_is_a_drop_in_for_the_scratch_one():
    """Same interface, so either can be swapped in without touching consumers."""
    pytest.importorskip("torchvision")
    backbone = v2.build_pretrained_backbone(cin=3)
    feats = backbone(torch.randn(1, 3, 224, 224))
    assert len(feats) == 4
    assert [f.shape[1] for f in feats] == backbone.widths
    assert backbone.out_dim == backbone.widths[-1]


def test_pretrained_backbone_adapts_to_a_different_band_count():
    """12-band input must reuse the RGB filters, not discard them.

    Re-initialising conv1 randomly would throw away exactly the thing a
    pretrained backbone is for.
    """
    pytest.importorskip("torchvision")
    backbone = v2.build_pretrained_backbone(cin=12)
    out = backbone(torch.randn(1, 12, 120, 120))
    assert out[-1].shape[1] == 2048
    assert torch.isfinite(out[-1]).all()


@pytest.mark.parametrize("builder,kwargs,shape", [
    ("build_grounding", {"vocab_size": 85, "dim": 128}, (2, 4)),
    ("build_caption", {"vocab_size": 200, "dim": 192}, (2, 24, 200)),
])
def test_pretrained_variants_keep_the_v1_forward_contract(builder, kwargs, shape):
    pytest.importorskip("torchvision")
    model = getattr(v2, builder)(**kwargs, pretrained=True)
    image = torch.randn(2, 3, 224, 224)
    if builder == "build_grounding":
        out = model(image, torch.randint(0, 85, (2, 16)))
    else:
        out = model(image, torch.randint(1, 200, (2, 24)))
    assert out.shape == shape


def test_the_projection_keeps_the_model_from_ballooning():
    """ResNet-50 is 2048 wide; sizing the decoder off that is the trap.

    Without a projection the pretrained caption model was 234M parameters on
    8,734 RSICD captions - a decoder eight times larger than the one that
    already overfits. The pretrained FEATURES are the point, not the width.
    """
    pytest.importorskip("torchvision")
    scratch = v2.build_caption(vocab_size=1781, dim=192, pretrained=False)
    pretrained = v2.build_caption(vocab_size=1781, dim=192, pretrained=True)
    ratio = params_m(pretrained) / params_m(scratch)
    assert ratio < 2.5, f"pretrained model is {ratio:.1f}x the scratch one"


@pytest.mark.parametrize("record_flag", [True, False])
def test_pretrained_caption_round_trips_through_the_tool(tmp_path, record_flag):
    """The checkpoint must carry `pretrained`, or the tool must infer it.

    Found by scripts/verify_v2_deploy.py on the lab box: grounding_pre and
    caption_pre both failed to load with a state-dict size mismatch, because
    the tool rebuilt the FROM-SCRATCH v2 (width 512) for weights written by
    the pretrained one (width 576). `--pretrained` selects a different
    backbone and so must travel with the weights, exactly like `arch`.

    `record_flag=False` is the two checkpoints already on disk, written
    before the flag was recorded: the tool infers it from the `proj.` keys
    only the pretrained variant has.
    """
    pytest.importorskip("torchvision")
    from satquery.tools.caption import _Handle
    from training.common.checkpointing import save_checkpoint
    from training.train_caption import build_model

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "a": 4, "b": 5}
    (tmp_path / "vocab.json").write_text(json_dumps(vocab), encoding="utf-8")

    model = build_model(len(vocab), 96, arch="v2", pretrained=True)
    extra = {"arch": "v2", "dim": 96}
    if record_flag:
        extra["pretrained"] = True
    save_checkpoint(tmp_path, 10, model, torch.optim.SGD(model.parameters(), lr=0.1),
                    extra=extra)

    loaded = _Handle(tmp_path).model
    assert type(loaded).__name__ == "CaptionV2"
    assert hasattr(loaded, "proj") and not isinstance(loaded.proj, torch.nn.Identity)
    saved, got = model.state_dict(), loaded.state_dict()
    assert set(saved) == set(got)


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj)
