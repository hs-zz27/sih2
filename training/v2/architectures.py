"""Drop-in v2 models: same forward signatures, enough capacity to learn the task.

WHY NEW ARCHITECTURES AND NOT LONGER TRAINING

The v1 models are not undertrained. They are too small, and in two cases the
architecture actively discards what the task needs:

| tool | v1 measured | published range | the actual cause |
|---|---|---|---|
| `landcover_v1` | mAP 0.2854 | ~0.65-0.85 | a 4-layer CNN at dim 64, 30k patches |
| `grounding_v1` | Acc@0.5 0.0762 | ~0.70-0.80 | **global-average-pools before regressing the box** |

`docs/model-cards.md` names the grounding cause outright: pooling the feature
map to a vector throws away every pixel coordinate, so the model can only
learn an average box. No amount of GPU time fixes that; the pooling has to go.
The land-cover case is plainer - a 0.42M-parameter model on half the prepared
patches, which is a capacity problem before it is a data problem.

So each model here changes exactly what the measurement said was wrong, and
nothing else. The forward signature, the loss, the data loader and the metric
are unchanged, which is what makes v1 and v2 comparable at all.

WHAT IS DELIBERATELY NOT HERE

No `trust_remote_code`. `training/train_grounding.py` records the decision to
build the fallback grounder rather than execute third-party Python fetched
from a model repo, and `scripts/fetch_models.py` excludes those files on
purpose. A bigger GPU is not a reason to reverse a security decision, so the
v2 grounder is still built here, from parts - it just no longer throws away
the spatial information.

No `timm` or `torchgeo` dependency either. Both were considered: pretrained
Sentinel-2 SSL weights would plausibly beat a from-scratch encoder. They are
not used because the campaign has to run on a cluster where the notebook may
have no outbound network, and a run that dies at `from_pretrained` after the
data is staged is the worst possible failure. The encoder is trained from
scratch on the 60,000 prepared BigEarthNet patches. That is a real caveat
rather than a comfortable one: 60k is enough for a 5M-parameter encoder but
it is not the ~549k of full BigEarthNet v2, so the published 0.65-0.85 range
is not a like-for-like target.

That trade is worth stating rather than presenting as free: on the two small
corpora it is probably the wrong one. SECOND has ~1,600 training pairs and v1
already measured a from-scratch encoder there at 0.1691 change-class mIoU
against a pretrained ResNet-18 that fixed it, which is why `configs/campaign.yaml`
runs `change_vqa` with `--pretrained` and treats the v2 arm as an ablation.

CONVENTIONS

Every builder returns a module whose `forward` matches its v1 counterpart
exactly, so the trainers call it at the same single call site. Sizes are
chosen so the largest model here is ~25M parameters - big enough to be a real
model, small enough that a full run is hours rather than days.
"""

from __future__ import annotations

import math


# --- Shared parts -----------------------------------------------------------


def _blocks():
    """Building blocks, imported lazily so this module is importable without torch.

    The whole repository follows this rule: torch is a training dependency and
    CI does not install it, so anything importable at module scope in a file
    the tests touch must not need it.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualBlock(nn.Module):
        """Pre-activation residual block. The depth v1 did not have."""

        def __init__(self, cin: int, cout: int, stride: int = 1):
            super().__init__()
            self.norm1 = nn.BatchNorm2d(cin)
            self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
            self.norm2 = nn.BatchNorm2d(cout)
            self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
            self.skip = (
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
                if (stride != 1 or cin != cout) else nn.Identity()
            )

        def forward(self, x):
            h = self.conv1(F.relu(self.norm1(x)))
            h = self.conv2(F.relu(self.norm2(h)))
            return h + self.skip(x)

    class Backbone(nn.Module):
        """A real four-stage residual trunk that keeps its feature map.

        `forward` returns the stages, not a pooled vector. Every consumer here
        needs spatial structure - the grounder needs it to place a box, the
        change decoder needs it to draw a mask - and v1's habit of pooling
        early is the defect this exists to avoid repeating.
        """

        def __init__(self, cin: int = 3, dim: int = 64, depths=(2, 2, 2, 2)):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(cin, dim, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            )
            widths = [dim, dim * 2, dim * 4, dim * 8]
            self.stages = nn.ModuleList()
            prev = dim
            for i, (width, depth) in enumerate(zip(widths, depths)):
                layers = [ResidualBlock(prev, width, stride=1 if i == 0 else 2)]
                layers += [ResidualBlock(width, width) for _ in range(depth - 1)]
                self.stages.append(nn.Sequential(*layers))
                prev = width
            self.widths = widths
            self.out_dim = widths[-1]

        def forward(self, x):
            x = self.stem(x)
            feats = []
            for stage in self.stages:
                x = stage(x)
                feats.append(x)
            return feats

    class TextEncoder(nn.Module):
        """Transformer over tokens, returning per-token states AND a summary.

        v1 used a GRU and kept only the final hidden state. Keeping the token
        states is what lets the grounder attend to "the aircraft on the left"
        word by word instead of compressing the phrase to one vector before it
        has seen the image.
        """

        def __init__(self, vocab_size: int, dim: int = 256, layers: int = 2,
                     heads: int = 4, max_tokens: int = 64, pad_id: int = 0):
            super().__init__()
            self.pad_id = pad_id
            self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
            self.pos = nn.Parameter(torch.zeros(1, max_tokens, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                dim, heads, dim * 4, dropout=0.1, batch_first=True,
                norm_first=True, activation="gelu",
            )
            # norm_first makes the nested-tensor fast path inapplicable;
            # asking for it only produces a warning per construction.
            self.encoder = nn.TransformerEncoder(layer, layers,
                                                 enable_nested_tensor=False)
            self.norm = nn.LayerNorm(dim)

        def forward(self, tokens):
            pad = tokens == self.pad_id
            x = self.embed(tokens) + self.pos[:, : tokens.shape[1]]
            x = self.norm(self.encoder(x, src_key_padding_mask=pad))
            # Masked mean, not the last state: with right-padding the last
            # state of a short phrase is a pad state.
            keep = (~pad).float().unsqueeze(-1)
            summary = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)
            return x, summary, pad

    return torch, nn, F, ResidualBlock, Backbone, TextEncoder


def build_pretrained_backbone(cin: int = 3, weights: str = "IMAGENET1K_V2"):
    """A torchvision ResNet-50 that returns its four stages, ImageNet-pretrained.

    WHY THIS NOW EXISTS, HAVING BEEN RULED OUT

    The module docstring above says pretrained weights were rejected because
    "a cluster notebook may have no outbound network, and a run that dies at
    `from_pretrained` after 63 GB is staged is the worst available failure".

    That premise turned out to be false for the machine this actually runs on.
    `compute01` reaches pypi.org, github.com and huggingface.co, and the whole
    environment was pip-installed over that network. The reason held when it
    was written and does not hold here.

    The evidence for the size of the effect is internal and direct. On SECOND,
    same data and same schedule:

        pretrained ResNet-18 stem   mIoU 0.2933
        from-scratch v2             mIoU 0.1730

    +0.12 mIoU from pretraining alone. Every v2 model still far below its
    published range - track_a 0.315 against 0.65-0.85, grounding 0.126 against
    0.70-0.80 - is from-scratch, and 40 extra epochs on track_a moved test mAP
    by 0.002 while training loss fell tenfold. Capacity and schedule are not
    what is missing.

    Interface matches `Backbone` exactly - `widths`, `out_dim`, and a forward
    returning one feature map per stage - so it is a drop-in for either.
    """
    import torch
    import torch.nn as nn
    import torchvision

    net = torchvision.models.resnet50(weights=weights)

    if cin != 3:
        # Keep the pretrained filters and adapt the channel count by averaging
        # across RGB, which is the standard way to carry ImageNet weights to a
        # different band count. Random re-init would discard exactly the thing
        # we came for. The 3/cin rescale keeps the summed activation magnitude
        # roughly where the pretrained batch-norm statistics expect it.
        old = net.conv1
        conv = nn.Conv2d(cin, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            mean = old.weight.mean(dim=1, keepdim=True)
            conv.weight.copy_(mean.repeat(1, cin, 1, 1) * (3.0 / cin))
        net.conv1 = conv

    class PretrainedBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.stages = nn.ModuleList([net.layer1, net.layer2,
                                         net.layer3, net.layer4])
            self.widths = [256, 512, 1024, 2048]
            self.out_dim = 2048

        def forward(self, x):
            x = self.stem(x)
            feats = []
            for stage in self.stages:
                x = stage(x)
                feats.append(x)
            return feats

    return PretrainedBackbone()


def _flatten_spatial(feat):
    """(B,C,H,W) -> (B,HW,C), the layout every attention module here wants."""
    b, c, h, w = feat.shape
    return feat.flatten(2).transpose(1, 2), h, w


# --- Track A: band-agnostic encoder -> landcover_v1 -------------------------


def build_track_a(n_bands: int = 12, dim: int = 96, gsd_conditioning: bool = True,
                  n_classes: int = 19, depths=(2, 2, 2, 2)):
    """v2 land-cover encoder. Signature and forward match `track_a_full.build_model`.

    Three things change, and each is the v1 comment taken seriously:

    * **Per-band tokens survive to depth.** v1 embedded each band with a
      shared stem and immediately averaged the bands together, so everything
      after the first layer saw one fused map. Here the masked mean happens
      after a per-band *residual* stage, so the model has room to learn what a
      band means before it is mixed.
    * **Band embeddings are added before mixing, and the mask is applied to
      the mean.** Unchanged in intent from v1, which got this right: absent
      bands are excluded from the average rather than contributing zeros,
      which is the property that lets a 4-band Cartosat input run at all.
    * **GSD conditioning is applied at every stage, not once.** v1 applied one
      FiLM before the trunk, so the deepest layers - the ones that decide
      class identity - had no access to scale. `docs/phase1-status.md` records
      resolution as the dominant transfer gap, which makes this the layer
      worth spending parameters on.
    """
    torch, nn, F, ResidualBlock, Backbone, _ = _blocks()

    class FiLM(nn.Module):
        def __init__(self, width: int, cond: int = 64):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(cond, cond), nn.ReLU(inplace=True), nn.Linear(cond, width * 2)
            )
            # Zero-init the output so an untrained FiLM is the identity. A
            # randomly-initialised scale/shift on every stage would perturb
            # the trunk before the conditioning has learned anything.
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

        def forward(self, x, cond):
            scale, shift = self.mlp(cond).chunk(2, dim=-1)
            return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

    class BandAgnosticEncoderV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.gsd_conditioning = gsd_conditioning
            self.n_bands = n_bands

            # Per-band stem: one band in, `dim` out. Shared across bands so
            # the parameter count does not grow with the band vocabulary and
            # an unseen band is representable.
            self.band_stem = nn.Sequential(
                nn.Conv2d(1, dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
                ResidualBlock(dim, dim),
            )
            self.band_embed = nn.Parameter(torch.zeros(n_bands, dim))
            nn.init.trunc_normal_(self.band_embed, std=0.02)

            widths = [dim, dim * 2, dim * 4, dim * 4]
            self.stages = nn.ModuleList()
            self.films = nn.ModuleList()
            prev = dim
            for i, (width, depth) in enumerate(zip(widths, depths)):
                layers = [ResidualBlock(prev, width, stride=2)]
                layers += [ResidualBlock(width, width) for _ in range(depth - 1)]
                self.stages.append(nn.Sequential(*layers))
                self.films.append(FiLM(width) if gsd_conditioning else nn.Identity())
                prev = width

            self.gsd_mlp = nn.Sequential(
                nn.Linear(1, 64), nn.ReLU(inplace=True), nn.Linear(64, 64)
            ) if gsd_conditioning else None

            self.norm = nn.BatchNorm2d(prev)
            self.head = nn.Linear(prev, n_classes)
            self.feature_dim = prev

        def encode(self, x, mask, gsd=None):
            """Feature map before the classifier. `optsar_fusion` reuses this."""
            b, c, h, w = x.shape
            x = x * mask[:, :, None, None]
            z = self.band_stem(x.reshape(b * c, 1, h, w))
            z = z.reshape(b, c, -1, h, w) + self.band_embed[None, :, :, None, None]

            m = mask[:, :, None, None, None]
            z = (z * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

            cond = None
            if self.gsd_conditioning and gsd is not None:
                # log GSD: 1.6 m and 10 m are a factor of six apart, which in
                # linear units swamps the 10 m-to-30 m distinction the model
                # also has to make.
                cond = self.gsd_mlp(torch.log(gsd.clamp(min=1e-3)).unsqueeze(-1))

            for stage, film in zip(self.stages, self.films):
                z = stage(z)
                if cond is not None:
                    z = film(z, cond)
            return z

        def forward(self, x, mask, gsd=None):
            z = F.relu(self.norm(self.encode(x, mask, gsd)))
            return self.head(F.adaptive_avg_pool2d(z, 1).flatten(1))

    return BandAgnosticEncoderV2()


# --- Grounding: the box regressor that keeps its coordinates ----------------


def build_grounding(vocab_size: int, dim: int = 128, max_tokens: int = 16,
                    pretrained: bool = False):
    """v2 referring grounder. `model(image, tokens) -> (B,4)` in cx,cy,w,h.

    THE DEFECT THIS EXISTS TO FIX

    `docs/model-cards.md`: v1 "global-average-pools the visual feature map
    before regressing the box, which discards exactly the spatial information
    localisation depends on, so it can only learn an average box." Acc@0.5
    came out at 0.0762 against a published 0.70-0.80.

    So the pooling is gone. The phrase attends over the feature map, and the
    box is read out of *where the attention landed* rather than out of a
    global summary:

    1. the backbone keeps a (B, C, H, W) map - 14x14 at 224 input;
    2. the phrase's token states cross-attend to those spatial tokens;
    3. a learned query pools the map **weighted by that attention**, so the
       pooled vector is position-dependent by construction;
    4. the box head reads centre and size from the attended vector, and the
       attention map itself is returned for the auxiliary loss below.

    WHY A HEATMAP COMES OUT TOO

    Regressing four numbers from a whole image is a weak learning signal - one
    gradient per example, and nothing tells the model *where* it went wrong.
    The attention map is supervised directly against the target box (a soft
    mask), which turns one scalar loss into H*W of them. The trainer can
    ignore `heatmap` and lose nothing; `train_grounding_v2` uses it.
    """
    torch, nn, F, ResidualBlock, Backbone, TextEncoder = _blocks()

    class GroundingV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = (build_pretrained_backbone(cin=3) if pretrained
                             else Backbone(cin=3, dim=dim // 2, depths=(2, 2, 2, 2)))
            # ResNet-50 ends 2048 wide. Sizing the attention and the text
            # encoder off that gives a 147M-parameter model on ~38k
            # referring expressions - the pretrained FEATURES are what we
            # came for, not a decoder eight times larger than the one that
            # already overfits. A 1x1 projection keeps the former and drops
            # the latter.
            width = min(self.backbone.out_dim, dim * 4)
            self.proj = (nn.Conv2d(self.backbone.out_dim, width, 1)
                         if self.backbone.out_dim != width else nn.Identity())
            self.text = TextEncoder(vocab_size, dim=width, layers=2, heads=4,
                                    max_tokens=max_tokens)

            self.visual_norm = nn.LayerNorm(width)
            # 2-D sine-cosine position encoding, added to the spatial tokens.
            # Without it the attention has no way to express "on the left":
            # the tokens are permutation-equivalent and the whole point of
            # this model is that position survives.
            self.pos_proj = nn.Linear(4, width)

            self.cross = nn.MultiheadAttention(width, 4, batch_first=True)
            self.query = nn.Parameter(torch.zeros(1, 1, width))
            nn.init.trunc_normal_(self.query, std=0.02)
            self.fuse_norm = nn.LayerNorm(width)

            self.box_head = nn.Sequential(
                nn.Linear(width, width), nn.ReLU(inplace=True),
                nn.Linear(width, width // 2), nn.ReLU(inplace=True),
                nn.Linear(width // 2, 4),
            )
            self.heatmap_head = nn.Linear(width, 1)

        def _positions(self, h, w, device, dtype):
            ys = torch.linspace(0, 1, h, device=device, dtype=dtype)
            xs = torch.linspace(0, 1, w, device=device, dtype=dtype)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            # sin/cos of both axes: smooth, bounded, and it gives the linear
            # projection something periodic to build "left of" out of.
            grid = torch.stack(
                [gx, gy, torch.sin(gx * math.pi), torch.sin(gy * math.pi)], dim=-1
            )
            return grid.reshape(1, h * w, 4)

        def forward(self, image, tokens, return_heatmap: bool = False):
            feats = self.proj(self.backbone(image)[-1])
            tokens_v, h, w = _flatten_spatial(feats)
            tokens_v = self.visual_norm(tokens_v)
            pos = self._positions(h, w, tokens_v.device, tokens_v.dtype)
            tokens_v = tokens_v + self.pos_proj(pos)

            text_states, text_summary, pad = self.text(tokens)

            # The phrase reads the image: visual tokens are the query so the
            # output is per-location, which is what the heatmap needs.
            attended, _ = self.cross(
                tokens_v, text_states, text_states, key_padding_mask=pad
            )
            grounded = self.fuse_norm(tokens_v + attended)

            heat = self.heatmap_head(grounded).squeeze(-1)          # (B, HW)
            weights = torch.softmax(heat, dim=-1).unsqueeze(-1)
            # Position-dependent pooling: this is the line v1 did not have.
            pooled = (grounded * weights).sum(1) + text_summary

            box = torch.sigmoid(self.box_head(pooled))
            if return_heatmap:
                return box, heat.reshape(-1, h, w)
            return box

    return GroundingV2()


# --- Change: mask, caption, VQA ---------------------------------------------


def build_change_mask(dim: int = 32, depths=(2, 2, 2, 2)):
    """v2 change detector. `model(a, b) -> (B,1,H,W)` logits at input resolution.

    v1 was a three-block siamese CNN that took the absolute difference of the
    deepest features and upsampled twice. Two consequences, both visible in a
    predicted mask: buildings smaller than the 4x downsample vanish, and the
    boundaries are blurred by bilinear upsampling with nothing to sharpen them.

    v2 keeps the two things v1 got right - **one shared encoder** (two encoders
    can drift and report change on identical imagery) and **absolute
    difference** (symmetric, so swapping the dates does not flip the sign) -
    and adds what it was missing: differences at every scale, and a decoder
    with skip connections so the fine scales survive to the output.
    """
    torch, nn, F, ResidualBlock, Backbone, _ = _blocks()

    class ChangeMaskV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Backbone(cin=3, dim=dim, depths=depths)
            widths = self.encoder.widths

            # One projection per scale, applied to |fa - fb|.
            self.diff_proj = nn.ModuleList([
                nn.Sequential(nn.Conv2d(w, dim * 2, 1, bias=False),
                              nn.BatchNorm2d(dim * 2), nn.ReLU(inplace=True))
                for w in widths
            ])
            self.decoder = nn.ModuleList([
                ResidualBlock(dim * 4 if i else dim * 2, dim * 2)
                for i in range(len(widths))
            ])
            self.head = nn.Sequential(
                ResidualBlock(dim * 2, dim), nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True), nn.Conv2d(dim, 1, 1),
            )

        def forward(self, a, b):
            fa, fb = self.encoder(a), self.encoder(b)
            diffs = [proj(torch.abs(x - y))
                     for proj, x, y in zip(self.diff_proj, fa, fb)]

            # Coarsest first, upsampling into the next finer difference. The
            # skip is a concatenation rather than a sum: a sum lets a strong
            # coarse response drown a weak fine one, which is the failure that
            # loses small buildings.
            z = diffs[-1]
            z = self.decoder[0](z)
            for i in range(len(diffs) - 2, -1, -1):
                z = F.interpolate(z, size=diffs[i].shape[-2:], mode="bilinear",
                                  align_corners=False)
                z = self.decoder[len(diffs) - 1 - i](torch.cat([z, diffs[i]], dim=1))

            z = F.interpolate(z, size=a.shape[-2:], mode="bilinear",
                              align_corners=False)
            return self.head(z)

    return ChangeMaskV2()


def build_change_caption(vocab_size: int, dim: int = 128, max_tokens: int = 64,
                         bos_id: int = 1, max_len: int = 24):
    """v2 change captioner. `model(a, b, mask, tokens) -> (B, T, V)` logits.

    The decoder is a masked transformer rather than v1's GRU, for one specific
    reason: a change caption is mostly a spatial statement ("buildings appeared
    in the lower left"), and cross-attention lets each generated word look at
    the part of the difference map it is describing. A GRU reading a single
    pooled vector cannot do that - it has one summary of the whole scene and
    has to say something plausible about it.

    The change mask is an input, not a prediction, and it is concatenated to
    the difference features rather than multiplied into them. Multiplying would
    make an empty mask erase the features entirely, and "nothing changed" is a
    caption the model has to be able to produce.
    """
    torch, nn, F, ResidualBlock, Backbone, _ = _blocks()

    class ChangeCaptionV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Backbone(cin=3, dim=dim // 2, depths=(2, 2, 2, 2))
            width = self.encoder.out_dim
            self.mask_stem = nn.Sequential(
                nn.Conv2d(1, dim // 2, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(dim // 2), nn.ReLU(inplace=True),
                ResidualBlock(dim // 2, dim // 2, stride=2),
                ResidualBlock(dim // 2, dim // 2, stride=2),
                ResidualBlock(dim // 2, dim // 2, stride=2),
            )
            self.fuse = nn.Sequential(
                nn.Conv2d(width + dim // 2, width, 1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            )
            self.memory_norm = nn.LayerNorm(width)
            self.pos = nn.Parameter(torch.zeros(1, 1024, width))
            nn.init.trunc_normal_(self.pos, std=0.02)

            self.embed = nn.Embedding(vocab_size, width, padding_idx=0)
            self.tok_pos = nn.Parameter(torch.zeros(1, max_tokens, width))
            nn.init.trunc_normal_(self.tok_pos, std=0.02)
            layer = nn.TransformerDecoderLayer(
                width, 4, width * 4, dropout=0.1, batch_first=True,
                norm_first=True, activation="gelu",
            )
            self.decoder = nn.TransformerDecoder(layer, 3)
            self.out = nn.Linear(width, vocab_size)

        def forward(self, a, b, mask, tokens):
            fa, fb = self.encoder(a)[-1], self.encoder(b)[-1]
            diff = torch.abs(fa - fb)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            m = self.mask_stem(mask.to(diff.dtype))
            if m.shape[-2:] != diff.shape[-2:]:
                m = F.interpolate(m, size=diff.shape[-2:], mode="bilinear",
                                  align_corners=False)
            memory, h, w = _flatten_spatial(self.fuse(torch.cat([diff, m], dim=1)))
            memory = self.memory_norm(memory) + self.pos[:, : h * w]

            t = tokens.shape[1]
            x = self.embed(tokens) + self.tok_pos[:, :t]
            # Causal mask: without it every position sees the token it is
            # supposed to predict and the training loss goes to zero while
            # generation produces nothing.
            causal = torch.triu(
                torch.ones(t, t, dtype=torch.bool, device=tokens.device),
                diagonal=1,
            )
            hidden = self.decoder(x, memory, tgt_mask=causal,
                                  tgt_key_padding_mask=tokens == 0)
            return self.out(hidden)

        def _memory(self, a, b, mask):
            fa, fb = self.encoder(a)[-1], self.encoder(b)[-1]
            diff = torch.abs(fa - fb)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            m = self.mask_stem(mask.to(diff.dtype))
            if m.shape[-2:] != diff.shape[-2:]:
                m = F.interpolate(m, size=diff.shape[-2:], mode="bilinear",
                                  align_corners=False)
            memory, h, w = _flatten_spatial(self.fuse(torch.cat([diff, m], dim=1)))
            return self.memory_norm(memory) + self.pos[:, : h * w]

        @torch.no_grad()
        def generate(self, a, b, mask, max_len: int = max_len):
            """Greedy decode, matching the v1 signature and output shape.

            WHY THIS EXISTS AS A SEPARATE METHOD

            `forward` is teacher-forced: it is handed the whole target
            sequence at once. Generation has no target, so each step must feed
            back what the model just produced. Omitting this is how both
            caption runs trained for their full schedule and then died in the
            evaluator - the weights were fine, there was simply no way to read
            them out.

            Returns `max_len` tokens without the BOS that seeded them, which
            is exactly what v1 returned; the caller's `decode` truncates at
            EOS.
            """
            memory = self._memory(a, b, mask)
            seq = torch.full((a.shape[0], 1), bos_id, dtype=torch.long,
                             device=a.device)
            produced = []
            for _ in range(max_len):
                t = seq.shape[1]
                x = self.embed(seq) + self.tok_pos[:, :t]
                causal = torch.triu(
                    torch.ones(t, t, dtype=torch.bool, device=seq.device),
                    diagonal=1,
                )
                hidden = self.decoder(x, memory, tgt_mask=causal)
                nxt = self.out(hidden[:, -1]).argmax(-1, keepdim=True)
                produced.append(nxt)
                seq = torch.cat([seq, nxt], dim=1)
            return torch.cat(produced, dim=1)

    return ChangeCaptionV2()


def build_change_vqa(dim: int = 32, n_classes: int = 7, depths=(2, 2, 2, 2)):
    """v2 semantic-change segmenter. `model(a, b) -> (logits_t1, logits_t2)`.

    Despite the name this is not a question-answering model. `change_vqa_v1`
    answers CDVQA questions from the *statistics* of two per-date semantic
    segmentations, which is why v1 returns two `(B, C, H, W)` maps and the
    trainer scores them with a confusion matrix. v2 keeps that exactly: same
    two outputs, same argmax over dim 1, same metric.

    v1's structure is kept where it was reasoned about and right:

    * **two decoders, never shared.** The v1 comment is correct and worth
      repeating - the same pixel is "buildings" at one date and "trees" at the
      other, and one shared decoder would have to resolve that from features
      alone.
    * **each decoder sees its own date's features and the difference**, so it
      can label what is there while knowing what moved.

    What changes is the same thing that changes in the mask model: the
    encoder has real depth, and the decoders take skip connections from every
    scale instead of upsampling twice from the coarsest one. Semantic change
    segmentation is where thin structures - a road, a field boundary - live or
    die, and they die at 4x bilinear upsampling.
    """
    torch, nn, F, ResidualBlock, Backbone, _ = _blocks()

    class SemanticChangeV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Backbone(cin=3, dim=dim, depths=depths)
            widths = self.encoder.widths

            def projections():
                # `own | diff` at every scale, projected to a common width.
                return nn.ModuleList([
                    nn.Sequential(nn.Conv2d(w * 2, dim * 2, 1, bias=False),
                                  nn.BatchNorm2d(dim * 2), nn.ReLU(inplace=True))
                    for w in widths
                ])

            def decoder():
                return nn.ModuleList([
                    ResidualBlock(dim * 4 if i else dim * 2, dim * 2)
                    for i in range(len(widths))
                ])

            self.proj1, self.proj2 = projections(), projections()
            self.dec1, self.dec2 = decoder(), decoder()
            self.head1 = nn.Conv2d(dim * 2, n_classes, 1)
            self.head2 = nn.Conv2d(dim * 2, n_classes, 1)

        @staticmethod
        def _decode(projections, blocks, head, own, diffs, size):
            feats = [p(torch.cat([o, d], dim=1))
                     for p, o, d in zip(projections, own, diffs)]
            z = blocks[0](feats[-1])
            for i in range(len(feats) - 2, -1, -1):
                z = F.interpolate(z, size=feats[i].shape[-2:], mode="bilinear",
                                  align_corners=False)
                z = blocks[len(feats) - 1 - i](torch.cat([z, feats[i]], dim=1))
            z = F.interpolate(z, size=size, mode="bilinear", align_corners=False)
            return head(z)

        def forward(self, a, b):
            fa, fb = self.encoder(a), self.encoder(b)
            diffs = [torch.abs(x - y) for x, y in zip(fa, fb)]
            size = a.shape[-2:]
            return (
                self._decode(self.proj1, self.dec1, self.head1, fa, diffs, size),
                self._decode(self.proj2, self.dec2, self.head2, fb, diffs, size),
            )

    return SemanticChangeV2()


# --- Optical/SAR fusion ------------------------------------------------------


def build_optsar_fusion(dim: int = 32, n_classes: int = 7, n_optical: int = 4,
                        n_sar: int = 1, depths=(2, 2, 2, 2)):
    """v2 dual-stream fusion. `model(optical, sar) -> (l_opt, l_sar, l_fused)`.

    The three-output shape is v1's and is kept, because it is what makes the
    fusion claim checkable: the per-stream heads say what each sensor alone
    supports, and the fused head has to beat both or the cross-attention is
    decoration. `docs/03` calls optical-SAR fusion PS-mandatory, so a fusion
    that cannot be shown to add anything is a real problem, not a cosmetic one.

    v1 concatenated the two pooled vectors. v2 cross-attends **before**
    pooling, in both directions, which is the difference between "both sensors
    contributed to the answer" and "both sensors saw the same scene": SAR
    responds to structure and roughness, optical to reflectance, and the
    useful signal is where one modality explains what the other could not
    resolve. That is a per-location relationship and it does not survive
    pooling.
    """
    torch, nn, F, ResidualBlock, Backbone, _ = _blocks()

    class OptSarFusionV2(nn.Module):
        def __init__(self):
            super().__init__()
            # Separate encoders on purpose - the opposite of the change model.
            # Speckle statistics and reflectance have nothing in common at the
            # first layer, and sharing weights would force one filter bank to
            # serve both.
            # 4-band optical (R,G,B,NIR) and single-channel backscatter, which
            # is what WHU-OPT-SAR actually supplies - not 3 and 2.
            self.optical = Backbone(cin=n_optical, dim=dim, depths=depths)
            self.sar = Backbone(cin=n_sar, dim=dim, depths=depths)
            width = self.optical.out_dim

            self.norm_o = nn.LayerNorm(width)
            self.norm_s = nn.LayerNorm(width)
            self.o_reads_s = nn.MultiheadAttention(width, 4, batch_first=True)
            self.s_reads_o = nn.MultiheadAttention(width, 4, batch_first=True)

            self.head_o = nn.Linear(width, n_classes)
            self.head_s = nn.Linear(width, n_classes)
            self.head_f = nn.Sequential(
                nn.Linear(width * 2, width), nn.ReLU(inplace=True),
                nn.Dropout(0.1), nn.Linear(width, n_classes),
            )

        def forward(self, optical, sar):
            fo, fs = self.optical(optical)[-1], self.sar(sar)[-1]
            to, h, w = _flatten_spatial(fo)
            ts, hs, ws = _flatten_spatial(fs)
            to, ts = self.norm_o(to), self.norm_s(ts)

            # If the two streams disagree on grid size the attention is
            # meaningless, so resample rather than silently broadcasting.
            if (h, w) != (hs, ws):
                fs_r = F.interpolate(fs, size=(h, w), mode="bilinear",
                                     align_corners=False)
                ts, _, _ = _flatten_spatial(fs_r)
                ts = self.norm_s(ts)

            o_aug, _ = self.o_reads_s(to, ts, ts)
            s_aug, _ = self.s_reads_o(ts, to, to)
            o_final, s_final = to + o_aug, ts + s_aug

            po, ps = o_final.mean(1), s_final.mean(1)
            # The per-stream heads read the PRE-attention pooling, so
            # "optical alone" means optical alone and the ablation the three
            # outputs exist to support stays honest.
            return (
                self.head_o(to.mean(1)),
                self.head_s(ts.mean(1)),
                self.head_f(torch.cat([po, ps], dim=-1)),
            )

    return OptSarFusionV2()


# --- Scene captioning --------------------------------------------------------


def build_caption(vocab_size: int, dim: int = 192, max_tokens: int = 64,
                  bos_id: int = 1, max_len: int = 24,
                  pretrained: bool = False):
    """v2 scene captioner. `model(image, tokens) -> (B, T, V)` logits."""
    torch, nn, F, ResidualBlock, Backbone, _ = _blocks()

    class CaptionV2(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = (build_pretrained_backbone(cin=3) if pretrained
                            else Backbone(cin=3, dim=dim // 3, depths=(2, 2, 2, 2)))
            # See build_grounding: 2048 would size a 234M decoder on 8,734
            # captions. Project, keep the features, keep the decoder sane.
            width = min(self.encoder.out_dim, dim * 3)
            self.proj = (nn.Conv2d(self.encoder.out_dim, width, 1)
                         if self.encoder.out_dim != width else nn.Identity())
            self.memory_norm = nn.LayerNorm(width)
            self.pos = nn.Parameter(torch.zeros(1, 1024, width))
            nn.init.trunc_normal_(self.pos, std=0.02)

            self.embed = nn.Embedding(vocab_size, width, padding_idx=0)
            self.tok_pos = nn.Parameter(torch.zeros(1, max_tokens, width))
            nn.init.trunc_normal_(self.tok_pos, std=0.02)
            layer = nn.TransformerDecoderLayer(
                width, 4, width * 4, dropout=0.1, batch_first=True,
                norm_first=True, activation="gelu",
            )
            self.decoder = nn.TransformerDecoder(layer, 3)
            self.out = nn.Linear(width, vocab_size)

        def forward(self, image, tokens):
            memory, h, w = _flatten_spatial(self.proj(self.encoder(image)[-1]))
            memory = self.memory_norm(memory) + self.pos[:, : h * w]
            t = tokens.shape[1]
            x = self.embed(tokens) + self.tok_pos[:, :t]
            causal = torch.triu(
                torch.ones(t, t, dtype=torch.bool, device=tokens.device),
                diagonal=1,
            )
            hidden = self.decoder(x, memory, tgt_mask=causal,
                                  tgt_key_padding_mask=tokens == 0)
            return self.out(hidden)

        @torch.no_grad()
        def generate(self, image, max_len: int = max_len):
            """Greedy decode. See `ChangeCaptionV2.generate` for why."""
            memory, h, w = _flatten_spatial(self.proj(self.encoder(image)[-1]))
            memory = self.memory_norm(memory) + self.pos[:, : h * w]
            seq = torch.full((image.shape[0], 1), bos_id, dtype=torch.long,
                             device=image.device)
            produced = []
            for _ in range(max_len):
                t = seq.shape[1]
                x = self.embed(seq) + self.tok_pos[:, :t]
                causal = torch.triu(
                    torch.ones(t, t, dtype=torch.bool, device=seq.device),
                    diagonal=1,
                )
                hidden = self.decoder(x, memory, tgt_mask=causal)
                nxt = self.out(hidden[:, -1]).argmax(-1, keepdim=True)
                produced.append(nxt)
                seq = torch.cat([seq, nxt], dim=1)
            return torch.cat(produced, dim=1)

    return CaptionV2()


# --- The registry the trainers and the tools both read ----------------------
#
# One name -> one builder. The trainers pass `--arch`, the tools read `arch`
# out of `run_metadata.json`, and both land here. A checkpoint written before
# this existed has no `arch` field, so the default is "v1" everywhere and the
# seven loadable v1 checkpoints keep loading with no migration.

BUILDERS = {
    "track_a": build_track_a,
    "grounding": build_grounding,
    "change_mask": build_change_mask,
    "change_caption": build_change_caption,
    "change_vqa": build_change_vqa,
    "optsar_fusion": build_optsar_fusion,
    "caption": build_caption,
}


def build(task: str, **kwargs):
    """Build the v2 model for `task`, or say which tasks exist."""
    if task not in BUILDERS:
        raise KeyError(
            f"no v2 architecture for '{task}'; have: {', '.join(sorted(BUILDERS))}"
        )
    return BUILDERS[task](**kwargs)
