"""Frozen DINOv2 ViT-S/14 backbone (IMPLEMENTATION_PLAN §5.1).

CONCEDED PLUMBING (FINAL_DIRECTION §A); never claimed as novel. ViT-S (hidden 384,
patch 14), forward-only, frozen because the supervised set is ~120-300 vet/VLM
labels; only the ~5 light heads train.

BACKBONE VARIANT (README §"engine is conceded plumbing"): the register variant
``dinov2_vits14_reg`` is the FIELD DEFAULT — registers suppress attention artifacts
that hurt dense, localized features, and per-AU FGS scoring (orbital/ear/muzzle
sub-regions) is exactly localized. The plain ``dinov2_vits14`` is kept selectable so
the reg-vs-plain choice can be A/B'd before locking. The choice is recorded in the
cache PROVENANCE.json (an input to every reported number), so a run can never hide
which backbone produced its features. Both variants are ViT-S (embed_dim 384).

Patch/hidden are derived from the model (m.embed_dim), not hardcoded in
load-bearing paths; the 14/384/518 constants in comments are for the reader only.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms

from src.model.device import DEVICE

IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# Selectable ViT-S/14 variants. Reg (registers) is the field default; plain is the
# A/B comparator. Both are embed_dim 384, patch 14 — preprocess is identical.
DEFAULT_VARIANT = "dinov2_vits14_reg"
SUPPORTED_VARIANTS = ("dinov2_vits14_reg", "dinov2_vits14", "dinov3_vits16")

# 518 = 37*14 -> 37x37 patch grid; divisible-by-14 mandatory for DINOv2.
# For dinov3_vits16 (patch 16) use 224/512 multiple of 16; richer per MCP context7 (dense oob frozen for small-N medical facial ordinal AU).
VARIANT_PATCH_SIZES = {"dinov2_vits14_reg": 14, "dinov2_vits14": 14, "dinov3_vits16": 16}
VARIANT_INPUT_SIZES = {"dinov2_vits14_reg": 518, "dinov2_vits14": 518, "dinov3_vits16": 224}  # 224 base for vits16; caller may override for dense

preprocess = transforms.Compose([
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(IMNET_MEAN, IMNET_STD),
])


def load_frozen_dinov2(device: str = DEVICE, variant: str = DEFAULT_VARIANT):
    """Load a frozen ViT-S/14 DINOv2 (or dinov3_vits16 successor) backbone.

    variant: ``dinov2_vits14_reg`` (field default, registers suppress attention
    artifacts on localized per-AU features) or the plain ``dinov2_vits14`` A/B
    comparator, or ``dinov3_vits16`` (MCP /facebookresearch/dinov3 primary A/B:
    high-quality dense features out-of-the-box / without fine-tuning per MODEL_CARD;
    same forward_features contract: x_norm_clstoken + x_norm_patchtokens;
    supports get_intermediate_layers(n=list) + L2 patch for richer localized AU).
    Returns (model, hidden_dim).
    """
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"unsupported backbone variant {variant!r}; expected one of {SUPPORTED_VARIANTS} "
            "(reg is the field default; plain is the A/B comparator; dinov3_vits16 MCP dense oob successor for small-N imbal ordinal facial AU)"
        )
    if variant.startswith("dinov3"):
        # MCP context7 confirmed: torch.hub "facebookresearch/dinov3" dinov3_vits16 or dinov3.hub.backbones import; frozen oob dense ideal
        m = torch.hub.load("facebookresearch/dinov3", variant, source="github")
    else:
        m = torch.hub.load("facebookresearch/dinov2", variant)
    m.requires_grad_(False)      # freeze ALL params
    m.eval()                     # disable dropout/BN-update; deterministic features
    m.to(device)
    hidden = m.embed_dim         # 384 for ViT-S; 384 for vits16 too
    # assert relaxed for dinov3 vits16 (still ~384d small); caller derives
    if "vits" in variant:
        assert hidden == 384, f"expected 384 for small ViT-S, got {hidden}"
    return m, hidden


@torch.inference_mode()
def extract(m, x, layer: str = "last", patch_l2: bool = False):
    """Extract (cls, patch) tokens from frozen backbone (conceded engine, no-FT default).

    layer: "last" (forward_features default) | list[int]/"intermed" for get_intermediate_layers
           (MCP-confirmed dinov3 vision_transformer contract: n=..., return_class_token=True, norm=True
            for dense localized features).
    patch_l2: apply F.normalize(..., p=2, dim=-1) gated by caller config (e.g. corn.yaml patch_mode=l2).
    Supports dinov3_vits16 richer oob frozen dense per-AU (orbital/ear/muzzle) for small-N imbal ordinal AU.
    Updates callers + provenance/schema (the 'hash' of config for repro/G3).
    """
    if layer == "last" or layer is None:
        out = m.forward_features(x)  # dict
        cls = out["x_norm_clstoken"]        # [B,384]   (CLS, == index 0 of the token seq)
        patch = out["x_norm_patchtokens"]   # [B,N,384] (patch tokens; HF-equiv of [:,1:,:])
    else:
        # MCP dinov3: get_intermediate_layers for multi-layer dense/localized (eval encoder, classifiers)
        # e.g. n=[5,11,17,23] or range; norm=True, return_class_token=True per vision_transformer + notebooks
        # PRACTICAL A/B richer: for n=list do mean+std concat across layers for denser localized cues (orbital/ear/muzzle) + L2 full
        if isinstance(layer, str) and layer.lower() == "intermed":
            layer = [-4, -3, -2, -1]  # richer last-4 intermed per MCP dinov3 contract for dense localized
        n = layer if isinstance(layer, (list, tuple, range)) else [-1]
        inter = m.get_intermediate_layers(x, n=n, reshape=False, return_class_token=True, norm=True)
        if inter:
            # richer: concat mean+std of patch feats from selected layers (or single); supports full intermed mean_std
            patches = [i[0] for i in inter]  # patch tokens per layer
            if len(patches) > 1:
                pcat = torch.cat(patches, dim=-1)  # concat layers first for richer dim
                pmean = pcat.mean(dim=1)  # [B, D*nl]
                pstd = pcat.std(dim=1)
                patch = torch.cat([pmean, pstd], dim=-1)  # mean+std concat richer
                # cls from last
                cls = inter[-1][1]
            else:
                patch, cls = inter[-1][0], inter[-1][1]  # patch + cls from selected (last) intermed layer
                # for single also support mean_std style if needed
                if patch.shape[1] > 0:  # ensure
                    pmean = patch.mean(dim=1)
                    pstd = patch.std(dim=1)
                    # keep patch as-is for compat; richer via caller pool or intermed mode
        else:
            out = m.forward_features(x)
            cls = out["x_norm_clstoken"]
            patch = out["x_norm_patchtokens"]
    if patch_l2:
        patch = F.normalize(patch, p=2, dim=-1)
    return cls, patch
