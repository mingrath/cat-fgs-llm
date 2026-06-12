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
from torchvision import transforms

from src.model.device import DEVICE

IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# Selectable ViT-S/14 variants. Reg (registers) is the field default; plain is the
# A/B comparator. Both are embed_dim 384, patch 14 — preprocess is identical.
DEFAULT_VARIANT = "dinov2_vits14_reg"
SUPPORTED_VARIANTS = ("dinov2_vits14_reg", "dinov2_vits14")

# 518 = 37*14 -> 37x37 patch grid; divisible-by-14 is mandatory for DINOv2.
preprocess = transforms.Compose([
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(IMNET_MEAN, IMNET_STD),
])


def load_frozen_dinov2(device: str = DEVICE, variant: str = DEFAULT_VARIANT):
    """Load a frozen ViT-S/14 DINOv2 backbone.

    variant: ``dinov2_vits14_reg`` (field default, registers suppress attention
    artifacts on localized per-AU features) or the plain ``dinov2_vits14`` A/B
    comparator. Returns (model, hidden_dim=384).
    """
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"unsupported backbone variant {variant!r}; expected one of {SUPPORTED_VARIANTS} "
            "(reg is the field default; plain is the A/B comparator)"
        )
    m = torch.hub.load("facebookresearch/dinov2", variant)
    m.requires_grad_(False)      # freeze ALL params
    m.eval()                     # disable dropout/BN-update; deterministic features
    m.to(device)
    hidden = m.embed_dim         # 384 for ViT-S; assert below
    assert hidden == 384, f"expected 384, got {hidden}"
    return m, hidden


@torch.inference_mode()
def extract(m, x):               # x: [B,3,H,W] already normalized + resized to /14 grid
    out = m.forward_features(x)  # dict
    cls = out["x_norm_clstoken"]        # [B,384]   (CLS, == index 0 of the token seq)
    patch = out["x_norm_patchtokens"]   # [B,N,384] (patch tokens; HF-equiv of [:,1:,:])
    return cls, patch
