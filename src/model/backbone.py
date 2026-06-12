"""Frozen DINOv2 ViT-S/14 backbone (IMPLEMENTATION_PLAN §5.1).

CONCEDED PLUMBING (FINAL_DIRECTION §A); never claimed as novel. dinov2_vits14,
hidden size 384, patch 14, forward-only. Frozen because the supervised set is
~120-300 vet/VLM labels; only the ~5 light heads train.

Patch/hidden are derived from the model (m.embed_dim), not hardcoded in
load-bearing paths; the 14/384/518 constants in comments are for the reader only.
"""

import torch
from torchvision import transforms

from src.model.device import DEVICE

IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# 518 = 37*14 -> 37x37 patch grid; divisible-by-14 is mandatory for DINOv2.
preprocess = transforms.Compose([
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(IMNET_MEAN, IMNET_STD),
])


def load_frozen_dinov2(device: str = DEVICE):
    # torch.hub ViT-S/14; no registers variant for the v1 spine
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
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
