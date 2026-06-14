"""Vendored CORN ordinal loss + decode (IMPLEMENTATION_PLAN §5.4).

CONCEDED PLUMBING. The frozen DINOv2 + CORN engine is NEVER claimed as novel
(FINAL_DIRECTION §A); it exists only to carry the binary-plus-wrapper spine, the
headline confound-attribution protocol, and the guarded kappa check.

Vendored verbatim (MIT, ludwig-ai/ludwig `corn.py`, torch + F only) so the
datasheet shows exactly what trained the heads. Cross-checked ONCE against
coral_pytorch.losses.corn_loss in the Gate-4 unit test (§5.7), then the dep is
dropped. Shi, Cao & Raschka 2021 (CORN, arXiv 2111.08851).

Decode paths:
  - corn_cumprobs  -> SOFT default path (FINAL_DIRECTION §E.1) feeding pmf/RPS/ECE.
  - corn_label_from_logits -> HARD decode, for the 0.39 POINT decision ONLY.
"""

import torch
import torch.nn.functional as F


def corn_loss(logits, y, num_classes):
    """Conditional ordinal (CORN) loss for ONE AU.

    logits: [B, num_classes-1]   y: [B] in {0,...,num_classes-1}
    Shi, Cao & Raschka 2021. Vendored (MIT, ludwig-ai/ludwig)."""
    sets = []
    for i in range(num_classes - 1):
        label_mask = (y > i - 1)                 # samples still "in play" at rank i
        label_tensor = (y[label_mask] > i).to(torch.int64)
        if label_mask.sum() == 0:
            continue
        sets.append((label_mask, label_tensor))
    losses = 0.0
    n = 0
    for i, (mask, lab) in enumerate(sets):
        pred = logits[mask, i]                    # conditional logit for rank i
        loss = -torch.sum(
            F.logsigmoid(pred) * lab + (F.logsigmoid(pred) - pred) * (1 - lab)
        )
        losses = losses + loss
        n += mask.sum().item()
    return losses / max(n, 1)


def corn_label_from_logits(logits):
    """Hard decode for the 0.39 POINT decision ONLY.

    logits: [B, num_classes-1] -> labels [B]. predict = sum_k( cumprod(sigmoid)[k] > 0.5 )."""
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)         # P(y>0), P(y>0 & y>1), ...
    return torch.sum(probas > 0.5, dim=1)


def corn_cumprobs(logits):
    """SOFT path (DEFAULT, FINAL_DIRECTION §E.1). Returns cumulative P(rank>k) per level.

    DO NOT hard-decode for calibration -- these soft probs feed pmf/RPS/ECE."""
    return torch.cumprod(torch.sigmoid(logits), dim=1)   # [B, num_classes-1]


def multi_corn_loss(logits_list, y, num_classes=3, au_weights=None):
    """Sum the 5 per-AU corn_loss, skipping AUs with the -1 sentinel.

    logits_list: 5 x [B,2]; y: [B,5] with -1 = AU not scored on that image.

    total starts as a graph-connected zero (not the float 0.0) so .backward()
    stays valid when every AU is sentinel-masked — reachable via a co-teaching
    selection of only all-sentinel vet-clean rows, or an empty k=0 selection on
    a size-1 tail batch."""
    total = logits_list[0].sum() * 0.0
    for a in range(len(logits_list)):
        mask = y[:, a] >= 0
        if mask.sum() == 0:
            continue
        au_loss = corn_loss(logits_list[a][mask], y[mask, a], num_classes)
        w = 1.0 if au_weights is None else au_weights[a]
        total = total + w * au_loss
    return total
