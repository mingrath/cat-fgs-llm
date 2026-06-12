"""Distributional decode (IMPLEMENTATION_PLAN §5.5).

DEFAULT path (FINAL_DIRECTION §E.1): keep soft cumulative P(rank>k); build each
AU's pmf over {0,1,2}; convolve the 5 pmfs into one pmf over the 0-10 sum. From
that distribution come RPS-on-the-sum (one scalar, bootstrap CI) + per-AU
ClasswiseECE (computed downstream in src/eval/). NO binned reliability diagram on
the 11-atom sum (degenerate at ~11 atoms).

argmax / hard-decode (point_sum) is reserved for the 0.39 POINT decision ONLY.
The 0-10 sum is INSPECTED-NOT-VALIDATED; no validated-claim number is emitted here.
"""

import numpy as np
import torch

from src.model.corn import corn_label_from_logits


def au_pmf_from_cumprobs(cum):          # cum: [B,2] = [P(y>0), P(y>0 & y>1)]
    p_gt0, p_gt1 = cum[:, 0], cum[:, 1]
    p0 = 1 - p_gt0
    p1 = p_gt0 - p_gt1                   # = P(y>0) - P(y>1)
    p2 = p_gt1
    pmf = torch.stack([p0, p1, p2], dim=1)          # [B,3]
    return torch.clamp(pmf, min=0)      # guard tiny negatives from float error


def sum_pmf(au_pmfs):                    # au_pmfs: list of 5 x [B,3] (numpy)
    # convolve 5 pmfs -> pmf over 0..10 (length 11)
    B = au_pmfs[0].shape[0]
    out = np.zeros((B, 11))
    for b in range(B):
        acc = np.array([1.0])
        for a in range(5):
            acc = np.convolve(acc, au_pmfs[a][b])
        out[b] = acc / acc.sum()         # renormalize
    return out                           # [B,11], sums to 1 over the 0..10 sum


def point_sum(logits_list):              # hard decode for the decision ONLY
    # threshold imported from src.vlm.aggregate — the single 0.39 definition in
    # the repo, shared with the VLM path so the two flags can never drift.
    from src.vlm.aggregate import analgesia_flag

    labels = [corn_label_from_logits(lg) for lg in logits_list]   # 5 x [B]
    s = torch.stack(labels, dim=1).sum(1)        # [B] in 0..10
    return s, analgesia_flag(s.float())          # painful flag
