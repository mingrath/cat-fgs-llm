"""The 5 CORN heads (IMPLEMENTATION_PLAN §5.3, layout B: one wide head).

CONCEDED PLUMBING (FINAL_DIRECTION §A); never claimed as novel. K=3 levels per
AU => K-1 = 2 logits per AU; Sigma(K_au-1) = 5*2 = 10. One matmul Linear(384,10),
then torch.split into 5 x [B,2] per-AU chunks fed to per-AU corn_loss.

The 5 AUs (FACTCHECK: Min-aggregation for whiskers AND head):
ear, orbital, muzzle, whiskers, head.

The wrapper's binary pain head (the v1 spine) is a sibling Linear(384,1) trained
with BCEWithLogitsLoss; it is independent of CORN and lives in src/wrapper/. The
0-10 CORN sum is the INSPECTED-NOT-VALIDATED path, never a validated-claim number.
"""

import torch
import torch.nn as nn

from src.constants import AU_ORDER

AUS = AU_ORDER
K = 3                              # levels 0/1/2 per AU
N_LOGITS = len(AUS) * (K - 1)      # = 10


class CornMultiHead(nn.Module):
    def __init__(self, in_dim=384, n_aus=5, k=3, p_drop=0.1):
        super().__init__()
        self.k = k
        self.n_aus = n_aus
        self.drop = nn.Dropout(p_drop)               # light reg on tiny linear probe
        self.proj = nn.Linear(in_dim, n_aus * (k - 1))   # 384 -> 10
        # trunc_normal_ init for stable small-data linear-probe
        # (BenediktAlkin/vtab1k-pytorch, GITHUB_MINE P2.2)
        nn.init.trunc_normal_(self.proj.weight, std=2e-5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat):                          # feat: [B,384] (CLS or mean-patch)
        logits = self.proj(self.drop(feat))           # [B,10]
        return list(torch.split(logits, self.k - 1, dim=1))  # 5 x [B,2]
