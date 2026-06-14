"""Co-teaching small-loss head trainer (IMPLEMENTATION_PLAN §5.6).

CONCEDED PLUMBING. The frozen DINOv2 + CORN engine is NEVER claimed as novel
(FINAL_DIRECTION §A). The validated-claim spine is the sibling BINARY pain head
built here; the 0-10 CORN sum is INSPECTED-NOT-VALIDATED and emits no
validated-claim number.

Method (FINAL_DIRECTION §6; bhanML/Co-teaching, BUILD_PLAN §3): train NOT
confirmed-only (too few) and NOT naive-all (bakes in the VLM under-estimation
bias). Two CornMultiHead nets, each selects the small-loss subset for the OTHER.
Keep-rate ramps 1.0 -> (1 - tau) over num_gradual=10 epochs (canonical Han et al.
2018 R(T): inclusive -> selective, exploiting the memorization effect). tau =
estimated VLM noise rate (Gate-1-B), NOT a fixed 0.5. vet-clean rows
(is_vet_clean==1) are NEVER dropped from either selection.

Cache-only loop: the backbone NEVER runs in the inner loop -- features are read
from the §5.2 npz cache, so the §5.6 pooling x loss x seed sweep runs in minutes.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model.corn import corn_loss, multi_corn_loss
from src.model.device import DEVICE
from src.model.heads import AUS, CornMultiHead, K

N_AUS = len(AUS)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_split(npz, train_folds=(0, 1, 2), pool="cls"):
    """Read the cached features + labels for the given train folds.

    The trainer reads `fold` from the cache (G3-hashed); it NEVER touches a raw
    test manifest. pool selects "cls" (default) or "patch_mean" -> [N,384].
    Returns (X, y, clean, y_pain) tensors for the selected folds.
    """
    d = np.load(npz, allow_pickle=True)
    # uniform hard _CACHE_SCHEMA no legacy (per FreshHandoffDINOv3RicherTiny + round_fresh_4 table#1 enforce; align separability)
    from src.model.cache_features import _CACHE_SCHEMA, SCHEMA_VERSION
    if d.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"cache schema_version mismatch (uniform enforce no legacy): {npz}")
    for k in ("n_aus", "au_hash", "k", "feature_dim"):
        if k not in d or d[k] != _CACHE_SCHEMA[k]:
            raise ValueError(f"_CACHE_SCHEMA key mismatch on {k} (enforce)")
    for k in ("variant", "layer", "patch_mode"):
        if k not in d:
            raise ValueError(f"_CACHE_SCHEMA key missing {k} (uniform enforce no legacy)")
    feat = d[pool]                          # "cls" or "patch_mean" -> [N,384]
    fold = d["fold"].astype(int)
    y = d["y"]                              # [N,5], -1 = AU not scored
    clean = d["is_vet_clean"]
    y_pain = d["y_pain"]
    tr = np.isin(fold, train_folds)
    return (
        torch.tensor(feat[tr]),
        torch.tensor(y[tr]),
        torch.tensor(clean[tr]),
        torch.tensor(y_pain[tr]),
    )


# ---------------------------------------------------------------------------
# Per-sample loss + selection helpers (~15 lines, kept in-file per §5.6)
# ---------------------------------------------------------------------------
def _persample(logits_au, y_au, num_classes=K):
    """Per-ROW corn_loss (reduction='none' equivalent) for ONE AU.

    logits_au: [B, K-1]; y_au: [B] with -1 = not scored (contributes 0).
    Loops rows so the vendored corn_loss stays the single source of truth for the
    ordinal objective; rows are tiny (~120-300) so this is cheap.
    """
    out = torch.zeros(logits_au.shape[0], device=logits_au.device)
    for j in range(logits_au.shape[0]):
        if int(y_au[j]) < 0:                       # sentinel -> AU unscored on this row
            continue
        out[j] = corn_loss(logits_au[j:j + 1], y_au[j:j + 1], num_classes)
    return out


def _persample_sum(logits_list, y):
    """Per-row CORN loss summed over the AUs scored on that row -> [B]."""
    return torch.stack(
        [_persample(logits_list[a], y[:, a]) for a in range(N_AUS)]
    ).sum(0)


def _select(loss_other, clean, k):
    """Indices to keep: vet-clean UNION smallest-loss-non-clean up to k.

    loss_other: per-sample loss from the PARTNER net (co-teaching cross-selection).
    clean: is_vet_clean flags. vet-clean rows are ALWAYS kept (never dropped);
    the remaining budget (k - n_clean) goes to the smallest-loss non-clean rows.
    """
    clean_idx = torch.nonzero(clean > 0, as_tuple=False).flatten()
    non_clean = torch.nonzero(clean == 0, as_tuple=False).flatten()
    budget = max(0, k - clean_idx.numel())
    if non_clean.numel() and budget:
        order = torch.argsort(loss_other[non_clean])         # ascending loss
        picked = non_clean[order[:budget]]
    else:
        picked = non_clean[:0]
    return torch.cat([clean_idx, picked])


# ---------------------------------------------------------------------------
# Loss switch (corn default vs CORAL ablation; one-flag, §5.6 table)
# ---------------------------------------------------------------------------
def _coral_multi_loss(logits_list, y, num_classes=K, au_weights=None):
    """CORAL ablation, sentinel-masked per AU (lazy dep; corn is the default).

    Uses coral_pytorch.losses.coral_loss with extended-binary level targets so the
    only thing that changes vs CORN is the per-AU objective -- everything else
    (heads, decode, selection) is shared. Imported lazily so import stays clean
    when the optional ablation dep is absent.
    """
    from coral_pytorch.dataset import levels_from_labelbatch
    from coral_pytorch.losses import coral_loss

    # graph-connected zero, same reason as multi_corn_loss: .backward() must stay
    # valid when every AU in the selected rows is sentinel-masked.
    total = logits_list[0].sum() * 0.0
    for a in range(len(logits_list)):
        mask = y[:, a] >= 0
        if mask.sum() == 0:
            continue
        levels = levels_from_labelbatch(y[mask, a], num_classes=num_classes).to(
            logits_list[a].device
        )
        au_loss = coral_loss(logits_list[a][mask], levels)
        w = 1.0 if au_weights is None else au_weights[a]
        total = total + w * au_loss
    return total


def _loss_fn(loss: str):
    """Return the multi-AU loss callable for the requested ablation flag."""
    if loss == "corn":
        return multi_corn_loss
    if loss == "coral":
        return _coral_multi_loss
    raise ValueError(f"unknown loss flag: {loss!r} (expected 'corn' or 'coral')")


# ---------------------------------------------------------------------------
# Binary pain head (the v1 SPINE) -- sibling Linear(384,1), independent of CORN
# ---------------------------------------------------------------------------
class BinaryPainHead(nn.Module):
    """v1 spine: Linear(384,1) trained with BCEWithLogitsLoss(pos_weight) on y_pain.

    Shares the cache, is independent of CORN, and is the path the calibration /
    abstention / decision-curve wrapper (Phase C) actually operates on. This is
    the validated-claim spine; the 0-10 CORN sum is the inspected path.
    """

    def __init__(self, in_dim=384, p_drop=0.1):
        super().__init__()
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(in_dim, 1)
        nn.init.trunc_normal_(self.fc.weight, std=2e-5)
        nn.init.zeros_(self.fc.bias)

    def forward(self, feat):                    # [B,384] -> [B] logit
        return self.fc(self.drop(feat)).squeeze(-1)


def train_binary_pain(X, y_pain, device=DEVICE, epochs=80, lr=1e-3,
                      weight_decay=1e-2, batch_size=32, p_drop=0.1,
                      pos_weight=None, num_workers=0, seed=0):
    """Train the sibling binary pain head on y_pain (the v1 spine)."""
    torch.manual_seed(seed)
    X = X.to(device)
    yp = y_pain.float().to(device)
    if pos_weight is None:                       # n_neg / n_pos from the train split
        n_pos = float((yp > 0).sum().item())
        n_neg = float((yp <= 0).sum().item())
        pos_weight = n_neg / max(n_pos, 1.0)
    pw = torch.tensor([pos_weight], device=device)
    net = BinaryPainHead(in_dim=X.shape[1], p_drop=p_drop).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    dl = DataLoader(TensorDataset(X, yp), batch_size=batch_size, shuffle=True,
                    num_workers=num_workers)
    net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            bce(net(xb), yb).backward()
            opt.step()
    return net


# ---------------------------------------------------------------------------
# Co-teaching CORN trainer
# ---------------------------------------------------------------------------
def train(npz, tau, device=DEVICE, epochs=80, pool="cls", seed=0,
          loss="corn", lr=1e-3, weight_decay=1e-2, batch_size=32, p_drop=0.1,
          head_init_std=2e-5, num_gradual=10, train_folds=(0, 1, 2),
          au_weights=None, num_workers=0):
    """Co-teaching small-loss trainer for the 5 CORN heads.

    tau = estimated VLM noise rate (Gate-1-B). 1 - tau is the floor keep-rate;
    the ramp goes 1.0 -> (1 - tau) over num_gradual epochs. vet-clean rows are
    never dropped. Returns netA (netB is the teaching partner).
    """
    torch.manual_seed(seed)
    Xtr, ytr, clean, _ = load_split(npz, train_folds=train_folds, pool=pool)
    Xtr = Xtr.to(device)
    dl = DataLoader(
        TensorDataset(Xtr, ytr.to(device), clean.to(device)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    loss_fn = _loss_fn(loss)

    # two heads for co-teaching. CornMultiHead (engine) already trunc_normal_-inits
    # its proj at std=2e-5; honor a config-supplied head_init_std without editing the
    # engine by re-initializing the proj weight post-construction.
    netA = CornMultiHead(in_dim=Xtr.shape[1], p_drop=p_drop).to(device)
    netB = CornMultiHead(in_dim=Xtr.shape[1], p_drop=p_drop).to(device)
    if head_init_std is not None:
        for net in (netA, netB):
            nn.init.trunc_normal_(net.proj.weight, std=head_init_std)
            nn.init.zeros_(net.proj.bias)
    optA = torch.optim.AdamW(netA.parameters(), lr=lr, weight_decay=weight_decay)
    optB = torch.optim.AdamW(netB.parameters(), lr=lr, weight_decay=weight_decay)

    netA.train()
    netB.train()
    for ep in range(epochs):
        # canonical Co-teaching R(T): keep-rate 1.0 -> (1-tau) over num_gradual epochs
        keep = 1.0 - tau * min(1.0, ep / num_gradual)   # inclusive -> selective
        for xb, yb, cb in dl:
            la = netA(xb)
            lb = netB(xb)
            with torch.no_grad():
                pa = _persample_sum(la, yb)              # A's per-row loss
                pb = _persample_sum(lb, yb)              # B's per-row loss
                k = int(keep * len(xb))
                # cross-selection: partner's small-loss subset; vet-clean always kept
                selA = _select(pb, cb, k)                # B selects for A
                selB = _select(pa, cb, k)                # A selects for B
            optA.zero_grad()
            loss_fn([lg[selA] for lg in la], yb[selA], au_weights=au_weights).backward()
            optA.step()
            optB.zero_grad()
            loss_fn([lg[selB] for lg in lb], yb[selB], au_weights=au_weights).backward()
            optB.step()
    return netA   # report netA; netB is the teaching partner
