"""Pillar 2 — distributional calibration of the 0-10 sum (IMPLEMENTATION_PLAN §6.2 / §7.2.3).

SUPPORTING (not a gate). The 0-10 CORN sum layer is INSPECTED-NOT-VALIDATED: the word
"graded" never appears in a validated-claim sentence, and the summed-CORN QWK (computed
elsewhere) is an internal inspection metric only — NEVER a validated-instrument claim.

Two metrics, both on the DISTRIBUTIONAL CORN path (FINAL_DIRECTION §E.1 default):
  - rps_sum()  : RPS-on-the-0-10-sum (single scalar, ordinal-aware Brier), with a
                 bootstrap-over-IMAGES CI (cat-grouped resampling supported).
  - classwise_ece_per_au() : per-AU ClasswiseECE via netcal (imported LAZILY), making
                 miscalibration attributable to a specific AU.

NO binned reliability diagram / binned ECE on the 11-atom sum: it is degenerate at ~11
atoms (per-bin SE +/-0.18-0.26), so a binned diagram would be noise dressed as signal.
The sum's calibration lives on this RPS / per-AU-ClasswiseECE path instead. argmax /
hard decode is reserved for the 0.39 POINT decision ONLY (see src.model.decode.point_sum).

The soft pmf path is imported from src.model.decode to AVOID DRIFT — the eval pmf must be
byte-for-byte the engine's decode, never a re-implementation.
"""

from __future__ import annotations

import numpy as np
import torch

# Import the soft pmf path from the engine to avoid drift (single source of truth).
from src.model.decode import au_pmf_from_cumprobs, sum_pmf
from src.model.corn import corn_cumprobs

# sum_pmf is re-exported via the import above (single source of truth = the engine).
__all__ = ["au_pmf", "sum_pmf", "rps_sum_one", "rps_sum", "classwise_ece_per_au"]


def au_pmf(logits2):
    """Per-AU pmf over {0,1,2} from CORN logits [..,2] (the §6.2 `au_pmf` sketch).

    Mirrors the engine SOFT path exactly: cumprob(sigmoid) -> pmf. Thin wrapper over
    src.model.corn.corn_cumprobs + src.model.decode.au_pmf_from_cumprobs so there is one
    decode, not two.
    """
    if not torch.is_tensor(logits2):
        logits2 = torch.as_tensor(logits2, dtype=torch.float32)
    logits2 = logits2.detach().cpu()  # eval is CPU/numpy; avoid an MPS->numpy footgun downstream
    cum = corn_cumprobs(logits2)  # [..,2] cumulative P(rank>k)
    return au_pmf_from_cumprobs(cum)  # [..,3] pmf over {0,1,2}


def rps_sum_one(pmf_sum: np.ndarray, y: int) -> float:
    """RPS for ONE image's 0-10 sum pmf against observed integer sum label y.

    RPS = sum_k (CDF_pred(k) - CDF_obs(k))^2 ; obs CDF is a step at y. Ordinal-aware
    Brier; CORN gives the predictive CDF for free.
    """
    cdf_p = np.cumsum(pmf_sum)
    cdf_o = (np.arange(len(pmf_sum)) >= y).astype(float)
    return float(((cdf_p - cdf_o) ** 2).sum())


def rps_sum(
    pmf_sum: np.ndarray,
    y,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
    groups=None,
) -> dict:
    """Mean RPS-on-the-0-10-sum over images, with a bootstrap CI (§6.2(b)).

    pmf_sum: [N,11] (each row sums to 1, from src.model.decode.sum_pmf).
    y:       [N] observed integer sums in 0..10.
    groups:  optional [N] cat_id for cat-grouped bootstrap (resample whole cats), so
             within-cat correlation is respected — the §6.2 note: "bootstrap over IMAGES
             (cat-grouped)".

    Returns dict(mean_rps, ci_lo, ci_hi, n).
    """
    pmf_sum = np.asarray(pmf_sum, dtype=float)
    y = np.asarray(y)
    per_img = np.array([rps_sum_one(pmf_sum[i], int(y[i])) for i in range(len(y))])

    rng = np.random.default_rng(seed)
    n = len(per_img)
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        boots = []
        for _ in range(n_boot):
            picked = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(groups == g)[0] for g in picked])
            boots.append(per_img[idx].mean())
    else:
        boots = [per_img[rng.integers(0, n, n)].mean() for _ in range(n_boot)]

    boots = np.asarray(boots, dtype=float)
    return {
        "mean_rps": float(per_img.mean()),
        "ci_lo": float(np.percentile(boots, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        "n": int(n),
    }


def classwise_ece_per_au(pmf_au: dict, vet_au: dict, bins: int = 5) -> dict:
    """Per-AU ClasswiseECE on each AU's 3-class pmf (§6.2(c)), via netcal (LAZY import).

    pmf_au[au]: [N,3] soft pmf over {0,1,2} for that AU.
    vet_au[au]: [N]   vet integer label in {0,1,2}.
    bins=5 (small n). Makes miscalibration ATTRIBUTABLE to a specific AU. Reported only
    for AUs that survive Gate 1-B; pathological ClasswiseECE on core AUs corroborates a
    pivot but is itself hygiene/supporting, never a gate.

    netcal is a heavy/optional dep, imported here so importing this module stays cheap.
    """
    from netcal.metrics import ClasswiseECE  # lazy: optional dep

    metric = ClasswiseECE(bins=bins)
    out = {}
    for au, pmf in pmf_au.items():
        p = np.asarray(pmf, dtype=float)
        labels = np.asarray(vet_au[au])
        out[au] = float(metric.measure(p, labels))
    return out
