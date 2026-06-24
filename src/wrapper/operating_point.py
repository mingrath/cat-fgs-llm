"""Operating-point selection — FIXED pain-recall >= 0.90 (IMPLEMENTATION_PLAN §6.3, §5.5).

The clinical cutoff on the calibrated P(pain) is selected at a **fixed pain-recall
>= 0.90** (Evangelista anchor, sens 90.7%) **inside train folds** — NOT Youden-J /
F1. A symmetric-cost knee is the wrong loss for a welfare instrument.

CIRCULARITY FIREWALL (IMPLEMENTATION_PLAN §6.3 / line 2302): sens/spec at the 0.39
point are estimated on **VET-CONFIRMED labels ONLY**. QWK-vs-VLM is never validation.
This module refuses to estimate sens/spec on any row that is not vet-confirmed.

The frozen DINOv2 + CORN engine that produces these scores is conceded plumbing, and
this wrapper is cited supporting evidence — not the headline (the headline is the
power-aware per-AU confound-attribution protocol in src/protocols/). We report the
SPECIFICITY the fixed-recall cutoff buys (with bootstrap CIs) and the cutoff's
fold-to-fold variance — never a "we beat X%" claim.
"""

from __future__ import annotations

import numpy as np
import yaml

from src.eval.bootstrap import bootstrap_ci as _grouped_bootstrap_ci


def _load_op_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["operating_point"]


def select_cutoff_at_recall(y_true, p_pain, target_recall: float) -> float:
    """Highest threshold on p_pain that still achieves pain-recall >= target_recall.

    Selected INSIDE train folds. NOT Youden-J / F1. ``y_true``/``p_pain`` are the
    TRAIN-fold rows only; sweeping thresholds high->low, we take the largest cutoff
    whose recall (sensitivity for pain) is still >= target. A higher cutoff buys more
    specificity, so we want the most-specific cutoff that still clears the recall floor.
    """
    y_true = np.asarray(y_true).astype(int)
    p_pain = np.asarray(p_pain, dtype=float)
    pos = y_true == 1
    n_pos = int(pos.sum())
    if n_pos == 0:
        raise ValueError("no pain-positive rows in train fold; cannot fix recall")

    # candidate cutoffs = the calibrated positive scores (recall only changes there)
    cand = np.unique(p_pain[pos])
    best = 0.0
    found = False
    for t in np.sort(cand)[::-1]:                 # high -> low cutoff
        recall = float((p_pain[pos] >= t).mean())
        if recall >= target_recall:
            best = float(t)
            found = True
            break
    if not found:                                 # even the lowest pos score misses target
        best = float(p_pain[pos].min())
    return best


def _sens_spec(y_true, p_pain, cutoff: float):
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(p_pain, dtype=float) >= cutoff).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def estimate_specificity_bought(
    y_vet,
    p_pain,
    vet_confirmed,
    cutoff: float,
    cat_id=None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
):
    """Sens/spec at ``cutoff``, estimated on VET-CONFIRMED rows ONLY (firewall).

    ``vet_confirmed`` is a boolean mask (is_vet_clean / vet-confirmed). Rows where it
    is False are DROPPED before any sens/spec is computed — this is the circularity
    firewall, enforced here, not merely commented. Returns the point estimates plus a
    percentile-bootstrap 95% CI on specificity. Pass ``cat_id`` (per-row individual
    ids) to resample whole cats — the project-standard grouped bootstrap; without it
    the CI is row-i.i.d. and optimistically narrow under within-cat correlation.
    """
    mask = np.asarray(vet_confirmed).astype(bool)
    if not mask.any():
        raise ValueError(
            "circularity firewall: no vet-confirmed rows; sens/spec cannot be "
            "estimated (QWK-vs-VLM is never validation)"
        )
    y = np.asarray(y_vet).astype(int)[mask]
    p = np.asarray(p_pain, dtype=float)[mask]

    sens, spec = _sens_spec(y, p, cutoff)

    # bootstrap the specificity over the vet-confirmed NEGATIVE rows
    neg = y == 0
    neg_correct = ((p < cutoff) & (y == 0)).astype(float)[neg]
    if len(neg_correct):
        # one bootstrap implementation for both paths (shared eval helper): grouped
        # by cat when cat_id is given, ungrouped (groups=None) otherwise. Avoids a
        # private copy that diverged in nan-handling and seed conventions.
        neg_cats = np.asarray(cat_id)[mask][neg] if cat_id is not None else None
        _, spec_lo, spec_hi = _grouped_bootstrap_ci(
            neg_correct, lambda a: float(np.mean(a)),
            n_boot=n_boot, alpha=alpha, seed=seed, groups=neg_cats,
        )
    else:
        spec_lo = spec_hi = float("nan")
    return {
        "cutoff": float(cutoff),
        "sensitivity": sens,
        "specificity": spec,
        "specificity_ci": (spec_lo, spec_hi),
        "n_vet_confirmed": int(mask.sum()),
    }


def operating_point_across_folds(
    fold_frames,
    config_path: str = _DEFAULT_CONFIG,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
):
    """Per-fold cutoff selection + cross-fold variance of the cutoff.

    ``fold_frames`` is an iterable of dicts, one per fold. The TRAIN arrays fix the
    cutoff; the HELD-OUT arrays score it — the two row sets must be disjoint (the
    cutoff is frozen on train, never tuned on the rows it is evaluated on):
        y_train  — train-fold labels used to FIX the recall cutoff
        p_train  — calibrated P(pain) for the SAME train-fold rows
        p_test   — calibrated P(pain) for the held-out rows of this fold
        y_vet    — vet labels for the held-out rows
        vet_confirmed — bool mask (firewall) for the held-out rows
        cat_id   — optional per-row individual ids for the held-out rows
                   (enables the project-standard cat-grouped specificity CI)
    The recall target is read from configs/wrapper.yaml (fixed 0.90 Evangelista anchor).
    """
    op = _load_op_config(config_path)
    assert op["metric"] == "pain_recall", "operating point must be pain-recall, not Youden-J/F1"
    target = float(op["target"])

    per_fold = []
    cutoffs = []
    for fr in fold_frames:
        cut = select_cutoff_at_recall(fr["y_train"], fr["p_train"], target)
        cutoffs.append(cut)
        est = estimate_specificity_bought(
            fr["y_vet"], fr["p_test"], fr["vet_confirmed"], cut,
            cat_id=fr.get("cat_id"), n_boot=n_boot, alpha=alpha, seed=seed,
        )
        est["target_recall"] = target
        per_fold.append(est)

    cutoffs = np.asarray(cutoffs, dtype=float)
    return {
        "target_recall": target,
        "per_fold": per_fold,
        "cutoff_mean": float(cutoffs.mean()) if len(cutoffs) else float("nan"),
        "cutoff_sd": float(cutoffs.std(ddof=1)) if len(cutoffs) > 1 else float("nan"),
        "spec_bought_mean": float(
            np.nanmean([f["specificity"] for f in per_fold])
        ) if per_fold else float("nan"),
    }
