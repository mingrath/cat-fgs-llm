"""Welfare-asymmetric net-benefit DECISION CURVE — HEADLINE wrapper artifact
(IMPLEMENTATION_PLAN §6.3, §5.5 line 2303).

**This, not ECE, is the headline wrapper artifact.** Net-benefit decision-curve
analysis (Vickers) via ``MSKCC-Epi-Bio/dcurves`` (imported lazily). The
undertreat:overtreat harm ratio is swept as a **RANGE** across the
threshold-probability axis — we have NO vet-elicited point ratio, so we never pin one.
The threshold-probability axis IS the harm-ratio sweep: pt = overtreat/(under+over).
Undertreatment >> overtreatment, so we emphasise the LOW-pt welfare region.

We bootstrap the net-benefit curve with **cat-grouped** resampling (group by cat_id,
never by clip) for a CI ribbon. Decision-support TRIAGE framing only — the output is
"grimace consistent with pain, X/10; recommend vet assessment," never an autonomous
analgesia trigger.

The frozen DINOv2 + CORN engine producing p_pain is conceded plumbing; this curve is
the headline frame. Benchmark FGS points are annotated as prior-art CONTEXT only —
never "we beat 77/79/95%".
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import yaml

_DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parents[2] / "configs" / "wrapper.yaml"
_OUTCOME = "vet_pain"        # circularity firewall: net benefit is read against VET labels
_MODELNAME = "p_pain"        # calibrated P(pain)


def _load_cfg(config_path: str = _DEFAULT_CONFIG) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _threshold_grid(cfg: dict) -> np.ndarray:
    """Low-pt welfare band. The plan's literal default is np.arange(0.01, 0.50, 0.01).

    We expose it via the harm_ratio_sweep range as a RANGE (pt = over/(under+over)),
    never a pinned ratio. If the config supplies explicit endpoints we honor them.
    """
    sweep = cfg.get("harm_ratio_sweep", {})
    rng = sweep.get("range")
    if rng:
        # undertreat:overtreat ratio r -> pt = over/(under+over) = 1/(1+r)
        ratios = np.linspace(float(rng[0]), float(rng[1]), int(sweep.get("num", 19)))
        return np.sort(1.0 / (1.0 + ratios))
    # default low-pt welfare band from §6.3 (pt 0.01..0.49 step 0.01)
    return np.arange(0.01, 0.50, 0.01)


def decision_curve(
    df: pd.DataFrame,
    thresholds: np.ndarray | None = None,
    config_path: str = _DEFAULT_CONFIG,
):
    """Run dcurves.dca on a frame with columns [vet_pain, p_pain].

    Returns the dca result object. ``dcurves`` is imported lazily so the module stays
    import-clean without the optional dependency installed.
    """
    cfg = _load_cfg(config_path)
    if thresholds is None:
        thresholds = _threshold_grid(cfg)

    from dcurves import dca  # lazy import (optional dependency)

    if _OUTCOME not in df.columns:
        raise KeyError(
            f"decision curve requires the VET outcome column '{_OUTCOME}' "
            "(circularity firewall: net benefit is computed against vet labels only)"
        )
    res = dca(
        data=df,
        outcome=_OUTCOME,
        modelnames=[_MODELNAME],
        thresholds=thresholds,
    )
    return res


def plot_decision_curve(res):
    """Net-benefit vs treat-all / treat-none. ``dcurves.plot_graphs`` lazily imported."""
    from dcurves import plot_graphs  # lazy import (optional dependency)

    return plot_graphs(res)


def _net_benefit(y_true, p_pain, pt: float) -> float:
    """Net benefit at threshold-probability pt: (TP/n) - (FP/n) * (pt/(1-pt))."""
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(p_pain, dtype=float) >= pt
    n = len(y_true)
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    w = pt / (1.0 - pt) if pt < 1.0 else np.inf
    return tp / n - (fp / n) * w


def bootstrap_net_benefit_ribbon(
    df: pd.DataFrame,
    cat_id_col: str = "cat_id",
    thresholds: np.ndarray | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    config_path: str = _DEFAULT_CONFIG,
):
    """Cat-grouped bootstrap CI ribbon for the model net-benefit curve (§6.3).

    Resampling is by INDIVIDUAL cat_id (never by clip): we resample cats with
    replacement and pool their rows, then recompute net benefit across the pt grid.
    Returns (pt_grid, nb_mean, nb_lo, nb_hi).
    """
    cfg = _load_cfg(config_path)
    if thresholds is None:
        thresholds = _threshold_grid(cfg)
    if cat_id_col not in df.columns:
        raise KeyError(
            f"cat-grouped bootstrap requires '{cat_id_col}' — never group by clip"
        )

    rng = np.random.default_rng(seed)
    cats = df[cat_id_col].unique()
    n_cats = len(cats)
    boot = np.empty((n_boot, len(thresholds)))
    for b in range(n_boot):
        drawn = rng.choice(cats, size=n_cats, replace=True)
        sub = pd.concat([df[df[cat_id_col] == c] for c in drawn], ignore_index=True)
        for j, pt in enumerate(thresholds):
            boot[b, j] = _net_benefit(sub[_OUTCOME], sub[_MODELNAME], float(pt))

    nb_mean = boot.mean(axis=0)
    nb_lo = np.quantile(boot, alpha / 2, axis=0)
    nb_hi = np.quantile(boot, 1 - alpha / 2, axis=0)
    return np.asarray(thresholds, dtype=float), nb_mean, nb_lo, nb_hi
