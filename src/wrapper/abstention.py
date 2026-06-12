"""Defer-to-vet ABSTENTION — one-sided 95% NPV LOWER-BOUND curve
(IMPLEMENTATION_PLAN §6.4, §5.5 line 2304).

We report a **one-sided 95% NPV lower-bound curve**: the NPV lower bound at each
abstention rate, honestly shown to clear >= target only at the abstention rates it
actually clears. The word **"guaranteed" is BANNED** in this module — say "lower
bound," never "guarantee."

WIRED PATH (what ``npv_lb_curve`` actually computes): for each target abstention
rate we abstain on the most-uncertain rows around the FIXED 0.39 point, classify
the rest by the 0.39 cutoff, and take the one-sided (1 - delta) Clopper-Pearson
LOWER bound on NPV over the non-abstained predicted-negatives (``_npv_lower_bound``;
delta=0.05 -> one-sided 95%). The 0.39 here is the band-CENTRE selector, NOT a
replacement for the §6.3 fixed-recall operating point — the two are different
quantities by design (the band brackets 0.39; the deployed cutoff is the recall-0.90
point in operating_point.py).

OPTIONAL / NOT WIRED INTO THE CURVE:
  - ``mapie_ltt_band`` — an alternative risk-controlled band via MAPIE 1.4.x
    Learn-then-Test (``BinaryClassificationController`` on
    ``negative_predictive_value``). Provided for the reviewer who wants the LTT
    framing; the shipped curve uses the Clopper-Pearson path above, not this.
  - ``aurc`` — risk-coverage AURC via ``torch-uncertainty`` (max-prob baseline).
    A reported extra, not part of the NPV curve.

Gate 0: the sample-size power calc runs BEFORE any vet spend. If the budget cannot
certify NPV >= target at <= the max abstention rate, the curve is EXPLORATORY-ONLY
(flag set) and the decision curve (§6.3) provides the operating point. Not a kill.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import yaml

# one-sided Clopper-Pearson lower bound — the SINGLE implementation, shared with
# Gate-0 power sizing so the certified band and the pre-registered budget use one math.
from src.eval.bootstrap import clopper_pearson_lower
# the fixed point the band brackets (band selector) — single 0.39 definition.
from src.vlm.aggregate import POINT_DECISION_THRESHOLD as _POINT

_DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parents[2] / "configs" / "wrapper.yaml"


def _load_abstention_cfg(config_path: str = _DEFAULT_CONFIG) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)["abstention"]


def npv_lb_curve(
    y_vet,
    p_pain,
    config_path: str = _DEFAULT_CONFIG,
):
    """Trace (abstention_rate, NPV_lower_bound) over the configured abstention grid.

    For each target abstention rate we abstain on the most-uncertain rows around the
    fixed 0.39 point (band centred on 0.39), classify the rest by the 0.39 cutoff, and
    report the one-sided 95% NPV LOWER BOUND on the non-abstained predicted-negatives.
    delta is read from config (ltt_delta, default 0.05 -> one-sided 95%).
    """
    cfg = _load_abstention_cfg(config_path)
    grid = list(cfg["grid"])
    delta = float(cfg.get("ltt_delta", 0.05))
    target_npv = float(cfg.get("npv_target", 0.90))

    y = np.asarray(y_vet).astype(int)
    p = np.asarray(p_pain, dtype=float)
    dist = np.abs(p - _POINT)             # distance from the fixed 0.39 point

    rows = []
    n = len(p)
    for rate in grid:
        k = int(round(rate * n))          # rows to abstain on (most-uncertain near 0.39)
        if k > 0:
            cut_idx = np.argsort(dist)[:k]
            keep = np.ones(n, dtype=bool)
            keep[cut_idx] = False
        else:
            keep = np.ones(n, dtype=bool)
        yk, pk = y[keep], p[keep]
        pred_neg = pk < _POINT
        tn = int((pred_neg & (yk == 0)).sum())
        fn = int((pred_neg & (yk == 1)).sum())
        # NPV lower bound: predicted-negatives as Bernoulli trials (success = true
        # negative), one-sided (1 - delta) Clopper-Pearson lower bound. n==0 -> 0.0.
        lb = clopper_pearson_lower(tn, tn + fn, alpha=delta)
        rows.append(
            {
                "abstention_rate": float(rate),
                "npv_lower_bound": lb,
                "clears_target": bool(lb >= target_npv),
                "n_pred_neg": int(tn + fn),
            }
        )
    return pd.DataFrame(rows)


def mapie_ltt_band(predict_proba_pos, X_cal, y_cal, X_test,
                   target_npv: float = 0.90, delta: float = 0.05):
    """Risk-controlled defer band via MAPIE risk control (lazy import; mapie 1.4.x).

    Controls ``negative_predictive_value`` at ``target_npv`` with confidence
    ``1 - delta`` (delta=0.05 -> one-sided 95% LOWER BOUND), calibrated on the
    vet-confirmed calibration split and applied to test.

    Args:
        predict_proba_pos: callable X -> P(pain) in [0,1] (e.g.
            ``lambda X: clf.predict_proba(X)[:, 1]``).
        X_cal, y_cal: calibration split (vet-confirmed labels only — firewall).
        X_test: rows to band.
        target_npv: NPV floor the controller certifies.
        delta: 1 - confidence of the bound (0.05 -> one-sided 95%).

    Returns the controller's predictions on X_test; rows it cannot certify form
    the DEFER-TO-VET band.
    """
    # import guarded: requires the mapie>=1.4,<2 pin in pyproject.
    from mapie.risk_control import (
        BinaryClassificationController,
        negative_predictive_value,
    )

    ctrl = BinaryClassificationController(
        predict_function=predict_proba_pos,
        risk=negative_predictive_value,
        target_level=target_npv,
        confidence_level=1 - delta,
    )
    ctrl.calibrate(X_cal, y_cal)          # learns the valid thresholds (lambdas)
    return ctrl.predict(X_test)


def aurc(confidence, error_indicator):
    """Risk-coverage AURC via torch-uncertainty (lazy import; torchmetrics-style).

    Max-prob confidence is the BASELINE; a learned selector would be the reported
    delta. ``error_indicator`` is (pred != vet_pain). Bootstrap-CI is computed by the
    caller (torch-uncertainty does not).
    """
    import torch
    from torch_uncertainty.metrics import AURC  # lazy import (optional dependency)

    m = AURC()
    m.update(
        torch.as_tensor(confidence, dtype=torch.float),
        torch.as_tensor(error_indicator),
    )
    return float(m.compute())


def gate0_exploratory_flag(
    npv_curve: pd.DataFrame,
    config_path: str = _DEFAULT_CONFIG,
) -> bool:
    """True => EXPLORATORY-ONLY framing (Gate 0 power budget unmet).

    If no abstention rate <= the configured maximum clears the target NPV lower bound,
    the abstention deliverable is exploratory-only and the decision curve (§6.3)
    provides the operating point. The word "guaranteed" never appears. Not a kill.
    """
    # Reuse the curve's own clears_target column (computed against the same config
    # in npv_lb_curve) instead of re-deriving 'lb >= target' from a second config
    # read — one definition of "clears", so the flag and the curve cannot disagree.
    cfg = _load_abstention_cfg(config_path)
    max_rate = max(cfg["grid"])
    within = npv_curve[npv_curve["abstention_rate"] <= max_rate]
    return bool(not within["clears_target"].any())
