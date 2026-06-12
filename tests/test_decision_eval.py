"""Phase A decision + eval unit tests (IMPLEMENTATION_PLAN §2.6/§2.7).

Runnable with only numpy + scikit-learn — synthetic scores/labels, never the real
export. Pins the two contracts that carry the validated Phase A numbers:

1. ``select_threshold`` returns the HIGHEST threshold still meeting the pain-recall
   floor (the fixed-recall operating point that buys specificity), never the
   degenerate lowest-passing threshold.
2. ``evaluate_fold`` tunes on train, freezes, and applies to test; ``pr_auc`` is
   Average Precision (not optimistic trapezoidal PR area).
"""

import numpy as np
from sklearn.metrics import average_precision_score

from src.detect.decision import (
    decide,
    image_pain_score,
    scores_from_predictions,
    select_threshold,
)
from src.detect.eval import aggregate_folds, bootstrap_ci, evaluate_fold, pr_auc


def test_image_pain_score_max_pain_conf():
    conf = [0.2, 0.9, 0.7]
    cls = ["pain", "no_pain", "pain"]
    assert image_pain_score(conf, cls) == 0.7


def test_image_pain_score_no_pain_box_is_zero():
    assert image_pain_score([0.9, 0.8], ["no_pain", "no_pain"]) == 0.0
    assert image_pain_score([], []) == 0.0


def test_scores_from_predictions_shape_and_values():
    preds = [
        ([0.3, 0.6], ["pain", "pain"]),
        ([0.9], ["no_pain"]),
    ]
    s = scores_from_predictions(preds)
    assert s.shape == (2,)
    assert s[0] == 0.6 and s[1] == 0.0


def test_select_threshold_highest_passing_not_degenerate_lowest():
    # 10 positives scored 0.1..1.0, negatives all 0.05. Recall >= 0.9 allows at
    # most one positive below the bar -> thresholds in (0.05, 0.2] pass; the
    # operating point is the HIGHEST passing candidate, 0.2. The degenerate
    # "lowest meeting the floor" reading would return 0.05 (everything pain).
    y = np.array([1] * 10 + [0] * 10)
    pos = np.arange(0.1, 1.01, 0.1)
    neg = np.full(10, 0.05)
    scores = np.concatenate([pos, neg])
    thr = select_threshold(y, scores, pain_recall_target=0.90)
    assert np.isclose(thr, 0.2)
    # the floor actually holds at the chosen threshold
    rec = ((scores >= thr) & (y == 1)).sum() / (y == 1).sum()
    assert rec >= 0.90


def test_select_threshold_no_positives_returns_zero():
    assert select_threshold(np.zeros(5), np.linspace(0, 1, 5)) == 0.0


def test_decide_applies_frozen_threshold():
    out = decide([0.1, 0.5, 0.9], 0.5)
    assert out.tolist() == [0, 1, 1]


def test_evaluate_fold_tunes_on_train_freezes_on_test():
    rng = np.random.default_rng(0)
    y_tr = np.array([1] * 20 + [0] * 80)
    s_tr = np.concatenate([rng.uniform(0.4, 1.0, 20), rng.uniform(0.0, 0.5, 80)])
    y_te = np.array([1] * 10 + [0] * 40)
    s_te = np.concatenate([rng.uniform(0.4, 1.0, 10), rng.uniform(0.0, 0.5, 40)])

    m = evaluate_fold(y_tr, s_tr, y_te, s_te, pain_recall_target=0.90)

    assert m["threshold"] == select_threshold(y_tr, s_tr, 0.90)  # tuned on TRAIN only
    expect = decide(s_te, m["threshold"])
    assert m["pain_recall"] == (
        ((expect == 1) & (y_te == 1)).sum() / (y_te == 1).sum()
    )
    assert 0.0 <= m["pr_auc"] <= 1.0
    assert 0.0 <= m["specificity"] <= 1.0


def test_pr_auc_is_average_precision():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.7])
    assert np.isclose(pr_auc(y, s), average_precision_score(y, s))


def test_aggregate_folds_mean_sd():
    folds = [{"pr_auc": 0.8}, {"pr_auc": 0.9}]
    agg = aggregate_folds(folds)
    assert np.isclose(agg["pr_auc"]["mean"], 0.85)
    assert np.isclose(agg["pr_auc"]["sd"], np.std([0.8, 0.9], ddof=1))


def test_bootstrap_ci_finite_and_ordered():
    rng = np.random.default_rng(1)
    y = np.array([1] * 30 + [0] * 70)
    s = np.concatenate([rng.uniform(0.5, 1.0, 30), rng.uniform(0.0, 0.6, 70)])
    ci = bootstrap_ci(y, s, pr_auc, n_boot=200)
    assert ci["n_resamples"] > 0
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_bootstrap_ci_degenerate_input_returns_nan_not_crash():
    # all-negative labels -> AP undefined on every resample; must not raise.
    y = np.zeros(10)
    s = np.linspace(0, 1, 10)
    ci = bootstrap_ci(y, s, pr_auc, n_boot=20)
    assert ci["n_resamples"] == 0 or np.isfinite(ci["mean"])
