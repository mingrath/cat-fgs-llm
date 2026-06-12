"""Phase A — evaluation (IMPLEMENTATION_PLAN §2.7).

PRIMARY metric is **PR-AUC / Average Precision** with pain = positive (the right
metric under heavy imbalance). NEVER rank on accuracy or ROC-AUC. The per-image
decision (max-conf pain box vs. tuned threshold, §2.6) is scored with PR metrics.

Operating point = FIXED pain-recall >= 0.90 (Evangelista anchor): the highest
threshold still meeting recall >= 0.90 is picked INSIDE the train folds, then frozen and
applied to the held-out test fold; the specificity it buys is reported with
bootstrap CIs. Results are reported as **mean +/- SD across the 5 cat-grouped folds**,
never a single split.

`torchmetrics.MeanAveragePrecision` is used for LOCALIZATION ONLY (box placement for
the Phase B crop candidate) — it is NOT a pain-skill verdict.

Anti-benchmark discipline (§2.6/§2.7/§2.9): never "we beat 77/79/95%". Report the
distinct-pain-CAT denominator with Clopper-Pearson / bootstrap CIs; augmented /
oversampled / copy-paste copies NEVER enter any reported N. Decision-support triage
framing only. The DINOv2-family backbone is plumbing, never claimed novel.
"""

import numpy as np
from sklearn.metrics import average_precision_score

from src.detect.decision import decide, select_threshold

DEFAULT_PAIN_RECALL_TARGET = 0.90
DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_SEED = 42


def pr_auc(y_true, scores):
    """PR-AUC as Average Precision, pain = 1. The rank metric (§2.7).

    Average Precision (step-wise sum) rather than trapezoidal area under the PR
    curve: linear interpolation between PR points is optimistically biased, which
    matters at this prevalence (~12.7%).
    """
    return float(average_precision_score(np.asarray(y_true), np.asarray(scores, dtype=float)))


def specificity(y_true, y_pred):
    """True-negative rate = TN / (TN + FP). no_pain = 0."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    denom = tn + fp
    return float(tn) / denom if denom else float("nan")


def recall(y_true, y_pred, positive=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == positive) & (y_true == positive)).sum())
    fn = int(((y_pred != positive) & (y_true == positive)).sum())
    denom = tp + fn
    return float(tp) / denom if denom else float("nan")


def precision(y_true, y_pred, positive=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == positive) & (y_true == positive)).sum())
    fp = int(((y_pred == positive) & (y_true != positive)).sum())
    denom = tp + fp
    return float(tp) / denom if denom else float("nan")


def f1(y_true, y_pred, positive=1):
    p = precision(y_true, y_pred, positive)
    r = recall(y_true, y_pred, positive)
    if not np.isfinite(p) or not np.isfinite(r) or (p + r) == 0:
        return float("nan")
    return 2 * p * r / (p + r)


def evaluate_fold(y_true_train, scores_train, y_true_test, scores_test,
                  pain_recall_target=DEFAULT_PAIN_RECALL_TARGET):
    """Score one cat-grouped fold: tune threshold on train, freeze, apply to test (§2.7).

    The threshold is selected INSIDE the train fold (highest thr still meeting
    pain-recall >= target), then frozen and applied to the held-out test fold —
    never tuned on test.

    Returns a dict of per-image decision metrics: pr_auc (rank metric on the test
    fold), the frozen threshold, and the pain-class recall/precision/F1 + specificity
    it buys on the held-out test fold.
    """
    thr = select_threshold(y_true_train, scores_train, pain_recall_target)
    y_pred_test = decide(scores_test, thr)
    return {
        "pr_auc": pr_auc(y_true_test, scores_test),     # PRIMARY rank metric
        "threshold": float(thr),
        "pain_recall": recall(y_true_test, y_pred_test, positive=1),
        "pain_precision": precision(y_true_test, y_pred_test, positive=1),
        "pain_f1": f1(y_true_test, y_pred_test, positive=1),
        "no_pain_recall": recall(y_true_test, y_pred_test, positive=0),
        "specificity": specificity(y_true_test, y_pred_test),
    }


def aggregate_folds(fold_metrics):
    """Mean +/- SD across the 5 cat-grouped folds (§2.7) — never a single split."""
    keys = fold_metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        out[k] = {"mean": float(np.nanmean(vals)), "sd": float(np.nanstd(vals, ddof=1))}
    return out


def bootstrap_ci(y_true, scores, metric_fn, n_boot=DEFAULT_N_BOOTSTRAP,
                 alpha=0.05, seed=DEFAULT_SEED):
    """Bootstrap 95% CI for a per-image metric (§2.7 reporting).

    Augmented/oversampled/copy-paste copies must NEVER enter the array passed here;
    only distinct images count toward the reported N.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            stat = metric_fn(y_true[idx], scores[idx])
        except ValueError:
            continue  # degenerate resample (e.g. one class absent) — skip
        if np.isfinite(stat):  # metrics may signal degeneracy with nan instead
            stats.append(stat)
    if not stats:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_resamples": 0}
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return {"mean": float(np.mean(stats)), "ci_low": lo, "ci_high": hi,
            "n_resamples": len(stats)}


def localization_map(preds, target):
    """Localization-only mAP via torchmetrics (§2.7) — box placement, NOT a pain verdict.

    Judges box placement for the Phase B crop candidate, not pain skill. `torch` /
    `torchmetrics` are imported lazily so this module imports without them.

    Args:
        preds: list of dicts with keys 'boxes', 'scores', 'labels' (torchmetrics format).
        target: list of dicts with keys 'boxes', 'labels'.

    Returns:
        dict with 'map', 'map_50', 'map_75', 'map_per_class'.
    """
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(class_metrics=True)
    metric.update(preds, target)
    res = metric.compute()
    return {
        "map": float(res["map"]),
        "map_50": float(res["map_50"]),
        "map_75": float(res["map_75"]),
        "map_per_class": res["map_per_class"].tolist()
        if hasattr(res["map_per_class"], "tolist")
        else res["map_per_class"],
    }
