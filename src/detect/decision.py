"""Phase A — per-image binary pain decision (IMPLEMENTATION_PLAN §2.6/§2.7).

The per-image score is the **max-confidence pain box** for that image (0 if the
detector emitted no pain box); the decision is that score vs. the tuned threshold.
The detector is detection-only (not crop-then-classify): the box doubles as the
Phase B face-crop CANDIDATE, so a separate classifier would be wasted work.

Operating point is FIXED pain-recall >= 0.90 (Evangelista anchor), NOT Youden-J /
F1 — undertreatment >> overtreatment for a welfare instrument. The threshold is set
LAST (§2.6 rung 4), swept on the *train-fold* val PR curve, then frozen and applied
to the held-out test fold. This module owns the threshold-selection rule and the
per-image scoring; the firewall (vet-confirmed labels only for the 0.39-side
sens/spec) is upstream of what enters here.
"""

import numpy as np

# pain is the positive class throughout Phase A.
PAIN_CLASS_NAME = "pain"


def image_pain_score(boxes_conf, boxes_class, pain_class_name=PAIN_CLASS_NAME):
    """Per-image score = max confidence over pain boxes, 0.0 if no pain box (§2.7).

    Args:
        boxes_conf: iterable of per-box confidences for one image.
        boxes_class: iterable of per-box class labels (names or ids) for one image.
        pain_class_name: the label that marks a pain box.

    Returns:
        float: the max pain-box confidence, or 0.0 when the image has no pain box.
    """
    pain_confs = [
        float(c) for c, cls in zip(boxes_conf, boxes_class) if cls == pain_class_name
    ]
    return max(pain_confs) if pain_confs else 0.0


def scores_from_predictions(predictions, pain_class_name=PAIN_CLASS_NAME):
    """Map a list of per-image predictions -> per-image pain scores.

    Args:
        predictions: list, one entry per image, each a (boxes_conf, boxes_class) pair.

    Returns:
        np.ndarray of shape [N] with the max pain-box confidence per image.
    """
    return np.array(
        [image_pain_score(conf, cls, pain_class_name) for conf, cls in predictions],
        dtype=float,
    )


def select_threshold(y_true, scores, pain_recall_target=0.90):
    """Pick the HIGHEST threshold still meeting pain-recall >= target (§2.6 rung 4).

    Swept on the train-fold val set; the chosen threshold is then frozen and applied
    to the held-out test fold. Operating point is fixed-recall, NEVER Youden-J / F1.

    Recall is non-increasing in the threshold, so the thresholds meeting the floor
    form an interval [min(scores), t*]. The operating point is t*: raise the
    confidence bar as far as the recall floor allows, which maximizes the
    specificity the fixed-recall point buys. (Any lower threshold also "meets"
    the floor but degenerates toward calling everything pain / specificity 0.)

    Args:
        y_true: binary array, pain = 1.
        scores: per-image pain scores (from ``scores_from_predictions``).
        pain_recall_target: fixed recall floor (Evangelista anchor, 0.90).

    Returns:
        float threshold. A box is called pain when ``score >= threshold``. Returns
        the highest candidate threshold that still achieves recall >= target; if no
        threshold reaches the target (only possible with zero positives), returns
        0.0 (decide pain for any positive score, the most permissive point).
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    # candidate thresholds = the distinct observed scores, ascending (lowest first).
    candidates = np.unique(scores)
    n_pos = float((y_true == 1).sum())
    if n_pos == 0:
        return 0.0
    best = 0.0
    found = False
    # ascending sweep: keep overwriting `best` while the floor holds, so the LAST
    # passing candidate — the highest threshold with recall >= target — survives.
    for thr in candidates:
        pred = scores >= thr
        recall = float((pred & (y_true == 1)).sum()) / n_pos
        if recall >= pain_recall_target:
            best = float(thr)
            found = True
        else:
            # recall is monotonically non-increasing in thr; once it drops below the
            # floor it stays below, so the sweep can stop here.
            break
    return best if found else 0.0


def decide(scores, threshold):
    """Apply a frozen threshold to per-image scores -> binary pain decisions."""
    return (np.asarray(scores, dtype=float) >= threshold).astype(int)
