"""F8 — frozen-feature separability probe (diagnostic floor, held-out + cat-grouped).

Pins that (a) separable features score a high held-out AUC, (b) the cat_id circularity
guard fires when an individual straddles the train/eval split, and (c) the result is
labelled a diagnostic floor, never a validated number.
"""

import numpy as np
import pytest

from src.eval.separability import probe_separability


def _build_npz(path, leak=False, seed=0):
    """Synthetic cache: pain rows clustered at +mu, no-pain at -mu, separable by design.

    4 cats x 10 rows. cats c0,c1 -> fold '0'; c2,c3 -> fold '1' (disjoint), unless
    `leak`, which puts c0 in both folds to trip the circularity guard.
    """
    rng = np.random.default_rng(seed)
    dim = 8
    cls, cat, fold, ypain = [], [], [], []
    plan = [("c0", "0"), ("c1", "0"), ("c2", "1"), ("c3", "1")]
    if leak:
        plan.append(("c0", "1"))  # same individual in both folds
    for cid, fld in plan:
        for i in range(10):
            pain = i % 2  # balanced within cat
            mu = 1.0 if pain else -1.0
            cls.append(rng.normal(mu, 0.1, size=dim))
            cat.append(cid)
            fold.append(fld)
            ypain.append(pain)
    cls = np.asarray(cls, dtype=np.float32)
    n = len(ypain)
    np.savez(
        path,
        cls=cls,
        patch_mean=cls.copy(),
        cat_id=np.array(cat),
        fold=np.array(fold),
        y=np.full((n, 5), -1, dtype=np.int64),
        y_pain=np.array(ypain, dtype=np.int64),
    )
    return path


def test_separable_features_score_high_auc(tmp_path):
    npz = _build_npz(tmp_path / "cache.npz")
    out = probe_separability(npz, pool="cls", train_folds=(0,), eval_folds=(1,), task="pain")
    assert out["linear_auc"] > 0.8
    assert out["knn_auc"] > 0.8
    assert out["n_train"] == 20 and out["n_eval"] == 20
    assert "not a validated result" in out["note"]


def test_circularity_guard_raises_on_cat_leak(tmp_path):
    npz = _build_npz(tmp_path / "leak.npz", leak=True)
    with pytest.raises(ValueError, match="leakage"):
        probe_separability(npz, pool="cls", train_folds=(0,), eval_folds=(1,), task="pain")


def test_overlapping_fold_spec_raises(tmp_path):
    npz = _build_npz(tmp_path / "cache2.npz")
    with pytest.raises(ValueError, match="overlap"):
        probe_separability(npz, train_folds=(0, 1), eval_folds=(1,))
