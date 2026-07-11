"""Frozen-feature separability probe (GITHUB_MINE Pass-3 P3.2 — tactical hardening).

A cheap, NON-PARAMETRIC sanity baseline on the cached frozen DINOv2 features: BEFORE
fitting the CORN / binary-pain heads, can a kNN or a linear probe separate pain from
no-pain at all? The trained head should BEAT this floor; if it does not, the problem is
the features (or the labels), not the head.

DIAGNOSTIC ONLY — NO claim ships from this number (THE LAW: the engine is conceded
plumbing, the graded layer is inspected-not-validated). It is a floor the head should
clear, never a reported result.

CIRCULARITY GUARD: fits on `train_folds`, scores on DISJOINT `eval_folds`, and ASSERTS
that no cat_id appears in both — the same StratifiedGroupKFold-by-cat_id discipline used
everywhere else (THE LAW: CV grouped by cat_id). Running a probe on the rows it was fit
on, or letting one cat straddle the split, would make the number circular. Reads the npz
written by src.model.cache_features (keys: cls, patch_mean, cat_id, fold, y, y_pain).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import AU_ORDER
from src.model.cache_features import _CACHE_SCHEMA

_POOLS = ("cls", "patch_mean", "patch_std")  # patch_std exposed for richer DINOv3 A/B per MCP dinov3 + schema enforcement


def _labels(data, task: str) -> np.ndarray:
    """Binary target: task='pain' -> y_pain; task=<AU name> -> (AU score >= 1)."""
    if task == "pain":
        return np.asarray(data["y_pain"]).astype(int)
    if task in AU_ORDER:
        col = list(AU_ORDER).index(task)
        return (np.asarray(data["y"])[:, col] >= 1).astype(int)
    raise ValueError(f"task must be 'pain' or one of {AU_ORDER}, got {task!r}")


def _auc_acc(clf, X_tr, y_tr, X_ev, y_ev) -> tuple[float, float]:
    """Fit clf on train, return (roc_auc, accuracy) on eval; auc is nan if single-class."""
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_ev)
    acc = float(accuracy_score(y_ev, pred))
    if len(np.unique(y_ev)) < 2:
        return float("nan"), acc
    proba = clf.predict_proba(X_ev)[:, 1]
    return float(roc_auc_score(y_ev, proba)), acc


def probe_separability(
    npz_path,
    pool: str = "cls",
    train_folds=(0, 1, 2),
    eval_folds=(3, 4),
    task: str = "pain",
    k: int = 5,
    seed: int = 42,
) -> dict:
    """kNN + linear-probe separability of frozen features on HELD-OUT folds (diagnostic).

    npz_path: cache from src.model.cache_features.build_cache.
    pool: 'cls' or 'patch_mean' (which pooled feature to probe).
    train_folds / eval_folds: disjoint fold ids; the probe fits on train, scores on eval.
    task: 'pain' (y_pain) or an AU name (binary AU-present >= 1).

    Returns dict(pool, task, n_train, n_eval, knn_auc, knn_acc, linear_auc, linear_acc,
    distinct_pain_cats_eval, note). The `note` states this is a diagnostic floor, not a
    validated result. Raises ValueError if a cat_id appears in both splits (circularity).
    """
    if pool not in _POOLS:
        raise ValueError(f"pool must be one of {_POOLS}, got {pool!r}")
    data = np.load(npz_path, allow_pickle=True)  # allow object for test synthetic npz (schema_version etc as U or str scalars); prod caches from cache_features use primitive arrays so safe either way. Enables the stricter _CACHE_SCHEMA enforce while keeping separability tests working.

    # Enforce schema_version + cross-prov + dim hygiene on read (builds on
    # protocols adapters + importlib-style seam for portable versioned caches; uniform with train_heads).
    # Supports A/B dinov3 richer (patch_std/layer/patch_mode per MCP). Preserves small-data repro. Hard on mismatch (no legacy tolerate).
    if "schema_version" not in data:
        raise ValueError(f"cache schema_version missing (uniform enforce): {npz_path}; run fresh cache build (supports richer intermed/L2)")
    # Full _CACHE_SCHEMA keys + PROVENANCE cross (FreshDINOv3RicherEnforcer tiny #1; uniform with train_heads). Supports A/B dinov3 richer (patch_std/layer/patch_mode per MCP). Hard on mismatch (no legacy tolerate).
    # PRACTICAL A/B uniform hard enforce: fixed keys exact match to _CACHE_SCHEMA; variable (variant/layer/patch_mode) require presence (actual value from build, cross prov for hash); supports intermed n=list mean+std richer + dinov3_vits16 vs v2_reg
    fixed_keys = ("n_aus", "au_hash", "k", "feature_dim")
    for schema_key in fixed_keys:
        if schema_key not in data or data[schema_key] != _CACHE_SCHEMA[schema_key]:
            raise ValueError(f"_CACHE_SCHEMA key mismatch on {schema_key} (enforce)")
    for schema_key in ("variant", "layer", "patch_mode"):
        if schema_key not in data:
            raise ValueError(f"_CACHE_SCHEMA key missing {schema_key} (uniform enforce no legacy)")
    # optional prov cross for A/B hash (layer/patch_mode/variant drive from corn.yaml)
    if "provenance" in data:
        prov = data["provenance"] if isinstance(data["provenance"], dict) else {}
        for schema_key in ("variant", "layer", "patch_mode"):
            if schema_key in prov and data.get(schema_key) != prov.get(schema_key):
                raise ValueError(f"PROVENANCE cross mismatch on {schema_key} (full hash enforce)")
    n_au = len(AU_ORDER)
    if "y" in data:
        yarr = np.asarray(data["y"])
        if yarr.ndim > 1 and yarr.shape[1] != n_au:
            raise ValueError(f"cache y dim {yarr.shape[1]} != AU_ORDER len {n_au}")
    # PROVENANCE sidecar cross (variant/registers/layer/patch_mode if present)
    if "provenance" in data and "variant" in data["provenance"] and data.get("variant") != data["provenance"]["variant"]:
        raise ValueError("PROVENANCE variant cross mismatch (full enforce)")
    # n_aus cross with protocols (generic surface)
    from src.protocols.adapters import get_au_names
    if len(get_au_names()) != n_au:
        raise ValueError("n_aus mismatch vs protocols generic (portable tie)")

    X = np.asarray(data[pool], dtype=np.float32)
    y = _labels(data, task)
    cat = np.asarray(data["cat_id"]).astype(str)
    fold = np.asarray(data["fold"]).astype(str)  # fold stored as strings in the cache

    train_set = {str(f) for f in train_folds}
    eval_set = {str(f) for f in eval_folds}
    if train_set & eval_set:
        raise ValueError(f"train_folds and eval_folds overlap: {train_set & eval_set}")

    tr = np.isin(fold, list(train_set))
    ev = np.isin(fold, list(eval_set))

    # CIRCULARITY GUARD: no individual may sit in both the fit and the score split.
    shared = set(cat[tr]) & set(cat[ev])
    if shared:
        raise ValueError(
            f"cat_id leakage across train/eval folds (probe would be circular): {sorted(shared)}"
        )

    knn_auc, knn_acc = _auc_acc(
        KNeighborsClassifier(n_neighbors=k), X[tr], y[tr], X[ev], y[ev]
    )
    linear_auc, linear_acc = _auc_acc(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        X[tr], y[tr], X[ev], y[ev],
    )

    return {
        "pool": pool,
        "task": task,
        "n_train": int(tr.sum()),
        "n_eval": int(ev.sum()),
        "knn_auc": knn_auc,
        "knn_acc": knn_acc,
        "linear_auc": linear_auc,
        "linear_acc": linear_acc,
        "distinct_pain_cats_eval": int(np.unique(cat[ev & (y == 1)]).size),
        # NOT a validated result: a diagnostic floor the trained head should beat.
        "note": "diagnostic separability floor (held-out, cat-grouped); not a validated result",
    }
