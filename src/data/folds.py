"""Re-split protocol & fold builder (IMPLEMENTATION_PLAN §1.6 / §2.2 / §7.2.1).

Replaces the leaky shipped split entirely. The keystone correction:

- Carve the dominant individual ``CAT_01`` out as a frozen leave-one-individual-out
  (LOIO) hold-out -- it is NEVER in a CV fold (it would degrade
  ``StratifiedGroupKFold`` to ``GroupKFold`` for a dominant group).
- Group the remaining individuals by ``cat_id`` (one cat across many clips ->
  ONE group) and run ``StratifiedGroupKFold(n_splits, shuffle, random_state)``.
- Image-level ``y`` (135/336 clips are class-mixed -> a clip cannot collapse to
  a single label).
- Assert per-fold cat-disjointness, that ``CAT_01`` never enters a CV fold, and
  the prevalence/min-positive floors. Per-fold pain counts are RECOMPUTED at
  runtime: the stale ``[39, 36, 27, 33, 38]`` was computed under the disqualified
  per-clip grouping and is intentionally NOT used here.

Reads all hyperparameters from ``configs/splits.yaml``. Writes
``folds.csv`` with columns ``(image_id, fold, y, cat_id)``. Data plumbing for
the binary spine; makes no validated claim.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "splits.yaml"

# LOIO rows are written to folds.csv with this sentinel fold value (never a CV
# fold index). They are reported leak-safe via LOIO, separately from CV.
LOIO_FOLD = -1


def load_splits_config(config_path: str | pathlib.Path = _DEFAULT_CONFIG) -> dict:
    """Read ``configs/splits.yaml`` (CV params, LOIO id, prevalence floors)."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_folds(
    df: pd.DataFrame,
    config_path: str | pathlib.Path = _DEFAULT_CONFIG,
    out_csv: str | pathlib.Path | None = None,
) -> pd.DataFrame:
    """Build the cat-grouped CV folds + LOIO hold-out.

    ``df`` must carry columns ``image_id``, ``y`` (image-level pain 1 / no_pain 0),
    and ``cat_id`` (the Gate-1 per-individual merge key, e.g. ``"CAT_01"`` for the
    dominant camera id). Returns a frame with an added ``fold`` column
    (``LOIO_FOLD`` for the hold-out, ``0..n_splits-1`` for CV) and, if ``out_csv``
    is given, writes ``(image_id, fold, y, cat_id)`` to it.
    """
    cfg = load_splits_config(config_path)
    cv = cfg["cv"]
    n_splits = int(cv["n_splits"])
    random_state = int(cv["random_state"])
    shuffle = bool(cv["shuffle"])
    loio_id = cfg["loio_holdout"]
    prevalence_target = float(cfg["prevalence_target"])
    min_pos = int(cfg["min_pos_per_fold"])
    tol = float(cfg.get("prevalence_tol", 0.05))

    required = {"image_id", "y", "cat_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_folds: df missing columns {sorted(missing)}")

    work = df.reset_index(drop=True).copy()
    work["fold"] = LOIO_FOLD  # default; CV rows overwritten below

    # 1) carve the dominant individual out as a frozen LOIO hold-out -- NEVER in CV.
    loio_mask = (work["cat_id"] == loio_id).to_numpy()
    cv_df = work.loc[~loio_mask].reset_index(drop=True)

    X = np.arange(len(cv_df))                       # image-index placeholder
    y = cv_df["y"].to_numpy()
    g = cv_df["cat_id"].to_numpy()

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )

    # map CV-frame position -> original work-frame index for fold assignment
    cv_to_work = work.loc[~loio_mask].index.to_numpy()

    # Per-fold pain counts are RECOMPUTED here (stale [39,36,27,33,38] left OUT,
    # since it was computed under the disqualified per-clip grouping).
    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups=g)):
        # G1 disjointness (per individual)
        assert set(g[tr]).isdisjoint(set(g[va])), f"fold {fold}: cat straddles train/val"
        # dominant individual never enters a CV fold
        assert loio_id not in set(g[va]), f"fold {fold}: {loio_id} leaked into a CV fold"
        # re-verify prevalence + min-positive floors after removing CAT_01
        pf = float(y[va].mean())
        n_pos = int(y[va].sum())
        assert abs(pf - prevalence_target) <= tol, (
            f"fold {fold}: prevalence {pf:.3f} off target {prevalence_target} (tol {tol})"
        )
        assert n_pos >= min_pos, f"fold {fold}: {n_pos} pain imgs < floor {min_pos}"
        work.loc[cv_to_work[va], "fold"] = fold

    out = work.loc[:, ["image_id", "fold", "y", "cat_id"]].copy()

    if out_csv is not None:
        out_csv = pathlib.Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)

    return out


def distinct_pain_cats(folds: pd.DataFrame) -> int:
    """Count of distinct individuals (incl. LOIO) with >=1 pain image.

    The anti-benchmark reporting denominator -- always printed alongside any
    rate. Augmented copies never enter this N (this counts real cat_ids only).
    """
    return int(folds.loc[folds["y"] == 1, "cat_id"].nunique())
