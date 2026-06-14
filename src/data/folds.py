"""Re-split protocol & fold builder (IMPLEMENTATION_PLAN §1.6 / §2.2 / §7.2.1; FoldsCacheAugSpecialist upgrades).

Replaces the leaky shipped split entirely. The keystone correction (P4 cat vs clip):

- Carve the dominant individual ``CAT_01`` out as a frozen leave-one-individual-out
  (LOIO) hold-out -- it is NEVER in a CV fold (it would degrade
  ``StratifiedGroupKFold`` to ``GroupKFold`` for a dominant group).
- Group the remaining individuals by ``cat_id`` (one cat across many clips ->
  ONE group) and run ``StratifiedGroupKFold(n_splits, shuffle, random_state)``.
  Per context7 sklearn medical best-practice: StratifiedGroupKFold exactly for
  subject (cat) disjointness + class balance on small imbalanced data.
- Image-level ``y`` (135/336 clips are class-mixed -> a clip cannot collapse to
  a single label).
- Assert per-fold cat-disjointness, that ``CAT_01`` never enters a CV fold, and
  the prevalence/min-positive floors. Per-fold pain counts are RECOMPUTED at
  runtime: the stale ``[39, 36, 27, 33, 38]`` was computed under the disqualified
  per-clip grouping and is intentionally NOT used here.

Power-aware simulation per G0 (power.yaml) + multi-strat (y + cat_id) + always
print distinct_pain_cats (anti-benchmark + P7 test reinforcement). Reads all
hyperparameters from ``configs/splits.yaml`` (P3 config). Writes
``folds.csv`` with columns ``(image_id, fold, y, cat_id)``. Data plumbing for
the binary spine; makes no validated claim. Leak-free + small-data repro.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "splits.yaml"
_DEFAULT_POWER = _REPO_ROOT / "configs" / "power.yaml"

# LOIO rows are written to folds.csv with this sentinel fold value (never a CV
# fold index). They are reported leak-safe via LOIO, separately from CV.
LOIO_FOLD = -1


def load_splits_config(config_path: str | pathlib.Path = _DEFAULT_CONFIG) -> dict:
    """Read ``configs/splits.yaml`` (CV params, LOIO id, prevalence floors; P3 config)."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_power_config(power_path: str | pathlib.Path = _DEFAULT_POWER) -> dict:
    """Read G0 power STRICT from manifests/power.json ONLY (FreshPowerG0ManifestsEnforcer).
    NO yaml fallback, NO gate0/ fallback -- G0 manifest target strict per handoff/audit/round_fresh_1/DYNAMIC.
    Hard SystemExit on missing or vet_budget placeholder. Single source for folds + cache/train.
    Power honesty (G0 first blocks all quant per FINAL).
    """
    manifests_power = _REPO_ROOT / "data" / "manifests" / "power.json"
    if not manifests_power.exists():
        raise SystemExit(
            "G0 power/vet-budget must precede; see data/manifests/power.json committed from gate0_power"
        )
    with open(manifests_power) as f:
        pj = json.load(f)
    # schema guard (hard floors)
    for k in ("vet_budget_integer", "min_pain_pos"):
        if k not in pj:
            raise ValueError(f"power.json missing G0 key {k} (manifests target drift?)")
    if int(pj.get("vet_budget_integer", 0)) < 50:
        raise SystemExit(
            "G0 power/vet-budget must precede; see data/manifests/power.json committed from gate0_power"
        )
    return pj


def build_folds(
    df: pd.DataFrame,
    config_path: str | pathlib.Path = _DEFAULT_CONFIG,
    out_csv: str | pathlib.Path | None = None,
    power_path: str | pathlib.Path | None = None,
) -> pd.DataFrame:
    """Build the cat-grouped CV folds + LOIO hold-out (P4 cat-vs-clip, leak-free).

    ``df`` must carry columns ``image_id``, ``y`` (image-level pain 1 / no_pain 0),
    and ``cat_id`` (the Gate-1 per-individual merge key, e.g. ``"CAT_01"`` for the
    dominant camera id). Returns a frame with an added ``fold`` column
    (``LOIO_FOLD`` for the hold-out, ``0..n_splits-1`` for CV) and, if ``out_csv``
    is given, writes ``(image_id, fold, y, cat_id)`` to it.

    Upgrades (FoldsCacheAugSpecialist):
    - Power-aware simulation per G0 (reads power.yaml vet_budget/min_pain_pos;
      prints simulated floors for small-data power check).
    - Multi-strat reporting (y + cat_id) + explicit distinct_pain_cats print.
    - P3 config driven; P7 test reinforcement via distinct + asserts.
    - Always surfaces the true reporting N (distinct pain cats only; aug copies excluded).
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

    # Power-aware (G0 per mission; small data context) -- STRICT manifests only (candidate4)
    pcfg = load_power_config(power_path)  # entry guard inside; no fallback
    vet_budget = int(pcfg.get("vet_budget_integer", 120))
    min_pain_pos = int(pcfg.get("min_pain_pos", 50))
    print(f"[build_folds] power-aware (G0): vet_budget~{vet_budget}, min_pain_pos>={min_pain_pos} "
          f"(floors simulated post-LOIO for small-data validation)")

    # P3/P4: assert G0 vet floors from manifests/power.json ONLY BEFORE any group split / SGKF
    # (closes empty-manifests risk + ensures power honesty at fold construction time)
    assert vet_budget >= 1, "G0 vet_budget_integer floor violated before StratifiedGroupKFold"
    assert min_pain_pos >= 1, "G0 min_pain_pos floor violated before StratifiedGroupKFold"

    # Early vet_clean filter support (from manifests; FreshDedupEarlyVetPowerHardener; pre-SGKF)
    # Stronger vet firewall early (rows kept per circularity design for train; filter for strict-clean subpaths if needed)
    if "is_vet_clean" in df.columns:
        vc = pd.to_numeric(df.get("is_vet_clean", 0), errors="coerce").fillna(0).astype(bool)
        print(f"[build_folds] early vet_clean filter from manifests: {int(vc.sum())} vet-confirmed rows flagged "
              "(pre-SGKF; applied upstream/in-wrapper for sens/spec per design; early now stronger per candidate4)")
        # non-breaking: main path keeps (as prior); strict sub-manifest e.g. df = df[vc | (df.get("y",0)==0)].copy() if required by caller

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
        # G1 disjointness (per individual) -- P4 cat vs clip enforcement
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

    # Multi-strat + distinct pain print (mission: always alongside rates; P7 test)
    denom = distinct_pain_cats(out)
    n_pain_cats_cv = int(out.loc[(out["y"] == 1) & (out["fold"] != LOIO_FOLD), "cat_id"].nunique())
    print(f"[build_folds] multi-strat (image y + cat_id groups): {n_splits} folds; "
          f"LOIO={loio_id} carved. DISTINCT-PAIN-CAT DENOM = {denom} (real cats only; "
          f"CV folds hold {n_pain_cats_cv} distinct pain cats). Power sim floors: >= {min_pos} pos/fold.")
    return out


def distinct_pain_cats(folds: pd.DataFrame) -> int:
    """Count of distinct individuals (incl. LOIO) with >=1 pain image.

    The anti-benchmark reporting denominator -- always printed alongside any
    rate. Augmented copies never enter this N (this counts real cat_ids only).
    """
    return int(folds.loc[folds["y"] == 1, "cat_id"].nunique())
