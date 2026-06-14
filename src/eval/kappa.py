"""GUARDED VLM-as-AU-rater reliability check — per-AU quadratic kappa (FINAL_DIRECTION §8).

This is the SUPPORTING leg, NOT the headline. The headline / spine of the paper is the
one-directional, power-conditioned confound-attribution protocol (src/eval/confound.py).
The kappa here is a guarded, inspected-not-validated reliability check whose result is
PENDING an independent vet anchor; it is ranked BELOW the confound protocol in prose and
kept as kill-tree insurance (sole-survivor headline only if the confound leg degrades).

PORTABLE / DATASET-AGNOSTIC reliability check (FINAL_DIRECTION §5/§8): scores any face
corpus's per-AU VLM (or other rater) labels vs a vet anchor; reports quadratic κ + CI
lower bound (cluster-bootstrap by groups e.g. cat_id recommended). Fires on CI-LB.
The CI-lower-bound acceptance gate is TEXTBOOK CLINIMETRICS, not a novel increment
(Tractenberg 2010; Donner & Rotondi 2010; Sim & Wright 2005). The rubric-independence
guard below is a published reliability test (Weng 2026), likewise not branded as novel.
A high κ measures weak-labeling capability ONLY if (i) anchor independent AND (ii) VLM
rubric differs from vet's (otherwise rubric-following). Borrowed clean patterns:
m-rewardbench (add quadratic weights), prometheus-eval (ordinal alpha complement).

Claim under check: a frozen VLM can weak-label feline FGS action units at human-rater
agreement. This is a CHECK WITH A RESULT PENDING — per-AU quadratic-weighted Cohen kappa
of VLM-vs-vet, gated on the bootstrap CI LOWER BOUND. A result requires an independent
vet anchor that does not yet exist; the artifact shipped today is the guarded check
itself, disclosed as inspected-not-validated.

CIRCULARITY FIREWALL (THE LAW, §4.7):
  - QWK-vs-VLM is NEVER validation. The kappa here measures VLM-vs-VET agreement, a
    labeler-agreement measurement, not a graded-instrument claim. It is NOT evidence
    that the 0.39 flag is correct.
  - A QWK computed against VLM-derived labels would only measure the model re-learning
    the VLM heuristic — forbidden as a validation metric.
  - sens/spec at 0.39 live elsewhere and are estimated on VET-CONFIRMED labels only.

INTERPRETATION GUARD (README §1):
  - A high kappa ONLY measures weak-labeling capability if BOTH conditions hold:
    (i) the vet anchor is independent (not authored by the same rater who scored
        the target data), AND (ii) the rubric handed to the VLM is different from
        the rubric the vet used to score the anchor.
  - If both conditions hold: kappa measures capability (can the VLM learn FGS?).
  - If the VLM's rubric matches the vet's rubric: kappa measures rubric-following,
    not weak-labeling skill. State which one a given run measures in the report.

The `m-rewardbench` row-paired pattern is NOMINAL (labels=[0,1,2] only fixes the class
set); adding weights="quadratic" makes it ORDINAL — that addition is the contribution.
The VLM emits ONLY the 5 AU atoms in {0,1,2}; the 0-10 sum / 0.39 flag are computed in
code, never by the VLM. The word "graded" never appears in a validated-claim sentence —
the per-AU kappa here is a labeler-agreement measurement, not a graded-instrument claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.constants import AU_ORDER

# Canonical AU order — MUST match the convolution order in src.model.decode.sum_pmf.
AU_NAMES = list(AU_ORDER)


def qwk(y_vlm, y_vet) -> float:
    """Quadratic-weighted Cohen kappa over the ordinal AU classes {0,1,2}.

    labels=[0,1,2] fixes the class set; weights="quadratic" makes it ORDINAL (the
    m-rewardbench bug-to-avoid). Convention follows §6.1: cohen_kappa_score(y1, y2)
    is symmetric in its two label arguments.
    """
    return cohen_kappa_score(y_vlm, y_vet, labels=[0, 1, 2], weights="quadratic")


def _boot_indices(rng, n: int, groups=None) -> np.ndarray:
    """Row indices for ONE bootstrap resample.

    Without `groups`: i.i.d. row resample (rng.integers). With `groups` (e.g. cat_id):
    CLUSTER resample — draw whole cats with replacement and gather their rows, so
    within-cat correlation is respected and the CI is not optimistically narrow. This
    mirrors src.eval.bootstrap.bootstrap_ci's cluster branch; the κ gate inherits the
    same cat_id grouping the wrapper curves and StratifiedGroupKFold already use (THE
    LAW: CV grouped by cat_id). Multiple images per cat are NOT independent draws.
    """
    if groups is None:
        return rng.integers(0, n, n)
    uniq = np.unique(groups)
    picked = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([np.where(groups == g)[0] for g in picked])


def bootstrap_qwk_lb(
    y_vlm,
    y_vet,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
    groups=None,
) -> tuple[float, float]:
    """Paired bootstrap one-sided 95% LOWER BOUND of the per-AU quadratic kappa.

    Resamples paired rows by default; pass `groups` (cat_id) to CLUSTER-resample whole
    cats instead — the gate-feeding lower bound must respect within-cat correlation, so
    Gate-1-B always passes groups=cat_id (image-i.i.d. resampling inflates the LB). The
    same cat_id grouping is used by the wrapper bootstraps and StratifiedGroupKFold.
    Degenerate resamples (a single class present -> kappa undefined) are skipped;
    sklearn signals these by RETURNING nan (it does not raise), so nan replicates are
    filtered explicitly — without that filter a near-constant AU silently yields a nan
    lower bound and the gate reports a quality FAIL for what is actually a
    degeneracy/power problem.
    Returns (point_kappa, ci_lower_bound); ci_lower_bound is nan ONLY when every
    resample was degenerate — callers must surface that as "degenerate", not "fail".

    GATE ON THE LOWER BOUND, NOT THE POINT (§4.6.3): at the pilot n a 0.60 floor is
    statistically indistinguishable from a true 0.47, so a point-estimate gate is not
    a gate. This CI-lower-bound acceptance rule is standard clinimetric practice
    (Tractenberg 2010; Donner & Rotondi 2010; Sim & Wright 2005), applied here as
    a real engineering constraint on the reliability check — not claimed as novel.
    """
    y_vlm = np.asarray(y_vlm)
    y_vet = np.asarray(y_vet)
    groups = None if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    n = len(y_vlm)
    stats_ = []
    for _ in range(n_boot):
        idx = _boot_indices(rng, n, groups)
        try:
            val = qwk(y_vlm[idx], y_vet[idx])
        except ValueError:
            continue  # degenerate resample (one class) -> skip
        if not np.isnan(val):  # sklearn returns nan (no raise) on degenerate input
            stats_.append(val)
    if not stats_:  # every resample degenerate (pilot n, near-constant labels)
        return qwk(y_vlm, y_vet), float("nan")
    lo = float(np.percentile(stats_, 100 * alpha))  # one-sided 95% LOWER BOUND
    return qwk(y_vlm, y_vet), lo


def bootstrap_qwk_ci(
    y_vlm,
    y_vet,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
    groups=None,
) -> tuple[float, float, float]:
    """Two-sided percentile CI of the per-AU quadratic kappa (§6.1 reporting form).

    Paired resample of rows by default; pass `groups` (cat_id) to cluster-resample
    whole cats (same discipline as bootstrap_qwk_lb). n_boot default 10000, percentile
    alpha/2 .. 1-alpha/2. Returns (kappa, lo, hi). The gate fires on the one-sided lower
    bound (bootstrap_qwk_lb); this two-sided CI is for the reported kappa table.
    """
    y_vlm = np.asarray(y_vlm)
    y_vet = np.asarray(y_vet)
    groups = None if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    n = len(y_vlm)
    boots = []
    for _ in range(n_boot):
        idx = _boot_indices(rng, n, groups)
        try:
            boots.append(qwk(y_vlm[idx], y_vet[idx]))
        except ValueError:
            continue
    boots = np.asarray(boots, dtype=float)
    return (
        qwk(y_vlm, y_vet),
        float(np.nanpercentile(boots, 100 * alpha / 2)),
        float(np.nanpercentile(boots, 100 * (1 - alpha / 2))),
    )


def per_au_kappa_table(
    df: pd.DataFrame,
    au_names: list[str] | None = None,
    cat_col: str = "cat_id",
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the per-AU kappa table (§4.6 / §6.1).

    Expects one row PER IMAGE with columns `{au}_vlm`, `{au}_vet` for each AU, and
    (optionally) a cat_id column so the distinct-pain-CAT denominator is reported
    alongside the kappa (anti-benchmark discipline, §4.8). Augmented copies must be
    excluded by the caller before this is called — they never enter the reported N.

    Returns columns: au, kappa, ci_lo, ci_hi, ci_lb (one-sided 95% lower bound), n,
    distinct_pain_cats.
    """
    au_names = au_names or AU_NAMES
    # cluster-bootstrap by cat_id when the column is present, so the reported CI and
    # the gate-firing lower bound both respect within-cat correlation (THE LAW).
    groups = df[cat_col].to_numpy() if cat_col in df.columns else None
    rows = []
    for au in au_names:
        v = df[f"{au}_vlm"].to_numpy()
        p = df[f"{au}_vet"].to_numpy()
        k, lo, hi = bootstrap_qwk_ci(v, p, n_boot=n_boot, alpha=alpha, seed=seed, groups=groups)
        # same n_boot/seed as the two-sided CI so the gate-firing lb and the
        # reported interval come from one RNG stream (reproducible together)
        _, lb = bootstrap_qwk_lb(v, p, n_boot=n_boot, alpha=alpha, seed=seed, groups=groups)
        n_cats = None
        if cat_col in df.columns:
            # distinct individuals among VLM-or-vet pain-positive (AU >= 1) rows
            pain = (v >= 1) | (p >= 1)
            n_cats = int(df.loc[pain, cat_col].nunique())
        rows.append(
            {
                "au": au,
                "kappa": k,
                "ci_lo": lo,
                "ci_hi": hi,
                "ci_lb": lb,
                "n": int(len(df)),
                "distinct_pain_cats": n_cats,
            }
        )
    return pd.DataFrame(rows).set_index("au")
