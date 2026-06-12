"""Anti-benchmark CI helpers (IMPLEMENTATION_PLAN §0.6 / §7.2).

Two CI primitives + a disciplined rate printer:
  - clopper_pearson()  : exact binomial interval for proportions (sens/spec/NPV).
  - bootstrap_ci()     : percentile bootstrap CI for any scalar statistic.
  - print_rate()       : ALWAYS shows the distinct-pain-CAT denominator next to the
                         rate, with a Clopper-Pearson interval. Augmented copies are
                         excluded from the reported N (they never enter any denominator).

ANTI-BENCHMARK INVARIANTS (THE LAW): never frame a number as "we beat 77/79/95%";
always print the distinct-pain-CAT denominator + a CI; augmented copies never enter
the reported N. Decision-support triage framing only.

CONCEDED-PLUMBING NOTE: these helpers are utility; the headline lives in kappa.py
(VLM-as-AU-rater kappa) and confound.py (confound-attribution protocol).
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import stats


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided 1-alpha interval for a binomial proportion.

    k successes out of n trials. Returns (lo, hi). Used for sens/spec/NPV proportions
    so the small-n FGS cells get an exact (not normal-approx) interval.
    """
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def clopper_pearson_lower(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided 1-alpha LOWER bound for a binomial proportion (e.g. NPV lower bound).

    Mirrors the abstention curve's one-sided 95% NPV LOWER BOUND. The word
    "guaranteed" is BANNED — this is a lower bound, not a guarantee.
    """
    if n == 0:
        return 0.0
    if k == 0:
        return 0.0
    return float(stats.beta.ppf(alpha, k, n - k + 1))


def bootstrap_ci(
    data: Sequence,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: Sequence | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a scalar statistic over rows of `data`.

    Resamples ROWS (or whole groups, when `groups` is given — pass cat_id to keep
    cat-grouped resampling so within-cat correlation is respected). Returns
    (point, lo, hi) where lo/hi are the two-sided alpha/2 .. 1-alpha/2 percentiles.

    For a one-sided LOWER bound, take percentile 100*alpha of the bootstrap draws
    (see kappa.bootstrap_qwk_lb) — this helper returns the two-sided interval.
    """
    arr = np.asarray(data)
    rng = np.random.default_rng(seed)
    point = float(stat_fn(arr))

    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        boots = []
        for _ in range(n_boot):
            picked = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(groups == g)[0] for g in picked])
            try:
                boots.append(stat_fn(arr[idx]))
            except ValueError:
                continue
    else:
        n = len(arr)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                boots.append(stat_fn(arr[idx]))
            except ValueError:
                continue

    boots = np.asarray(boots, dtype=float)
    lo = float(np.nanpercentile(boots, 100 * alpha / 2))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return point, lo, hi


def distinct_pain_cats(cat_ids: Iterable, pain_mask: Iterable[bool]) -> int:
    """Count distinct pain-positive individuals (cat_id) — the anti-benchmark denominator.

    The pilot's positives may concentrate in a few individuals (e.g. CAT_01, 84 clips);
    this denominator is printed next to every rate so concentration is never hidden.
    """
    cats = np.asarray(list(cat_ids))
    mask = np.asarray(list(pain_mask), dtype=bool)
    return int(len(np.unique(cats[mask])))


def print_rate(
    label: str,
    k: int,
    n: int,
    cat_ids: Iterable | None = None,
    pain_mask: Iterable[bool] | None = None,
    alpha: float = 0.05,
) -> dict:
    """Print a rate WITH its Clopper-Pearson CI and the distinct-pain-CAT denominator.

    This is the disciplined reporting primitive (THE LAW §0.6/§7.2): no bare rate ever
    ships. Augmented copies must be excluded by the caller before passing k/n — they
    never enter the reported N. Returns the same numbers as a dict for downstream JSON.
    """
    lo, hi = clopper_pearson(k, n, alpha)
    rate = (k / n) if n else float("nan")
    n_cats = (
        distinct_pain_cats(cat_ids, pain_mask)
        if (cat_ids is not None and pain_mask is not None)
        else None
    )
    cat_str = "" if n_cats is None else f" | distinct-pain-CATs={n_cats}"
    print(
        f"{label}: {rate:.3f} [{lo:.3f}, {hi:.3f}] "
        f"({k}/{n}, {100 * (1 - alpha):.0f}% Clopper-Pearson){cat_str}"
    )
    return {
        "label": label,
        "rate": rate,
        "ci_lo": lo,
        "ci_hi": hi,
        "k": k,
        "n": n,
        "distinct_pain_cats": n_cats,
        "alpha": alpha,
    }
