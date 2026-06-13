"""F1 — the per-AU kappa bootstrap must cluster-resample by cat_id, not by image.

Within-cat correlation (many images per individual) means image-i.i.d. resampling
understates variance and INFLATES the one-sided 95% lower bound that fires Gate-1-B.
These tests pin that passing groups=cat_id widens the interval (lowers the LB) and that
per_au_kappa_table threads cat_id into the bootstrap automatically.
"""

import numpy as np
import pandas as pd

from src.eval.kappa import bootstrap_qwk_ci, bootstrap_qwk_lb, per_au_kappa_table


def _clustered_pairs():
    """5 cats x 20 identical rows each: 3 agreeing cats + 2 disagreeing cats.

    Rows within a cat are perfectly correlated (the worst case for i.i.d. resampling).
    Returns (y_vlm, y_vet, cat_id) with all of {0,1,2} present so qwk is well-defined.
    """
    per_cat = {
        "A": (0, 0),  # agree at 0
        "B": (1, 1),  # agree at 1
        "C": (2, 2),  # agree at 2
        "D": (1, 0),  # disagree
        "E": (1, 2),  # disagree
    }
    y_vlm, y_vet, cat = [], [], []
    for c, (v, p) in per_cat.items():
        y_vlm += [v] * 20
        y_vet += [p] * 20
        cat += [c] * 20
    return np.array(y_vlm), np.array(y_vet), np.array(cat)


def test_cluster_bootstrap_lowers_the_lower_bound():
    y_vlm, y_vet, cat = _clustered_pairs()
    _, lb_iid = bootstrap_qwk_lb(y_vlm, y_vet, n_boot=2000, seed=42)
    _, lb_clustered = bootstrap_qwk_lb(y_vlm, y_vet, n_boot=2000, seed=42, groups=cat)
    # cluster resampling draws whole cats -> far higher variance -> lower (honest) LB
    assert lb_clustered < lb_iid, (lb_clustered, lb_iid)


def test_cluster_ci_is_wider_than_iid_ci():
    y_vlm, y_vet, cat = _clustered_pairs()
    _, lo_iid, hi_iid = bootstrap_qwk_ci(y_vlm, y_vet, n_boot=2000, seed=42)
    _, lo_cl, hi_cl = bootstrap_qwk_ci(y_vlm, y_vet, n_boot=2000, seed=42, groups=cat)
    assert (hi_cl - lo_cl) > (hi_iid - lo_iid)


def test_groups_path_is_deterministic_under_seed():
    y_vlm, y_vet, cat = _clustered_pairs()
    a = bootstrap_qwk_lb(y_vlm, y_vet, n_boot=1000, seed=7, groups=cat)
    b = bootstrap_qwk_lb(y_vlm, y_vet, n_boot=1000, seed=7, groups=cat)
    assert a == b


def test_per_au_table_threads_cat_id_into_the_bootstrap():
    y_vlm, y_vet, cat = _clustered_pairs()
    df = pd.DataFrame({"ear_vlm": y_vlm, "ear_vet": y_vet, "cat_id": cat})
    clustered = per_au_kappa_table(df, au_names=["ear"], n_boot=1500, seed=42)
    # cat_col absent -> groups=None -> i.i.d. (the old, inflated behaviour)
    iid = per_au_kappa_table(df, au_names=["ear"], cat_col="missing", n_boot=1500, seed=42)
    assert clustered.loc["ear", "ci_lb"] < iid.loc["ear", "ci_lb"]
    # the distinct-pain-cat denominator is still reported when cat_id is present
    assert clustered.loc["ear", "distinct_pain_cats"] is not None
