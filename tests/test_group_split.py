"""Group-parse + fold-disjointness unit tests (IMPLEMENTATION_PLAN §1.1 step 4,
§0.3 tests/test_group_split.py intent).

Runnable with only stdlib + numpy + pandas + scikit-learn + pyyaml -- builds a
synthetic dataframe and never requires the real export. Two halves:

1. The four ``group_id`` asserts from §1.1 step 4.
2. A synthetic split test: one cat across many clips lands in ONE CV group and
   folds are cat-disjoint with CAT_01 carved out as the LOIO hold-out.
"""

import numpy as np
import pandas as pd

from src.data.folds import LOIO_FOLD, build_folds, distinct_pain_cats
from src.data.parse import group_id

_H = "0123456789abcdef0123456789abcdef"  # placeholder 32-hex Roboflow hash


def test_group_id_cam_prefix():
    # CAT_01_..  ->  "CAT01_00000100"
    assert group_id(f"CAT_01_00000100_007_png.rf.{_H}.jpg") == "CAT01_00000100"


def test_group_id_plain_prefix():
    # plain  ->  "P_00000100"
    assert group_id(f"00000100_007_png.rf.{_H}.jpg") == "P_00000100"


def test_clip_00000100_two_distinct_groups():
    # the lone numeric collision: same clip number, two distinct groups.
    cam = group_id(f"CAT_01_00000100_007_png.rf.{_H}.jpg")
    plain = group_id(f"00000100_007_png.rf.{_H}.jpg")
    assert len({cam, plain}) == 2


def test_all_synthetic_names_parse():
    # every well-formed name parses; an unparseable name must raise.
    names = [
        f"CAT_01_00000100_007_png.rf.{_H}.jpg",
        f"CAT_02_00000200_001_png.rf.{_H}.jpg",
        f"00000100_007_png.rf.{_H}.jpg",
        f"00000300_042_png.rf.{_H}.jpg",
    ]
    for n in names:
        assert group_id(n)  # non-empty, no raise
    try:
        group_id("not_a_valid_name.jpg")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unparseable filename must raise ValueError")


def _synthetic_df(n_cats: int = 40, imgs_per_cat: int = 30, seed: int = 0) -> pd.DataFrame:
    """One individual across many clips -> one cat_id; image-level y.

    CAT_01 is the dominant individual (LOIO hold-out). The rest get ~12.7%
    pain prevalence assigned per individual so per-fold floors clear.
    """
    rng = np.random.default_rng(seed)
    rows = []
    # dominant LOIO individual
    for f in range(imgs_per_cat * 4):
        rows.append(
            {"image_id": f"CAT01_00000100_{f:03d}", "cat_id": "CAT_01", "y": int(f % 8 == 0)}
        )
    # remaining individuals across several clips each
    for c in range(2, n_cats + 1):
        cat_id = f"cat_{c:03d}"
        # assign each individual a base pain rate; clips share the cat_id (ONE group)
        pos_rate = 0.127
        for clip in range(3):
            for f in range(imgs_per_cat // 3):
                y = int(rng.random() < pos_rate)
                rows.append(
                    {
                        "image_id": f"{cat_id}_clip{clip}_{f:03d}",
                        "cat_id": cat_id,
                        "y": y,
                    }
                )
    return pd.DataFrame(rows)


def test_one_cat_lands_in_one_group_and_folds_are_cat_disjoint(tmp_path):
    df = _synthetic_df()
    out_csv = tmp_path / "folds.csv"
    folds = build_folds(df, out_csv=out_csv)

    # LOIO carve-out: CAT_01 is never in a CV fold (sentinel fold only).
    cat01 = folds[folds["cat_id"] == "CAT_01"]
    assert (cat01["fold"] == LOIO_FOLD).all()
    assert (folds.loc[folds["fold"] != LOIO_FOLD, "cat_id"] != "CAT_01").all()

    # one cat across many clips -> exactly one fold assignment (ONE group).
    per_cat_folds = folds.groupby("cat_id")["fold"].nunique()
    assert (per_cat_folds == 1).all(), "an individual straddles folds"

    # folds are cat-disjoint across the CV partition.
    cv = folds[folds["fold"] != LOIO_FOLD]
    fold_cats = {f: set(g["cat_id"]) for f, g in cv.groupby("fold")}
    fold_ids = sorted(fold_cats)
    for i in range(len(fold_ids)):
        for j in range(i + 1, len(fold_ids)):
            assert fold_cats[fold_ids[i]].isdisjoint(fold_cats[fold_ids[j]])

    # written CSV carries exactly the four columns in order.
    written = pd.read_csv(out_csv)
    assert list(written.columns) == ["image_id", "fold", "y", "cat_id"]

    # the reporting denominator is computable and positive.
    assert distinct_pain_cats(folds) > 0
