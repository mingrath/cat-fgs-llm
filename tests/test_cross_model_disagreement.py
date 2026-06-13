"""F3 — cross-model (epistemic) disagreement per AU, complementary to alpha_per_au.

Pins that two disagreeing models surface a non-zero per-AU gap, agreeing AUs surface 0,
and a single-model parquet yields nan (no between-model signal). pyarrow/krippendorff
are skipped if absent so the suite still runs on a minimal env.
"""

import pandas as pd
import pytest

pytest.importorskip("krippendorff")
pytest.importorskip("pyarrow")

from src.vlm.consistency import cross_model_disagreement_per_au  # noqa: E402
from src.vlm.schema import AU_NAMES  # noqa: E402


def _write_parquet(tmp_path, rows):
    df = pd.DataFrame(rows)
    p = tmp_path / "runs.parquet"
    df.to_parquet(p)
    return p


def _row(image, model, **au_scores):
    r = {"custom_id": f"{image}__{model}__run0"}
    # every AU column must exist (the function loops AU_NAMES); default agree at 0
    for au in AU_NAMES:
        r[f"{au}_score"] = au_scores.get(au, 0)
    return r


def test_two_models_disagree_on_ear_agree_on_orbital(tmp_path):
    rows = [
        _row("i1", "m1", ear=0, orbital=1),
        _row("i1", "m2", ear=2, orbital=1),  # ear off by 2, orbital identical
        _row("i2", "m1", ear=0, orbital=1),
        _row("i2", "m2", ear=2, orbital=1),
    ]
    out = cross_model_disagreement_per_au(_write_parquet(tmp_path, rows))
    assert out["ear"] == 2.0       # |0-2| on both images
    assert out["orbital"] == 0.0   # models agree
    assert out["muzzle"] == 0.0    # untouched AUs default to agreement


def test_single_model_yields_nan(tmp_path):
    rows = [_row("i1", "m1", ear=1), _row("i2", "m1", ear=2)]
    out = cross_model_disagreement_per_au(_write_parquet(tmp_path, rows))
    assert out["ear"] != out["ear"]  # nan: no between-model signal
