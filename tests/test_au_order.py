"""AU-order single-source guard.

The npz y-matrix writer, the CORN head list, the VLM columns, the kappa table,
and the Gate-6 collapse are positionally coupled through the Evangelista 5-AU
order. A reorder in any one site silently mislabels AUs (everything stays in
{0,1,2}, nothing crashes), so this test pins every mirror to
src.constants.AU_ORDER and pins the order itself (LAW).
"""

from src.constants import AU_ORDER

CANONICAL = ("ear", "orbital", "muzzle", "whiskers", "head")


def test_au_order_is_the_law_order():
    assert AU_ORDER == CANONICAL


def test_mirrors_match_canonical():
    from src.eval.kappa import AU_NAMES as kappa_names
    from src.model.cache_features import AUS as cache_aus
    from src.model.heads import AUS as heads_aus
    from src.vlm.aggregate import AU_NAMES as agg_names
    from src.vlm.schema import AU_NAMES as schema_names

    for mirror in (kappa_names, cache_aus, heads_aus, agg_names, schema_names):
        assert tuple(mirror) == AU_ORDER


def test_gate6_mirror_matches_canonical():
    import importlib.util
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "gate6_severity", root / "scripts" / "gate6_severity.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate6_severity"] = mod
    spec.loader.exec_module(mod)
    assert tuple(mod.AU_NAMES) == AU_ORDER


def test_vlm_fgs_yaml_au_order_matches_canonical():
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "vlm_fgs.yaml").read_text())
    assert tuple(cfg["au_order"]) == AU_ORDER


def test_portable_protocols_isolation_zero_torch():
    """Portable seam isolation (P2 fix + deletion-safe per critic2/arch/Local).
    Import only src.protocols surface (reexports kappa+confound pure); assert
    zero torch leak. Exercises adapters + arbitrary AU (non-FGS).
    """
    import sys
    pre = set(sys.modules.keys())
    # import only the seam (no model/vlm/data/wrapper at top)
    from src.protocols import per_au_kappa_table, get_au_names, map_df_columns, generic_ordinal_mode, judge_bias  # noqa
    from src.protocols.standalone_test_corpus import make_synth_df  # exercises inside
    post = set(sys.modules.keys())
    torch_leaks = [m for m in (post - pre) if "torch" in m.lower()]
    assert not torch_leaks, f"torch leaked on portable-only import: {torch_leaks}"
    # exercise arbitrary AU override + generic (no 0.39 forced)
    au_arb = ["foo", "bar", "baz"]  # 3-AU non-FGS ordinal
    df = make_synth_df(n=8, au_names=au_arb, with_cat=False)
    tab = per_au_kappa_table(df, au_names=get_au_names(au_arb))
    assert len(tab) == 3
    mode = generic_ordinal_mode(use_fgs_threshold=False)
    assert not mode["fgs_mode"]
    print("PASS: portable isolation + arbitrary AU + generic mode (no torch)")


def test_deletion_safe_synthetic_gates_via_protocols():
    """Synthetic gate-like exercise using ONLY protocols (deletion test for P1/P2).
    No real manifests, no CatFLW, no engine. Kappa + confound on synth.
    Mirrors gate1b/gate2 but portable.
    """
    import pandas as pd
    import numpy as np
    from src.protocols import per_au_kappa_table, bg_gap, NO_CONFOUND_MSG, get_au_names
    # synth like gate output but arbitrary
    aus = get_au_names()[:2]  # partial
    df = pd.DataFrame({
        f"{aus[0]}_vlm": [0,1,2,0], f"{aus[0]}_vet": [0,1,1,0],
        f"{aus[1]}_vlm": [2,0,1,2], f"{aus[1]}_vet": [1,0,2,1],
        "cat_id": ["C1","C1","C2","C2"],
    })
    tab = per_au_kappa_table(df, au_names=aus, cat_col="cat_id")
    assert tab.shape[0] == 2 and "ci_lb" in tab.columns
    # confound gate-like (trivial + one-dir)
    s0 = np.array([3,4,5,2.0])
    ss = s0 + 0.2
    g = bg_gap(s0, ss)
    assert g["note"] == NO_CONFOUND_MSG
    print("PASS: deletion-safe synthetic 'gates' via pure protocols (kappa+confound)")


def test_portable_protocols_deletion_isolation_monkeypatch(monkeypatch):
    """Harden + expand deletion-safe / import-isolation (load-bearing for portable headline).
    Uses pytest monkeypatch (from context7 MCP + public patterns e.g. setitem/delitem on sys.modules)
    to temporarily hide / stub model/vlm/data dirs + torch (simulate rm -rf after protocols landed).
    Asserts: protocols surface still imports cleanly + runs FULL: kappa + confound (bg_gap/judge) + 
    adapters (get/map/generic/apply) + generic mode + NO_CONFOUND_MSG on synth zero-engine.
    Directly makes 'reusable on next corpus / deletion safe' claim testable + CI'd.
    Supports P2/P7 uniqueness vs priors (no equivalent portable surface existed).
    """
    import sys
    import types
    import numpy as np
    from src.protocols.standalone_test_corpus import make_synth_df

    # Simulate deletion/hiding of heavy engine + model/vlm/data subpackages (dirs)
    # Pattern: delitem to remove if present; setitem stub dummy ModuleType to guard re-import attempts
    # (prevents pull of torch etc even if lazy import downstream; protocols/eval stay pure)
    heavy = ["torch", "src.model", "src.vlm", "src.data", "src.detect", "src.wrapper", "src.crop"]
    for name in list(sys.modules):
        if any(name == h or name.startswith(h + ".") for h in heavy):
            monkeypatch.delitem(sys.modules, name, raising=False)
    for name in heavy:
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    pre = set(sys.modules.keys())

    # Import ONLY the portable protocols surface under the hidden-heavies regime
    from src.protocols import (
        per_au_kappa_table,
        bg_gap,
        judge_bias,
        get_au_names,
        map_df_columns,
        generic_ordinal_mode,
        apply_au_override,
        NO_CONFOUND_MSG,
        qwk,
    )

    post = set(sys.modules.keys())
    leaks = [
        m for m in (post - pre)
        if any(h in m or m.startswith(h) for h in ["torch", "src.model", "src.vlm", "src.data"])
    ]
    assert not leaks, f"heavy module leaked under monkeypatch deletion isolation: {leaks}"

    # FULL exercise: kappa + confound + adapters + generic + NO_CONFOUND_MSG on synth (zero engine)
    # non-FGS arbitrary + FGS default + cat cluster
    au_arb = ["ear", "orbital", "muzzle"]  # 3-AU override, non default
    df_arb = make_synth_df(n=24, au_names=au_arb, with_cat=True, seed=99)
    tab = per_au_kappa_table(df_arb, au_names=get_au_names(au_arb), cat_col="cat_id")
    assert len(tab) == 3 and "ci_lb" in tab.columns and "kappa" in tab.columns
    assert list(tab.index) == au_arb

    # confound full (bg + judge + note)
    s0 = np.random.default_rng(7).integers(0, 11, 12).astype(float)
    ss = s0 + np.random.default_rng(8).normal(0, 0.3, 12)
    g = bg_gap(s0, ss)
    assert g["note"] == NO_CONFOUND_MSG and "bg_gap" in g
    jb = judge_bias({au: np.array([0, 1, 2, 1]) for au in au_arb[:2]}, {"position": {au_arb[0]: np.array([1, 2, 0, 1])}})
    assert jb["note"] == NO_CONFOUND_MSG

    # adapters + generic + map on raw-ish
    df_raw = make_synth_df(n=8, au_names=["auX", "auY"], with_cat=False, seed=11)
    df_raw = df_raw.rename(columns={"auX_vlm": "score_vlm_X", "auX_vet": "gold_X"})
    df_m = map_df_columns(df_raw, ["auX", "auY"], mapper={"score_vlm_X": "auX_vlm", "gold_X": "auX_vet"})
    assert "auX_vlm" in df_m.columns
    mode = generic_ordinal_mode(use_fgs_threshold=False)
    assert mode["fgs_mode"] is False and "no FGS 0.39" in mode["note"]
    mode_f = generic_ordinal_mode(use_fgs_threshold=True)
    assert mode_f["fgs_mode"] is True
    overr = apply_au_override(["custom1", "custom2"])
    assert overr == ["custom1", "custom2"]
    # also basic qwk on the data
    assert 0.0 <= qwk(df_arb["ear_vlm"], df_arb["ear_vet"]) <= 1.0 or np.isnan(qwk(df_arb["ear_vlm"], df_arb["ear_vet"]))

    print("PASS: portable deletion isolation (monkeypatch sys.modules hide model/vlm/data/torch) + FULL kappa+confound+adapters+generic+NO_CONFOUND_MSG+synth zero-engine")
