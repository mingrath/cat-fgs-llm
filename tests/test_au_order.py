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
