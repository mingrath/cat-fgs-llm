"""Backbone-variant contract (README §"engine is conceded plumbing").

The register variant ``dinov2_vits14_reg`` is the FIELD DEFAULT (registers suppress
attention artifacts on the localized per-AU features); the plain ``dinov2_vits14`` is
the A/B comparator. These tests pin that contract WITHOUT downloading weights (no
torch.hub fetch), and pin that the chosen variant is single-sourced in corn.yaml so
the cache provenance can record it.
"""

import pathlib

import yaml

from src.model.backbone import DEFAULT_VARIANT, SUPPORTED_VARIANTS, load_frozen_dinov2


def test_reg_variant_is_the_field_default():
    assert DEFAULT_VARIANT == "dinov2_vits14_reg"
    assert "dinov2_vits14_reg" in SUPPORTED_VARIANTS
    assert "dinov2_vits14" in SUPPORTED_VARIANTS  # plain kept as the A/B comparator


def test_unsupported_variant_is_rejected_before_any_download():
    # validation happens before torch.hub.load, so this raises without network.
    try:
        load_frozen_dinov2(device="cpu", variant="dinov2_vitb14")
    except ValueError as e:
        assert "unsupported backbone variant" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for an unsupported variant")


def test_corn_yaml_backbone_name_is_a_supported_hub_id():
    root = pathlib.Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "corn.yaml").read_text())
    name = cfg["backbone"]["name"]
    assert name in SUPPORTED_VARIANTS, (
        f"corn.yaml backbone.name={name!r} is not a torch.hub id the loader accepts; "
        f"expected one of {SUPPORTED_VARIANTS}"
    )
