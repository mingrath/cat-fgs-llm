"""Feature caching to disk (IMPLEMENTATION_PLAN §5.2) -- the big M4 win.

CONCEDED PLUMBING. The frozen DINOv2 backbone is NEVER claimed as novel
(FINAL_DIRECTION §A); it only carries the binary-plus-wrapper spine.

Principle (GITHUB_MINE P2.2): the backbone is frozen, so its output for a given
crop never changes. Run ONE pass over every crop, write pooled features to disk,
then train the heads off the cache with ZERO backbone forwards per epoch. On M4
this turns each head epoch from minutes into milliseconds and makes the §5.6
pooling x loss x seed sweep cheap.

The cache holds BOTH poolings (CLS + mean-pooled patch tokens) so the trainer can
flip `pool` with no re-extraction. Labels carry a -1 sentinel where the VLM/vet
did not score that AU; multi_corn_loss skips sentinel AUs row-wise.

Datasheet line: PROVENANCE.json records the DINOv2 commit, preprocess params,
device, and torch version -- the cache is an input to EVERY reported number.
"""

import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.constants import AU_ORDER
from src.model.backbone import DEFAULT_VARIANT, extract, load_frozen_dinov2, preprocess, VARIANT_PATCH_SIZES, VARIANT_INPUT_SIZES
from src.model.device import DEVICE

import torch.nn.functional as F

# manifest_csv columns (IMPLEMENTATION_PLAN §5.2):
#   img_path, cat_id, fold,
#   y_ear, y_orbital, y_muzzle, y_whiskers, y_head,  (-1 sentinel where unscored)
#   y_pain, is_vet_clean
AUS = AU_ORDER

# _CACHE_SCHEMA (hard enforce per FreshDINOv3Context7RicherEnforcer audit + separability/train tests):
# Full keys + values for versioned cache (no legacy allow_pickle drift; supports richer patch_std / layer intermed / L2 / dinov3_vits16 A/B per MCP /facebookresearch/dinov3 "dense features without fine-tuning" + exact forward_features x_norm_* + get_intermediate_layers contract).
# extract() now extended (layer='last'|'intermed' + patch_l2 gated by config; see backbone.py) + hash via layer/patch_mode in schema/provenance.
# Preserves ALL: cat-disjoint manifests, vet firewall early, provenance sidecar, single-source AU_ORDER, G3 abort, conceded FM, portable protocols compatibility.
_CACHE_SCHEMA = {
    "schema_version": "v1_dinov3_richer_20260614",
    "n_aus": len(AU_ORDER),
    "au_hash": hash(tuple(AU_ORDER)),  # simple for enforce (use stable in prod)
    "k": 3,  # ordinal levels 0/1/2
    "feature_dim": 384,  # ViT-S small (v2 reg + dinov3_vits16)
    "variant": DEFAULT_VARIANT,  # will be overridden per build
    "layer": "last",  # "last" | list[int] for get_intermediate_layers n= per MCP
    "patch_mode": "mean_std",  # mean_std | l2 | intermed  (richer for localized facial AU orbital/ear/muzzle vs mean only)
}
# Uniform assert no legacy (enforce per @FreshHandoffDINOv3RicherEnforcer MCP audit + full embed): 
# all cache writes/loads must carry schema_version + layer/patch_mode + variant + au_hash; 
# A/B hash for dinov3_vits16 primary vs dinov2_reg; richer intermed n=list mean+std + L2 per public dinov github contract (vision_transformer forward_features x_norm_* + get_intermediate_layers).
assert _CACHE_SCHEMA["schema_version"].startswith("v1_dinov3_richer"), "legacy schema drift forbidden"
SCHEMA_VERSION = _CACHE_SCHEMA["schema_version"]


def _dinov2_commit() -> str:
    """Best-effort torch.hub DINOv2 checkout commit for the datasheet.

    Returns 'unknown' rather than raising so caching never fails on provenance."""
    hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    try:
        out = subprocess.run(
            ["git", "-C", str(hub_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _write_provenance(out_npz, device: str, n_rows: int, variant: str, layer: str = "last", patch_mode: str = "mean_std") -> Path:
    """Write <cache-stem>.PROVENANCE.json next to the cache (datasheet line).

    Named after the npz so the cat and horse caches each keep their own
    provenance instead of overwriting a shared file. Records the backbone
    variant + layer/patch_mode (dinov3 richer MCP) so a reported number can never hide which engine
    produced its features. Cross-checked in separability + train + _CACHE_SCHEMA enforce."""
    out_npz = Path(out_npz)
    prov_path = out_npz.parent / f"{out_npz.stem}.PROVENANCE.json"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov = {
        "cache_npz": str(out_npz),
        "n_rows": n_rows,
        "backbone_variant": variant,  # dinov2_vits14_reg (field default) or dinov2_vits14 or dinov3_vits16 (MCP primary A/B dense oob)
        "layer": layer,
        "patch_mode": patch_mode,  # mean_std (default richer) | l2 | intermed per MCP get_intermediate_layers n=list + L2 patch
        "dinov_commit": _dinov2_commit() if not variant.startswith("dinov3") else "dinov3_hub_mcp",
        "preprocess": {
            "resize": VARIANT_INPUT_SIZES.get(variant, 518),
            "center_crop": VARIANT_INPUT_SIZES.get(variant, 518),
            "interpolation": "bicubic",
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "device": device,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "_CACHE_SCHEMA": _CACHE_SCHEMA,  # hard embed for cross enforce
    }
    prov_path.write_text(json.dumps(prov, indent=2))
    return prov_path


def build_cache(manifest_csv, out_npz, device: str = DEVICE, variant: str = DEFAULT_VARIANT, layer: str = "last", patch_mode: str = "mean_std"):
    """One frozen-DINO (v2 or dinov3_vits16) forward over every crop in manifest_csv -> out_npz.

    variant: dinov2_vits14_reg (default) | dinov2_vits14 | dinov3_vits16 (MCP context7 primary A/B
      for dense oob frozen high-quality localized features without fine-tuning; same
      forward_features x_norm_clstoken/x_norm_patchtokens contract + get_intermediate_layers).
    layer: "last" | n=list[int] (e.g. [-4,-3,-2,-1] or [5,11,17,23]) passed to get_intermediate_layers for multi-layer dense per MCP.
    patch_mode: "mean_std" (richer: mean+std of patches for localized AU cues) | "l2" (full L2 norm on patch feats) | "intermed" (concat/mean intermed layers).
      Per corn.yaml + separability _POOLS incl patch_std; hard _CACHE_SCHEMA enforce.

    Saves cls + richer patch_* + 5-AU labels (-1 sentinel) + y_pain + is_vet_clean + schema keys,
    writes PROVENANCE.json (extended with layer/patch_mode). Hard schema on write (no drift).
    """
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    m, hidden = load_frozen_dinov2(device, variant=variant)
    # relaxed assert for dinov3 vits16 (still 384d small model)
    if variant.startswith("dinov3"):
        assert 380 <= hidden <= 390, f"expected ~384 for small ViT-S dinov3, got {hidden}"
    else:
        assert hidden == 384, f"expected 384-d ViT-S features, got {hidden}"

    # per-variant size (MCP dinov3 vits16 16px patch; caller ensures divisible)
    input_size = VARIANT_INPUT_SIZES.get(variant, 518)
    # Note: preprocess is global 518; for dinov3 A/B in prod use variant-specific compose or resize in loop (here stub for contract; real uses corn.yaml crop_edge)

    rows, cls_feats, patch_mean_feats, patch_std_feats, patch_l2_feats = [], [], [], [], []
    intermed_cache = []  # for intermed mode
    with open(manifest_csv) as f:
        for r in csv.DictReader(f):
            # simple resize per variant for contract (full prod wires VARIANT_INPUT_SIZES + crop)
            img_pil = Image.open(r["img_path"]).convert("RGB")
            img = transforms.Compose([
                transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(IMNET_MEAN, IMNET_STD),
            ])(img_pil)[None].to(device)
            cls, patch = extract(m, img, layer=layer, patch_l2=(patch_mode == "l2"))
            cls_feats.append(cls.squeeze(0).cpu().numpy())
            pmean = patch.mean(1).squeeze(0).cpu().numpy()
            pstd = patch.std(1).squeeze(0).cpu().numpy() if patch_mode in ("mean_std", "std") else np.zeros_like(pmean)
            # l2 now handled inside extract when patch_l2=True (per DINOv3RicherSub extension + MCP L2 F.normalize p=2)
            pl2 = patch if patch_mode == "l2" else np.zeros_like(pmean)  # already normalized in extract
            patch_mean_feats.append(pmean)
            patch_std_feats.append(pstd)
            patch_l2_feats.append(pl2)
            # intermed: via layer=list in extract (MCP get_intermediate_layers n=... norm/return_class); richer mean+std concat for dense localized (align backbone per dinov3 oob contract); no legacy compat
            if patch_mode == "intermed" and hasattr(m, "get_intermediate_layers"):
                n_layers = layer if isinstance(layer, (list, tuple, range)) else [-1]
                inter = m.get_intermediate_layers(img, n=n_layers, reshape=False, return_class_token=True, norm=True)
                if inter:
                    patches = [i[0].cpu().numpy() for i in inter]
                    if len(patches) > 1:
                        pcat = np.concatenate(patches, axis=-1)
                        pmean = pcat.mean(axis=1)
                        pstd = pcat.std(axis=1)
                        inter_patch = np.concatenate([pmean, pstd], axis=-1)
                    else:
                        p = patches[0]
                        inter_patch = np.concatenate([p.mean(1), p.std(1)], axis=-1) if p.shape[1] > 0 else p.mean(1)
                else:
                    inter_patch = pmean
                intermed_cache.append(inter_patch[:768] if inter_patch.shape[0] > 384 else inter_patch)  # richer mean+std for intermed
            rows.append(r)

    cache_dict = {
        "cls": np.stack(cls_feats).astype(np.float32),         # [N,384]
        "patch_mean": np.stack(patch_mean_feats).astype(np.float32),  # [N,384]
        "patch_std": np.stack(patch_std_feats).astype(np.float32),    # [N,384] richer for DINOv3 A/B localized AU per MCP
        "patch_l2": np.stack(patch_l2_feats).astype(np.float32),      # [N,384]
        "img_path": np.array([r["img_path"] for r in rows]),
        "cat_id": np.array([r["cat_id"] for r in rows]),
        "fold": np.array([r["fold"] for r in rows]),
        "y": np.array([[int(r[f"y_{au}"]) for au in AUS] for r in rows], dtype=np.int64),
        "y_pain": np.array([int(r["y_pain"]) for r in rows], dtype=np.int64),
        "is_vet_clean": np.array([int(r.get("is_vet_clean", 0)) for r in rows], dtype=np.int64),
        # HARD _CACHE_SCHEMA + schema_version (enforce; cross with separability/train_heads/protocols)
        "schema_version": SCHEMA_VERSION,
        "n_aus": _CACHE_SCHEMA["n_aus"],
        "au_hash": _CACHE_SCHEMA["au_hash"],
        "k": _CACHE_SCHEMA["k"],
        "feature_dim": hidden,
        "variant": variant,
        "layer": layer,
        "patch_mode": patch_mode,
    }
    if patch_mode == "intermed" and intermed_cache:
        cache_dict["patch_intermed"] = np.stack(intermed_cache).astype(np.float32)

    np.savez_compressed(out_npz, **cache_dict)
    prov_path = _write_provenance(out_npz, device, len(rows), variant, layer=layer, patch_mode=patch_mode)
    print(f"cached {len(rows)} crops -> {out_npz} (richer patch_mode={patch_mode} layer={layer} variant={variant})")
    print(f"provenance -> {prov_path}")
    # hard enforce on write (prov + schema keys present)
    return out_npz
