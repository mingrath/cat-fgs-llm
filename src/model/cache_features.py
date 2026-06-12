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
from src.model.backbone import extract, load_frozen_dinov2, preprocess
from src.model.device import DEVICE

# manifest_csv columns (IMPLEMENTATION_PLAN §5.2):
#   img_path, cat_id, fold,
#   y_ear, y_orbital, y_muzzle, y_whiskers, y_head,  (-1 sentinel where unscored)
#   y_pain, is_vet_clean
AUS = AU_ORDER


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


def _write_provenance(out_npz, device: str, n_rows: int) -> Path:
    """Write <cache-stem>.PROVENANCE.json next to the cache (datasheet line).

    Named after the npz so the cat and horse caches each keep their own
    provenance instead of overwriting a shared file."""
    out_npz = Path(out_npz)
    prov_path = out_npz.parent / f"{out_npz.stem}.PROVENANCE.json"
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    prov = {
        "cache_npz": str(out_npz),
        "n_rows": n_rows,
        "dinov2_commit": _dinov2_commit(),
        "preprocess": {
            "resize": 518,
            "center_crop": 518,
            "interpolation": "bicubic",
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "device": device,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    prov_path.write_text(json.dumps(prov, indent=2))
    return prov_path


def build_cache(manifest_csv, out_npz, device: str = DEVICE):
    """One frozen-DINOv2 forward over every crop in manifest_csv -> out_npz.

    Saves CLS + mean-pooled patch tokens + 5-AU labels (-1 sentinel where
    unscored) + y_pain + is_vet_clean, then writes PROVENANCE.json.
    """
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    m, hidden = load_frozen_dinov2(device)
    assert hidden == 384, f"expected 384-d ViT-S features, got {hidden}"

    rows, cls_feats, patch_feats = [], [], []
    with open(manifest_csv) as f:
        for r in csv.DictReader(f):
            img = preprocess(Image.open(r["img_path"]).convert("RGB"))[None].to(device)
            cls, patch = extract(m, img)
            cls_feats.append(cls.squeeze(0).cpu().numpy())
            patch_feats.append(patch.mean(1).squeeze(0).cpu().numpy())  # mean-pool patches
            rows.append(r)

    np.savez_compressed(
        out_npz,
        cls=np.stack(cls_feats).astype(np.float32),         # [N,384]
        patch_mean=np.stack(patch_feats).astype(np.float32),  # [N,384]
        img_path=np.array([r["img_path"] for r in rows]),
        cat_id=np.array([r["cat_id"] for r in rows]),
        fold=np.array([r["fold"] for r in rows]),
        # labels: -1 sentinel where the VLM/vet did not score this AU on this image
        y=np.array(
            [[int(r[f"y_{au}"]) for au in AUS] for r in rows], dtype=np.int64
        ),
        y_pain=np.array([int(r["y_pain"]) for r in rows], dtype=np.int64),
        is_vet_clean=np.array([int(r.get("is_vet_clean", 0)) for r in rows], dtype=np.int64),
    )
    prov_path = _write_provenance(out_npz, device, len(rows))
    print(f"cached {len(rows)} crops -> {out_npz}")
    print(f"provenance -> {prov_path}")
    return out_npz
