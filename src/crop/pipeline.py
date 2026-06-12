"""Production face-crop pipeline + manifest (IMPLEMENTATION_PLAN §3.5).

detect -> quality-gate -> expand -> (align) -> letterbox -> resize 518 RGB.

Produces the exact crop tensor the frozen DINOv2 ViT-S/14 engine consumes, plus
``crop_manifest.parquet`` (one row per source image). Rows with
``route_vet=True`` are EXCLUDED from feature caching and from every sens/spec/κ
denominator downstream (§3.5) — "detector/quality failure -> defer-to-vet" is an
explicit, counted abstention channel, not a silent drop. Augmented copies never
enter any reported N (anti-benchmark; §3.5).

This is conceded preprocessing plumbing, not a claimed-novel artifact (§3).
"""

from __future__ import annotations

import pathlib

import cv2
import numpy as np
import pandas as pd
import yaml

from src.crop.align_eyes import align_by_eyes
from src.crop.expand import expand_box
from src.crop.quality_gate import (
    GateResult,
    QualityThresholds,
    quality_gate,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "crop.yaml"

# Crop manifest columns (§3.5). One row per source image.
MANIFEST_COLUMNS = [
    "image_id", "clip_id", "cat_id",
    "box_xyxy", "expand", "clipped",
    "align_mode", "interocular_px",
    "route_vet", "reasons",
    "crop_path",
]


def load_crop_config(config_path: str | pathlib.Path = _DEFAULT_CONFIG) -> dict:
    """Read configs/crop.yaml (output contract, expand, gate thresholds, align)."""
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def letterbox_square(img: np.ndarray) -> np.ndarray:
    """Pad to square BEFORE resize with replicate-edge pad (§3.0).

    Preserves aspect ratio so ear tips/whiskers are not stretched; replicate
    avoids a black border the backbone reads as a feature. Returns a square
    image; the caller does the single resize to `edge`.
    """
    h, w = img.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    return cv2.copyMakeBorder(
        img, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE
    )


def resize_to_edge(square_bgr: np.ndarray, edge: int) -> np.ndarray:
    """Single resize of a square crop to (edge, edge).

    INTER_AREA on downscale, INTER_CUBIC on upscale (quality on small faces;
    §3.0). Input must already be square (letterboxed).
    """
    h, w = square_bgr.shape[:2]
    interp = cv2.INTER_AREA if edge < max(h, w) else cv2.INTER_CUBIC
    return cv2.resize(square_bgr, (edge, edge), interpolation=interp)


def finalize_crop(crop_bgr: np.ndarray, edge: int, patch: int = 14) -> np.ndarray:
    """letterbox -> resize -> BGR->RGB; assert the (edge,edge,3) /14 contract (§3.6).

    Returns an (edge, edge, 3) uint8 RGB array ready to be written as PNG.
    """
    assert edge % patch == 0, f"edge {edge} not divisible by patch {patch}"
    square = letterbox_square(crop_bgr)
    resized = resize_to_edge(square, edge)
    # OpenCV decodes BGR; DINOv2 expects RGB — convert before save (§3.0).
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    assert rgb.shape == (edge, edge, 3), f"bad crop shape {rgb.shape}, want {(edge, edge, 3)}"
    return rgb


def process_image(
    img_bgr: np.ndarray,
    box_norm: tuple[float, float, float, float],
    *,
    cfg: dict,
    thresholds: QualityThresholds,
    haar_cascade=None,
    align_mode: str = "box",
    landmarks: dict | None = None,
):
    """Run one image through detect-result -> gate -> expand -> (align) -> 518 RGB.

    Parameters
    ----------
    img_bgr : np.ndarray
        Full source image (OpenCV BGR).
    box_norm : (cx, cy, w, h)
        Normalized YOLO face box for this image.
    cfg : dict
        configs/crop.yaml.
    thresholds : QualityThresholds
        Frozen §3.2 operating values.
    haar_cascade : cv2.CascadeClassifier | None
        Loaded cat-face cascade (detect-failure abstention).
    align_mode : str
        "box" (box_ok) or "eye_aligned" (need_aligner) — the Gate-5 decision.
    landmarks : dict | None
        Optional pixel-coord landmarks in the EXPANDED-CROP frame:
        {nose, left_eye, right_eye, eye_lids:(top,bottom,inner,outer)}. Used for
        the yaw/roll/EAR checks and (in eye_aligned mode) the alignment. When
        absent those checks are skipped (the Haar detect-failure check still
        applies); see the runtime TODO in the module docstring of pipeline CLI.

    Returns
    -------
    (crop_rgb_or_None, gate, box_xyxy, clipped, interocular_px)
        crop_rgb is None iff the image is routed to vet (no crop is cached).
    """
    img_h, img_w = img_bgr.shape[:2]
    cx, cy, w, h = box_norm
    expand = float(cfg["expand"])
    height_mult = float(cfg["height_expand_mult"])
    edge = int(cfg["edge"])
    patch = int(cfg["patch"])

    # §3.1 EXPAND (never shrink) + clip flag
    (x1, y1, x2, y2), clipped = expand_box(
        cx, cy, w, h, img_w, img_h, expand=expand, height_mult=height_mult
    )
    box_xyxy = (x1, y1, x2, y2)
    crop = img_bgr[y1:y2, x1:x2]

    lm = landmarks or {}
    gate: GateResult = quality_gate(
        crop,
        thresholds,
        nose=lm.get("nose"),
        left_eye=lm.get("left_eye"),
        right_eye=lm.get("right_eye"),
        eye_lids=lm.get("eye_lids"),
        haar_cascade=haar_cascade,
        clipped=clipped,
    )

    interocular_px: float | None = None
    if gate.route_vet:
        # routed to vet -> NO crop cached (excluded from feature cache / denominators)
        return None, gate, box_xyxy, clipped, interocular_px

    # §3.4 alignment fallback only when Gate 5 decided need_aligner
    if align_mode == "eye_aligned" and lm.get("left_eye") is not None \
            and lm.get("right_eye") is not None:
        align_cfg = cfg["align"]
        aligned = align_by_eyes(
            crop,
            lm["left_eye"], lm["right_eye"],
            out=edge,
            left_frac=tuple(align_cfg["left_target"]),
            right_frac=tuple(align_cfg["right_target"]),
        )
        crop_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        assert crop_rgb.shape == (edge, edge, 3), f"bad aligned shape {crop_rgb.shape}"
        # post-align interocular = interocular_frac * edge (eyes hit canonical line)
        interocular_px = float(align_cfg["interocular_frac"]) * edge
    else:
        crop_rgb = finalize_crop(crop, edge, patch=patch)
        if lm.get("left_eye") is not None and lm.get("right_eye") is not None:
            # measured in crop frame; scale-after-letterbox+resize is handled at
            # audit time (§3.3). Here we record the in-crop interocular as-is.
            le = np.asarray(lm["left_eye"], dtype=np.float64)
            re = np.asarray(lm["right_eye"], dtype=np.float64)
            interocular_px = float(np.linalg.norm(le - re))

    return crop_rgb, gate, box_xyxy, clipped, interocular_px


def write_manifest(rows: list[dict], out_path: str | pathlib.Path) -> pd.DataFrame:
    """Write crop_manifest.parquet with the §3.5 columns; return the DataFrame."""
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df
