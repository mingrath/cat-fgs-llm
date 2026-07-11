# Key source excerpts
## src/crop/pipeline.py
"""Production face-crop pipeline + manifest (IMPLEMENTATION_PLAN §3.5; FoldsCacheAugSpecialist).

detect -> quality-gate -> expand -> (align) -> letterbox -> resize 518 RGB.

Produces the exact crop tensor the frozen DINOv2/DINOv3-ready ViT-S/14 engine consumes (the
``_reg`` register variant ``dinov2_vits14_reg`` (or dinov3) is the field default — registers
suppress attention artifacts that hurt the dense, localized per-AU features; context7
confirms patch+cls for localization on facial cues),
plus ``crop_manifest.parquet`` (one row per source image). Rows with
``route_vet=True`` are EXCLUDED from feature caching and from every sens/spec/κ
denominator downstream (§3.5) — "detector/quality failure -> defer-to-vet" is an
explicit, counted abstention channel, not a silent drop. Augmented copies never
enter any reported N (anti-benchmark; §3.5).

FGS-safe: THIS STAGE IS DETERMINISTIC CLEAN CROPS ONLY (no aug). Light geometry
only upstream in detector (see train_yolo FGS_SAFE_AUG + expand). Heavy aug
banned — would destroy whiskers/orbital/ear AUs (Steagall alignment insight).
Test-time aug (TTA) for robustness at infer: apply light hflip/scale at Phase B
wrapper/decision time (never pollutes cache or denominators). Copy-paste pain
with care: detector imbalance only.

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
## src/detect/train_rfdetr.py
"""Phase A — RF-DETR-Nano fold trainer (IMPLEMENTATION_PLAN §2.4).

Phase A is the **binary spine** (pain / no_pain) and the only Phase that emits a
*validated* number under the firewall: detector P/R and the fixed-recall operating
point are reported on vet-confirmable binary labels, NEVER on VLM-derived AU labels.
The RF-DETR backbone (DINOv2-family) is **plumbing, never claimed novel**.

Full 5-fold training runs on Colab T4 (CUDA): there the `device` arg is OMITTED so
RF-DETR auto-detects CUDA. The *same* call with `device="mps"`, `epochs=1`, and a
small subset is the M4 smoke run (Gate-4 MPS-correctness half for Phase A). CUDA
pins live ONLY in notebooks/colab_train_rfdetr.ipynb, never here.

`rfdetr` is an opt-in `detect`-group dependency; it is imported LAZILY inside the
trainer so this module imports clean without it.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "detect_rfdetr.yaml"


def load_config(config_path=DEFAULT_CONFIG):
    with open(config_path) as f:
        return yaml.safe_load(f)


def num_classes_from_class_names(class_names):
    """Derive ``num_classes`` from the data, never hard-code 2 (the §2.4 footgun box).

    RF-DETR sizes its classification head from ``num_classes`` and the value must
    cover the max COCO ``category_id`` present (RF-DETR reserves the high index for
    background; a 2-category 1-based file can require 3). We let the data set it:
    ``len(class_names)`` for the sorted ['no_pain','pain'] case; override only if
    the head errors against the actual ``category_id`` scheme.
    """
    return len(class_names)


def train_fold(
    fold: int,
    dataset_dir=None,
    class_names=None,
    device=None,
    config_path=DEFAULT_CONFIG,
    output_dir=None,
):
    """Train RF-DETR-Nano on one cat-grouped fold (run once per k in {0..4}).

    Args:
        fold: fold index K. Used to resolve ``./datasets/fold{K}`` and ``./out/fold{K}``.
        dataset_dir: COCO fold dir (train/ valid/ test/ each with _annotations.coco.json).
            Defaults to ``datasets/fold{K}`` relative to cwd (the Colab layout).
        class_names: sorted COCO class names from the datamodule, e.g. ['no_pain','pain'].
            ``num_classes`` is derived from this (footgun box); pass it after
            ``dm.setup('fit'); dm.class_names``.
        device: OMIT (None) on Colab to auto-detect CUDA; pass "mps" for the M4 smoke run.
        config_path: configs/detect_rfdetr.yaml (knobs read from here, never hardcoded).
        output_dir: checkpoint/log dir; defaults to ``out/fold{K}``.
    """
    # Lazy import: rfdetr is the opt-in 'detect' group dep (heavy). Keeps this module
    # import-clean without it. Swap RFDETRSmall only if Nano's pain-recall plateaus.
    from rfdetr import RFDETRNano

    cfg = load_config(config_path)

    if class_names is None:
        raise ValueError(
            "class_names is required: derive num_classes from datamodule.class_names "
            "(the §2.4 footgun box). Build the COCO datamodule, then pass "
            "dm.class_names (expected sorted ['no_pain','pain'])."
        )
    num_classes = num_classes_from_class_names(class_names)  # let the data set it

    if dataset_dir is None:
        dataset_dir = f"./datasets/fold{fold}"
    if output_dir is None:
        output_dir = f"./out/fold{fold}"

    model = RFDETRNano(num_classes=num_classes)  # COCO weights auto-download (cat = COCO id 15)

    train_kwargs = dict(
        dataset_dir=str(dataset_dir),
        resolution=cfg["resolution"],            # 512 default; 576 if T4 holds it at batch 4
        epochs=cfg["epochs"],                    # expect earlier convergence than 100
        batch_size=cfg["batch_size"],
        grad_accum_steps=cfg["grad_accum_steps"],  # effective batch = batch_size * grad_accum_steps
        lr=cfg["lr"],
        lr_encoder=cfg["lr_encoder"],            # backbone/encoder LR; drop to 7.5e-5 if unstable
        weight_decay=cfg.get("weight_decay", 1e-4),
        use_ema=cfg.get("use_ema", True),
        early_stopping=cfg.get("early_stopping", True),
        early_stopping_patience=cfg.get("early_stopping_patience", 10),
        early_stopping_min_delta=cfg.get("early_stopping_min_delta", 0.005),  # 0.5% mAP; 0.001 is val noise
        early_stopping_use_ema=cfg.get("early_stopping_use_ema", True),
        num_workers=cfg.get("num_workers", 2),
        output_dir=str(output_dir),
        progress_bar=cfg.get("progress_bar", "rich"),
        tensorboard=cfg.get("tensorboard", True),
        seed=cfg.get("seed", 42),
    )

    # device omitted -> auto-detect CUDA on Colab; device="mps" only for the M4 smoke run.
    if device is not None:
        train_kwargs["device"] = device

    model.train(**train_kwargs)
    return model
## src/detect/train_yolo.py
"""Phase A — YOLOv11s RUNNER-UP runner (IMPLEMENTATION_PLAN §2.5).

Thin wrapper that builds/prints the `yolo detect train` command with the FGS-safe
augmentation knobs (§2.3). YOLOv11s is the fallback only if RF-DETR training is
unstable on this tiny, imbalanced set. ALWAYS fine-tune from the COCO `.pt` —
never random init. Backbone is plumbing, never claimed novel.

FGS-safe augmentation: subtle orbital/muzzle/whisker cues are the whole signal,
so `flipud=0.0` (banned — inverts grimace geometry), `degrees<=10`, `hsv_h~0`
(coat-color is a confound, don't amplify). Selection metric is PAIN RECALL, never
mAP; operating point = FIXED pain-recall >= 0.90 (set LAST, threshold §2.6 rung 4).

Colab T4 primary, M4/MPS fallback (`device=mps`). `ultralytics` is the opt-in
'detect' group dep; this wrapper only assembles a shell command and does not import it.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "detect_yolo.yaml"

# FGS-safe augmentation knobs (§2.3 + FoldsCacheAugSpecialist upgrade).
# LIGHT GEOMETRY ONLY: no heavy that destroy AUs (whiskers/orbital/ear cues).
# Steagall alignment: preserve subtle grimace geometry; copy-paste pain with care
# (detector imbalance only; never on Phase B clean crops for cache).
# Test-time aug (TTA) for robustness at infer: recommended in wrapper/decision
# (e.g. hflip ensemble on crops at inference time; not baked into cache source).
FGS_SAFE_AUG = {
    "fliplr": 0.5,      # safe (face is L/R symmetric)
    "flipud": 0.0,      # BANNED — inverts grimace geometry
    "degrees": 10,      # larger destroys ear/whisker angle cues
    "translate": 0.1,
    "scale": 0.4,
    "hsv_h": 0.0,       # coat-color is a confound, don't amplify
    "hsv_s": 0.4,       # modest
    "hsv_v": 0.3,       # modest
    "close_mosaic": 10,  # final epochs see the real distribution
    # TTA note: at infer, lightweight hflip/scale jitter on detector crops before
    # Phase B cache/infer improves robustness without retraining cache.
}


def load_config(config_path=DEFAULT_CONFIG):
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_command(fold: int, device="0", config_path=DEFAULT_CONFIG):
    """Assemble the `yolo detect train` command for one cat-grouped fold.

    Args:
        fold: fold index K -> ``fold{K}.yaml`` data file.
        device: "0" on Colab T4 (CUDA); "mps" on the M4 fallback.
    """
    cfg = load_config(config_path)
    parts = [
        "yolo", "detect", "train",
        f"model={cfg['model']}.pt",               # fine-tune from COCO — never random init
        f"data=fold{fold}.yaml",
        f"imgsz={cfg['resolution']}",
        f"batch={cfg['batch_size']}",
        f"epochs={cfg['epochs']}",
        f"patience={cfg.get('patience', 25)}",
        "optimizer=auto",
    ]
    for k, v in FGS_SAFE_AUG.items():
        parts.append(f"{k}={v}")
    parts.append(f"device={device}")
    return " ".join(parts)


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Print the YOLOv11s runner-up train command (§2.5).")
    ap.add_argument("--fold", type=int, required=True, help="fold index K in {0..4}")
    ap.add_argument("--device", default="0", help='"0" for Colab T4 / CUDA, "mps" for M4 fallback')
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    print(build_command(args.fold, device=args.device, config_path=args.config))


if __name__ == "__main__":
    main()
## src/model/backbone.py
"""Frozen DINOv2 ViT-S/14 backbone (IMPLEMENTATION_PLAN §5.1).

CONCEDED PLUMBING (FINAL_DIRECTION §A); never claimed as novel. ViT-S (hidden 384,
patch 14), forward-only, frozen because the supervised set is ~120-300 vet/VLM
labels; only the ~5 light heads train.

BACKBONE VARIANT (README §"engine is conceded plumbing"): the register variant
``dinov2_vits14_reg`` is the FIELD DEFAULT — registers suppress attention artifacts
that hurt dense, localized features, and per-AU FGS scoring (orbital/ear/muzzle
sub-regions) is exactly localized. The plain ``dinov2_vits14`` is kept selectable so
the reg-vs-plain choice can be A/B'd before locking. The choice is recorded in the
cache PROVENANCE.json (an input to every reported number), so a run can never hide
which backbone produced its features. Both variants are ViT-S (embed_dim 384).

Patch/hidden are derived from the model (m.embed_dim), not hardcoded in
load-bearing paths; the 14/384/518 constants in comments are for the reader only.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms

from src.model.device import DEVICE

IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# Selectable ViT-S/14 variants. Reg (registers) is the field default; plain is the
# A/B comparator. Both are embed_dim 384, patch 14 — preprocess is identical.
DEFAULT_VARIANT = "dinov2_vits14_reg"
SUPPORTED_VARIANTS = ("dinov2_vits14_reg", "dinov2_vits14", "dinov3_vits16")

# 518 = 37*14 -> 37x37 patch grid; divisible-by-14 mandatory for DINOv2.
# For dinov3_vits16 (patch 16) use 224/512 multiple of 16; richer per MCP context7 (dense oob frozen for small-N medical facial ordinal AU).
VARIANT_PATCH_SIZES = {"dinov2_vits14_reg": 14, "dinov2_vits14": 14, "dinov3_vits16": 16}
VARIANT_INPUT_SIZES = {"dinov2_vits14_reg": 518, "dinov2_vits14": 518, "dinov3_vits16": 224}  # 224 base for vits16; caller may override for dense

preprocess = transforms.Compose([
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(IMNET_MEAN, IMNET_STD),
])


def load_frozen_dinov2(device: str = DEVICE, variant: str = DEFAULT_VARIANT):
    """Load a frozen ViT-S/14 DINOv2 (or dinov3_vits16 successor) backbone.

    variant: ``dinov2_vits14_reg`` (field default, registers suppress attention
    artifacts on localized per-AU features) or the plain ``dinov2_vits14`` A/B
    comparator, or ``dinov3_vits16`` (MCP /facebookresearch/dinov3 primary A/B:
    high-quality dense features out-of-the-box / without fine-tuning per MODEL_CARD;
    same forward_features contract: x_norm_clstoken + x_norm_patchtokens;
    supports get_intermediate_layers(n=list) + L2 patch for richer localized AU).
    Returns (model, hidden_dim).
    """
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"unsupported backbone variant {variant!r}; expected one of {SUPPORTED_VARIANTS} "
            "(reg is the field default; plain is the A/B comparator; dinov3_vits16 MCP dense oob successor for small-N imbal ordinal facial AU)"
        )
    if variant.startswith("dinov3"):
        # MCP context7 confirmed: torch.hub "facebookresearch/dinov3" dinov3_vits16 or dinov3.hub.backbones import; frozen oob dense ideal
        m = torch.hub.load("facebookresearch/dinov3", variant, source="github")
    else:
        m = torch.hub.load("facebookresearch/dinov2", variant)
    m.requires_grad_(False)      # freeze ALL params
    m.eval()                     # disable dropout/BN-update; deterministic features
    m.to(device)
    hidden = m.embed_dim         # 384 for ViT-S; 384 for vits16 too
    # assert relaxed for dinov3 vits16 (still ~384d small); caller derives
    if "vits" in variant:
        assert hidden == 384, f"expected 384 for small ViT-S, got {hidden}"
    return m, hidden


@torch.inference_mode()
def extract(m, x, layer: str = "last", patch_l2: bool = False):
    """Extract (cls, patch) tokens from frozen backbone (conceded engine, no-FT default).

    layer: "last" (forward_features default) | list[int]/"intermed" for get_intermediate_layers
           (MCP-confirmed dinov3 vision_transformer contract: n=..., return_class_token=True, norm=True
            for dense localized features).
    patch_l2: apply F.normalize(..., p=2, dim=-1) gated by caller config (e.g. corn.yaml patch_mode=l2).
    Supports dinov3_vits16 richer oob frozen dense per-AU (orbital/ear/muzzle) for small-N imbal ordinal AU.
    Updates callers + provenance/schema (the 'hash' of config for repro/G3).
    """
    if layer == "last" or layer is None:
        out = m.forward_features(x)  # dict
        cls = out["x_norm_clstoken"]        # [B,384]   (CLS, == index 0 of the token seq)
        patch = out["x_norm_patchtokens"]   # [B,N,384] (patch tokens; HF-equiv of [:,1:,:])
    else:
        # MCP dinov3: get_intermediate_layers for multi-layer dense/localized (eval encoder, classifiers)
        # e.g. n=[5,11,17,23] or range; norm=True, return_class_token=True per vision_transformer + notebooks
        # PRACTICAL A/B richer: for n=list do mean+std concat across layers for denser localized cues (orbital/ear/muzzle) + L2 full
        if isinstance(layer, str) and layer.lower() == "intermed":
            layer = [-4, -3, -2, -1]  # richer last-4 intermed per MCP dinov3 contract for dense localized
        n = layer if isinstance(layer, (list, tuple, range)) else [-1]
        inter = m.get_intermediate_layers(x, n=n, reshape=False, return_class_token=True, norm=True)
        if inter:
            # richer: concat mean+std of patch feats from selected layers (or single); supports full intermed mean_std
            patches = [i[0] for i in inter]  # patch tokens per layer
            if len(patches) > 1:
                pcat = torch.cat(patches, dim=-1)  # concat layers first for richer dim
                pmean = pcat.mean(dim=1)  # [B, D*nl]
                pstd = pcat.std(dim=1)
                patch = torch.cat([pmean, pstd], dim=-1)  # mean+std concat richer
                # cls from last
                cls = inter[-1][1]
            else:
                patch, cls = inter[-1][0], inter[-1][1]  # patch + cls from selected (last) intermed layer
                # for single also support mean_std style if needed
                if patch.shape[1] > 0:  # ensure
                    pmean = patch.mean(dim=1)
                    pstd = patch.std(dim=1)
                    # keep patch as-is for compat; richer via caller pool or intermed mode
        else:
            out = m.forward_features(x)
            cls = out["x_norm_clstoken"]
            patch = out["x_norm_patchtokens"]
    if patch_l2:
        patch = F.normalize(patch, p=2, dim=-1)
    return cls, patch
## src/model/heads.py
"""The 5 CORN heads (IMPLEMENTATION_PLAN §5.3, layout B: one wide head).

CONCEDED PLUMBING (FINAL_DIRECTION §A); never claimed as novel. K=3 levels per
AU => K-1 = 2 logits per AU; Sigma(K_au-1) = 5*2 = 10. One matmul Linear(384,10),
then torch.split into 5 x [B,2] per-AU chunks fed to per-AU corn_loss.

The 5 AUs (FACTCHECK: Min-aggregation for whiskers AND head):
ear, orbital, muzzle, whiskers, head.

The wrapper's binary pain head (the v1 spine) is a sibling Linear(384,1) trained
with BCEWithLogitsLoss; it is independent of CORN and lives in src/wrapper/. The
0-10 CORN sum is the INSPECTED-NOT-VALIDATED path, never a validated-claim number.
"""

import torch
import torch.nn as nn

from src.constants import AU_ORDER

AUS = AU_ORDER
K = 3                              # levels 0/1/2 per AU
N_LOGITS = len(AUS) * (K - 1)      # = 10


class CornMultiHead(nn.Module):
    def __init__(self, in_dim=384, n_aus=5, k=3, p_drop=0.1):
        super().__init__()
        self.k = k
        self.n_aus = n_aus
        self.drop = nn.Dropout(p_drop)               # light reg on tiny linear probe
        self.proj = nn.Linear(in_dim, n_aus * (k - 1))   # 384 -> 10
        # trunc_normal_ init for stable small-data linear-probe
        # (BenediktAlkin/vtab1k-pytorch, GITHUB_MINE P2.2)
        nn.init.trunc_normal_(self.proj.weight, std=2e-5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat):                          # feat: [B,384] (CLS or mean-patch)
        logits = self.proj(self.drop(feat))           # [B,10]
        return list(torch.split(logits, self.k - 1, dim=1))  # 5 x [B,2]
## src/model/corn.py
"""Vendored CORN ordinal loss + decode (IMPLEMENTATION_PLAN §5.4).

CONCEDED PLUMBING. The frozen DINOv2 + CORN engine is NEVER claimed as novel
(FINAL_DIRECTION §A); it exists only to carry the binary-plus-wrapper spine, the
headline confound-attribution protocol, and the guarded kappa check.

Vendored verbatim (MIT, ludwig-ai/ludwig `corn.py`, torch + F only) so the
datasheet shows exactly what trained the heads. Cross-checked ONCE against
coral_pytorch.losses.corn_loss in the Gate-4 unit test (§5.7), then the dep is
dropped. Shi, Cao & Raschka 2021 (CORN, arXiv 2111.08851).

Decode paths:
  - corn_cumprobs  -> SOFT default path (FINAL_DIRECTION §E.1) feeding pmf/RPS/ECE.
  - corn_label_from_logits -> HARD decode, for the 0.39 POINT decision ONLY.
"""

import torch
import torch.nn.functional as F


def corn_loss(logits, y, num_classes):
    """Conditional ordinal (CORN) loss for ONE AU.

    logits: [B, num_classes-1]   y: [B] in {0,...,num_classes-1}
    Shi, Cao & Raschka 2021. Vendored (MIT, ludwig-ai/ludwig)."""
    sets = []
    for i in range(num_classes - 1):
        label_mask = (y > i - 1)                 # samples still "in play" at rank i
        label_tensor = (y[label_mask] > i).to(torch.int64)
        if label_mask.sum() == 0:
            continue
        sets.append((label_mask, label_tensor))
    losses = 0.0
    n = 0
    for i, (mask, lab) in enumerate(sets):
        pred = logits[mask, i]                    # conditional logit for rank i
        loss = -torch.sum(
            F.logsigmoid(pred) * lab + (F.logsigmoid(pred) - pred) * (1 - lab)
        )
        losses = losses + loss
        n += mask.sum().item()
    return losses / max(n, 1)


def corn_label_from_logits(logits):
    """Hard decode for the 0.39 POINT decision ONLY.

    logits: [B, num_classes-1] -> labels [B]. predict = sum_k( cumprod(sigmoid)[k] > 0.5 )."""
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)         # P(y>0), P(y>0 & y>1), ...
    return torch.sum(probas > 0.5, dim=1)


def corn_cumprobs(logits):
    """SOFT path (DEFAULT, FINAL_DIRECTION §E.1). Returns cumulative P(rank>k) per level.

    DO NOT hard-decode for calibration -- these soft probs feed pmf/RPS/ECE."""
    return torch.cumprod(torch.sigmoid(logits), dim=1)   # [B, num_classes-1]


def multi_corn_loss(logits_list, y, num_classes=3, au_weights=None):
    """Sum the 5 per-AU corn_loss, skipping AUs with the -1 sentinel.

    logits_list: 5 x [B,2]; y: [B,5] with -1 = AU not scored on that image.

    total starts as a graph-connected zero (not the float 0.0) so .backward()
    stays valid when every AU is sentinel-masked — reachable via a co-teaching
    selection of only all-sentinel vet-clean rows, or an empty k=0 selection on
    a size-1 tail batch."""
    total = logits_list[0].sum() * 0.0
    for a in range(len(logits_list)):
        mask = y[:, a] >= 0
        if mask.sum() == 0:
            continue
        au_loss = corn_loss(logits_list[a][mask], y[mask, a], num_classes)
        w = 1.0 if au_weights is None else au_weights[a]
        total = total + w * au_loss
    return total
## src/model/decode.py
"""Distributional decode (IMPLEMENTATION_PLAN §5.5).

PORTABLE PLUMBING — supporting infrastructure for the headline confound-attribution
protocol and its guarded kappa check, not a contribution in its own right.

DEFAULT path (FINAL_DIRECTION §E.1): keep soft cumulative P(rank>k); build each
AU's pmf over {0,1,2}; convolve the 5 pmfs into one pmf over the 0-10 sum. From
that distribution come RPS-on-the-sum (one scalar, bootstrap CI) + per-AU
ClasswiseECE (computed downstream in src/eval/). NO binned reliability diagram on
the 11-atom sum (degenerate at ~11 atoms).

argmax / hard-decode (point_sum) is reserved for the 0.39 POINT decision ONLY.
The 0-10 sum is INSPECTED-NOT-VALIDATED; no validated-claim number is emitted here.
"""

import numpy as np
import torch

from src.constants import AU_ORDER
from src.model.corn import corn_label_from_logits


N_DEFAULT = len(AU_ORDER)


def au_pmf_from_cumprobs(cum):          # cum: [B,2] = [P(y>0), P(y>0 & y>1)]
    p_gt0, p_gt1 = cum[:, 0], cum[:, 1]
    p0 = 1 - p_gt0
    p1 = p_gt0 - p_gt1                   # = P(y>0) - P(y>1)
    p2 = p_gt1
    pmf = torch.stack([p0, p1, p2], dim=1)          # [B,3]
    return torch.clamp(pmf, min=0)      # guard tiny negatives from float error


def sum_pmf(au_pmfs):                    # au_pmfs: list of 5 x [B,3] (numpy)
    # convolve per-AU pmfs -> pmf over 0..(n_aus * max_au_score)
    B = au_pmfs[0].shape[0]
    n_aus = len(au_pmfs)
    max_sum = sum(int(p.shape[1]) - 1 for p in au_pmfs)
    out = np.zeros((B, max_sum + 1))
    for b in range(B):
        acc = np.array([1.0])
        for a in range(n_aus):
            acc = np.convolve(acc, au_pmfs[a][b])
        out[b] = acc / acc.sum()         # renormalize
    return out                           # [B,max_sum+1], sums to 1 over the sum support


def point_sum(logits_list):              # hard decode for the decision ONLY
    # threshold imported from src.vlm.aggregate — the single 0.39 definition in
    # the repo, shared with the VLM path so the two flags can never drift.
    from src.vlm.aggregate import analgesia_flag

    labels = [corn_label_from_logits(lg) for lg in logits_list]   # 5 x [B]
    s = torch.stack(labels, dim=1).sum(1)        # [B] in 0..10
    return s, analgesia_flag(s.float())          # painful flag


def pmf_entropy(pmf):
    """Entropy of pmf (scalar or per-row).

    Used for active VLM/abstention: high entropy = high uncertainty (CORN pmf driver).
    Per-AU: call on au_pmf_from_cumprobs output; image-level: on sum_pmf output.
    Preserves atoms-only contract: unc computed in code, never from VLM.
    Pure-np portable derive (N/k from input shape; use protocols + adapters.generic_ordinal_mode for non-FGS).
    """
    p = np.asarray(pmf, dtype=float)
    if p.ndim == 1:
        p = p / (p.sum() + 1e-12)
        p = np.clip(p, 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))
    else:
        # batch: per-row entropy
        p = p / (p.sum(axis=-1, keepdims=True) + 1e-12)
        p = np.clip(p, 1e-12, 1.0)
        return -np.sum(p * np.log(p), axis=-1)


def au_pmf_entropies(au_pmfs_list):
    """List of per-AU mean entropy over batch (for triage/score)."""
    return [float(np.mean(pmf_entropy(au))) for au in au_pmfs_list]
## src/model/train_heads.py
"""Co-teaching small-loss head trainer (IMPLEMENTATION_PLAN §5.6).

CONCEDED PLUMBING. The frozen DINOv2 + CORN engine is NEVER claimed as novel
(FINAL_DIRECTION §A). The validated-claim spine is the sibling BINARY pain head
built here; the 0-10 CORN sum is INSPECTED-NOT-VALIDATED and emits no
validated-claim number.

Method (FINAL_DIRECTION §6; bhanML/Co-teaching, BUILD_PLAN §3): train NOT
confirmed-only (too few) and NOT naive-all (bakes in the VLM under-estimation
bias). Two CornMultiHead nets, each selects the small-loss subset for the OTHER.
Keep-rate ramps 1.0 -> (1 - tau) over num_gradual=10 epochs (canonical Han et al.
2018 R(T): inclusive -> selective, exploiting the memorization effect). tau =
estimated VLM noise rate (Gate-1-B), NOT a fixed 0.5. vet-clean rows
(is_vet_clean==1) are NEVER dropped from either selection.

Cache-only loop: the backbone NEVER runs in the inner loop -- features are read
from the §5.2 npz cache, so the §5.6 pooling x loss x seed sweep runs in minutes.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model.corn import corn_loss, multi_corn_loss
from src.model.device import DEVICE
from src.model.heads import AUS, CornMultiHead, K

N_AUS = len(AUS)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_split(npz, train_folds=(0, 1, 2), pool="cls"):
    """Read the cached features + labels for the given train folds.

    The trainer reads `fold` from the cache (G3-hashed); it NEVER touches a raw
    test manifest. pool selects "cls" (default) or "patch_mean" -> [N,384].
    Returns (X, y, clean, y_pain) tensors for the selected folds.
    """
    d = np.load(npz, allow_pickle=True)
    # uniform hard _CACHE_SCHEMA no legacy (per FreshHandoffDINOv3RicherTiny + round_fresh_4 table#1 enforce; align separability)
    from src.model.cache_features import _CACHE_SCHEMA, SCHEMA_VERSION
    if d.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"cache schema_version mismatch (uniform enforce no legacy): {npz}")
    for k in ("n_aus", "au_hash", "k", "feature_dim"):
        if k not in d or d[k] != _CACHE_SCHEMA[k]:
            raise ValueError(f"_CACHE_SCHEMA key mismatch on {k} (enforce)")
    for k in ("variant", "layer", "patch_mode"):
        if k not in d:
            raise ValueError(f"_CACHE_SCHEMA key missing {k} (uniform enforce no legacy)")
    feat = d[pool]                          # "cls" or "patch_mean" -> [N,384]
    fold = d["fold"].astype(int)
    y = d["y"]                              # [N,5], -1 = AU not scored
    clean = d["is_vet_clean"]
    y_pain = d["y_pain"]
    tr = np.isin(fold, train_folds)
    return (
        torch.tensor(feat[tr]),
        torch.tensor(y[tr]),
        torch.tensor(clean[tr]),
        torch.tensor(y_pain[tr]),
    )


# ---------------------------------------------------------------------------
# Per-sample loss + selection helpers (~15 lines, kept in-file per §5.6)
# ---------------------------------------------------------------------------
def _persample(logits_au, y_au, num_classes=K):
    """Per-ROW corn_loss (reduction='none' equivalent) for ONE AU.

    logits_au: [B, K-1]; y_au: [B] with -1 = not scored (contributes 0).
    Loops rows so the vendored corn_loss stays the single source of truth for the
    ordinal objective; rows are tiny (~120-300) so this is cheap.
    """
    out = torch.zeros(logits_au.shape[0], device=logits_au.device)
    for j in range(logits_au.shape[0]):
        if int(y_au[j]) < 0:                       # sentinel -> AU unscored on this row
            continue
        out[j] = corn_loss(logits_au[j:j + 1], y_au[j:j + 1], num_classes)
    return out


def _persample_sum(logits_list, y):
    """Per-row CORN loss summed over the AUs scored on that row -> [B]."""
    return torch.stack(
        [_persample(logits_list[a], y[:, a]) for a in range(N_AUS)]
    ).sum(0)


def _select(loss_other, clean, k):
    """Indices to keep: vet-clean UNION smallest-loss-non-clean up to k.

    loss_other: per-sample loss from the PARTNER net (co-teaching cross-selection).
    clean: is_vet_clean flags. vet-clean rows are ALWAYS kept (never dropped);
    the remaining budget (k - n_clean) goes to the smallest-loss non-clean rows.
    """
    clean_idx = torch.nonzero(clean > 0, as_tuple=False).flatten()
    non_clean = torch.nonzero(clean == 0, as_tuple=False).flatten()
    budget = max(0, k - clean_idx.numel())
    if non_clean.numel() and budget:
        order = torch.argsort(loss_other[non_clean])         # ascending loss
        picked = non_clean[order[:budget]]
    else:
        picked = non_clean[:0]
    return torch.cat([clean_idx, picked])


# ---------------------------------------------------------------------------
# Loss switch (corn default vs CORAL ablation; one-flag, §5.6 table)
# ---------------------------------------------------------------------------
def _coral_multi_loss(logits_list, y, num_classes=K, au_weights=None):
    """CORAL ablation, sentinel-masked per AU (lazy dep; corn is the default).

    Uses coral_pytorch.losses.coral_loss with extended-binary level targets so the
    only thing that changes vs CORN is the per-AU objective -- everything else
    (heads, decode, selection) is shared. Imported lazily so import stays clean
    when the optional ablation dep is absent.
    """
    from coral_pytorch.dataset import levels_from_labelbatch
    from coral_pytorch.losses import coral_loss

    # graph-connected zero, same reason as multi_corn_loss: .backward() must stay
    # valid when every AU in the selected rows is sentinel-masked.
    total = logits_list[0].sum() * 0.0
    for a in range(len(logits_list)):
        mask = y[:, a] >= 0
        if mask.sum() == 0:
            continue
        levels = levels_from_labelbatch(y[mask, a], num_classes=num_classes).to(
            logits_list[a].device
        )
        au_loss = coral_loss(logits_list[a][mask], levels)
        w = 1.0 if au_weights is None else au_weights[a]
        total = total + w * au_loss
    return total


def _loss_fn(loss: str):
    """Return the multi-AU loss callable for the requested ablation flag."""
    if loss == "corn":
        return multi_corn_loss
    if loss == "coral":
        return _coral_multi_loss
    raise ValueError(f"unknown loss flag: {loss!r} (expected 'corn' or 'coral')")


# ---------------------------------------------------------------------------
# Binary pain head (the v1 SPINE) -- sibling Linear(384,1), independent of CORN
# ---------------------------------------------------------------------------
class BinaryPainHead(nn.Module):
    """v1 spine: Linear(384,1) trained with BCEWithLogitsLoss(pos_weight) on y_pain.

    Shares the cache, is independent of CORN, and is the path the calibration /
    abstention / decision-curve wrapper (Phase C) actually operates on. This is
    the validated-claim spine; the 0-10 CORN sum is the inspected path.
    """

    def __init__(self, in_dim=384, p_drop=0.1):
        super().__init__()
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(in_dim, 1)
        nn.init.trunc_normal_(self.fc.weight, std=2e-5)
        nn.init.zeros_(self.fc.bias)

    def forward(self, feat):                    # [B,384] -> [B] logit
        return self.fc(self.drop(feat)).squeeze(-1)


def train_binary_pain(X, y_pain, device=DEVICE, epochs=80, lr=1e-3,
                      weight_decay=1e-2, batch_size=32, p_drop=0.1,
                      pos_weight=None, num_workers=0, seed=0):
    """Train the sibling binary pain head on y_pain (the v1 spine)."""
    torch.manual_seed(seed)
    X = X.to(device)
    yp = y_pain.float().to(device)
    if pos_weight is None:                       # n_neg / n_pos from the train split
        n_pos = float((yp > 0).sum().item())
        n_neg = float((yp <= 0).sum().item())
        pos_weight = n_neg / max(n_pos, 1.0)
    pw = torch.tensor([pos_weight], device=device)
    net = BinaryPainHead(in_dim=X.shape[1], p_drop=p_drop).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    dl = DataLoader(TensorDataset(X, yp), batch_size=batch_size, shuffle=True,
                    num_workers=num_workers)
    net.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            bce(net(xb), yb).backward()
            opt.step()
    return net


# ---------------------------------------------------------------------------
# Co-teaching CORN trainer
# ---------------------------------------------------------------------------
def train(npz, tau, device=DEVICE, epochs=80, pool="cls", seed=0,
          loss="corn", lr=1e-3, weight_decay=1e-2, batch_size=32, p_drop=0.1,
          head_init_std=2e-5, num_gradual=10, train_folds=(0, 1, 2),
          au_weights=None, num_workers=0):
    """Co-teaching small-loss trainer for the 5 CORN heads.

    tau = estimated VLM noise rate (Gate-1-B). 1 - tau is the floor keep-rate;
    the ramp goes 1.0 -> (1 - tau) over num_gradual epochs. vet-clean rows are
    never dropped. Returns netA (netB is the teaching partner).
    """
    torch.manual_seed(seed)
    Xtr, ytr, clean, _ = load_split(npz, train_folds=train_folds, pool=pool)
    Xtr = Xtr.to(device)
    dl = DataLoader(
        TensorDataset(Xtr, ytr.to(device), clean.to(device)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    loss_fn = _loss_fn(loss)

    # two heads for co-teaching. CornMultiHead (engine) already trunc_normal_-inits
    # its proj at std=2e-5; honor a config-supplied head_init_std without editing the
    # engine by re-initializing the proj weight post-construction.
    netA = CornMultiHead(in_dim=Xtr.shape[1], p_drop=p_drop).to(device)
    netB = CornMultiHead(in_dim=Xtr.shape[1], p_drop=p_drop).to(device)
    if head_init_std is not None:
        for net in (netA, netB):
            nn.init.trunc_normal_(net.proj.weight, std=head_init_std)
            nn.init.zeros_(net.proj.bias)
    optA = torch.optim.AdamW(netA.parameters(), lr=lr, weight_decay=weight_decay)
    optB = torch.optim.AdamW(netB.parameters(), lr=lr, weight_decay=weight_decay)

    netA.train()
    netB.train()
    for ep in range(epochs):
        # canonical Co-teaching R(T): keep-rate 1.0 -> (1-tau) over num_gradual epochs
        keep = 1.0 - tau * min(1.0, ep / num_gradual)   # inclusive -> selective
        for xb, yb, cb in dl:
            la = netA(xb)
            lb = netB(xb)
            with torch.no_grad():
                pa = _persample_sum(la, yb)              # A's per-row loss
                pb = _persample_sum(lb, yb)              # B's per-row loss
                k = int(keep * len(xb))
                # cross-selection: partner's small-loss subset; vet-clean always kept
                selA = _select(pb, cb, k)                # B selects for A
                selB = _select(pa, cb, k)                # A selects for B
            optA.zero_grad()
            loss_fn([lg[selA] for lg in la], yb[selA], au_weights=au_weights).backward()
            optA.step()
            optB.zero_grad()
            loss_fn([lg[selB] for lg in lb], yb[selB], au_weights=au_weights).backward()
            optB.step()
    return netA   # report netA; netB is the teaching partner
## src/vlm/rubric.py
"""The rubric system prompt -- one paragraph per AU (IMPLEMENTATION_PLAN §4.2).

Verbatim-Evangelista 0/1/2 descriptors (FACTCHECK C50: ``0=absent; 1=moderate OR
uncertain; 2=marked/obvious``), AU anatomy grounded in CatFACS. Lives in a
cache-controlled system block so the ~2040-image bulk run pays for it once.

The rubric instructs default-ambiguity-to-1 (low confidence) and abstain on
occlusion/blur/non-frontal, and explicitly states the model may NOT output a total
or any pain/treatment decision -- the sum and the >=0.39 flag are computed in code
(src.vlm.aggregate), keeping the engine decode path the single source of truth for
the threshold.
"""

RUBRIC = """You are scoring the Feline Grimace Scale (FGS, Evangelista et al. 2019) on ONE cat face.
Score 5 action units, each on a 0/1/2 ordinal scale, where 0 = action unit ABSENT,
1 = action unit MODERATELY present OR you are UNCERTAIN, 2 = action unit MARKEDLY/OBVIOUSLY present.
Write the rationale BEFORE the score, citing the specific visible feature. Score each AU INDEPENDENTLY.

EAR POSITION: 0 = ears facing forward; 1 = ears slightly pulled apart or moderately rotated;
2 = ears flattened and rotated outwards.
ORBITAL TIGHTENING: 0 = eyes opened; 1 = eyes partially opened OR eye squinting beginning;
2 = eyes squinted/closed.
MUZZLE TENSION: 0 = relaxed, round muzzle; 1 = mild tension/oval; 2 = tense, elliptical muzzle.
WHISKERS CHANGE: 0 = loose and curved whiskers; 1 = slight straightening/forward;
2 = straight and moving forward.
HEAD POSITION: 0 = head above shoulder line; 1 = head aligned with shoulder line;
2 = head below shoulder line OR tilted down.

RULES:
- When a feature is genuinely ambiguous, DEFAULT THE SCORE TO 1 and set confidence='low'.
- If an AU region is occluded, blurred, out-of-frame, or the face is NON-FRONTAL, set abstain=true
  for THAT AU (still emit a best-guess score, but it will be down-weighted/routed to vet).
- Set image_quality='unusable' only if the whole face cannot be assessed.
- You may NOT output a total score or any pain/treatment decision. Output only the 5 AU records."""

# Cache-controlled system block. ttl="1h" (not the 5m default): the async Message
# Batches window runs for hours, and a 5-minute TTL with silent edit-invalidation
# collapses the rubric-cache hit-rate so every request re-pays full input-token price
# for the rubric. 1h TTL may require the beta header extended-cache-ttl-2025-04-11 on
# some model ids. Single-sourced here -> propagates to batch_submit.py and call.py.
SYSTEM_BLOCKS = [
    {
        "type": "text",
        "text": RUBRIC,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }
]
## src/vlm/aggregate.py
"""In-code aggregation: the 0-10 sum and the 0.39 flag (IMPLEMENTATION_PLAN §4.3).

THIS is the single threshold definition reused by both the VLM path and the engine
(CORN) decode path -- the Gate-4 unit test pins ``decode -> per-AU 0-2 -> sum -> 0.39``
against this exact contract so there is ONE threshold definition in the repo.

The VLM NEVER computes any of these; it emits only the 5 atoms in {0,1,2}. The
0.39 flag is triage decision-support only -- never an autonomous analgesia trigger
-- and the sum it rides on is an INSPECTED-NOT-VALIDATED quantity.
"""

# Fixed Evangelista 5-AU order — canonical declaration in src.constants.AU_ORDER
# (a leaf module, so this import adds no schema/anthropic-stack dependency).
from src.constants import AU_ORDER

AU_NAMES = list(AU_ORDER)

# THE one 0.39 definition (Evangelista sum/10 cut, ~4/10). Engine decode
# (src.model.decode.point_sum) and the abstention band import it from here;
# never re-declare the literal elsewhere.
POINT_DECISION_THRESHOLD = 0.39


def fgs_sum(result_dict):
    """Sum the 5 per-AU scores into 0..10. ``result_dict[au]['score']`` in {0,1,2}."""
    return sum(result_dict[au]["score"] for au in AU_NAMES)


def analgesia_flag(s):
    """Triage flag: ratio = sum/10; clinical cut at >= 0.39 (~4/10).

    Works on scalars, numpy arrays, and torch tensors (s/10.0 promotes to float)."""
    return (s / 10.0) >= POINT_DECISION_THRESHOLD


def any_abstain(result_dict):
    """For routing/triage only, not a clinical output."""
    return any(result_dict[au]["abstain"] for au in AU_NAMES) or (
        result_dict["image_quality"] != "frontal_clear"
    )
## src/wrapper/operating_point.py
"""Operating-point selection — FIXED pain-recall >= 0.90 (IMPLEMENTATION_PLAN §6.3, §5.5).

The clinical cutoff on the calibrated P(pain) is selected at a **fixed pain-recall
>= 0.90** (Evangelista anchor, sens 90.7%) **inside train folds** — NOT Youden-J /
F1. A symmetric-cost knee is the wrong loss for a welfare instrument.

CIRCULARITY FIREWALL (IMPLEMENTATION_PLAN §6.3 / line 2302): sens/spec at the 0.39
point are estimated on **VET-CONFIRMED labels ONLY**. QWK-vs-VLM is never validation.
This module refuses to estimate sens/spec on any row that is not vet-confirmed.

The frozen DINOv2 + CORN engine that produces these scores is conceded plumbing, and
this wrapper is cited supporting evidence — not the headline (the headline is the
power-aware per-AU confound-attribution protocol in src/protocols/). We report the
SPECIFICITY the fixed-recall cutoff buys (with bootstrap CIs) and the cutoff's
fold-to-fold variance — never a "we beat X%" claim.
"""

from __future__ import annotations

import pathlib

import numpy as np
import yaml

from src.eval.bootstrap import bootstrap_ci as _grouped_bootstrap_ci


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _get_default_wrapper_config() -> pathlib.Path:
    return _REPO_ROOT / "configs" / "wrapper.yaml"


_DEFAULT_CONFIG = _get_default_wrapper_config()


def _load_op_config(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["operating_point"]


def select_cutoff_at_recall(y_true, p_pain, target_recall: float) -> float:
    """Highest threshold on p_pain that still achieves pain-recall >= target_recall.

    Selected INSIDE train folds. NOT Youden-J / F1. ``y_true``/``p_pain`` are the
    TRAIN-fold rows only; sweeping thresholds high->low, we take the largest cutoff
    whose recall (sensitivity for pain) is still >= target. A higher cutoff buys more
    specificity, so we want the most-specific cutoff that still clears the recall floor.
    """
    y_true = np.asarray(y_true).astype(int)
    p_pain = np.asarray(p_pain, dtype=float)
    pos = y_true == 1
    n_pos = int(pos.sum())
    if n_pos == 0:
        raise ValueError("no pain-positive rows in train fold; cannot fix recall")

    # candidate cutoffs = the calibrated positive scores (recall only changes there)
    cand = np.unique(p_pain[pos])
    best = 0.0
    found = False
    for t in np.sort(cand)[::-1]:                 # high -> low cutoff
        recall = float((p_pain[pos] >= t).mean())
        if recall >= target_recall:
            best = float(t)
            found = True
            break
    if not found:                                 # even the lowest pos score misses target
        best = float(p_pain[pos].min())
    return best


def _sens_spec(y_true, p_pain, cutoff: float):
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(p_pain, dtype=float) >= cutoff).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def estimate_specificity_bought(
    y_vet,
    p_pain,
    vet_confirmed,
    cutoff: float,
    cat_id=None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
):
    """Sens/spec at ``cutoff``, estimated on VET-CONFIRMED rows ONLY (firewall).

    ``vet_confirmed`` is a boolean mask (is_vet_clean / vet-confirmed). Rows where it
    is False are DROPPED before any sens/spec is computed — this is the circularity
    firewall, enforced here, not merely commented. Returns the point estimates plus a
    percentile-bootstrap 95% CI on specificity. Pass ``cat_id`` (per-row individual
    ids) to resample whole cats — the project-standard grouped bootstrap; without it
    the CI is row-i.i.d. and optimistically narrow under within-cat correlation.
    """
    mask = np.asarray(vet_confirmed).astype(bool)
    if not mask.any():
        raise ValueError(
            "circularity firewall: no vet-confirmed rows; sens/spec cannot be "
            "estimated (QWK-vs-VLM is never validation)"
        )
    y = np.asarray(y_vet).astype(int)[mask]
    p = np.asarray(p_pain, dtype=float)[mask]

    sens, spec = _sens_spec(y, p, cutoff)

    # bootstrap the specificity over the vet-confirmed NEGATIVE rows
    neg = y == 0
    neg_correct = ((p < cutoff) & (y == 0)).astype(float)[neg]
    if len(neg_correct):
        # one bootstrap implementation for both paths (shared eval helper): grouped
        # by cat when cat_id is given, ungrouped (groups=None) otherwise. Avoids a
        # private copy that diverged in nan-handling and seed conventions.
        neg_cats = np.asarray(cat_id)[mask][neg] if cat_id is not None else None
        _, spec_lo, spec_hi = _grouped_bootstrap_ci(
            neg_correct, lambda a: float(np.mean(a)),
            n_boot=n_boot, alpha=alpha, seed=seed, groups=neg_cats,
        )
    else:
        spec_lo = spec_hi = float("nan")
    return {
        "cutoff": float(cutoff),
        "sensitivity": sens,
        "specificity": spec,
        "specificity_ci": (spec_lo, spec_hi),
        "n_vet_confirmed": int(mask.sum()),
    }


def operating_point_across_folds(
    fold_frames,
    config_path: str = _DEFAULT_CONFIG,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
):
    """Per-fold cutoff selection + cross-fold variance of the cutoff.

    ``fold_frames`` is an iterable of dicts, one per fold. The TRAIN arrays fix the
    cutoff; the HELD-OUT arrays score it — the two row sets must be disjoint (the
    cutoff is frozen on train, never tuned on the rows it is evaluated on):
        y_train  — train-fold labels used to FIX the recall cutoff
        p_train  — calibrated P(pain) for the SAME train-fold rows
        p_test   — calibrated P(pain) for the held-out rows of this fold
        y_vet    — vet labels for the held-out rows
        vet_confirmed — bool mask (firewall) for the held-out rows
        cat_id   — optional per-row individual ids for the held-out rows
                   (enables the project-standard cat-grouped specificity CI)
    The recall target is read from configs/wrapper.yaml (fixed 0.90 Evangelista anchor).
    """
    op = _load_op_config(config_path)
    assert op["metric"] == "pain_recall", "operating point must be pain-recall, not Youden-J/F1"
    target = float(op["target"])

    per_fold = []
    cutoffs = []
    for fr in fold_frames:
        cut = select_cutoff_at_recall(fr["y_train"], fr["p_train"], target)
        cutoffs.append(cut)
        est = estimate_specificity_bought(
            fr["y_vet"], fr["p_test"], fr["vet_confirmed"], cut,
            cat_id=fr.get("cat_id"), n_boot=n_boot, alpha=alpha, seed=seed,
        )
        est["target_recall"] = target
        per_fold.append(est)

    cutoffs = np.asarray(cutoffs, dtype=float)
    return {
        "target_recall": target,
        "per_fold": per_fold,
        "cutoff_mean": float(cutoffs.mean()) if len(cutoffs) else float("nan"),
        "cutoff_sd": float(cutoffs.std(ddof=1)) if len(cutoffs) > 1 else float("nan"),
        "spec_bought_mean": float(
            np.nanmean([f["specificity"] for f in per_fold])
        ) if per_fold else float("nan"),
    }
## src/gates/orchestrator.py
#!/usr/bin/env python3
"""Central Gate E2E Orchestrator (P1/P7 focus).

Enforces the canonical run order from Makefile/README (G0 first — power tie-in).
Scripts-only (no src locality changes per P1; delegates to existing gate*.py and helpers).
- Writes standardized run artifacts (artifacts/gates_run_<ts>.json + per-gate).
- Aborts on first failure (non-zero exit or missing expected PASS artifact).
- Supports --synthetic: generates minimal toy data (deletion-safe; no real manifests, no CatFLW, no API keys for VLM pilot).
  Toy path exercises G0-G6 + VLM-pilot protocol (via gate1b on synth) + wrapper smoke (no full labels batch).
- python -m src.gates.orchestrator (or via make).
- Repro aid (P3/P4): deterministic seeds, explicit order, artifact hashes, full-pipeline e2e test entry.
- Gates remain the enforcement of uniqueness/strict honesty: committed manifests, single-source thresholds (via aggregate), pre-reg power (G0), circularity firewall, one-directional verdicts, hash guards (G3), CI aborts.

Usage (synthetic for tests/CI smoke):
  python -m src.gates.orchestrator --synthetic
  uv run python -m src.gates.orchestrator --synthetic --include-wrapper

Full (requires prior gates data + keys for VLM):
  python -m src.gates.orchestrator

See Makefile: gate-pipeline / gate-e2e-synthetic targets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.seed import seed_everything  # noqa: E402
from src.vlm.aggregate import AU_NAMES, POINT_DECISION_THRESHOLD  # noqa: E402  # single-source honesty

# Canonical order (per Makefile + README; G0 FIRST — power pre-reg blocks everything quantitative)
GATE_ORDER: list[str] = ["gate0", "gate1", "gate2", "gate3", "gate4", "gate5", "gate1b", "gate6"]

# Script mapping (scripts-only; thin delegation; gate1b is composite)
GATE_SCRIPTS: dict[str, list[str]] = {
    "gate0": ["scripts/gate0_power.py"],
    "gate1": ["scripts/gate1_merge.py"],
    "gate2": ["scripts/gate2_confound.py"],
    "gate3": ["scripts/gate3_holdout.py"],
    "gate4": ["scripts/gate4_mps_check.py"],  # internally runs pytest decode+mps + corn smoke
    "gate5": ["scripts/gate5_nme.py"],
    "gate1b": ["scripts/run_vlm_labels.py", "scripts/gate1b_kappa_pilot.py"],  # pilot protocol only in synthetic
    "gate6": ["scripts/gate6_severity.py"],
}

# Expanded contracts/registry for real paths + manifests enforcement (FreshFullGateWiringManifestsEnforcer cand1)
# Real entries/scripts must target committed manifests/ (power.json etc) or SystemExit before compute.
GATE_PRECONDS: dict[str, list[str]] = {
    "gate0": [],  # G0 is the source
    "gate1": ["data/manifests/power.json"],
    "gate2": ["data/manifests/power.json"],
    "gate3": ["data/manifests/power.json", "data/manifests/folds.csv"],
    "gate4": ["data/manifests/power.json"],
    "gate5": ["data/manifests/power.json"],
    "gate1b": ["data/manifests/power.json"],
    "gate6": ["data/manifests/power.json", "data/manifests/severity.json"],
}

from src.gates.manifests import enforce_g0_manifests as _enforce_g0_manifests  # noqa: E402

# Expected side-effect artifacts (for verification + honesty; Gx writes immutable reports)
EXPECTED_ARTIFACTS: dict[str, list[str]] = {
    "gate0": ["data/manifests/power.json"],
    "gate1": ["data/manifests/folds.csv", "data/manifests/cat_id_map.csv"],
    "gate2": ["artifacts/gate2/confound_audit.json"],
    "gate3": ["data/manifests/test_manifest.json", "data/manifests/test_manifest.sha256"],
    "gate4": ["artifacts/gate4.txt"],
    "gate5": ["reports/gate5_nme.json"],
    "gate1b": ["artifacts/gate1b/kappa_report.json"],
    "gate6": ["data/manifests/severity.json"],
}


def _run_cmd(cmd: list[str], cwd: Path = ROOT, env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run subprocess; capture combined out/err. Return (code, output)."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    proc = subprocess.run(cmd, cwd=str(cwd), env=full_env, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def _synthetic_toy_manifests(tmp: Path) -> dict[str, Path]:
    """Generate minimal deletion-safe toy data for --synthetic e2e (G0-G6 + pilot + wrapper).
    No real Roboflow/CatFLW/exports; pure in-mem derived CSVs + synth scores.
    Enforces reproducibility via fixed seed.
    """
    seed_everything(42)
    import numpy as np
    import pandas as pd

    # Toy for G1/G3/G2/G6 (tiny cat-disjoint, ~13% prevalence feel)
    n = 60
    cats = [f"CAT_{i:02d}" for i in range(6)]
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        cat = cats[i % len(cats)]
        y = 1 if rng.random() < 0.13 else 0
        # filename parseable by src.data.parse.group_id (CAT_ style)
        fname = f"{cat}_20240101_{i:03d}.png"
        rows.append({"filename": fname, "y": y, "image_id": f"img_{i}", "cat_id": cat})
    manifest_df = pd.DataFrame(rows)
    man_path = tmp / "toy_manifest.csv"
    manifest_df.to_csv(man_path, index=False)

    # Pre-synth folds.csv (G1 output) + minimal test_manifest for G3 in synthetic mode
    # (G1 CLI would build it; here pre-populate to keep synthetic deletion-safe + script delegation)
    folds_df = manifest_df[["image_id", "cat_id", "y"]].copy()
    folds_df["fold"] = [i % 5 for i in range(len(folds_df))]  # 5 toy folds
    folds_path = tmp / "folds.csv"
    folds_df.to_csv(folds_path, index=False)

    # Minimal G3 frozen test manifest stub (hash would be real in full; toy for e2e)
    test_man = {
        "version": "toy-synthetic@v0",
        "test_groups": ["CAT_00"],
        "fold_csv_sha256": "deadbeef-toy",
    }
    (tmp / "test_manifest.json").write_text(json.dumps(test_man))
    (tmp / "test_manifest.sha256").write_text("deadbeef-toy")

    # Toy vet-confirmed merged for G1b/G6 (per-image {au}_vlm/{au}_vet + cat_id)
    # Small N ~ pilot size; some disagreement to exercise kappa LB + noise rate
    n_pilot = 30
    au_data = {"cat_id": []}
    for au in AU_NAMES:
        au_data[f"{au}_vlm"] = []
        au_data[f"{au}_vet"] = []
    for i in range(n_pilot):
        cat = cats[i % len(cats)]
        au_data["cat_id"].append(cat)
        for au in AU_NAMES:
            true = rng.integers(0, 3)
            # VLM noisy (simulates pilot disagreement; overall ~0.25-0.35 for realism)
            vlm = true if rng.random() > 0.30 else rng.integers(0, 3)
            au_data[f"{au}_vlm"].append(int(vlm))
            au_data[f"{au}_vet"].append(int(true))
    pilot_df = pd.DataFrame(au_data)
    pilot_path = tmp / "toy_vet_pilot.csv"
    pilot_path.parent.mkdir(parents=True, exist_ok=True)
    pilot_df.to_csv(pilot_path, index=False)

    # Toy for G2 confound probe (needs pain, cat_id + scalar features)
    probe_df = manifest_df[["y", "cat_id"]].copy()
    probe_df = probe_df.rename(columns={"y": "pain"})
    probe_df["brightness"] = rng.normal(0.5, 0.2, len(probe_df))
    probe_df["blur"] = rng.normal(0.1, 0.05, len(probe_df))
    probe_df["aspect_ratio"] = rng.uniform(0.8, 1.2, len(probe_df))
    probe_path = tmp / "toy_features_probe.csv"
    probe_df.to_csv(probe_path, index=False)

    # Toy vet csv for G6 (same columns as pilot)
    sev_path = tmp / "toy_vet_severity.csv"
    pilot_df.to_csv(sev_path, index=False)  # reuse; counts will be small -> collapse likely

    return {
        "manifest": man_path,
        "folds": folds_path,
        "pilot": pilot_path,
        "probe": probe_path,
        "severity": sev_path,
        "tmp": tmp,
    }


def _verify_artifact(path_str: str, run_dir: Path = ROOT) -> bool:
    p = run_dir / path_str
    return p.exists() and p.stat().st_size > 0


def run_pipeline(synthetic: bool = False, include_wrapper: bool = False, dry_run: bool = False, portable_only: bool = False) -> dict[str, Any]:
    """Core orchestrator. Returns summary dict (also written to artifacts). Aborts on fail."""
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary: dict[str, Any] = {
        "ts": ts,
        "synthetic": bool(synthetic),
        "portable_only": bool(portable_only),
        "order": GATE_ORDER[:],
        "results": {},
        "artifacts_verified": {},
        "aborted_at": None,
        "power_first": True,  # G0 tie-in explicit
        "honesty_notes": [
            "G0 power pre-registration first (blocks quantitative).",
            "Single-source 0.39 via src.vlm.aggregate (decode + VLM paths).",
            "One-directional G2; CI-LB G1b; hash G3; BLOCKING G4.",
            "All via committed manifests/artifacts (uniqueness enforcement).",
            "Portable protocols (src.protocols) surface: deletion-safe / import-isolated kappa+confound+adapters reusable on the next corpus.",
        ],
    }

    toy = None
    tmp_ctx = None
    if synthetic:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="gate_e2e_toy_")
        tmp = Path(tmp_ctx.name)
        toy = _synthetic_toy_manifests(tmp)
        summary["toy_dir"] = str(toy["tmp"])
    if portable_only:
        print("[orchestrator] --portable-only: light mode for portable protocols claim (see test-portable, standalone_test_corpus, monkeypatch isolation tests). Skips full heavy gate exec; synthetic protocol paths (e.g. G1b kappa pilot, G2 confound) still validate surface.")

    try:
        for g in GATE_ORDER:
            _enforce_g0_manifests(g, synthetic=synthetic)  # FreshFullGateWiringManifestsEnforcer: hard G0/committed manifests before any gate compute (real paths contract)
            if portable_only and g not in ("gate1b", "gate2", "gate0"):
                # portable-only: only exercise protocol-bearing gates (G1b uses kappa pilot on synth; G2 confound; G0 power is pure)
                print(f"[orchestrator] portable-only: skipping heavy {g} (only portable surface paths run)")
                summary["results"][g] = {"skipped": "portable_only", "note": "see src.protocols for full isolation"}
                continue
            scripts = GATE_SCRIPTS[g]
            print(f"\n[orchestrator] === {g.upper()} (synthetic={synthetic}) ===")
            g_outs = []
            g_code = 0
            for i, script in enumerate(scripts):
                cmd = ["uv", "run", "python", str(ROOT / script)]
                extra_args: list[str] = []
                env_override: dict[str, str] | None = None

                # Gate-specific arg wiring (scripts-only; no locality edits)
                if g == "gate1":
                    if synthetic:
                        # Pre-placed folds; skip full CLI exec for synthetic (G1 plumbing covered in unit tests; toy ensures downstream)
                        print("[orchestrator] G1 synthetic: using pre-generated toy folds (core G1 code exercised in dedicated tests)")
                        g_outs.append("SYNTHETIC-STUB")
                        continue
                    extra_args = ["--manifest", str(ROOT / "data/manifests/folds.csv")]
                elif g == "gate2":
                    extra_args = ["--features-csv", str(toy["probe"] if synthetic else (ROOT / "artifacts/gate2/toy_features.csv"))]
                elif g == "gate3":
                    if synthetic:
                        print("[orchestrator] G3 synthetic: using pre-placed toy test_manifest (G3 hash guard exercised in test_no_test_leak + dedicated)")
                        g_outs.append("SYNTHETIC-STUB")
                        continue
                    folds_for_g3 = str(ROOT / "data/manifests/folds.csv")
                    extra_args = ["--folds-csv", folds_for_g3, "--test-groups", "CAT_00"]
                elif g == "gate5":
                    if synthetic:
                        print("[orchestrator] G5 synthetic: stub (NME audit requires CatFLW; core pipeline + G4/G0/G1b/G6 exercised)")
                        g_outs.append("SYNTHETIC-STUB (G5 NME)")
                        continue
                    extra_args = ["--n", "1"]
                elif g == "gate1b":
                    # Synthetic: bypass run_vlm_labels (needs key + real manifest); directly exercise pilot on toy vet csv
                    if synthetic and i == 0:
                        print("[orchestrator] G1b synthetic: skipping real VLM batch (no key); direct kappa pilot on toy")
                        continue
## src/eval/separability.py
"""Frozen-feature separability probe (GITHUB_MINE Pass-3 P3.2 — tactical hardening).

A cheap, NON-PARAMETRIC sanity baseline on the cached frozen DINOv2 features: BEFORE
fitting the CORN / binary-pain heads, can a kNN or a linear probe separate pain from
no-pain at all? The trained head should BEAT this floor; if it does not, the problem is
the features (or the labels), not the head.

DIAGNOSTIC ONLY — NO claim ships from this number (THE LAW: the engine is conceded
plumbing, the graded layer is inspected-not-validated). It is a floor the head should
clear, never a reported result.

CIRCULARITY GUARD: fits on `train_folds`, scores on DISJOINT `eval_folds`, and ASSERTS
that no cat_id appears in both — the same StratifiedGroupKFold-by-cat_id discipline used
everywhere else (THE LAW: CV grouped by cat_id). Running a probe on the rows it was fit
on, or letting one cat straddle the split, would make the number circular. Reads the npz
written by src.model.cache_features (keys: cls, patch_mean, cat_id, fold, y, y_pain).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import AU_ORDER
from src.model.cache_features import _CACHE_SCHEMA

_POOLS = ("cls", "patch_mean", "patch_std")  # patch_std exposed for richer DINOv3 A/B per MCP dinov3 + schema enforcement


def _labels(data, task: str) -> np.ndarray:
    """Binary target: task='pain' -> y_pain; task=<AU name> -> (AU score >= 1)."""
    if task == "pain":
        return np.asarray(data["y_pain"]).astype(int)
    if task in AU_ORDER:
        col = list(AU_ORDER).index(task)
        return (np.asarray(data["y"])[:, col] >= 1).astype(int)
    raise ValueError(f"task must be 'pain' or one of {AU_ORDER}, got {task!r}")


def _auc_acc(clf, X_tr, y_tr, X_ev, y_ev) -> tuple[float, float]:
    """Fit clf on train, return (roc_auc, accuracy) on eval; auc is nan if single-class."""
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_ev)
    acc = float(accuracy_score(y_ev, pred))
    if len(np.unique(y_ev)) < 2:
        return float("nan"), acc
    proba = clf.predict_proba(X_ev)[:, 1]
    return float(roc_auc_score(y_ev, proba)), acc


def probe_separability(
    npz_path,
    pool: str = "cls",
    train_folds=(0, 1, 2),
    eval_folds=(3, 4),
    task: str = "pain",
    k: int = 5,
    seed: int = 42,
) -> dict:
    """kNN + linear-probe separability of frozen features on HELD-OUT folds (diagnostic).

    npz_path: cache from src.model.cache_features.build_cache.
    pool: 'cls' or 'patch_mean' (which pooled feature to probe).
    train_folds / eval_folds: disjoint fold ids; the probe fits on train, scores on eval.
    task: 'pain' (y_pain) or an AU name (binary AU-present >= 1).

    Returns dict(pool, task, n_train, n_eval, knn_auc, knn_acc, linear_auc, linear_acc,
    distinct_pain_cats_eval, note). The `note` states this is a diagnostic floor, not a
    validated result. Raises ValueError if a cat_id appears in both splits (circularity).
    """
    if pool not in _POOLS:
        raise ValueError(f"pool must be one of {_POOLS}, got {pool!r}")
    data = np.load(npz_path, allow_pickle=True)  # allow object for test synthetic npz (schema_version etc as U or str scalars); prod caches from cache_features use primitive arrays so safe either way. Enables the stricter _CACHE_SCHEMA enforce while keeping separability tests working.

    # Enforce schema_version + cross-prov + dim hygiene on read (builds on
    # protocols adapters + importlib-style seam for portable versioned caches; uniform with train_heads).
    # Supports A/B dinov3 richer (patch_std/layer/patch_mode per MCP). Preserves small-data repro. Hard on mismatch (no legacy tolerate).
    if "schema_version" not in data:
        raise ValueError(f"cache schema_version missing (uniform enforce): {npz_path}; run fresh cache build (supports richer intermed/L2)")
    # Full _CACHE_SCHEMA keys + PROVENANCE cross (FreshDINOv3RicherEnforcer tiny #1; uniform with train_heads). Supports A/B dinov3 richer (patch_std/layer/patch_mode per MCP). Hard on mismatch (no legacy tolerate).
    # PRACTICAL A/B uniform hard enforce: fixed keys exact match to _CACHE_SCHEMA; variable (variant/layer/patch_mode) require presence (actual value from build, cross prov for hash); supports intermed n=list mean+std richer + dinov3_vits16 vs v2_reg
    fixed_keys = ("n_aus", "au_hash", "k", "feature_dim")
    for schema_key in fixed_keys:
        if schema_key not in data or data[schema_key] != _CACHE_SCHEMA[schema_key]:
            raise ValueError(f"_CACHE_SCHEMA key mismatch on {schema_key} (enforce)")
    for schema_key in ("variant", "layer", "patch_mode"):
        if schema_key not in data:
            raise ValueError(f"_CACHE_SCHEMA key missing {schema_key} (uniform enforce no legacy)")
    # optional prov cross for A/B hash (layer/patch_mode/variant drive from corn.yaml)
    if "provenance" in data:
        prov = data["provenance"] if isinstance(data["provenance"], dict) else {}
        for schema_key in ("variant", "layer", "patch_mode"):
            if schema_key in prov and data.get(schema_key) != prov.get(schema_key):
                raise ValueError(f"PROVENANCE cross mismatch on {schema_key} (full hash enforce)")
    n_au = len(AU_ORDER)
    if "y" in data:
        yarr = np.asarray(data["y"])
        if yarr.ndim > 1 and yarr.shape[1] != n_au:
            raise ValueError(f"cache y dim {yarr.shape[1]} != AU_ORDER len {n_au}")
    # PROVENANCE sidecar cross (variant/registers/layer/patch_mode if present)
    if "provenance" in data and "variant" in data["provenance"] and data.get("variant") != data["provenance"]["variant"]:
        raise ValueError("PROVENANCE variant cross mismatch (full enforce)")
    # n_aus cross with protocols (generic surface)
    from src.protocols.adapters import get_au_names
    if len(get_au_names()) != n_au:
        raise ValueError("n_aus mismatch vs protocols generic (portable tie)")

    X = np.asarray(data[pool], dtype=np.float32)
    y = _labels(data, task)
    cat = np.asarray(data["cat_id"]).astype(str)
    fold = np.asarray(data["fold"]).astype(str)  # fold stored as strings in the cache

    train_set = {str(f) for f in train_folds}
    eval_set = {str(f) for f in eval_folds}
    if train_set & eval_set:
        raise ValueError(f"train_folds and eval_folds overlap: {train_set & eval_set}")

    tr = np.isin(fold, list(train_set))
    ev = np.isin(fold, list(eval_set))

    # CIRCULARITY GUARD: no individual may sit in both the fit and the score split.
    shared = set(cat[tr]) & set(cat[ev])
    if shared:
        raise ValueError(
            f"cat_id leakage across train/eval folds (probe would be circular): {sorted(shared)}"
        )

    knn_auc, knn_acc = _auc_acc(
        KNeighborsClassifier(n_neighbors=k), X[tr], y[tr], X[ev], y[ev]
    )
    linear_auc, linear_acc = _auc_acc(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)),
        X[tr], y[tr], X[ev], y[ev],
    )

    return {
        "pool": pool,
        "task": task,
        "n_train": int(tr.sum()),
        "n_eval": int(ev.sum()),
        "knn_auc": knn_auc,
        "knn_acc": knn_acc,
        "linear_auc": linear_auc,
        "linear_acc": linear_acc,
        "distinct_pain_cats_eval": int(np.unique(cat[ev & (y == 1)]).size),
        # NOT a validated result: a diagnostic floor the trained head should beat.
        "note": "diagnostic separability floor (held-out, cat-grouped); not a validated result",
    }
