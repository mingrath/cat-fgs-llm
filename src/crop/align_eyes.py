"""2-point eye-similarity alignment fallback (IMPLEMENTATION_PLAN §3.4).

Switched on only if the Gate 5 NME audit (§3.3) fails. It is a closed-form
similarity transform (rotate + uniform scale + translate) from the two detected
eye centers to two CANONICAL target points — no learned warp, no per-image
optimization, MPS-irrelevant (pure NumPy/OpenCV on CPU).

Eye-source priority (§3.4): (1) CatFLW / Martvel-detector eye centers when
available; (2) the local Zhang-2008 archive 2 eye landmarks ONLY as a fallback
if CatFLW access stalls. Haar is never the landmark source (Steagall-trap).
Interocular-normalization + PCK/NME implementation follows ref/cat-landmark-align.

This is conceded preprocessing plumbing, not a claimed-novel artifact (§3).
"""

from __future__ import annotations

import cv2
import numpy as np

OUT = 518  # default output edge; pass `out` to override with a legal x14 value.

# Canonical eye targets: eyes on a horizontal line, interocular = 38% of edge,
# centered, slightly high. Keeps ears/whiskers inside the frame after rotation
# (replicate border on corners that rotate out). Fractions of `out` (§3.4).
LEFT_TARGET_FRAC = (0.31, 0.42)
RIGHT_TARGET_FRAC = (0.69, 0.42)


def canonical_targets(out: int = OUT,
                      left_frac: tuple[float, float] = LEFT_TARGET_FRAC,
                      right_frac: tuple[float, float] = RIGHT_TARGET_FRAC):
    """Canonical (left, right) eye target pixel coords for an `out`-edge frame."""
    left_t = np.float32([out * left_frac[0], out * left_frac[1]])
    right_t = np.float32([out * right_frac[0], out * right_frac[1]])
    return left_t, right_t


def align_by_eyes(img, left_eye, right_eye, out: int = OUT,
                  left_frac: tuple[float, float] = LEFT_TARGET_FRAC,
                  right_frac: tuple[float, float] = RIGHT_TARGET_FRAC):
    """Similarity-align an image so the eyes hit the canonical targets (§3.4).

    Parameters
    ----------
    img : np.ndarray
        Expanded crop (H, W, 3) — eyes given in this image's pixel coords.
    left_eye, right_eye : array-like
        Eye-center pixel coordinates in `img`.
    out : int
        Output edge (square). Must be divisible by 14 upstream; default 518.
    left_frac, right_frac : tuple[float, float]
        Canonical eye-target fractions of `out` (from config; default §3.4).

    Returns
    -------
    np.ndarray
        Aligned (out, out, 3) image, BORDER_REPLICATE on rotated-out corners.
    """
    left_t, right_t = canonical_targets(out, left_frac, right_frac)
    src = np.float32([left_eye, right_eye])
    dst = np.float32([left_t, right_t])
    # similarity (rot + uniform scale + trans), NO shear
    M, _ = cv2.estimateAffinePartial2D(src, dst)
    return cv2.warpAffine(
        img, M, (out, out),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
