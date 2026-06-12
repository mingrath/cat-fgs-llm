"""Frontal-pose / quality gate — route non-frontal/occluded faces to vet (§3.2).

FGS assumes a near-frontal view. Profile / tilted / occluded / eyes-closed faces
are NOT scored: they are routed to the vet queue and never enter the CORN
feature cache or any reported denominator (anti-benchmark; §3.5). This is a
deterministic, CPU-only filter applied per expanded crop — no training.

All thresholds are read from config (configs/crop.yaml `quality_gate`). They are
tuned ONCE on the Gate 5 audit set then FROZEN before scoring so the gate cannot
be re-tuned to inflate prevalence (§3.2). Every decision is emitted as
(route_vet: bool, reasons: list[str]) into crop_manifest.parquet (§3.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class QualityThresholds:
    """Frozen §3.2 operating values (loaded from configs/crop.yaml quality_gate)."""

    yaw_ratio_min: float = 0.6        # < min  -> route_vet (~ -25 deg yaw)
    yaw_ratio_max: float = 1.67       # > max  -> route_vet (~ +25 deg yaw)
    roll_abs_max_deg: float = 25.0    # in-plane tilt; aligner fixes <=25 deg
    ear_closed_min: float = 0.12      # eye-aspect-ratio; < min -> eyes closed
    blur_var_min: float = 60.0        # variance-of-Laplacian; < min -> blur defer
    haar_scale_factor: float = 1.1
    haar_min_neighbors: int = 3
    haar_min_size: tuple[int, int] = (48, 48)

    @classmethod
    def from_config(cls, cfg: dict) -> "QualityThresholds":
        """Build from the `quality_gate` sub-dict of configs/crop.yaml."""
        return cls(
            yaw_ratio_min=float(cfg["yaw_ratio_min"]),
            yaw_ratio_max=float(cfg["yaw_ratio_max"]),
            roll_abs_max_deg=float(cfg["roll_abs_max_deg"]),
            ear_closed_min=float(cfg["ear_closed_min"]),
            blur_var_min=float(cfg["blur_var_min"]),
            haar_scale_factor=float(cfg.get("haar_scale_factor", 1.1)),
            haar_min_neighbors=int(cfg.get("haar_min_neighbors", 3)),
            haar_min_size=tuple(cfg.get("haar_min_size", (48, 48))),  # type: ignore[arg-type]
        )


@dataclass
class GateResult:
    route_vet: bool = False
    reasons: list[str] = field(default_factory=list)


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Blur metric: variance of the Laplacian on a grayscale crop (§3.2)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def yaw_ratio(nose: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    """Ratio of (nose->left-eye) / (nose->right-eye) distances (§3.2 frontal/yaw).

    A near-frontal face gives ~1.0; profile views push the ratio away from 1.
    """
    nose = np.asarray(nose, dtype=np.float64)
    dl = float(np.linalg.norm(nose - np.asarray(left_eye, dtype=np.float64)))
    dr = float(np.linalg.norm(nose - np.asarray(right_eye, dtype=np.float64)))
    if dr == 0.0:
        return math.inf
    return dl / dr


def roll_deg(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    """In-plane tilt: angle (deg) of the inter-ocular line off horizontal (§3.2)."""
    left_eye = np.asarray(left_eye, dtype=np.float64)
    right_eye = np.asarray(right_eye, dtype=np.float64)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    return float(math.degrees(math.atan2(dy, dx)))


def eye_aspect_ratio(top: np.ndarray, bottom: np.ndarray,
                     inner: np.ndarray, outer: np.ndarray) -> float:
    """Eye-aspect-ratio (EAR): vertical lid opening / horizontal eye width (§3.2).

    < ear_closed_min -> eyes-closed, orbital AU unreadable -> route to vet.
    """
    vert = float(np.linalg.norm(np.asarray(top, dtype=np.float64)
                                - np.asarray(bottom, dtype=np.float64)))
    horiz = float(np.linalg.norm(np.asarray(inner, dtype=np.float64)
                                 - np.asarray(outer, dtype=np.float64)))
    if horiz == 0.0:
        return 0.0
    return vert / horiz


def haar_present(gray: np.ndarray, cascade: "cv2.CascadeClassifier",
                 thresholds: QualityThresholds) -> bool:
    """Coarse Haar `frontalcatface` localizer — detect-failure abstention only.

    This is the detect-failure -> abstention signal (§3.2; BUILD_PLAN P2.2), NOT
    the landmark path (Steagall-trap avoidance: Haar is never used to place
    landmarks). Returns False -> abstention reason 'haar_miss'.
    """
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=thresholds.haar_scale_factor,
        minNeighbors=thresholds.haar_min_neighbors,
        minSize=thresholds.haar_min_size,
    )
    return len(faces) > 0


def load_haar(cascade_path: str) -> "cv2.CascadeClassifier":
    """Load the cat-face Haar cascade; raise if the XML failed to parse."""
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise FileNotFoundError(f"Haar cascade failed to load: {cascade_path}")
    return cascade


def quality_gate(
    crop_bgr: np.ndarray,
    thresholds: QualityThresholds,
    *,
    nose: np.ndarray | None = None,
    left_eye: np.ndarray | None = None,
    right_eye: np.ndarray | None = None,
    eye_lids: tuple | None = None,
    haar_cascade: "cv2.CascadeClassifier | None" = None,
    clipped: bool = False,
) -> GateResult:
    """Run the deterministic §3.2 checks on one expanded crop.

    Parameters
    ----------
    crop_bgr : np.ndarray
        Expanded crop in OpenCV BGR (H, W, 3) uint8.
    thresholds : QualityThresholds
        Frozen operating values.
    nose, left_eye, right_eye : np.ndarray | None
        Landmark points (pixel coords in the crop frame) for yaw/roll. If
        absent (no landmark source), the yaw/roll checks are skipped — the Haar
        detect-failure check still applies.
    eye_lids : tuple | None
        (top, bottom, inner, outer) lid points for EAR; skipped if None.
    haar_cascade : cv2.CascadeClassifier | None
        Loaded cat-face cascade; if None the haar_miss check is skipped.
    clipped : bool
        From expand_box — propagate the ear/whisker clip flag as a reason.

    Returns
    -------
    GateResult(route_vet, reasons)
    """
    result = GateResult()
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # clipped ear/whisker (from §3.1 expand_box) — a clipped grimace AU cannot
    # be scored honestly.
    if clipped:
        result.reasons.append("clipped")

    # frontal / yaw — needs all three points (nose vs eye line)
    if nose is not None and left_eye is not None and right_eye is not None:
        r = yaw_ratio(nose, left_eye, right_eye)
        if r < thresholds.yaw_ratio_min or r > thresholds.yaw_ratio_max:
            result.reasons.append("yaw")

    # in-plane tilt (roll): needs only the two eyes, so check it independently of
    # nose presence. Aligner fixes <=25 deg; beyond is a pose problem.
    if left_eye is not None and right_eye is not None:
        if abs(roll_deg(left_eye, right_eye)) > thresholds.roll_abs_max_deg:
            result.reasons.append("roll")

    # eyes closed (EAR)
    if eye_lids is not None:
        top, bottom, inner, outer = eye_lids
        if eye_aspect_ratio(top, bottom, inner, outer) < thresholds.ear_closed_min:
            result.reasons.append("eyes_closed")

    # occlusion / blur
    if variance_of_laplacian(gray) < thresholds.blur_var_min:
        result.reasons.append("blur")

    # detector miss (Haar) -> abstention
    if haar_cascade is not None:
        if not haar_present(gray, haar_cascade, thresholds):
            result.reasons.append("haar_miss")

    result.route_vet = len(result.reasons) > 0
    return result
