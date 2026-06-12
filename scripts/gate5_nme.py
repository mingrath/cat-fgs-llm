#!/usr/bin/env python3
"""Gate 5 NME / resolution audit driver (IMPLEMENTATION_PLAN §3.3).

BLOCKING gate: no CORN head is trained and no vet hour is spent until this
passes (FINAL_DIRECTION Gate order). CPU-only on the M4 (~2 min on 50 imgs).

Answers two yes/no questions on 30-50 CatFLW images:

  1. Is the cheap detector-box crop's landmark layout close enough to a canonical
     eye-aligned layout? -> median NME_A <= 0.08 AND gap(A-B) <= 0.02.
  2. Is the face big enough after resize? -> median inter-ocular >= 40 px.

We do NOT train a landmark detector. We evaluate LANDMARK STABILITY under the two
crop transforms: how much the SAME CatFLW ground-truth landmarks, re-expressed in
each candidate crop's coordinate frame, drift relative to a canonical eye-aligned
layout. NME is interocular-normalized (the standard cat-landmark metric;
ref/cat-landmark-align).

  Crop A (box-as-cropper): expand CatFLW bbox by 13.5%, letterbox -> 518; map the
                           48 GT landmarks into this frame. The eye-aligned layout
                           (Crop B) IS the canonical reference; NME_A = Crop A's
                           drift from it.
  NME_B (residual floor):  best-fit similarity of the Crop-A layout onto the
                           canonical layout, residual NME — i.e. the part of the
                           drift a 2-point aligner could NOT fix. gap = NME_A -
                           NME_B is the similarity-recoverable drift, the quantity
                           the box-vs-aligner decision actually rides on.

Decision in {box_ok, need_aligner, need_higher_res} written to reports/gate5_nme.json.

This audit certifies conceded preprocessing plumbing; it makes no validated
clinical claim. QWK-vs-VLM is never validation; this is crop-quality only.

CRITICAL TODO (gate-exit blocker, §3.6): the L/R EYE-CENTER indices into the
48-point CatFLW landmark array are NOT confirmed against the CatFLW index map.
configs/crop.yaml `audit.eye_index_pinned: false` MUST become true (after visual
confirmation on the audit set) and the chosen indices recorded into the output
JSON BEFORE any NME number is trusted. This script refuses to declare a pass
while `eye_index_pinned` is false.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

import cv2
import numpy as np
import yaml

import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `python scripts/gate5_nme.py` without PYTHONPATH

from src.crop.align_eyes import canonical_targets  # noqa: E402
from src.crop.expand import expand_box  # noqa: E402


def load_crop_config(path: pathlib.Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _list_pairs(catflw_dir: pathlib.Path):
    """Return [(image_path, label_path)] for CatFLW images with a JSON label.

    Prefers images that also carry a ``CAT_`` id so the audit set overlaps our
    corpus morphology (§3.3); falls back to all paired images otherwise.
    """
    img_dir = catflw_dir / "images"
    lbl_dir = catflw_dir / "labels"
    pairs = []
    for img in sorted(img_dir.glob("*.png")):
        lbl = lbl_dir / (img.stem + ".json")
        if lbl.exists():
            pairs.append((img, lbl))
    cat_pairs = [p for p in pairs if p[0].name.upper().startswith("CAT_")]
    return cat_pairs if cat_pairs else pairs


def _load_label(label_path: pathlib.Path):
    """CatFLW JSON -> (landmarks [48,2] float, bbox [x1,y1,x2,y2] float) in pixels."""
    d = json.loads(label_path.read_text())
    lm = np.asarray(d["labels"], dtype=np.float64)          # [48,2]
    bb = np.asarray(d["bounding_boxes"], dtype=np.float64)  # [x1,y1,x2,y2]
    return lm, bb


def _bbox_to_norm(bbox, img_w, img_h):
    """Pixel [x1,y1,x2,y2] -> normalized YOLO (cx,cy,w,h) for expand_box()."""
    x1, y1, x2, y2 = bbox
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    return cx, cy, w, h


def _map_points_box_frame(pts, box_xyxy, edge):
    """Map source-pixel landmarks into the Crop-A (letterbox->resize) frame.

    Replicates the §3.0 transform geometrically (translate by crop origin, pad to
    square, uniform resize) so we can re-express GT landmarks in the 518 frame.
    """
    x1, y1, x2, y2 = box_xyxy
    cw, ch = (x2 - x1), (y2 - y1)
    side = max(cw, ch)
    pad_left = (side - cw) // 2
    pad_top = (side - ch) // 2
    scale = edge / side
    out = np.empty_like(pts)
    out[:, 0] = (pts[:, 0] - x1 + pad_left) * scale
    out[:, 1] = (pts[:, 1] - y1 + pad_top) * scale
    return out


def _map_points_affine(pts, M):
    """Apply a 2x3 affine M (from estimateAffinePartial2D) to [N,2] points."""
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homo = np.hstack([pts, ones])           # [N,3]
    return homo @ M.T                        # [N,2]


def _interocular(lm, li, ri):
    return float(np.linalg.norm(lm[li] - lm[ri]))


def _nme(pred_pts, ref_pts, d_interocular):
    """Interocular-normalized NME between a layout and a canonical reference."""
    if d_interocular <= 0:
        return float("nan")
    per_pt = np.linalg.norm(pred_pts - ref_pts, axis=1)
    return float(per_pt.mean() / d_interocular)


def run_audit(catflw_dir: pathlib.Path, cfg: dict, n: int, seed: int,
              edge: int, expand: float):
    """Compute NME_A, NME_B, gap, interocular-px-after-resize per image."""
    acfg = cfg["audit"]
    li = int(acfg["left_eye_idx"])
    ri = int(acfg["right_eye_idx"])
    eye_index_pinned = bool(acfg.get("eye_index_pinned", False))
    height_mult = float(cfg["height_expand_mult"])
    align_cfg = cfg["align"]
    left_t, right_t = canonical_targets(
        edge, tuple(align_cfg["left_target"]), tuple(align_cfg["right_target"])
    )

    pairs = _list_pairs(catflw_dir)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    pairs = pairs[:n]

    per_image = []
    for img_path, lbl_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        lm, bbox = _load_label(lbl_path)

        # --- Crop A: box-as-cropper (expand 13.5%, letterbox -> edge) ---
        cx, cy, bw, bh = _bbox_to_norm(bbox, img_w, img_h)
        box_xyxy, _clipped = expand_box(
            cx, cy, bw, bh, img_w, img_h, expand=expand, height_mult=height_mult
        )
        lm_A = _map_points_box_frame(lm, box_xyxy, edge)
        interocular_A = _interocular(lm_A, li, ri)

        # --- Crop B: 2-point eye-aligned reference (similarity onto canonical) ---
        src_eyes = np.float32([lm[li], lm[ri]])
        dst_eyes = np.float32([left_t, right_t])
        M, _ = cv2.estimateAffinePartial2D(src_eyes, dst_eyes)
        lm_B = _map_points_affine(lm, M)
        interocular_B = _interocular(lm_B, li, ri)  # ~= interocular_frac*edge

        # Canonical reference layout = the eye-aligned layout (Crop B). NME_A is
        # how far the box-crop layout drifts from that canonical; NME_B ~ 0 by
        # construction at the two eye points and measures residual non-similar
        # deformation across the other 46 points after best-fit alignment.
        nme_A = _nme(lm_A, lm_B, interocular_B)
        # NME_B: re-express Crop-A layout's own canonical fit; report the eye-line
        # residual as the achievable floor (the aligner's own NME).
        # Best-fit similarity of lm_A onto lm_B, residual normalized.
        M_AB, _ = cv2.estimateAffinePartial2D(
            lm_A.astype(np.float32), lm_B.astype(np.float32)
        )
        lm_A_fit = _map_points_affine(lm_A, M_AB)
        nme_B = _nme(lm_A_fit, lm_B, interocular_B)

        per_image.append({
            "image_id": img_path.stem,
            "nme_A": nme_A,
            "nme_B": nme_B,
            "gap": nme_A - nme_B,
            "interocular_px_after_resize": interocular_A,
        })

    return per_image, eye_index_pinned, li, ri


def summarize(per_image: list[dict], cfg: dict, eye_index_pinned: bool):
    """Aggregate per-image rows into the §3.3 audit decision."""
    acfg = cfg["audit"]
    nme_pass = float(acfg["nme_pass_max"])
    gap_pass = float(acfg["gap_pass_max"])
    px_min = float(acfg["interocular_px_min"])

    nmes_A = np.asarray([r["nme_A"] for r in per_image], dtype=np.float64)
    gaps = np.asarray([r["gap"] for r in per_image], dtype=np.float64)
    px = np.asarray([r["interocular_px_after_resize"] for r in per_image], dtype=np.float64)

    median_nme_A = float(np.nanmedian(nmes_A)) if len(nmes_A) else float("nan")
    median_nme_B = float(np.nanmedian([r["nme_B"] for r in per_image])) \
        if per_image else float("nan")
    median_gap = float(np.nanmedian(gaps)) if len(gaps) else float("nan")
    median_px = float(np.nanmedian(px)) if len(px) else float("nan")
    worst_decile_nme = float(np.nanpercentile(nmes_A, 90)) if len(nmes_A) else float("nan")

    nme_ok = (median_nme_A <= nme_pass) and (median_gap <= gap_pass)
    res_ok = median_px >= px_min

    # decision (§3.3 fail actions):
    #   resolution fail FIRST -> need_higher_res (must be resolved before cropping)
    #   else NME fail          -> need_aligner (switch on §3.4 aligner)
    #   else                   -> box_ok
    if not res_ok:
        decision = "need_higher_res"
    elif not nme_ok:
        decision = "need_aligner"
    else:
        decision = "box_ok"

    # A pass is NOT declarable until the eye-center indices are pinned (§3.6):
    # the decision string itself is withheld so no downstream consumer can read
    # box_ok off the report while the eye indices are best-guess placeholders.
    eye_index_blocked = (decision == "box_ok") and not eye_index_pinned
    if eye_index_blocked:
        decision = "blocked_eye_index_unpinned"

    return {
        "n": len(per_image),
        "median_nme_box": median_nme_A,      # NME_A (box-as-cropper)
        "median_nme_aligned": median_nme_B,  # NME_B (eye-aligned reference floor)
        "gap": median_gap,
        "median_interocular_px_after_resize": median_px,
        "worst_decile_nme": worst_decile_nme,
        "thresholds": {
            "nme_pass_max": nme_pass,
            "gap_pass_max": gap_pass,
            "interocular_px_min": px_min,
        },
        "nme_ok": nme_ok,
        "resolution_ok": res_ok,
        "decision": decision,
        "eye_index_pinned": eye_index_pinned,
        "eye_index_blocked": eye_index_blocked,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate 5 NME / resolution audit (§3.3)")
    ap.add_argument("--catflw", default=str(ROOT / "datasets" / "catflw" / "CatFLW dataset"),
                    help="CatFLW dataset dir containing images/ and labels/")
    ap.add_argument("--n", type=int, default=None, help="audit image count (default: config audit.n)")
    ap.add_argument("--seed", type=int, default=None, help="sample seed (default: config audit.seed)")
    ap.add_argument("--crop-edge", type=int, default=None, help="output edge (default: config edge)")
    ap.add_argument("--expand", type=float, default=None, help="margin (default: config expand)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "crop.yaml"))
    ap.add_argument("--out", nargs="+",
                    default=[str(ROOT / "reports" / "gate5_nme.json"),
                             str(ROOT / "reports" / "gate5_nme.csv")],
                    help="output paths: <json> [<csv>]")
    args = ap.parse_args()

    cfg = load_crop_config(pathlib.Path(args.config))
    n = args.n if args.n is not None else int(cfg["audit"]["n"])
    seed = args.seed if args.seed is not None else int(cfg["audit"]["seed"])
    edge = args.crop_edge if args.crop_edge is not None else int(cfg["edge"])
    expand = args.expand if args.expand is not None else float(cfg["expand"])

    catflw_dir = pathlib.Path(args.catflw)
    per_image, eye_index_pinned, li, ri = run_audit(catflw_dir, cfg, n, seed, edge, expand)
    summary = summarize(per_image, cfg, eye_index_pinned)
    summary["eye_center_indices"] = {"left": li, "right": ri}
    summary["crop_edge"] = edge
    summary["expand"] = expand

    out_json = pathlib.Path(args.out[0])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))

    if len(args.out) > 1:
        import csv
        out_csv = pathlib.Path(args.out[1])
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            fields = ["image_id", "nme_A", "nme_B", "gap", "interocular_px_after_resize"]
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for row in per_image:
                w.writerow(row)

    print(json.dumps(summary, indent=2))
    if summary["eye_index_blocked"]:
        raise SystemExit(
            "\nGATE 5 REFUSED: metrics would pass (box_ok) but the eye-center "
            "indices are NOT pinned (configs/crop.yaml audit.eye_index_pinned=false), "
            "so every NME number above is computed against best-guess eye indices. "
            "Pin L/R against the CatFLW 48-point index map, set eye_index_pinned: "
            "true, and re-run (§3.6)."
        )


if __name__ == "__main__":
    main()
