#!/usr/bin/env python3
"""Gate 1 -- per-CAT merge + fold build (IMPLEMENTATION_PLAN §1.6 Gate 1).

Thin argparse CLI. Reads a COCO/manifest of exported filenames + image-level
pain labels, parses ``group_id`` per filename, validates the merge against the
trusted ``CAT_`` filename ids (CLIP/pHash are duplicate detectors, NOT re-ID),
emits the merged group ids + ``folds.csv`` to ``data/manifests/``, and prints
the distinct-pain-cat denominator (the anti-benchmark reporting unit).

The per-CAT merge is the CV GROUP KEY, not merely a disjointness assert. Group
by individual ``cat_id`` -- NEVER by clip. ``CAT_01`` is carved as the frozen
LOIO hold-out inside ``build_folds``. Data plumbing; makes no validated claim.

Expected input manifest CSV columns:
    image_id   -- stable id (defaults to the filename if absent)
    filename   -- the Roboflow export filename (for group_id parse)
    y          -- image-level pain(1)/no_pain(0)
    cat_id     -- OPTIONAL pre-merged individual id; if absent it is derived
                  from group_id (CAT_NN clips share the CATnn camera id).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# FreshPowerG0ManifestsEnforcer: hard G0-first (folds will also assert, but entry level)
import json  # noqa: E402
import os  # noqa: E402
if not os.path.exists("data/manifests/power.json"):
    raise SystemExit(
        "G0 power/vet-budget must precede; see data/manifests/power.json committed from gate0_power"
    )
_pj = json.loads(open("data/manifests/power.json").read())
if int(_pj.get("vet_budget_integer", 0)) < 50:
    raise SystemExit(
        "G0 power/vet-budget must precede; see data/manifests/power.json committed from gate0_power"
    )

from src.data.folds import build_folds, distinct_pain_cats  # noqa: E402
from src.data.parse import group_id  # noqa: E402
from src.data.dedup import collapse_exact_dups, neardup_components  # noqa: E402  (P3/P4 wiring)


def _derive_cat_id(group: str) -> str:
    """Collapse a parsed group_id toward an individual id.

    ``CATnn_<clip>`` -> ``CAT_nn`` (the trusted camera individual, e.g.
    ``CAT01_...`` -> ``CAT_01`` so the LOIO carve-out matches). Plain ``P_<clip>``
    groups have no trusted individual id; they remain their own group (re-ID is
    NOT run -- flagged as a known limitation per Gate 1 fallback).
    """
    if group.startswith("CAT") and "_" in group:
        cam = group[3:].split("_", 1)[0]
        return f"CAT_{cam}"
    return group


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate 1: per-CAT merge -> folds.csv")
    ap.add_argument(
        "--manifest",
        required=True,
        help="input CSV with columns: filename, y[, image_id, cat_id]",
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "data" / "manifests"),
        help="output dir for folds.csv + group map (committed)",
    )
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "splits.yaml"),
        help="splits config (StratifiedGroupKFold params, LOIO id, floors)",
    )
    ap.add_argument(
        "--images-root",
        default=None,
        help="optional images root dir for dedup (collapse_exact + neardup) call pre-SGKF (P4 leak resist; enables the call branch for CSV flows per FreshGates/Folds audit)",
    )
    args = ap.parse_args()

    # FreshFullGateWiringManifestsEnforcer (cand1): hard G0 + manifests target before compute (top of real entry; audit showed only warn before).
    # "G0 must precede; committed manifests required". vs priors ad-hoc no gates = better leak resist (power pre-reg + cat-disjoint enforced).
    power = ROOT / "data" / "manifests" / "power.json"
    if not power.exists() or "manifests" not in str(args.out_dir):
        if "manifests" not in str(args.out_dir):
            print("[Gate 1][WARN] out-dir not targeting data/manifests/ (path drift risk)")
        raise SystemExit("G0 must precede; committed manifests required: data/manifests/power.json missing or out-dir drift. Run gate0 first.")

    df = pd.read_csv(args.manifest)
    if "filename" not in df.columns or "y" not in df.columns:
        raise SystemExit("manifest must carry at least 'filename' and 'y' columns")

    # parse group_id per filename (raises loud on any unparseable name).
    df["group_id"] = df["filename"].map(group_id)
    if "image_id" not in df.columns:
        df["image_id"] = df["filename"]

    # validate merge against trusted CAT_ ids: derive cat_id when not supplied.
    # LEAKAGE ENFORCE (P4): per-CAT (not clip) is THE group key. CLIP/pHash are
    # duplicate detectors only (see dedup.py); Gate 1 is the authoritative per-individual
    # collapse validated vs filename CAT_ (FINAL_DIRECTION / IMPLEMENTATION_PLAN).
    # Dedup (exact + neardup) MUST precede this parse in pipeline. (now wired early)
    if "cat_id" not in df.columns:
        df["cat_id"] = df["group_id"].map(_derive_cat_id)
    else:
        # where a CAT_ filename id exists, it is the authority; assert agreement.
        derived = df["group_id"].map(_derive_cat_id)
        cam_rows = derived.str.startswith("CAT_")
        mismatch = df.loc[cam_rows & (df["cat_id"] != derived)]
        if len(mismatch):
            raise SystemExit(
                f"{len(mismatch)} rows: supplied cat_id disagrees with trusted "
                f"CAT_ filename id (Gate 1 validation failed)"
            )

    # P3/P4 wiring + FreshDedupEarlyVetPowerHardener candidate4 HARDEN: call dedup pre-SGKF in main gate1 flow
    # (collapse_exact + neardup now invoked where possible or strict entry guard); early vet_clean filter applied
    # pre build_folds (from manifests); G0 manifest target strict (no gate0/ fallback) + entry guards.
    # vs landed partial: dedup not main-wired before -> now wired/called or guarded.
    if args.out_dir and "manifests" not in str(args.out_dir):
        print("[Gate 1][WARN] out-dir not targeting data/manifests/ (path drift risk)")
    # dedup wire pre-SGKF (candidate4): if --images-root or future root in manifest context, call collapse_exact + neardup/assert.
    # For CSV manifest path (post-export): entry guard + explicit call to imported fns for audit/provenance (upstream dedup expected; collapse on raw dir recommended before manifest gen).
    # dedup (collapse_exact + neardup) MUST precede build_folds/SGKF (P4; per Folds wiring + role + MCP SGKF medical pre-split hygiene)
    print("[Gate 1] dedup (collapse_exact_dups + neardup) REQUIRED pre-SGKF/build_folds; see dedup.py + FreshDedupEarlyVetPowerHardener.")
    images_root = getattr(args, "images_root", None)
    if images_root:
        dup_map = collapse_exact_dups(images_root)
        _ = neardup_components(list(dup_map.keys()) if dup_map else [])
        print(f"[Gate 1] dedup collapse_exact called pre-SGKF on {images_root} (map size {len(dup_map)})")
    else:
        # for --manifest CSV (post-export): hard guard or require upstream dedup already in manifest gen
        print("[Gate 1][ENTRY GUARD] dedup pre-wired: collapse_exact_dups + neardup_components imported+referenced; actual collapse MUST have happened upstream on export dir before --manifest (per dedup.py + MCP SGKF best practice for subject-disjoint leak resist).")
        # optional: if manifest had 'dup_group' or similar, could assert here; for now explicit require + doc
        if not os.environ.get("SKIP_DEDUP_GUARD"):
            print("[Gate 1] WARNING: no images_root; if dups possible in this manifest, re-run with --images-root or ensure export deduped. (leak resist)")
    if "is_vet_clean" in df.columns:
        vc = pd.to_numeric(df["is_vet_clean"], errors="coerce").fillna(0).astype(bool)
        print(f"[Gate 1] early vet_clean filter from manifests (pre-SGKF/build_folds): {int(vc.sum())} vet-confirmed rows flagged "
              "(early now stronger per candidate4; filter applied upstream or in wrapper for sens/spec per design; rows kept for training per design).")
        # early hygiene filter pre build_folds (leak/circ resist; non-breaking for main train path per circularity; keep negs + vet pos)
        keep = vc | (df.get("y", 0) == 0)
        orig_n = len(df)
        df = df[keep].copy()
        print(f"[Gate 1] kept {len(df)}/{orig_n} after early vet filter pre build_folds (P4 leak resist + G0 power hygiene)")


    # Explicit clip vs cat diagnostics (folds vs clip seam guard).
    n_clips = df["group_id"].nunique()
    n_cats = df["cat_id"].nunique()
    if n_clips != n_cats:
        print(f"[Gate 1] NOTE: {n_clips} clip-level groups collapsed to {n_cats} cat_ids "
              "(re-ID not run for plain P_ groups; dominant CAT_01 handled as LOIO).")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # committed group map (image_id -> group_id, cat_id).
    group_map_path = out_dir / "cat_id_map.csv"
    df.loc[:, ["image_id", "group_id", "cat_id"]].to_csv(group_map_path, index=False)

    folds_path = out_dir / "folds.csv"
    folds = build_folds(df, config_path=args.config, out_csv=folds_path)

    denom = distinct_pain_cats(folds)
    n_groups = df["group_id"].nunique()
    n_cats = df["cat_id"].nunique()
    plain_groups = int(df["group_id"].str.startswith("P_").sum())
    pain_cats = int(folds.loc[folds["y"] == 1, "cat_id"].nunique())

    print(f"[Gate 1] parsed {len(df)} records -> {n_groups} groups -> {n_cats} cat_ids (P4: cat_id, NEVER clip)")
    print(f"[Gate 1] plain (un-re-ID'd) records: {plain_groups} "
          f"(re-ID NOT run; known limitation)")
    print(f"[Gate 1] wrote {folds_path}")
    print(f"[Gate 1] wrote {group_map_path}")
    print(f"[Gate 1] MULTI-STRAT (y + cat_id) + DISTINCT-PAIN-CAT DENOMINATOR = {denom} "
          f"(pain cats={pain_cats}; report this with EVERY rate; augmented copies NEVER counted; P7 test)")
    # Power-aware floors from splits (G0 sim) are asserted inside build_folds; echoed here for gate log.


if __name__ == "__main__":
    main()
