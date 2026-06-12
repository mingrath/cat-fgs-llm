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

from src.data.folds import build_folds, distinct_pain_cats  # noqa: E402
from src.data.parse import group_id  # noqa: E402


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
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if "filename" not in df.columns or "y" not in df.columns:
        raise SystemExit("manifest must carry at least 'filename' and 'y' columns")

    # parse group_id per filename (raises loud on any unparseable name).
    df["group_id"] = df["filename"].map(group_id)
    if "image_id" not in df.columns:
        df["image_id"] = df["filename"]

    # validate merge against trusted CAT_ ids: derive cat_id when not supplied.
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

    print(f"[Gate 1] parsed {len(df)} records -> {n_groups} groups -> {n_cats} cat_ids")
    print(f"[Gate 1] plain (un-re-ID'd) records: {plain_groups} "
          f"(re-ID NOT run; known limitation)")
    print(f"[Gate 1] wrote {folds_path}")
    print(f"[Gate 1] wrote {group_map_path}")
    print(f"[Gate 1] DISTINCT-PAIN-CAT DENOMINATOR = {denom} "
          f"(report this with every rate; augmented copies never counted)")


if __name__ == "__main__":
    main()
