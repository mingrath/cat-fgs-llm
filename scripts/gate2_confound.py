#!/usr/bin/env python3
"""Gate 2 — capture-condition confound audit (one-directional, §1.6 Gate 2 / §6.5).

HEADLINE METHOD #2 (portable PROTOCOL). Train a TRIVIAL classifier on
brightness / blur / box-aspect / CLIP-embedding features to predict pain. If it beats
chance (cat-grouped CV ROC-AUC CI lower bound > 0.5), pain is entangled with acquisition
context. ONE-DIRECTIONAL: well-powered to DETECT, underpowered to RULE OUT — every line
says "no confound detected at this power", NEVER "no confound" / "ruled out".

Also surfaces the transportable attribution probes (the actual deliverable, not
"CAT_01 is confounded"): FGS-BG-Gap (mean |0-10 sum shift| on bg swap + per-AU pain-flip
rate) and per-AU EBPG (energy_in_ROI / energy_whole). Those live in src.eval.confound; this
script wires the trivial probe + logs the protocol one-directionally.

CPU, no vet, no GPU — the cheapest kill/reframe; runs BEFORE any training or vet hour.
sklearn/numpy/pandas at top; open_clip imported LAZILY (only if CLIP features requested).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `python scripts/gate2_confound.py` without PYTHONPATH

from src.eval.bootstrap import bootstrap_ci  # noqa: E402
from src.eval.confound import NO_CONFOUND_MSG  # noqa: E402

# Trivial probe feature columns (cheap acquisition-context proxies). CLIP columns, if
# present, are any column prefixed "clip_". brightness/blur/aspect are scalar columns.
SCALAR_FEATURES = ["brightness", "blur", "aspect_ratio"]


def _feature_matrix(df: pd.DataFrame, use_clip: bool) -> np.ndarray:
    cols = [c for c in SCALAR_FEATURES if c in df.columns]
    blocks = [df[cols].to_numpy(dtype=float)] if cols else []
    if use_clip:
        clip_cols = [c for c in df.columns if c.startswith("clip_")]
        if clip_cols:
            blocks.append(df[clip_cols].to_numpy(dtype=float))
    if not blocks:
        raise ValueError("no trivial-probe features found (brightness/blur/aspect_ratio/clip_*)")
    return np.concatenate(blocks, axis=1)


def trivial_probe(
    df: pd.DataFrame,
    use_clip: bool = False,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Cat-grouped CV ROC-AUC of a trivial classifier predicting pain from context features.

    Groups by cat_id (NEVER by clip; §1.6) so the probe can't cheat via same-individual
    leakage. Pools out-of-fold predictions, reports ROC-AUC with a cat-grouped bootstrap CI.
    A LOWER-BOUND > 0.5 => confound DETECTED at this power; otherwise NO_CONFOUND_MSG.
    """
    X = _feature_matrix(df, use_clip)
    y = df["pain"].to_numpy(dtype=int)
    groups = df["cat_id"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, va in sgkf.split(X, y, groups=groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]

    mask = ~np.isnan(oof)
    y_m, p_m, g_m = y[mask], oof[mask], groups[mask]

    paired = np.rec.fromarrays([y_m, p_m], names="y,p")

    def auc_stat(rows) -> float:
        yy, pp = rows["y"], rows["p"]
        if len(np.unique(yy)) < 2:
            raise ValueError("single-class resample")
        return roc_auc_score(yy, pp)

    point, lo, hi = bootstrap_ci(paired, auc_stat, n_boot=5000, seed=seed, groups=g_m)
    detected = lo > 0.5  # gate on the LOWER BOUND, like every other gate in this project
    return {
        "probe": "trivial brightness/blur/aspect" + ("/CLIP" if use_clip else ""),
        "oof_roc_auc": float(point),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": int(mask.sum()),
        "distinct_pain_cats": int(pd.unique(g_m[y_m == 1]).size),
        # ONE-DIRECTIONAL verdict:
        "verdict": (
            "confound DETECTED at this power (trivial features beat chance; pain is "
            "entangled with acquisition context)"
            if detected
            else NO_CONFOUND_MSG
        ),
        "confound_detected": bool(detected),
    }


def run_gate(features_csv: str, out_json: str, use_clip: bool = False) -> dict:
    df = pd.read_csv(features_csv)  # rows = NON-augmented images; cols: pain, cat_id, features
    result = trivial_probe(df, use_clip=use_clip)

    out = {
        "gate": "2",
        "direction": "one_directional",
        "trivial_probe": result,
        # FGS-BG-Gap and per-AU EBPG (the transportable deliverable) are computed by
        # src.eval.confound.bg_gap_per_au / ebpg from the bg-swap composites + saliency maps;
        # wire them here once those artifacts exist.
        "bg_gap": None,  # TODO(runtime): src.eval.confound.bg_gap_per_au on bg-swap composites
        "ebpg": None,    # TODO(runtime): src.eval.confound.ebpg on per-AU saliency vs CatFLW ROI
        "judge_bias": None,  # TODO(runtime): src.eval.confound.judge_bias on baseline + perturbed
                             # (position/verbosity/self_enhancement) VLM-rater reruns
        "note": NO_CONFOUND_MSG,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2))

    print(
        f"trivial probe OOF ROC-AUC={result['oof_roc_auc']:.3f} "
        f"[{result['ci_lo']:.3f}, {result['ci_hi']:.3f}] "
        f"(n={result['n']}, distinct-pain-CATs={result['distinct_pain_cats']})"
    )
    print(result["verdict"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-csv", required=True,
                    help="non-augmented images; cols: pain, cat_id, brightness/blur/aspect_ratio[/clip_*]")
    ap.add_argument("--out-json", default=str(ROOT / "artifacts" / "gate2" / "confound_audit.json"))
    ap.add_argument("--use-clip", action="store_true", help="include clip_* embedding columns in the probe")
    args = ap.parse_args()
    run_gate(args.features_csv, args.out_json, use_clip=args.use_clip)


if __name__ == "__main__":
    main()
