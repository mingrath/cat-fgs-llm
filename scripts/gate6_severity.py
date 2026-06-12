"""Gate 6 — severity-cell collapse (after the anchor; pure counting).

IMPLEMENTATION_PLAN section 7.1 G6 / FINAL_DIRECTION Gate 6. Counts vet-confirmed
AU=2 (severe) cells per AU. If ANY single AU's =2 count is single-digit (< 10), the
high-end of the ordinal scale is structurally underpowered and we COLLAPSE it (merge
AU levels 1+2, i.e. report only painful / not-above-threshold). This is a
PRE-COMMITTED STRUCTURAL DECISION, not a caveat — it is expected to fire at the pain
base rate.

Circularity firewall: counts come from VET-CONFIRMED labels ONLY ({au}_vet columns).
This script deliberately does NOT print a per-cell severe sens/calibration number — at
single-digit n the CI spans roughly [0.35, 0.97], so a point number would mislead. It
emits the collapse DECISION (which AUs collapse) and the per-AU severe COUNT only as a
power signal, never as a performance metric.
"""

import argparse
import json
import pathlib
import sys

import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))  # allow `python scripts/gate6_severity.py` without PYTHONPATH

from src.constants import AU_ORDER  # noqa: E402

# The 5 AUs (FACTCHECK: Min-aggregation for whiskers AND head).
AU_NAMES = AU_ORDER
SINGLE_DIGIT = 10   # < 10 vet-confirmed AU=2 cells => collapse the high end


def count_severe_cells(df):
    """Per-AU count of vet-confirmed AU=2 cells (severe). VET labels only."""
    counts = {}
    for au in AU_NAMES:
        col = f"{au}_vet"
        if col not in df.columns:
            counts[au] = None   # AU not vet-scored in this pilot; flagged below
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        counts[au] = int((vals == 2).sum())
    return counts


def collapse_decision(counts):
    """Per-AU collapse decision. Any single-digit severe count => collapse high end."""
    decision = {}
    fired = False
    for au, c in counts.items():
        if c is None:
            decision[au] = "no-vet-column"
            continue
        if c < SINGLE_DIGIT:
            decision[au] = "collapse-high-end"   # merge AU levels 1+2
            fired = True
        else:
            decision[au] = "keep-0-1-2"
    return decision, fired


def main():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Gate 6 — severity-cell collapse (pure counting; vet labels only).")
    ap.add_argument("--vet-csv", required=True,
                    help="CSV with one row per vet-confirmed image and {au}_vet columns in {0,1,2}.")
    ap.add_argument("--out", default=str(repo_root / "data" / "manifests" / "severity.json"),
                    help="output JSON in the COMMITTED manifests dir (same home as power.json).")
    args = ap.parse_args()

    df = pd.read_csv(args.vet_csv)
    counts = count_severe_cells(df)
    decision, fired = collapse_decision(counts)

    report = {
        "single_digit_threshold": SINGLE_DIGIT,
        "severe_cell_counts": counts,        # power signal only, NOT a performance metric
        "collapse_decision": decision,
        "gate6_fired": fired,
        "scale_after_collapse": ("painful / not-above-threshold (AU 1+2 merged)"
                                 if fired else "0/1/2 retained on all AUs"),
        "note": ("Pre-committed STRUCTURAL decision (not a caveat). Counts are from "
                 "VET-CONFIRMED labels only. No per-cell severe sens/calibration number "
                 "is emitted: at single-digit n the CI spans ~[0.35,0.97]."),
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"[gate6] gate6_fired = {fired}")
    for au in AU_NAMES:
        # print the DECISION per AU; the count is the power signal, never a metric.
        print(f"[gate6] {au:>9}: {decision[au]}")
    print(f"[gate6] scale_after_collapse = {report['scale_after_collapse']}")
    print(f"[gate6] wrote {out_path}")


if __name__ == "__main__":
    main()
