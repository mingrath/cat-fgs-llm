#!/usr/bin/env python3
"""Gate 1-B — per-AU VLM-vs-vet quadratic kappa CHECK, GATED ON THE CI LOWER BOUND (§4.6.3).

GUARDED, INSPECTED-NOT-VALIDATED reliability check (result pending an independent vet anchor) —
NOT the paper's headline. The headline / spine is the power-conditioned per-AU confound-attribution
protocol (see scripts/gate2_confound.py); this kappa sits BELOW it as a supporting reliability
check and kill-tree insurance (it becomes the sole-survivor headline only if the confound leg
degrades). The CI-lower-bound gate is textbook clinimetrics (Tractenberg 2010; Donner & Rotondi
2010; Sim & Wright 2005) and the rubric-independence guard is published (Weng et al. 2026) — neither
is branded as a novel increment. Reads per-AU kappa floors from the Gate-0 power manifest
(data/manifests/power.json -> "au_kappa_floors"), falling back to the FINAL_DIRECTION defaults
(orbital/ear/head 0.60, muzzle/whiskers 0.40) ONLY when Gate 0 has not yet committed them. Writes a
JSON report measuring VLM-vs-vet agreement; interpretation guard (anchor independence + rubric
divergence) must be stated in the report.

GATE ON THE LOWER BOUND, NOT THE POINT (§4.6.3 / §6.1): at the pilot n a 0.60 floor is
statistically indistinguishable from a true 0.47, so a point-estimate gate is not a gate.
graded_go requires orbital/ear/head LB >= floor. muzzle/whiskers below their floor are
flagged drop_head, not a hard fail.

CIRCULARITY FIREWALL: this kappa is VLM-vs-VET agreement — a guarded labeler-quality
check (result pending the independent vet anchor). It is NEVER validation of the
0.39 flag (QWK-vs-VLM is never validation).

Only VET-CONFIRMED rows enter the kappa. Augmented copies never enter the pilot N.
Input merged CSV has one row PER IMAGE with columns {au}_vlm, {au}_vet (+ optional cat_id).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.kappa import AU_NAMES, bootstrap_qwk_lb  # noqa: E402
POWER_JSON = ROOT / "data" / "manifests" / "power.json"

# FreshPowerG0ManifestsEnforcer: hard G0-first at load; manifests single source.
if not POWER_JSON.exists():
    raise SystemExit(
        "G0 power/vet-budget must precede; see data/manifests/power.json committed from gate0_power"
    )
_pj = json.loads(POWER_JSON.read_text())
if int(_pj.get("vet_budget_integer", 0)) < 50:
    raise SystemExit(
        "G0 power/vet-budget must precede; see data/manifests/power.json committed from gate0_power"
    )

# FINAL_DIRECTION / BUILD_PLAN defaults — placeholders ONLY; Gate 0 overwrites them.
DEFAULT_FLOORS = {
    "orbital": 0.60,
    "ear": 0.60,
    "head": 0.60,  # gate on LB >= 0.60
    "muzzle": 0.40,
    "whiskers": 0.40,  # 0.40-0.60 = acceptable-with-caveat
}
# AUs whose LB failing their floor PIVOTS the whole 0-10 layer to the binary spine.
CORE_AUS = ["orbital", "ear", "head"]
CAVEAT_AUS = ["muzzle", "whiskers"]


def load_floors(power_json: Path = POWER_JSON) -> dict:
    """Per-AU kappa floors from Gate-0 (data/manifests/power.json), else the defaults.

    TODO(Gate 0): writes "au_kappa_floors" + pilot n into power.json; until then the
    DEFAULT_FLOORS placeholders are used and the report flags floors_source="default".
    """
    try:
        return json.loads(Path(power_json).read_text())["au_kappa_floors"], "power.json"
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return DEFAULT_FLOORS, "default"


def run_gate(merged_csv: str, out_json: str, n_boot: int = 5000, alpha: float = 0.05) -> dict:
    """Compute per-AU kappa + one-sided 95% CI lower bound; gate on the LOWER BOUND."""
    df = pd.read_csv(merged_csv)  # one row/image: {au}_vlm, {au}_vet
    floors, floors_source = load_floors()

    report, decision = {}, {}
    noise_rates = {}
    # Cluster the gate-feeding bootstrap by cat_id so the one-sided 95% LB respects
    # within-cat correlation (pilot positives concentrate in few cats, e.g. CAT_01);
    # image-i.i.d. resampling would inflate the LB and make the gate too easy (THE LAW).
    groups = df["cat_id"].to_numpy() if "cat_id" in df.columns else None
    for au in AU_NAMES:
        y_vlm = df[f"{au}_vlm"].to_numpy()
        y_vet = df[f"{au}_vet"].to_numpy()
        k, lb = bootstrap_qwk_lb(y_vlm, y_vet, n_boot=n_boot, alpha=alpha, groups=groups)
        floor = floors.get(au, DEFAULT_FLOORS[au])
        # nan lb = every bootstrap resample was degenerate (near-constant AU) — a
        # power/degeneracy outcome, NOT a VLM-quality FAIL; surfaced separately.
        degenerate = bool(np.isnan(lb))
        n_cats = None
        if "cat_id" in df.columns:
            pain = (y_vlm >= 1) | (y_vet >= 1)
            n_cats = int(df.loc[pain, "cat_id"].nunique())
        # per-AU VLM mislabel fraction on the vet-confirmed pilot: the co-teaching
        # tau input (train_corn reads est_noise_rate.overall from this report).
        noise_rates[au] = float(np.mean(y_vlm != y_vet))
        report[au] = {
            "kappa": k,
            "ci_lb": lb,  # one-sided 95% LOWER BOUND
            "floor": floor,
            "pass": (not degenerate) and lb >= floor,  # GATE ON THE LOWER BOUND, not the point
            "degenerate": degenerate,
            "n": int(len(df)),
            "distinct_pain_cats": n_cats,  # anti-benchmark denominator (§4.8)
        }

    # graded_go: orbital/ear/head LB >= floor. (Internal inspection gate — passing it does
    # NOT make the 0-10 layer validated; the layer ships inspected-not-validated regardless.)
    decision["graded_go"] = all(report[au]["pass"] for au in CORE_AUS)
    decision["muzzle_whiskers"] = {
        au: ("ok" if report[au]["ci_lb"] >= report[au]["floor"] else "drop_head")
        for au in CAVEAT_AUS
    }
    decision["on_core_fail"] = (
        "PIVOT to binary spine (drop the 0-10 layer); confound-attribution headline is "
        "unaffected, and this kappa check remains kill-tree insurance"
        if not decision["graded_go"]
        else None
    )

    out = {
        "gate": "1B",
        "floors_source": floors_source,
        "n_boot": n_boot,
        "alpha": alpha,
        "per_au": report,
        "est_noise_rate": {  # tau for co-teaching: VLM-vs-vet disagreement on the pilot
            "per_au": noise_rates,
            "overall": float(np.mean(list(noise_rates.values()))),
        },
        "decision": decision,
        "interpretation_guard": {
            "note": "A high kappa only measures capability if BOTH: (i) the vet anchor is independent, AND (ii) the VLM rubric differs from the vet rubric. Otherwise it measures rubric-following. See README section 1 Interpretation Guard.",
            "anchor_independent": None,  # Document at runtime if anchor is independent (e.g., true/false/unknown)
            "rubric_match": None,  # Document at runtime if VLM rubric matches vet rubric (e.g., true/false/unknown)
        },
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2))

    # Anti-benchmark console: print the lower bound + denominator, never a bare point kappa.
    for au in AU_NAMES:
        r = report[au]
        cats = "" if r["distinct_pain_cats"] is None else f" | distinct-pain-CATs={r['distinct_pain_cats']}"
        verdict = "DEGENERATE (near-constant AU; power problem, not a quality FAIL)" \
            if r["degenerate"] else ("PASS" if r["pass"] else "FAIL")
        print(
            f"{au:>9}: kappa={r['kappa']:.3f}  LB(95%)={r['ci_lb']:.3f}  "
            f"floor={r['floor']:.2f}  {verdict}{cats}"
        )
    print(f"graded_go (orbital/ear/head LB>=floor): {decision['graded_go']}")
    if decision["on_core_fail"]:
        print(decision["on_core_fail"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged-csv", required=True, help="per-image {au}_vlm/{au}_vet CSV (vet-confirmed)")
    ap.add_argument("--out-json", default=str(ROOT / "artifacts" / "gate1b" / "kappa_report.json"))
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    # FreshFullGateWiringManifestsEnforcer cand1: G0 hard in gate1b (before compute/load; audit had fallback only). Require committed power, no default.
    power = ROOT / "data" / "manifests" / "power.json"
    if not power.exists() or not power.is_file():
        raise SystemExit("G0 must precede; committed manifests required: data/manifests/power.json missing. Run gate0_power.py (floors committed; no default fallback for full wiring).")
    run_gate(args.merged_csv, args.out_json, n_boot=args.n_boot, alpha=args.alpha)


if __name__ == "__main__":
    main()
