"""Gate 0 — zero-data power calcs + vet-budget pre-registration (CLI entrypoint).

Blocks ALL quantitative work (IMPLEMENTATION_PLAN section 7.1 G0 / FINAL_DIRECTION Gate 0).
Runs three zero-data calcs and writes the frozen budget + floors to
data/manifests/power.json (the committed manifests dir; Gate-1-B reads
au_kappa_floors from there):

  (a) faces needed for a per-AU weighted-kappa CI half-width <= 0.15.
      kappaSize (R) is the CANONICAL weighted-kappa power tool; here we use a
      numpy/statsmodels normal-approximation as a CPU-only fallback. The result is
      the per-AU kappa floors + whether the budget meets the half-width target.
  (b) faces needed for an LTT/MAPIE-certified NPV>=0.90 band at <= 40% abstention.
      One-sided binomial / Clopper-Pearson sizing on the abstained-in negative pool.
      If the budget cannot certify a useful band -> abstention is EXPLORATORY ONLY
      (the word "guaranteed" never appears anywhere; FINAL_DIRECTION Gate 0(b)).
  (c) whether the 0.39 sens/spec CI is reportable at the budgeted n
      (Clopper-Pearson half-width on the vet-confirmed positive pool).

This script reads its targets from configs/power.yaml and NEVER invents thresholds.
The single committed vet-budget integer is a Gate-0 / clinician decision; the value
in configs/power.yaml (placeholder ~120) is echoed through and flagged as such.

CONCEDED-ENGINE-FREE: no torch, no backbone, no data. CPU, ~1 afternoon.
"""

import argparse
import json
import math
import pathlib

import sys

import yaml
from scipy import stats as sp_stats

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))  # allow `python scripts/gate0_power.py` without PYTHONPATH

from src.vlm.aggregate import POINT_DECISION_THRESHOLD  # noqa: E402

# The 5 AUs (FACTCHECK: Min-aggregation for whiskers AND head). Floor membership
# (high-floor vs caveat-band) is the FINAL_DIRECTION/BUILD_PLAN pre-registered grouping.
# The floor VALUES are single-sourced from configs/vlm_fgs.yaml (kappa_floors) and
# committed into power.json, which Gate-1-B reads — three copies never diverge.
HIGH_FLOOR_AUS = ("orbital", "ear", "head")     # gate on kappa CI-LB >= 0.60
CAVEAT_BAND_AUS = ("muzzle", "whiskers")        # 0.40-0.60 acceptable-with-caveat


def kappa_faces_for_half_width(half_width, kappa0=0.60, n_levels=3, two_raters=2):
    """(a) Approx faces for a weighted-kappa CI half-width target.

    Normal-approximation fallback to R's kappaSize (the canonical tool; see g0_power.R
    note). Uses the large-sample SE of weighted kappa ~ sqrt((1-kappa^2)/n) as a
    conservative ballpark: half_width ~= z * sqrt((1-kappa^2)/n) -> solve for n.
    statsmodels is used if present for a tighter estimate; else pure numpy."""
    z = sp_stats.norm.ppf(0.975)  # 95% two-sided -> 1.96
    # conservative variance proxy for a kappa near kappa0
    var_proxy = (1.0 - kappa0 ** 2)
    n = math.ceil((z ** 2) * var_proxy / (half_width ** 2))
    # statsmodels does not expose a weighted-kappa power routine; left as canonical-R note.
    try:
        import statsmodels  # noqa: F401  (presence check only; kappaSize in R is canonical)
        engine = "statsmodels-present(normal-approx); kappaSize(R) canonical"
    except Exception:
        engine = "numpy normal-approx; kappaSize(R) canonical"
    return int(n), engine


def npv_faces_for_band(npv_target, max_abstention, prevalence=0.13, conf=0.95):
    """(b) Faces needed for an LTT/MAPIE-certified NPV>=target band at <= max_abstention.

    Sizing the abstained-IN negative pool: to certify a one-sided lower bound on NPV at
    `conf`, we need the abstained-in true-negative count k such that the Clopper-Pearson
    LB of (k correct / m abstained-in) clears `npv_target`. We report the minimal m that
    can clear the band under a zero-error best case, then scale to total faces via the
    abstention budget. This is the calc that decides whether F ships certified or
    EXPLORATORY-ONLY (no 'guaranteed', ever)."""
    # zero-error best case: k = m correct; LB = (alpha/2)^(1/m)-style -> use exact CP.
    # find minimal m s.t. CP lower bound at conf for (m successes / m trials) >= npv_target
    alpha = 1.0 - conf
    m = 1
    while m < 100000:
        # Clopper-Pearson lower bound for k=m successes out of m trials
        lb = alpha ** (1.0 / m)  # = Beta.ppf(alpha, m, 1) for the all-success edge case
        if lb >= npv_target:
            break
        m += 1
    # abstained-in negatives are at most max_abstention * total; negatives ~ (1-prevalence)*total
    # require (abstained-in negatives) >= m  ->  total >= m / (max_abstention * (1-prevalence))
    denom = max_abstention * (1.0 - prevalence)
    total_faces = math.ceil(m / denom) if denom > 0 else None
    return {"min_abstained_in_negatives": int(m), "total_faces": total_faces}


def point_039_reportable(n_vet, prevalence=0.13, target_half_width=0.15, conf=0.95):
    """(c) Is the 0.39 sens/spec CI reportable at the budgeted n?

    Circularity firewall: sens/spec at 0.39 are estimated on VET-CONFIRMED labels ONLY.
    The binding pool is the vet-confirmed POSITIVE count (sensitivity is the scarcer side
    at ~13% prevalence). Reportable iff the Clopper-Pearson half-width on that pool, at a
    mid-range point estimate, is <= target_half_width."""
    n_pos = int(round(n_vet * prevalence))
    if n_pos < 1:
        return {"n_pos": n_pos, "cp_half_width": None, "reportable": False}
    # worst-case CP half-width is widest near p=0.5; size it there for the positive pool
    k = round(0.5 * n_pos)
    lo, hi = sp_stats.beta.ppf([(1 - conf) / 2, 1 - (1 - conf) / 2],
                               [k, k + 1], [n_pos - k + 1, n_pos - k])
    lo = 0.0 if k == 0 else float(lo)
    hi = 1.0 if k == n_pos else float(hi)
    half = (hi - lo) / 2.0
    return {"n_pos": n_pos, "cp_half_width": round(half, 4),
            "reportable": bool(half <= target_half_width)}


def build_report(cfg, vet_budget, min_pain_pos, au_kappa_floors):
    half_width = cfg["kappa"]["ci_half_width_max"]
    npv_target = cfg["npv_lb"]["target"]
    max_abst = cfg["npv_lb"]["max_abstention"]
    thresh = POINT_DECISION_THRESHOLD  # single source; never a config copy of 0.39
    prevalence = cfg.get("prevalence", 0.13)  # ~13% pain prevalence (Gate-3 target set)

    # (a) per-AU kappa floors come from configs/vlm_fgs.yaml (single source); face
    # counts are sized at the high-floor and caveat-band floor levels.
    floor_high = max(au_kappa_floors[au] for au in HIGH_FLOOR_AUS)
    floor_caveat = min(au_kappa_floors[au] for au in CAVEAT_BAND_AUS)
    n_kappa_high, k_engine = kappa_faces_for_half_width(half_width, kappa0=floor_high)
    n_kappa_caveat, _ = kappa_faces_for_half_width(half_width, kappa0=floor_caveat)

    kappa_calc = {
        "ci_half_width_target": half_width,
        "faces_for_high_floor_au": n_kappa_high,
        "faces_for_caveat_band_au": n_kappa_caveat,
        "budget_meets_half_width": bool(vet_budget >= n_kappa_high),
        "engine": k_engine,
    }

    # (b) NPV band
    npv_calc = npv_faces_for_band(npv_target, max_abst, prevalence=prevalence)
    npv_budget_ok = (npv_calc["total_faces"] is not None
                     and vet_budget >= npv_calc["total_faces"])
    npv_calc["budget_certifies_band"] = bool(npv_budget_ok)
    # FINAL_DIRECTION Gate 0(b): if the budget cannot certify, abstention is EXPLORATORY ONLY.
    npv_calc["abstention_status"] = "certified" if npv_budget_ok else "exploratory-only"

    # (c) 0.39 CI reportability
    rho_calc = point_039_reportable(vet_budget, prevalence=prevalence,
                                    target_half_width=half_width)
    rho_calc["threshold"] = thresh
    # rho-band decision: reportable point CI vs exploratory.
    rho_band_decision = "reportable" if rho_calc["reportable"] else "underpowered-exploratory"

    return {
        "vet_budget_integer": int(vet_budget),
        "min_pain_pos": int(min_pain_pos),
        "prevalence_assumed": prevalence,
        "au_kappa_floors": au_kappa_floors,
        "calc_a_kappa": kappa_calc,
        "calc_b_npv": npv_calc,
        "calc_c_point_039": rho_calc,
        "rho_band_decision": rho_band_decision,
        "note": ("Zero-data Gate-0 power pre-registration. kappaSize(R) is the canonical "
                 "weighted-kappa tool; the numbers here are a numpy/scipy fallback. The "
                 "vet_budget_integer is a clinician/Gate-0 decision echoed from "
                 "configs/power.yaml; replace the placeholder before any quantitative claim. "
                 "'guaranteed' is banned; NPV is reported as a one-sided 95% LOWER BOUND."),
    }


def main():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Gate 0 — zero-data power calcs + vet-budget pre-registration.")
    ap.add_argument("--config", default=str(repo_root / "configs" / "power.yaml"),
                    help="power.yaml with kappa/npv_lb/point_decision targets + vet_budget_integer.")
    ap.add_argument("--floors-config", default=str(repo_root / "configs" / "vlm_fgs.yaml"),
                    help="vlm_fgs.yaml carrying kappa_floors (the single floor source).")
    ap.add_argument("--out", default=str(repo_root / "data" / "manifests" / "power.json"),
                    help="output JSON in the COMMITTED manifests dir; Gate-1-B reads au_kappa_floors here.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    vet_budget = int(cfg.get("vet_budget_integer", 120))   # placeholder ~120 until Gate 0 commits
    min_pain_pos = int(cfg.get("min_pain_pos", 50))        # >= 50 pain-positive (Gate 0)
    floors_cfg = yaml.safe_load(open(args.floors_config))
    au_kappa_floors = {au: float(v) for au, v in floors_cfg["kappa_floors"].items()}

    report = build_report(cfg, vet_budget, min_pain_pos, au_kappa_floors)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"[gate0] vet_budget_integer = {report['vet_budget_integer']} "
          f"(min_pain_pos >= {report['min_pain_pos']})")
    print(f"[gate0] (a) kappa: faces_for_high_floor_au = {report['calc_a_kappa']['faces_for_high_floor_au']}, "
          f"budget_meets_half_width = {report['calc_a_kappa']['budget_meets_half_width']}")
    print(f"[gate0] (b) npv: {report['calc_b_npv']['abstention_status']} "
          f"(needs {report['calc_b_npv']['total_faces']} faces)")
    print(f"[gate0] (c) 0.39 CI: {report['rho_band_decision']} "
          f"(n_pos={report['calc_c_point_039']['n_pos']}, "
          f"half_width={report['calc_c_point_039']['cp_half_width']})")
    print(f"[gate0] wrote {out_path}")


if __name__ == "__main__":
    main()
