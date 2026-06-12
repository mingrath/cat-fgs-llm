"""HEADLINE METHOD #2 — confound-attribution PROTOCOL (IMPLEMENTATION_PLAN §6.5, portable).

ONE-DIRECTIONAL BY CONSTRUCTION (THE LAW): well-powered to DETECT confounding,
underpowered to RULE IT OUT. Every output string says "no confound detected at this
power" — NEVER "ruled out", never "guaranteed". The deliverable is the reusable
PROTOCOL, not the claim "CAT_01 is confounded".

Two probes:
  - bg_gap()  : FGS-BG-Gap background counterfactual. Mean |score shift| on a background
                swap over the 0-10 sum, PLUS the pain-flip rate (fraction whose 0.39
                binary decision flips on bg swap), per AU. A large gap on a specific AU =
                that head is reading the cage/clinic, not the cat.
  - ebpg()    : per-AU energy-based pointing game. energy_in_ROI / energy_whole for a
                per-AU saliency map vs a CatFLW-landmark AU ROI mask. Answers "does the
                ear head fire on the ear?" — per-AU localization faithfulness. If landmark
                coverage is partial, EBPG is reported only on the covered subset (state
                the denominator).
"""

from __future__ import annotations

import numpy as np

# Standard message reused everywhere so no caller can phrase a one-directional result
# as a two-directional one. ("ruled out" / "guaranteed" are BANNED.)
NO_CONFOUND_MSG = "no confound detected at this power"


def ebpg(sal_map, roi_mask) -> float:
    """Energy-based pointing game: fraction of saliency energy inside the ROI (§6.5(b)).

    sal_map, roi_mask: both HxW, sal_map >= 0 (saliency / attention-rollout / Score-CAM),
    roi_mask is 0/1 over the AU box from the CatFLW 48-landmark map. Returns
    energy_in_ROI / energy_whole in [0,1]. Higher = the head localizes onto its AU.
    """
    sal = np.asarray(sal_map, dtype=float)
    roi = np.asarray(roi_mask, dtype=float)
    return float((sal * roi).sum() / (sal.sum() + 1e-8))


def bg_gap(
    score_orig: np.ndarray,
    score_swapped: np.ndarray,
    decision_orig=None,
    decision_swapped=None,
) -> dict:
    """FGS-BG-Gap: mean |0-10 sum shift| on a background swap, + pain-flip rate (§6.5(a)).

    score_orig, score_swapped: [N] 0-10 sum scores for the original and background-swapped
        composites (mixed_rand / only_bg_t style). Pass per-AU arrays to attribute the gap
        to a specific head (call once per AU).
    decision_orig, decision_swapped: optional [N] boolean 0.39 pain decisions; the pain-flip
        rate is the fraction whose binary decision flips on the bg swap.

    Returns dict(bg_gap, pain_flip_rate, n, note). `note` is the one-directional string —
    a large gap means "confound DETECTED here", a small gap means only NO_CONFOUND_MSG,
    never "no confound".
    """
    so = np.asarray(score_orig, dtype=float)
    ss = np.asarray(score_swapped, dtype=float)
    gap = float(np.abs(ss - so).mean())

    pain_flip = None
    if decision_orig is not None and decision_swapped is not None:
        do = np.asarray(decision_orig, dtype=bool)
        ds = np.asarray(decision_swapped, dtype=bool)
        pain_flip = float((do != ds).mean())

    return {
        "bg_gap": gap,
        "pain_flip_rate": pain_flip,
        "n": int(len(so)),
        # ONE-DIRECTIONAL: a small gap does NOT rule out a confound, it only fails to
        # detect one at this power.
        "note": NO_CONFOUND_MSG,
    }


def bg_gap_per_au(
    score_orig_by_au: dict,
    score_swapped_by_au: dict,
    decision_orig=None,
    decision_swapped=None,
) -> dict:
    """Run bg_gap() per AU and tag the whole table with the one-directional note.

    score_orig_by_au[au] / score_swapped_by_au[au]: [N] sum-score arrays per AU. Returns
    {au: bg_gap(...)} plus a top-level "note" = NO_CONFOUND_MSG so the protocol's
    one-directional framing is attached to the attribution table, not just per cell.
    """
    table = {
        au: bg_gap(
            score_orig_by_au[au],
            score_swapped_by_au[au],
            decision_orig,
            decision_swapped,
        )
        for au in score_orig_by_au
    }
    return {"per_au": table, "note": NO_CONFOUND_MSG}
