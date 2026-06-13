"""HEADLINE METHOD #2 — confound-attribution PROTOCOL (IMPLEMENTATION_PLAN §6.5, portable).

ONE-DIRECTIONAL BY CONSTRUCTION (THE LAW): well-powered to DETECT confounding,
underpowered to RULE IT OUT. Every output string says "no confound detected at this
power" — NEVER "ruled out", never "guaranteed". The deliverable is the reusable
PROTOCOL, not the claim "CAT_01 is confounded".

Three probes (all one-directional):
  - bg_gap()  : FGS-BG-Gap background counterfactual. Mean |score shift| on a background
                swap over the 0-10 sum, PLUS the pain-flip rate (fraction whose 0.39
                binary decision flips on bg swap), per AU. A large gap on a specific AU =
                that head is reading the cage/clinic, not the cat.
  - ebpg()    : per-AU energy-based pointing game. energy_in_ROI / energy_whole for a
                per-AU saliency map vs a CatFLW-landmark AU ROI mask. Answers "does the
                ear head fire on the ear?" — per-AU localization faithfulness. If landmark
                coverage is partial, EBPG is reported only on the covered subset (state
                the denominator).
  - judge_bias() : JUDGE-side confound axis for the VLM-as-AU-rater. Mean |0/1/2 shift| +
                AU-flip rate when the rater's OWN prompt is perturbed in content-preserving
                ways (position-swap of the AU order, verbosity inflation, self-enhancement
                preamble). Covers the LLM-as-judge biases that bg_gap/ebpg (image-side
                confounds) do not. Compares the rater to itself, so it needs NO vet anchor
                and is NOT gated by the pending per-AU vet anchor.
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


# Named, content-preserving perturbations of the VLM-as-AU-rater prompt (a DiffuJudge-
# style named-perturbation template). The probe CONSUMES scores from a baseline prompt
# and from each named perturbed rerun — it never calls the model itself, so it stays a
# pure, unit-testable attribution function like bg_gap().
JUDGE_BIAS_PERTURBATIONS = (
    "position",          # permute the AU enumeration order (ear,orbital,muzzle,whiskers,head)
    "verbosity",         # verbosity-inflated rubric text — longer descriptors, same meaning
    "self_enhancement",  # self-referential "you are an expert" preamble before scoring
)


def judge_bias_shift(score_baseline, score_perturbed) -> dict:
    """One judge-bias cell: mean |0/1/2 score shift| + AU-flip rate under one perturbation.

    score_baseline, score_perturbed: [N] per-AU ordinal scores from the baseline prompt
    and from ONE named perturbation of the VLM-as-AU-rater prompt. A large shift means the
    rater's score depends on prompt framing, not the cat — a JUDGE-side confound DETECTED
    here. ONE-DIRECTIONAL: a small shift is only NO_CONFOUND_MSG, never "no bias".
    """
    sb = np.asarray(score_baseline, dtype=float)
    sp = np.asarray(score_perturbed, dtype=float)
    return {
        "shift": float(np.abs(sp - sb).mean()),
        "flip_rate": float((sb != sp).mean()),
        "n": int(len(sb)),
        # ONE-DIRECTIONAL: a small shift does NOT rule out a judge bias, it only fails to
        # detect one at this power.
        "note": NO_CONFOUND_MSG,
    }


def judge_bias(baseline_by_au: dict, perturbed_by_au: dict) -> dict:
    """Judge-bias confound axis over the VLM-as-AU-rater (extends the confound protocol).

    Probes whether the rater's per-AU 0/1/2 scores move under named, content-preserving
    perturbations of its OWN prompt — the LLM-as-judge biases (position / verbosity /
    self-enhancement) that the image-side bg_gap/ebpg probes do not cover. Compares the
    rater to itself, so it needs NO vet anchor and is NOT gated by the pending per-AU vet
    anchor (distinct from the kappa-validation question).

    baseline_by_au[au]:        [N] scores from the baseline prompt.
    perturbed_by_au[pert][au]: [N] scores from named perturbation `pert` (one of
                               JUDGE_BIAS_PERTURBATIONS), aligned row-for-row to baseline.

    Returns {"per_perturbation": {pert: {au: judge_bias_shift(...)}}, "note"}.
    ONE-DIRECTIONAL (NO_CONFOUND_MSG): a small shift fails to detect a judge confound at
    this power; it does not rule one out.
    """
    table = {
        pert: {
            au: judge_bias_shift(baseline_by_au[au], pert_scores[au])
            for au in baseline_by_au
            if au in pert_scores
        }
        for pert, pert_scores in perturbed_by_au.items()
    }
    return {"per_perturbation": table, "note": NO_CONFOUND_MSG}
