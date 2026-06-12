"""In-code aggregation: the 0-10 sum and the 0.39 flag (IMPLEMENTATION_PLAN §4.3).

THIS is the single threshold definition reused by both the VLM path and the engine
(CORN) decode path -- the Gate-4 unit test pins ``decode -> per-AU 0-2 -> sum -> 0.39``
against this exact contract so there is ONE threshold definition in the repo.

The VLM NEVER computes any of these; it emits only the 5 atoms in {0,1,2}. The
0.39 flag is triage decision-support only -- never an autonomous analgesia trigger
-- and the sum it rides on is an INSPECTED-NOT-VALIDATED quantity.
"""

# Fixed Evangelista 5-AU order — canonical declaration in src.constants.AU_ORDER
# (a leaf module, so this import adds no schema/anthropic-stack dependency).
from src.constants import AU_ORDER

AU_NAMES = list(AU_ORDER)

# THE one 0.39 definition (Evangelista sum/10 cut, ~4/10). Engine decode
# (src.model.decode.point_sum) and the abstention band import it from here;
# never re-declare the literal elsewhere.
POINT_DECISION_THRESHOLD = 0.39


def fgs_sum(result_dict):
    """Sum the 5 per-AU scores into 0..10. ``result_dict[au]['score']`` in {0,1,2}."""
    return sum(result_dict[au]["score"] for au in AU_NAMES)


def analgesia_flag(s):
    """Triage flag: ratio = sum/10; clinical cut at >= 0.39 (~4/10).

    Works on scalars, numpy arrays, and torch tensors (s/10.0 promotes to float)."""
    return (s / 10.0) >= POINT_DECISION_THRESHOLD


def any_abstain(result_dict):
    """For routing/triage only, not a clinical output."""
    return any(result_dict[au]["abstain"] for au in AU_NAMES) or (
        result_dict["image_quality"] != "frontal_clear"
    )
