"""Pin the SINGLE 0.39 definition (IMPLEMENTATION_PLAN §4.3 / Gate-4 decode contract).

The VLM weak-label path (src.vlm.aggregate.analgesia_flag) and the engine decode path
(src.model.decode.point_sum) must produce the SAME analgesia flag from the SAME 0-10
sum, because both read one threshold constant — POINT_DECISION_THRESHOLD in
src.vlm.aggregate. This test fails the moment a second, divergent 0.39 is introduced.

Runnable with only torch + numpy.
"""

import torch

from src.model.decode import point_sum
from src.vlm.aggregate import POINT_DECISION_THRESHOLD, analgesia_flag


def test_threshold_constant_is_039():
    assert POINT_DECISION_THRESHOLD == 0.39


def test_aggregate_flag_boundary():
    # 4/10 = 0.40 >= 0.39 -> flagged; 3/10 = 0.30 -> not.
    assert analgesia_flag(4) is True or bool(analgesia_flag(4))
    assert not analgesia_flag(3)


def test_engine_and_vlm_paths_agree_on_every_sum():
    # For each integer sum 0..10, the engine flag (from one-hot logits decoding to
    # that sum) must equal the VLM-aggregate flag on the same scalar sum.
    for target in range(11):
        # build 5 per-AU CORN logit pairs that hard-decode to scores summing to target
        per_au = _scores_for_sum(target)
        logits = [_corn_logits_for_score(v) for v in per_au]
        s, engine_flag = point_sum(logits)
        assert int(s.item()) == target
        assert bool(engine_flag.item()) == bool(analgesia_flag(target)), target


def _scores_for_sum(target):
    """Five AU scores in {0,1,2} summing to target (0..10)."""
    base, rem = divmod(target, 5)
    return [base + (1 if i < rem else 0) for i in range(5)]


def _corn_logits_for_score(v):
    """CORN logits [1,2] that hard-decode to ordinal v in {0,1,2}.

    corn_label_from_logits counts ranks with cumulative P(rank>k) > 0.5. Big positive
    logit -> P>0.5 (rank passed); big negative -> not. v passed ranks => first v
    logits positive, rest negative.
    """
    big = 10.0
    return torch.tensor([[big if k < v else -big for k in range(2)]], dtype=torch.float32)
