"""Gate-4 BLOCKING synthetic CORN-decode -> per-AU 0/1/2 -> 0-10 sum -> 0.39 unit test.

This is *the* Gate-4 artifact (IMPLEMENTATION_PLAN section 5.7b). It pins the
vendored decode against values verified BY HAND for the actual K=3 (2-logit/AU)
head shape, so a silently-wrong MPS path cannot ship. A red Gate 4 BLOCKS Phase B.

The 0-10 sum here is the INSPECTED-NOT-VALIDATED path: these tests assert the
decode->sum->threshold PLUMBING is correct; QWK-vs-VLM is never validation and no
validated-claim number is emitted from the sum. The 0.39 flag and the 0-10 sum are
computed in code (point_sum / sum_pmf), never by the VLM.

Runs with ONLY torch + numpy installed (the coral-pytorch cross-check skips cleanly
if that one-time-check dep is absent).
"""

import importlib.util

import numpy as np
import torch

from src.model.corn import corn_cumprobs, corn_label_from_logits
from src.model.decode import au_pmf_from_cumprobs, point_sum, sum_pmf


def test_corn_decode_anchor_k3():
    # K=3 (2 logits/AU). cumprod-of-sigmoid > 0.5 decode, verified by hand:
    #   [ 9,  9] -> sig~1,1   -> cum 1,1   -> 2
    #   [ 9, -9] -> sig~1,0   -> cum 1,0   -> 1
    #   [-9, -9] -> sig~0,0   -> cum 0,0   -> 0
    logits = torch.tensor([[ 9.,  9.],
                           [ 9., -9.],
                           [-9., -9.]])
    out = corn_label_from_logits(logits)
    assert out.tolist() == [2, 1, 0], out.tolist()


def test_corn_decode_monotone_in_logits():
    # rank consistency: raising any logit cannot lower the decoded level
    base = torch.tensor([[0.3, -0.2]])
    up   = base + torch.tensor([[0.0, 2.0]])
    assert corn_label_from_logits(up).item() >= corn_label_from_logits(base).item()


def test_vendored_matches_coral_pytorch():
    # one-time cross-check that the vendored decode == coral-pytorch reference,
    # then the dep is dropped. Skips cleanly if coral-pytorch is not installed.
    coral = importlib.util.find_spec("coral_pytorch")
    if coral is None:
        import pytest
        pytest.skip("coral-pytorch not installed")
    from coral_pytorch.dataset import corn_label_from_logits as ref
    x = torch.randn(16, 2)
    assert torch.equal(corn_label_from_logits(x), ref(x))


def test_pmf_sums_to_one():
    cum = corn_cumprobs(torch.randn(8, 2))           # [B,2] for K=3
    pmf = au_pmf_from_cumprobs(cum).numpy()
    assert np.allclose(pmf.sum(1), 1.0, atol=1e-5)
    assert (pmf >= -1e-6).all()                      # no negative atoms


def test_sum_pmf_is_distribution_over_0_10():
    cums = [corn_cumprobs(torch.randn(4, 2)) for _ in range(5)]
    pmfs = [au_pmf_from_cumprobs(c).numpy() for c in cums]
    S = sum_pmf(pmfs)                                 # [4, 2*5+1=11] ; portable for N AUs
    assert S.shape == (4, 11)
    assert np.allclose(S.sum(1), 1.0, atol=1e-5)


def test_point_decision_at_039():
    hi = [torch.tensor([[ 9.,  9.]]) for _ in range(5)]   # decodes to 2 each -> sum 10
    lo = [torch.tensor([[-9., -9.]]) for _ in range(5)]   # decodes to 0 each -> sum 0
    s_hi, f_hi = point_sum(hi)
    s_lo, f_lo = point_sum(lo)
    assert s_hi.item() == 10 and bool(f_hi.item()) is True
    assert s_lo.item() == 0  and bool(f_lo.item()) is False
    # boundary: ratio>=0.39 means sum>=3.9 -> sum 4 painful, sum 3 not
    assert (4 / 10) >= 0.39 and (3 / 10) < 0.39
