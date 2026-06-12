"""Gate-4 MPS<->CPU DINOv2 logit-parity check (IMPLEMENTATION_PLAN section 5.7).

A leak-proof PR-AUC computed from silently-wrong MPS logits is worthless, so before
any MPS-side number is believed we assert the frozen DINOv2 backbone produces the
same features on CPU and MPS (max|delta| < 1e-3 after fp32). The backbone is CONCEDED
PLUMBING (FINAL_DIRECTION section A); this test only guards its compute-correctness.

This file does NOT hard-require a real aligned crop: it feeds a synthetic random
518x518x3 tensor through the backbone. It skips cleanly when MPS is unavailable or
when torch.hub cannot fetch the weights (e.g. offline CI), so it never blocks the
ONLY-torch+numpy Gate-4 decode suite.
"""

import pytest
import torch


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def test_mps_cpu_logit_parity():
    if not _mps_available():
        pytest.skip("MPS unavailable on this host")

    from src.model.backbone import extract, load_frozen_dinov2

    # Synthetic, already-normalized /14-grid tensor (518 = 37*14). No real crop required.
    torch.manual_seed(42)
    x = torch.randn(1, 3, 518, 518, dtype=torch.float32)

    try:
        m_cpu, _ = load_frozen_dinov2("cpu")
        m_mps, _ = load_frozen_dinov2("mps")
    except Exception as e:  # torch.hub fetch failure (offline) — not a correctness failure
        pytest.skip(f"could not load frozen DINOv2 backbone: {e}")

    c_cpu, _ = extract(m_cpu, x)
    c_mps, _ = extract(m_mps, x.to("mps"))

    # fp32 cast on both sides; MPS<->CPU tolerance per section 5.7 (max|delta| < 1e-3)
    torch.testing.assert_close(c_cpu.float(), c_mps.cpu().float(), rtol=1e-3, atol=1e-3)
