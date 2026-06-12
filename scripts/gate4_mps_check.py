"""Gate 4 — MPS compute-correctness driver (BLOCKING).

Runs the Gate-4 artifacts (IMPLEMENTATION_PLAN section 5.7) and writes a single
PASS/FAIL verdict to artifacts/gate4.txt. A red Gate 4 BLOCKS Phase B: a leak-proof
PR-AUC computed from silently-wrong MPS logits is worthless.

Three checks:
  (1) tests/test_gate4_decode.py  — synthetic CORN-decode -> per-AU 0/1/2 -> 0-10 sum
      -> 0.39, pinned against hand-verified values (runs with ONLY torch + numpy).
  (2) tests/test_mps_parity.py    — MPS<->CPU DINOv2 logit parity (skips cleanly if
      MPS unavailable / weights unfetchable).
  (3) a 1-epoch CORN smoke on SYNTHETIC cached features: cache -> CornMultiHead ->
      multi_corn_loss -> decode -> sum on synthetic 0/1/2 labels; asserts the loss is
      finite and DECREASES and that corn_label_from_logits stays in-range {0,1,2}. No
      data / no backbone forward (features are random), so it runs anywhere.

The CORN engine here is CONCEDED PLUMBING (FINAL_DIRECTION section A); this driver only
proves it computes correctly. The 0-10 sum it exercises is INSPECTED-NOT-VALIDATED.
"""

import argparse
import pathlib
import subprocess
import sys


def run_pytests(repo_root, test_files):
    """Run the Gate-4 pytest files; return (ok, captured_text)."""
    cmd = [sys.executable, "-m", "pytest", "-q", *[str(f) for f in test_files]]
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out


def corn_smoke(seed=42, n=64, in_dim=384, epochs_check=8):
    """1-epoch (multi-step) CORN smoke on synthetic features + synthetic 0/1/2 labels.

    Returns (ok, msg). ok iff: loss finite, loss decreases over the short run, and the
    hard decode stays in {0,1,2}. Mirrors the cache -> CornMultiHead -> multi_corn_loss
    -> decode path without any backbone forward (features are random)."""
    import torch

    from src.model.corn import corn_label_from_logits, multi_corn_loss
    from src.model.heads import AUS, CornMultiHead

    torch.manual_seed(seed)
    feats = torch.randn(n, in_dim)
    # synthetic genuine 0/1/2 labels for all 5 AUs (no -1 sentinel needed for the smoke)
    y = torch.randint(0, 3, (n, len(AUS)))

    head = CornMultiHead(in_dim=in_dim)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)

    losses = []
    for _ in range(epochs_check):
        opt.zero_grad()
        logits_list = head(feats)                  # 5 x [n,2]
        loss = multi_corn_loss(logits_list, y, num_classes=3)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    finite = all(map(lambda v: v == v and abs(v) != float("inf"), losses))
    decreased = losses[-1] < losses[0]

    # decode stays in range {0,1,2} per AU
    with torch.no_grad():
        logits_list = head(feats)
        decoded = [corn_label_from_logits(lg) for lg in logits_list]
    in_range = all(int(d.min()) >= 0 and int(d.max()) <= 2 for d in decoded)

    ok = finite and decreased and in_range
    msg = (f"loss[0]={losses[0]:.4f} -> loss[-1]={losses[-1]:.4f} "
           f"(finite={finite}, decreased={decreased}, decode_in_range={in_range})")
    return ok, msg


def main():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Gate 4 — MPS compute-correctness driver (BLOCKING).")
    ap.add_argument("--out", default=str(repo_root / "artifacts" / "gate4.txt"),
                    help="PASS/FAIL verdict path.")
    args = ap.parse_args()

    test_files = [
        repo_root / "tests" / "test_gate4_decode.py",
        repo_root / "tests" / "test_mps_parity.py",
    ]

    lines = ["Gate 4 — MPS compute-correctness (BLOCKING)", "=" * 44]

    tests_ok, tests_out = run_pytests(repo_root, test_files)
    lines.append(f"[1+2] pytest decode + mps-parity: {'PASS' if tests_ok else 'FAIL'}")
    lines.append(tests_out.strip())

    try:
        smoke_ok, smoke_msg = corn_smoke()
    except Exception as e:
        smoke_ok, smoke_msg = False, f"CORN smoke raised: {e!r}"
    lines.append(f"[3] 1-epoch CORN smoke (synthetic features): {'PASS' if smoke_ok else 'FAIL'}")
    lines.append(smoke_msg)

    overall = tests_ok and smoke_ok
    verdict = "PASS" if overall else "FAIL"
    lines.append("-" * 44)
    lines.append(f"GATE 4: {verdict}")
    if not overall:
        lines.append("BLOCKING: no Phase-B number is believed until Gate 4 is green.")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"[gate4] wrote {out_path}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
