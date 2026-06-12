#!/usr/bin/env python3
"""Thin CLI over src.model.train_heads (IMPLEMENTATION_PLAN §5.6).

Runs the co-teaching CORN sweep -- pooling (cls/patch_mean) x loss (corn/coral) x
5 seeds -- off the cached features, and reports mean +/- bootstrap CI of the swept
per-seed quantity. Also trains the sibling BINARY pain head (the v1 spine) per seed.

GATE: this sweep only runs if Gate-1-B is GO (the per-AU kappa CI lower bound
cleared its floors). The verdict is verified against the Gate-1-B ARTIFACT
(artifacts/gate1b/kappa_report.json decision.graded_go) — the --gate1b-go flag
is an explicit operator acknowledgment, not the proof. tau (estimated VLM noise
rate) is read from the same report (est_noise_rate.overall) unless --tau
overrides it; there is NO config placeholder fallback.

The whole sweep runs in MINUTES on M4 because the frozen backbone never re-runs --
features come from the §5.2 npz cache.

LAW: the 0-10 CORN sum is INSPECTED-NOT-VALIDATED. The per-seed CORN train-fold
loss printed here is an inspection signal, NOT a validated claim. QWK-vs-VLM is
NEVER validation. Validated sens/spec at 0.39, calibration, decision curve, and
abstention are Phase C, fit on VET-CONFIRMED labels only (circularity firewall).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gate3_holdout import abort_if_test_manifest_readable  # noqa: E402
from src.data.seed import seed_everything  # noqa: E402
from src.model.device import DEVICE  # noqa: E402
from src.model.train_heads import (  # noqa: E402
    _persample_sum,
    load_split,
    train,
    train_binary_pain,
)


def _bootstrap_ci(vals, n_boot=10000, alpha=0.05, seed=42):
    """Percentile bootstrap CI of the mean over the (few) per-seed values."""
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(n_boot, len(vals)))].mean(1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)


def _train_fold_corn_loss(net, npz, pool, train_folds, device):
    """Inspection signal: mean per-row CORN loss of netA on the train folds.

    NOT a validated metric -- it exists only to summarize a seed for the sweep
    table. The 0-10 sum stays inspected-not-validated."""
    X, y, _, _ = load_split(npz, train_folds=tuple(train_folds), pool=pool)
    net.eval()
    with torch.no_grad():
        logits = net(X.to(device))
        per_row = _persample_sum(logits, y.to(device))
    return float(per_row.mean().item())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=str(ROOT / "artifacts" / "cache" / "cat_features.npz"))
    ap.add_argument("--config", default=str(ROOT / "configs" / "corn.yaml"))
    ap.add_argument("--global-config", default=str(ROOT / "configs" / "global.yaml"))
    ap.add_argument("--pool", choices=["cls", "patch_mean"], default=None,
                    help="single pooling override; default sweeps both")
    ap.add_argument("--loss", choices=["corn", "coral"], default=None,
                    help="single loss override; default sweeps both")
    ap.add_argument("--tau", type=float, default=None,
                    help="estimated VLM noise rate; overrides the Gate-1-B report value")
    ap.add_argument("--gate1b-go", action="store_true",
                    help="REQUIRED: acknowledge Gate-1-B GO (verified against the report artifact)")
    ap.add_argument("--gate1b-report",
                    default=str(ROOT / "artifacts" / "gate1b" / "kappa_report.json"),
                    help="Gate-1-B kappa report; decision.graded_go must be true")
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts" / "train"),
                    help="where trained heads + the sweep summary are persisted")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # Gate-3 firewall: a training run that can read the frozen test manifest is a
    # leak by construction — refuse before any head trains.
    abort_if_test_manifest_readable()

    if not args.gate1b_go:
        raise SystemExit(
            "REFUSED: Gate-1-B not GO. The CORN sweep runs only after the per-AU "
            "kappa CI LOWER BOUND clears its floors. Pass --gate1b-go once the "
            "Gate-1-B verdict is GO."
        )

    # The flag alone is an honor system; verify the Gate-1-B ARTIFACT. A failed
    # kappa pilot must structurally block the graded head (kill/pivot rule), not
    # rely on the operator reading the console correctly.
    report_path = Path(args.gate1b_report)
    if not report_path.exists():
        raise SystemExit(
            f"REFUSED: Gate-1-B report not found at {report_path}. Run "
            "scripts/gate1b_kappa_pilot.py first; --gate1b-go does not substitute "
            "for the artifact."
        )
    gate1b = json.loads(report_path.read_text())
    if not gate1b.get("decision", {}).get("graded_go", False):
        raise SystemExit(
            f"REFUSED: {report_path} records graded_go=false (core-AU kappa CI "
            "lower bound below floor). PIVOT to the binary spine; the graded head "
            "does not train on a failed pilot."
        )

    cfg = yaml.safe_load(Path(args.config).read_text())
    gcfg = yaml.safe_load(Path(args.global_config).read_text())
    seed_everything(int(gcfg.get("seed", 42)))
    device = args.device or gcfg.get("device") or DEVICE  # null config -> auto (mps else cpu)

    tcfg = cfg["train"]
    ctcfg = cfg["co_teach"]
    # tau precedence: explicit --tau, else the Gate-1-B measured noise rate.
    # NEVER a config placeholder — a fabricated tau silently drops good rows.
    tau = args.tau if args.tau is not None else (
        gate1b.get("est_noise_rate", {}).get("overall")
    )
    if tau is None:
        raise SystemExit(
            "tau (est. VLM noise rate) missing: the Gate-1-B report carries no "
            "est_noise_rate.overall (re-run gate1b_kappa_pilot.py) or pass --tau."
        )

    # default: sweep BOTH poolings and BOTH losses (§5.6 ablation); --pool/--loss pin one
    pools = [args.pool] if args.pool else ["cls", "patch_mean"]
    losses = [args.loss] if args.loss else ["corn", "coral"]
    seeds = list(tcfg.get("seeds", [0, 1, 2, 3, 4]))
    au_weights = tcfg.get("au_weights")
    train_folds = tuple(tcfg.get("train_folds", [0, 1, 2]))

    common = dict(
        device=device,
        epochs=int(tcfg.get("epochs", 80)),
        lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 1e-2)),
        batch_size=int(tcfg.get("batch_size", 32)),
        p_drop=float(tcfg.get("dropout", 0.1)),
        head_init_std=float(tcfg.get("head_init_std", 2e-5)),
        num_gradual=int(ctcfg.get("num_gradual", 10)),
        train_folds=train_folds,
        au_weights=au_weights,
        num_workers=int(tcfg.get("num_workers", 0)),
    )
    pain_pw = cfg.get("binary_pain", {}).get("pos_weight")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_corn] device={device} tau={tau} seeds={seeds} "
          f"pools={pools} losses={losses} folds={train_folds}")
    summary = {
        "npz": str(args.npz),
        "tau": float(tau),
        "tau_source": "cli" if args.tau is not None else "gate1b_report",
        "gate1b_report": str(report_path),
        "device": device,
        "train_folds": list(train_folds),
        "seeds": seeds,
        "sweeps": [],
    }
    for pool in pools:
        # the binary pain head is loss-independent: train once per (pool, seed)
        X, _, _, y_pain = load_split(args.npz, train_folds=train_folds, pool=pool)
        pain_paths = {}
        for s in seeds:
            pain_net = train_binary_pain(
                X, y_pain, device=device, epochs=common["epochs"],
                lr=common["lr"], weight_decay=common["weight_decay"],
                batch_size=common["batch_size"], p_drop=common["p_drop"],
                pos_weight=pain_pw, num_workers=common["num_workers"], seed=s,
            )
            # v1 spine: PERSIST the validated-claim head — Phase C (calibration /
            # operating point / abstention / decision curve) loads it from here.
            pain_paths[s] = out_dir / f"pain_head_{pool}_seed{s}.pt"
            torch.save(pain_net.state_dict(), pain_paths[s])
        for loss in losses:
            per_seed = []
            for s in seeds:
                netA = train(args.npz, tau=float(tau), pool=pool, seed=s,
                             loss=loss, **common)
                corn_path = out_dir / f"corn_head_{pool}_{loss}_seed{s}.pt"
                torch.save(netA.state_dict(), corn_path)
                val = _train_fold_corn_loss(netA, args.npz, pool, train_folds, device)
                per_seed.append(
                    {"seed": s, "train_corn_loss": val,
                     "corn_head": str(corn_path), "pain_head": str(pain_paths[s])}
                )
                print(f"  pool={pool} loss={loss} seed={s} "
                      f"train_corn_loss={val:.4f}  (INSPECTED, not validated)")
            mean, lo, hi = _bootstrap_ci(
                [r["train_corn_loss"] for r in per_seed], seed=int(gcfg.get("seed", 42))
            )
            summary["sweeps"].append(
                {"pool": pool, "loss": loss, "per_seed": per_seed,
                 "train_corn_loss_mean": mean, "ci95": [lo, hi]}
            )
            print(f"[sweep] pool={pool} loss={loss} "
                  f"train_corn_loss mean={mean:.4f} 95%CI=[{lo:.4f},{hi:.4f}] "
                  f"(INSPECTED-NOT-VALIDATED; QWK-vs-VLM is never validation)")

    summary_path = out_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[train_corn] heads + summary persisted -> {out_dir}")


if __name__ == "__main__":
    main()
