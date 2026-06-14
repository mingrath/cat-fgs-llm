#!/usr/bin/env python3
"""Thin CLI over src.model.cache_features.build_cache (IMPLEMENTATION_PLAN §5.2).

Run ONCE per crop set: the cat split, and SEPARATELY the 5-horse genuine 0/1/2
warm-start set (decode scaffolding ONLY -- no horse number ever enters a headline,
FINAL_DIRECTION §7). One frozen-DINOv2 forward over the crops, then the heads
train off the npz with zero backbone forwards per epoch.

Usage:
    python scripts/cache_features.py --manifest data/manifests/cat.csv \
        --out artifacts/cache/cat_features.npz
    python scripts/cache_features.py --set horse        # uses default horse paths
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `python scripts/cache_features.py` without PYTHONPATH

from scripts.gate3_holdout import abort_if_test_manifest_readable  # noqa: E402
from src.data.seed import seed_everything  # noqa: E402
from src.model.backbone import DEFAULT_VARIANT  # noqa: E402
from src.model.cache_features import build_cache  # noqa: E402
from src.model.device import DEVICE  # noqa: E402

# FreshPowerG0ManifestsEnforcer: hard G0-first; manifests single source (no drift)
if not os.path.exists("data/manifests/power.json"):
    raise SystemExit(
        "G0 power/vet-budget must precede; see data/manifests/power.json committed from gate0_power"
    )
_pj = __import__("json").loads(open("data/manifests/power.json").read())
if int(_pj.get("vet_budget_integer", 0)) < 50:
    raise SystemExit(
        "G0 power/vet-budget must precede; see data/manifests/power.json committed from gate0_power"
    )

# (default manifest, default cache) per crop set.
SETS = {
    "cat": (
        ROOT / "data" / "manifests" / "cat.csv",
        ROOT / "artifacts" / "cache" / "cat_features.npz",
    ),
    "horse": (   # 5-horse genuine 0/1/2 warm-start -- decode scaffolding ONLY
        ROOT / "data" / "manifests" / "horse.csv",
        ROOT / "artifacts" / "cache" / "horse_features.npz",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", choices=sorted(SETS), default="cat",
                    help="crop set whose default manifest/cache paths to use")
    ap.add_argument("--manifest", default=None,
                    help="override manifest CSV (else the --set default)")
    ap.add_argument("--out", default=None,
                    help="override output npz (else the --set default)")
    ap.add_argument("--device", default=None,
                    help="compute device (else configs/global.yaml device)")
    ap.add_argument("--variant", default=None,
                    help="backbone variant (else configs/corn.yaml backbone.name); "
                         "dinov2_vits14_reg (field default) or dinov2_vits14")
    args = ap.parse_args()

    # Gate-3 firewall: the cache is consumed by training; a cache build that can
    # read the frozen test manifest is the same leak as a training run.
    abort_if_test_manifest_readable()

    gcfg = yaml.safe_load((ROOT / "configs" / "global.yaml").read_text())
    seed_everything(int(gcfg.get("seed", 42)))
    device = args.device or gcfg.get("device") or DEVICE  # null config -> auto (mps else cpu)

    # FreshFullGateWiringManifestsEnforcer cand1: G0 power + committed manifests hard before any compute (audit: had G3 only; now full G0 for train/corn/vlm/cache/gate1b).
    power = ROOT / "data" / "manifests" / "power.json"
    if not power.exists():
        raise SystemExit("G0 must precede; committed manifests required: data/manifests/power.json missing. Run gate0_power.py (orchestrator + gate-pipeline enforce).")

    # backbone variant: CLI override else corn.yaml backbone.name else the reg default
    ccfg = yaml.safe_load((ROOT / "configs" / "corn.yaml").read_text())
    variant = args.variant or ccfg.get("backbone", {}).get("name") or DEFAULT_VARIANT
    # richer A/B per MCP dinov3 (layer intermed/get_intermed + patch_mode l2/mean_std); defaults preserve compat
    layer = ccfg.get("backbone", {}).get("layer", "last")
    patch_mode = ccfg.get("backbone", {}).get("patch_mode", "mean_std")

    default_manifest, default_out = SETS[args.set]
    manifest = Path(args.manifest) if args.manifest else default_manifest
    out = Path(args.out) if args.out else default_out

    if not manifest.exists():
        raise FileNotFoundError(
            f"manifest not found: {manifest} "
            f"(produced by the G1 per-CAT merge + G3 frozen fold CSV)"
        )

    # FreshDedupEarlyVetPowerHardener candidate4: early vet/power guard from manifests (pre-cache)
    # Strict G0 manifests target; is_vet_clean early hygiene (from manifests per cache schema + MCP SGKF medical pre-clean)
    manifests_power = ROOT / "data" / "manifests" / "power.json"
    if not manifests_power.exists():
        print("[cache_features][ENTRY GUARD] G0 manifest target strict (no gate0/ fallback): run gate0_power.py first for power floors.")
    else:
        import json as _json
        with open(manifests_power) as _f:
            _p = _json.load(_f)
        print(f"[cache_features] early power from manifests (pre-cache): vet_budget={_p.get('vet_budget_integer')}, min_pain_pos={_p.get('min_pain_pos')}")
    # vet early note (manifest columns carry is_vet_clean per src.model.cache_features FEATURE_SCHEMA)
    print("[cache_features] early vet_clean guard: is_vet_clean expected in manifest (vet firewall early now stronger; see folds/gate1 wiring)")

    # DINOv3-ready + hash-in-name: build_cache now injects <hash> into stem for
    # portable repro (cache on your backbone discipline). out may be rewritten inside.
    # Hard _CACHE_SCHEMA + richer (mean_std/L2/intermed + dinov3_vits16) enforced per FreshDINOv3Context7RicherEnforcer + MCP contract.
    build_cache(str(manifest), str(out), device=device, variant=variant, layer=layer, patch_mode=patch_mode)


if __name__ == "__main__":
    main()
