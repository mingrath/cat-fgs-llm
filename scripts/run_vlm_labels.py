#!/usr/bin/env python3
"""CLI: VLM 5-AU weak-labeling over a crop manifest (IMPLEMENTATION_PLAN §4.4).

Thin argparse wrapper that submits a Message Batch over a crop manifest and (by
default) polls + streams results to parquet. The bulk ~2040-image run is BLOCKED
until Gate 1-B returns GO on orbital/ear/head -- this script does not enforce that
gate; the orchestrator does.

The VLM emits ONLY the 5 AU atoms in {0,1,2}; the 0-10 sum and the >=0.39 analgesia
flag are computed in code at collect time (src.vlm.aggregate). The engine
(DINOv2+CORN) is conceded plumbing -- nothing here trains or claims it.

Requires ANTHROPIC_API_KEY in the environment (.env is gitignored). Model id and AU
order are read from configs/vlm_fgs.yaml by the library code.

Examples
--------
  # submit + wait + collect
  python scripts/run_vlm_labels.py --manifest data/pilot/manifest.csv \\
      --out data/weak_labels/labels.parquet

  # N>=3 self-consistency subset (alpha; §4.5)
  python scripts/run_vlm_labels.py --manifest data/pilot/manifest.csv \\
      --out data/weak_labels/alpha.parquet --n-runs 3

  # submit only (collect a known batch id later)
  python scripts/run_vlm_labels.py --manifest data/pilot/manifest.csv --submit-only

  # collect only, for a previously-submitted batch
  python scripts/run_vlm_labels.py --batch-id msgbatch_... \\
      --out data/weak_labels/labels.parquet --collect-only
"""

import argparse
import json
import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))  # allow `python scripts/run_vlm_labels.py` without PYTHONPATH

from src.vlm.batch_collect import wait_and_collect  # noqa: E402
from src.vlm.batch_submit import submit  # noqa: E402

# FreshPowerG0ManifestsEnforcer: hard G0-first; manifests single source (no drift)
if not os.path.exists("data/manifests/power.json"):
    raise SystemExit(
        "G0 power/vet-budget must precede; see data/manifests/power.json committed from gate0_power"
    )
_pj = json.loads(open("data/manifests/power.json").read())
if int(_pj.get("vet_budget_integer", 0)) < 50:
    raise SystemExit(
        "G0 power/vet-budget must precede; see data/manifests/power.json committed from gate0_power"
    )


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", help="crop manifest CSV (columns: image_id, path)")
    p.add_argument("--out", help="output parquet path for collected results")
    p.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="runs per image; >=3 for the self-consistency subset (temp spread to 1.0). Default 1.",
    )
    p.add_argument("--model", default=None, help="override VLM model id (default: configs/vlm_fgs.yaml).")
    p.add_argument("--batch-id", default=None, help="existing batch id (for --collect-only).")
    p.add_argument("--poll-s", type=int, default=60, help="poll interval seconds while waiting. Default 60.")
    p.add_argument("--submit-only", action="store_true", help="submit the batch and print its id; do not wait/collect.")
    p.add_argument("--collect-only", action="store_true", help="collect an existing --batch-id to --out; do not submit.")
    # P8 VLM active wire (pmf_entropy from decode + protocols): prioritize high unc for small-N G0 vet focus + G1B CI-LB tie (full only if floors pass per FINAL; atoms-only preserved)
    # Full wire per FreshHandoffVLMActivePmfFullWire + batch:50+ : ent=pmf_entropy(sum_pmf or per-au), dist=abs(p-0.39), score=ent+dist+(1-consist); sort high-unc first; trim budget or defer; pre-CORN pool (cache+decode) + G1B tie + hybrid MAPIE LTT/wrapper consistency. Reexport pmf_entropy in protocols.
    p.add_argument("--prioritize-unc", action="store_true", help="Prioritize high-uncertainty (pmf_entropy(sum_pmf(au_pmfs/prior or consistency) + |p-0.39| + low_consist) rows first/top-k or defer to vet pre/post VLM. decode portable; use generic_ordinal_mode for non-FGS.")
    p.add_argument("--unc-budget", type=int, default=None, help="If --prioritize-unc, limit to top-N highest-unc (focus vet budget).")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    # FreshFullGateWiringManifestsEnforcer (G0 hard before compute in vlm entry per audit/ kickoff CONCRETE; orch comment says orch blocks but real scripts now also enforce for bypass resist).
    # "G0 must precede; committed manifests required". Unique strict 8-gate vs priors.
    import pathlib as _p
    _root = _p.Path(__file__).resolve().parents[1]
    _power = _root / "data" / "manifests" / "power.json"
    if not _power.exists():
        raise SystemExit("G0 must precede; committed manifests required: data/manifests/power.json missing for VLM run. Run gate0 first.")

    if args.submit_only and args.collect_only:
        print("error: --submit-only and --collect-only are mutually exclusive", file=sys.stderr)
        return 2

    if args.collect_only:
        if not args.batch_id or not args.out:
            print("error: --collect-only requires --batch-id and --out", file=sys.stderr)
            return 2
        out = wait_and_collect(args.batch_id, args.out, poll_s=args.poll_s)
        print("wrote", out)
        return 0

    if not args.manifest:
        print("error: --manifest is required (unless --collect-only)", file=sys.stderr)
        return 2

    # VLM active wire: pass prioritize_unc / unc_budget (G1B tie: only full if CI-LB floors; see orch gate1b; G0 power first)
    # Full wire per FreshHandoffVLMActivePmfFullWire + batch:50+ : ent=pmf_entropy(sum_pmf or per-au), dist=abs(p-0.39), score=ent+dist+(1-consist); sort high-unc first; trim budget or defer; pre-CORN pool (cache+decode) + G1B tie + hybrid MAPIE LTT/wrapper consistency. Reexport pmf_entropy in protocols.
    batch_id = submit(
        args.manifest,
        n_runs=args.n_runs,
        model=args.model,
        prioritize_unc=getattr(args, "prioritize_unc", False),
        unc_budget=getattr(args, "unc_budget", None),
    )

    if args.submit_only:
        print("submitted batch_id =", batch_id)
        return 0

    if not args.out:
        print("error: --out is required to collect results (or pass --submit-only)", file=sys.stderr)
        return 2

    out = wait_and_collect(batch_id, args.out, poll_s=args.poll_s)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
