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
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))  # allow `python scripts/run_vlm_labels.py` without PYTHONPATH

from src.vlm.batch_collect import wait_and_collect  # noqa: E402
from src.vlm.batch_submit import submit  # noqa: E402


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
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

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

    batch_id = submit(args.manifest, n_runs=args.n_runs, model=args.model)

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
