#!/usr/bin/env python3
"""Gate 3 -- frozen, hashed, cat-disjoint hold-out + CI-abort guard
(IMPLEMENTATION_PLAN §1.6 Gate 3 / §7.2.1).

Builds the boundary-rich, true-~13%-prevalence test manifest from merged
(cat-disjoint) groups, then freezes and sha256-hashes it. The test set is
FROZEN and never touched during selection. A CI check fails the build if any
train/sweep/threshold script can read the test manifest -- structurally
preventing the post-hoc goalpost move.

Writes:
    data/manifests/test_manifest.json   -- {version, test_groups, fold_csv_sha256}
    data/manifests/test_manifest.sha256 -- sha256 of the canonical json blob

``abort_if_test_manifest_readable`` is the guard helper that train/tune scripts
import at startup. Data plumbing; makes no validated claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
TEST_MANIFEST_JSON = MANIFESTS / "test_manifest.json"
TEST_MANIFEST_SHA = MANIFESTS / "test_manifest.sha256"


def freeze_test_manifest(
    version: str,
    test_groups: list[str],
    folds_csv: pathlib.Path,
    out_json: pathlib.Path = TEST_MANIFEST_JSON,
    out_sha: pathlib.Path = TEST_MANIFEST_SHA,
) -> str:
    """Freeze + hash the cat-disjoint test manifest. Returns the manifest hash."""
    folds_csv = pathlib.Path(folds_csv)
    manifest = {
        "version": version,
        "test_groups": sorted(test_groups),  # cat-disjoint from all train/val folds
        "fold_csv_sha256": hashlib.sha256(folds_csv.read_bytes()).hexdigest(),
    }
    blob = json.dumps(manifest, sort_keys=True).encode()
    manifest_hash = hashlib.sha256(blob).hexdigest()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_bytes(blob)
    out_sha.write_text(manifest_hash)
    return manifest_hash


def abort_if_test_manifest_readable(
    manifest_path: pathlib.Path = TEST_MANIFEST_JSON,
) -> None:
    """CI-abort guard: any train/sweep/threshold run that can READ the frozen
    test manifest must refuse to run (structurally defeats post-hoc
    goalpost-moving). Call this at the top of every training/tuning entrypoint.

    Raises ``SystemExit`` if the manifest is present and readable.
    """
    manifest_path = pathlib.Path(manifest_path)
    if manifest_path.exists():
        try:
            with open(manifest_path, "rb"):
                pass
        except OSError:
            return  # exists but not readable -> fine for a train run
        raise SystemExit(
            f"ABORT (Gate 3): {manifest_path} is readable from this process. "
            f"Training/tuning/threshold runs must not be able to read the frozen "
            f"test manifest. Wall it off (chmod/CI sandbox) before proceeding."
        )


def verify_fold_csv_unmutated(
    folds_csv: pathlib.Path,
    manifest_path: pathlib.Path = TEST_MANIFEST_JSON,
) -> None:
    """Hold-out eval guard: refuse to run if the working-tree fold CSV was
    mutated after freeze (its sha256 no longer matches the frozen manifest).
    """
    manifest = json.loads(pathlib.Path(manifest_path).read_text())
    cur = hashlib.sha256(pathlib.Path(folds_csv).read_bytes()).hexdigest()
    if cur != manifest["fold_csv_sha256"]:
        raise SystemExit(
            "ABORT (Gate 3): folds.csv sha256 differs from the frozen test "
            "manifest -- the split was mutated after freeze. Re-freeze or revert."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate 3: freeze + hash cat-disjoint hold-out")
    ap.add_argument(
        "--version",
        default="cat-pain-ul7lu-p3rtl@v2",
        help="dataset version string (goes in the manifest + W&B artifacts)",
    )
    ap.add_argument(
        "--folds-csv",
        default=str(MANIFESTS / "folds.csv"),
        help="path to the Gate-1 folds.csv (its sha256 is bound into the manifest)",
    )
    ap.add_argument(
        "--test-groups",
        nargs="+",
        required=True,
        help="cat-disjoint test group ids (selected boundary-rich, ~13%% prevalence)",
    )
    args = ap.parse_args()

    folds_csv = pathlib.Path(args.folds_csv)
    if not folds_csv.exists():
        raise SystemExit(f"folds.csv not found: {folds_csv} (run gate1_merge first)")

    manifest_hash = freeze_test_manifest(
        version=args.version,
        test_groups=args.test_groups,
        folds_csv=folds_csv,
    )
    print(f"[Gate 3] froze {TEST_MANIFEST_JSON}")
    print(f"[Gate 3] wrote {TEST_MANIFEST_SHA}")
    print(f"[Gate 3] manifest sha256 = {manifest_hash}")
    print(f"[Gate 3] {len(args.test_groups)} cat-disjoint test groups frozen; "
          f"never touched during selection.")


if __name__ == "__main__":
    main()
