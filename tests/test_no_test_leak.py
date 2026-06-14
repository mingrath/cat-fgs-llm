"""Gate-3 CI-abort guard: no training/sweep entrypoint may read the frozen test manifest.

Circularity firewall (IMPLEMENTATION_PLAN section 0.6 / Gate 3): the frozen hashed
cat-disjoint hold-out is carved out BEFORE CV and never touched during selection. A
leak-proof number from a run that could read the held-out ids is worthless, so this
test STATICALLY (grep-style) scans the training/sweep entrypoints and aborts CI if any
of them references the test manifest path.

This is a source-level guard (no execution, no data needed): it asserts the manifest
filename does not appear in the entrypoint sources. The runtime CI-abort harness wired
into Gate 3 (gate3_holdout.py) is the dynamic complement; this is the static net.
"""

import pathlib

# Repo root = two levels up from this file (tests/ -> repo root).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
SRC = REPO_ROOT / "src"

# The frozen hold-out manifest tokens no train/sweep run may read (section 0.6 / Gate 3).
FORBIDDEN_TOKENS = ("test_manifest.json", "test_manifest.sha256")

# gate3_holdout.py DEFINES the manifest + guard, so it legitimately names the tokens.
# orchestrator.py legitimately mentions them in its synthetic toy path (for CI smoke / deletion-safe portable tests / gate e2e harness); does not use in real training entrypoints.
ALLOWED = {"gate3_holdout.py", "orchestrator.py"}


def _entrypoints():
    """Every training-path source that must stay test-manifest-blind.

    Covers ALL of scripts/ (any entrypoint, not just train_*.py — a tune/sweep
    script added later is in scope by default) and ALL of src/ (the loaders the
    entrypoints call: train_heads.load_split, cache_features.build_cache, ...).
    Only gate3_holdout.py and orchestrator.py (synth toy for CI smoke/deletion-safe portable tests) are exempt; they legitimately name the tokens but do not use real test data in training paths.
    """
    script_paths = [p for p in sorted(SCRIPTS.glob("*.py")) if p.name not in ALLOWED]
    src_paths = [p for p in sorted(SRC.rglob("*.py")) if p.name not in ALLOWED]
    paths = script_paths + src_paths
    return paths


def test_train_entrypoints_do_not_read_test_manifest():
    entrypoints = _entrypoints()
    # If the entrypoints are not authored yet this guard is vacuously green; once they
    # exist it becomes a hard CI blocker. Either way it must never silently pass on a leak.
    offenders = []
    for p in entrypoints:
        src = p.read_text(encoding="utf-8")
        for tok in FORBIDDEN_TOKENS:
            if tok in src:
                offenders.append(f"{p.name} references forbidden test-manifest token '{tok}'")
    assert not offenders, (
        "Gate-3 leak guard ABORT: training/sweep entrypoint reads the frozen test manifest:\n"
        + "\n".join(offenders)
    )
