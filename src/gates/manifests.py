"""G0 manifest enforcement helpers (single source).

Extracted from src/gates/orchestrator.py to eliminate duplication.
All real gate scripts and the orchestrator should import from here.

G0 must precede all quantitative work; committed manifests under data/manifests/
are the contract. Synthetic mode bypasses enforcement for CI/toy runs.
"""

from __future__ import annotations

import pathlib
from typing import Final

# Canonical precondition manifests per gate (G0 is the bootstrap source)
GATE_PRECONDS: Final[dict[str, list[str]]] = {
    "gate0": [],  # G0 produces power.json; no precondition
    "gate1": ["data/manifests/power.json"],
    "gate2": ["data/manifests/power.json"],
    "gate3": ["data/manifests/power.json", "data/manifests/folds.csv"],
    "gate4": ["data/manifests/power.json"],
    "gate5": ["data/manifests/power.json"],
    "gate1b": ["data/manifests/power.json"],
    "gate6": ["data/manifests/power.json", "data/manifests/severity.json"],
}


def enforce_g0_manifests(gate: str, synthetic: bool = False) -> None:
    """Hard block if not synthetic and required G0 manifests missing or empty.

    Enforces 'G0 must precede; committed manifests required'.
    Raises SystemExit on violation (matches orchestrator contract).
    """
    if synthetic:
        return

    root = pathlib.Path(__file__).resolve().parents[2]
    for p in GATE_PRECONDS.get(gate, []):
        mp = root / p
        if not mp.exists() or mp.stat().st_size == 0:
            raise SystemExit(
                f"G0 must precede; committed manifests required: "
                f"missing or empty {p} for {gate}. "
                f"Run gate0 first (and gate1 for folds)."
            )
        if "manifests" not in str(mp):
            raise SystemExit(
                f"Manifests target drift: {p} must be under data/manifests/ "
                f"(single-source enforcement)."
            )

    # Explicit power.json check for G0-tied gates
    power = root / "data" / "manifests" / "power.json"
    if gate != "gate0" and not power.exists():
        raise SystemExit(
            "G0 must precede; committed manifests required: "
            "data/manifests/power.json missing. "
            "Run scripts/gate0_power.py first."
        )
