"""Portable protocols seam.

Asymmetric structure:
- HEADLINE (spine): capture-condition confound-attribution protocol
  (FGS-BG-Gap / per-AU EBPG / judge-bias; one-directional by const). The novelty
  is the assembly + per-AU EBPG-as-confound-evidence + equivalence-style audit
  hygiene, NOT any individual primitive and NOT the one-dir/power-conditioned stats.
- GUARDED SUPPORTING CHECK (ranked below): VLM-as-AU-rater kappa protocol
  (reusable on arbitrary ordinal rater corpus). Inspected-not-validated reliability
  check, result pending an independent vet anchor. CI-LB gate is textbook
  clinimetrics and the rubric-paraphrase guard is published; neither is novel.
  Kept as kill-tree insurance, not a co-equal pillar.

Thin adapters for portability (AU override, col mapper, no-FGS-0.39 generic ordinal mode).
Pure re-exports: from .kappa import ... (numpy + sklearn only; no engine/torch unless optional).
from .confound import ...

Standalone test corpus runner exercises non-FGS or arbitrary AUs with zero torch leak.

Usage on new corpus:
  from src.protocols import per_au_kappa_table, bg_gap, judge_bias, map_to_au_cols, get_au_names
  # or for generic:
  from src.protocols.adapters import generic_ordinal_kappa  # no 0.39 assumption

Re-exports keep the original modules import-pure (no engine pulled at top of kappa/confound).
The decode 0-10 / 0.39 is conceded plumbing (use only if attaching graded inspection; optional import).

See standalone_test_corpus.py for deletion-safe example (import *only* protocols; assert 'torch' not in sys.modules).
"""
from __future__ import annotations

# Pure re-exports (stay numpy-only at import time; engine optional downstream)
from src.eval.kappa import (
    per_au_kappa_table,
    bootstrap_qwk_lb,
    bootstrap_qwk_ci,
    qwk,
    AU_NAMES as FGS_AU_NAMES,
)
from src.eval.confound import (
    bg_gap,
    bg_gap_per_au,
    ebpg,
    judge_bias,
    judge_bias_shift,
    NO_CONFOUND_MSG,
    JUDGE_BIAS_PERTURBATIONS,
)

# Thin adapters + generic mode
from .adapters import (
    get_au_names,
    map_df_columns,
    generic_ordinal_mode,
    apply_au_override,
)

# Portable reexports for active VLM pmf unc + abstention hybrid (per FreshHandoffVLMActivePmfFullWire + decode:72):
# pmf_entropy / sum_pmf / au_pmf_from_cumprobs are pure-np portable derives (no hard 5/11; N/k from len/shape).
# Used when prioritize_unc + au_pmfs/consist for ent=pmf_entropy(sum_pmf or per-au), dist=abs(p-0.39), score=ent+dist+(1-consist).
# Full pre-CORN pool path + G1B CI-LB tie + wrapper MAPIE LTT hybrid. Atoms-only (VLM 5 only; unc in code).
# Reexport keeps protocols surface citable for non-FGS ordinal reuse too (with adapters generic_ordinal_mode).
# DEL-SAFE GUARD (FreshHandoffPortableDeletionVerifier): try/except so "from src.protocols import (kappa names...)" + "python -m src.protocols.standalone_test_corpus"
# and monkeypatch deletion tests succeed with model/vlm/wrapper deleted (rm -rf engine still passes for core portable kappa+confound+adapters surface).
# pmf_* remain available (or None) only when engine present; pure portable path (standalone_test_corpus) never pulls them.
try:
    from src.model.decode import pmf_entropy, sum_pmf, au_pmf_from_cumprobs  # noqa: E402 (optional for unc consumers; engine optional at import for pure path)
except ImportError:
    # deletion-safe portable consumers (CI rm -rf src/model; test-portable monkey; standalone zero-engine) succeed
    pmf_entropy = sum_pmf = au_pmf_from_cumprobs = None  # type: ignore[assignment]

__all__ = [
    "per_au_kappa_table",
    "bootstrap_qwk_lb",
    "bootstrap_qwk_ci",
    "qwk",
    "FGS_AU_NAMES",
    "bg_gap",
    "bg_gap_per_au",
    "ebpg",
    "judge_bias",
    "judge_bias_shift",
    "NO_CONFOUND_MSG",
    "JUDGE_BIAS_PERTURBATIONS",
    "get_au_names",
    "map_df_columns",
    "generic_ordinal_mode",
    "apply_au_override",
    "pmf_entropy",
    "sum_pmf",
    "au_pmf_from_cumprobs",
    # reexport for decode if needed (pure consumers use optional)
]
