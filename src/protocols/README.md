# src/protocols/ — Portable Protocols (Confound Attribution headline + guarded Kappa check)

**Portable API surface for the codebase, structured asymmetrically.**

The headline (spine) method and a guarded supporting check are exposed here:
- **Headline:** the one-directional, power-conditioned, per-AU capture-condition confound-attribution protocol (FGS-BG-Gap counterfactual + per-AU EBPG saliency-as-confound-evidence + judge-bias probe). This is the strong leg: validated today on planted positive+negative controls in `standalone_test_corpus.py`. The novelty is the *assembly* plus per-AU EBPG-as-confound-evidence plus the instantiation of equivalence-style audit hygiene — not any individual primitive, and not the one-directional / power-conditioned / equivalence statistics themselves (those are prior art; see Adebayo et al. 2023 for the "at this power" framing, Xiao et al. 2021 / Moayeri et al. 2022 for bg-swap, Huang & Hooker 2026 / Singh et al. 2023 for power-conditioned audit hygiene).
- **Guarded supporting check:** the VLM-as-AU-rater kappa protocol (per-AU quadratic weighted kappa + bootstrap CI lower-bound; reusable on arbitrary ordinal rater corpus). This is an inspected-not-validated reliability check (result pending an independent vet anchor), ranked below the confound protocol. Its CI-lower-bound gate is textbook clinimetrics (Tractenberg et al. 2010, Donner & Rotondi 2010, Sim & Wright 2005) and the rubric-paraphrase guard is published (Weng et al. 2026); neither is a novel increment. Kept as kill-tree insurance (sole-survivor headline if the confound leg degrades), not as a co-equal pillar.

Thin adapters + `generic_ordinal_mode` make it dataset-agnostic (override AU names/order, col mapper for your {au}_vlm/{au}_vet, opt-in/out of FGS 0.39 assumption).

Pure re-exports keep `src/eval/kappa.py` + `src/eval/confound.py` import-pure (numpy + sklearn only at protocol import time; engine/torch optional and never leaked).

**Standalone test corpus runner** (deletion-safe): exercises everything on synthetic arbitrary-AU data; asserts zero torch leak post-import.

See also: main README.md (portable section), `src/constants.py` (single-source AU_ORDER + POINT_DECISION_THRESHOLD=0.39, never leaked into portable), wrapper/ (decision_curve + operating_point + abstention for welfare frame; cited supporting plumbing, not a headline contribution).

## Quickstart / Usage Examples

### Basic import + FGS default (5-AU Evangelista)
```python
from src.protocols import (
    per_au_kappa_table,
    bootstrap_qwk_lb,
    bootstrap_qwk_ci,
    qwk,
    bg_gap,
    bg_gap_per_au,
    ebpg,
    judge_bias,
    judge_bias_shift,
    FGS_AU_NAMES,
    NO_CONFOUND_MSG,
    get_au_names,
    map_df_columns,
    generic_ordinal_mode,
)

# kappa table (per-AU QWK + CIs; cat-grouped bootstrap supported)
# df must have {au}_vlm, {au}_vet (and optional cat_id for grouped)
tab = per_au_kappa_table(df, au_names=get_au_names(), cat_col="cat_id")
print(tab)  # columns: kappa, ci_lo, ci_hi, ci_lb, n, ...

lb, _ = bootstrap_qwk_lb(df["ear_vlm"], df["ear_vet"], n_boot=200, groups=df.get("cat_id"))
```

### Non-FGS / arbitrary AU override (generic ordinal, no 0.39)
```python
au3 = ["ear", "muzzle", "custom_orbital"]  # or 7-AU etc.
df3 = ...  # your corpus cols
tab3 = per_au_kappa_table(df3, au_names=get_au_names(au3))
print("kappa on arbitrary 3:", tab3.shape)

mode = generic_ordinal_mode(use_fgs_threshold=False)
assert not mode["fgs_mode"]  # disables 0.39 / FGS sum logic downstream if caller respects
```

### Col mapper + confound on arbitrary scores
```python
df_raw = ...  # e.g. "my_vlm_score1", "gold1"
df_mapped = map_df_columns(
    df_raw, ["au1", "au2"],
    mapper={"my_vlm_score1": "au1_vlm", "gold1": "au1_vet"}
)

# pure confound (no engine, no 0.39 needed)
scores = ...  # 0-10 or arbitrary
sw = ...
g = bg_gap(scores, sw)  # or bg_gap_per_au, judge_bias, ebpg
assert g["note"] == NO_CONFOUND_MSG
```

### Gate 2 lightweight synth demo (v1 reusable one-dir portable protocol)
Per FINAL/GAP: cheap pre-train (CPU, no vet, no GPU; runs before any spend; one-directional "no confound detected at this power" — NEVER "no confound"/"ruled out"). Trivial probe + judge-bias demo (synth baseline/perturbed) exercise the FGS-BG-Gap / EBPG / judge-bias surface + NO_CONFOUND_MSG. Reusable on ANY corpus (dataset-agnostic via adapters/generic_ordinal_mode; zero-torch del-safe).

Synth run + output:
- `python -m src.protocols.standalone_test_corpus` (exercises judge_bias/bg_gap_per_au/ebpg + asserts note==NO_CONFOUND_MSG on arbitrary synth; PASS lines ~82/113/134/150).
- `uv run python -m src.gates.orchestrator --synthetic --portable-only` (light mode; exercises gate2 toy probe path + protocols surface; summary["results"]["gate2"] carries "note", "judge_bias":{"demo":true}, "verdict" or NO).
- Real/synth gate2 produces `artifacts/gate2/confound_audit.json` (trivial AUC+CI+verdict; "note": NO_CONFOUND_MSG; judge demo).

See: scripts/gate2_confound.py (trivial_probe:70, _demo_judge:124, run_gate:145+), src/eval/confound.py:3 (PORTABLE/DATASET-AGNOSTIC + ONE-DIR LAW), :37 (NO_CONFOUND_MSG), protocols/__init__.py:33 (reexports bg_gap/ebpg/judge/NO), standalone_test_corpus.py:82, src/gates/orchestrator.py:237 (portable_only gate2), test_e2e_gates_pipeline.py:93 (demo assert). Lightweight affirmed for v1 (honest stubs for unpowered bg/ebpg; full real future). Makes the headline confound-attribution protocol citable/executable; the novelty is the assembly + per-AU EBPG-as-confound-evidence, never the one-directional/power-conditioned statistics themselves.

### Full surface + adapters (apply_au_override)
```python
from src.protocols.adapters import apply_au_override

over = apply_au_override(["x1", "x2", "x3"])
# re-exports also include FGS_AU_NAMES, JUDGE_BIAS_PERTURBATIONS etc.
```

### Standalone / deletion-safe test (zero torch leak proof)
```bash
# direct
python -m src.protocols.standalone_test_corpus

# or with project env
uv run python -m src.protocols.standalone_test_corpus

# after pip install -e . (see below)
python -m src.protocols.standalone_test_corpus
```
This runs synth corpus (FGS default + 3-AU override + col map + generic mode + full kappa/confound/ebpg/bootstrap/adapters). Prints PASSes. If `rm -rf src/model src/vlm ...` (keeping protocols + eval + constants) and it still passes → portable claim holds.
# Deletion-safety is verified by a separate import-isolation check (rm + monkeypatch PASS; guard at src/protocols/__init__.py:56 try/except). The deletion-safe portable surface (confound headline + guarded kappa check + adapters + standalone) supports the protocol's portability claim; do not restate it as a "first/only" superlative. It also feeds the training pipeline (kappa CI-LB as a reliability gate for VLM labels; confound pre-filter; generic mode for new-corpus reuse; CI deletion isolation).

Import isolation assert is first action.

## Pyproject / "pip install -e" Reuse Surface
From repo root:
```bash
pip install -e .
# or uv sync / uv pip install -e .

# Then (PYTHONPATH may be needed depending on layout; uv run handles):
PYTHONPATH=. python -c "
from src.protocols import per_au_kappa_table, bg_gap, get_au_names, map_df_columns, generic_ordinal_mode
from src.protocols.adapters import apply_au_override
# + standalone runner as -m
python -m src.protocols.standalone_test_corpus
"
# Also reuses wrapper (operating_point/decision_curve/abstention) + protocols together for full frame.
# See pyproject.toml (name=cat-fgs-llm, dependencies include sklearn/krippendorff for protocols; no engine forced at protocol layer).
# pytest config sets pythonpath=["."]; gates/Makefile use uv run.
# Explicit config_path= in wrapper calls for hygiene when using installed data.
```

This makes kappa/confound protocols + adapters + standalone first-class reusable (dataset-agnostic) on any ordinal AU corpus. The 0-10/0.39/decode are **optional compat layer** (conceded plumbing; opt-in only).

## Relation to Rest of Codebase (Preserved Constraints)
- **Wrapper (cited supporting plumbing, not a headline contribution)**: `src/wrapper/decision_curve.py` (harm-ratio range as pt sweep + cat-grouped net-benefit ribbon; dcurves), `operating_point.py` (fixed pain-recall >=0.90, never Youden), `abstention.py` (one-sided 95% NPV LB Clopper-Pearson; MAPIE LTT alt via BinaryClassificationController + negative_predictive_value; "guaranteed" banned). All param-only / importlib.resources clean (no __file__ deprecate).
- Single-source 0.39 (`src/constants.py` POINT_DECISION_THRESHOLD; used in abstention band center + decode + tests; **never leaked** into protocols).
- Vet firewall + circularity (sens/spec on vet-confirmed only; Gate order strict).
- All gates / welfare / conceded engine constraints preserved.
- Relation to priors: see findings + paper sections (the confound-attribution assembly is the headline; the kappa check, wrapper, and 0.39 compat layer are supporting/plumbing; the one-directional and CI-LB statistics are prior art and credited, not claimed as novel).

## Testing
- `make test-portable` or direct -m as above.
- `uv run pytest tests/test_au_order.py tests/test_kappa_cluster_bootstrap.py ...` (protocols exercised).
- Full e2e: `make gate-orchestrate` (but protocols surface independent).

## Notes for Pipeline / Training Use
See main findings + paper sections for how this surface relates to priors (distinguished from Lencioni et al. 2025 per-AU saliency, BECKI 2025 VLM context bias, the FGS-chatbot agreement study 2025, and the FGS domain anchors Evangelista 2021 / Cheng 2020).
- This surface feeds training/eval: the confound audit acts as a pre-train filter; the kappa check (CI-LB gated) is a reliability check on VLM label quality, not a validation claim.
- Explicit generic + standalone = reusable on future corpora without engine lock-in.
- Combined with the wrapper (cited plumbing) = welfare-aware + risk-controlled + portable evaluation. The reframed honesty position: the headline is the assembly of the confound-attribution protocol; the kappa check, wrapper, and 0.39 compat layer are supporting evidence/plumbing.

For deeper: `src/protocols/__init__.py` (reexports + doc), `adapters.py`, `standalone_test_corpus.py`, main README + IMPLEMENTATION_PLAN.

Ready for "pip install -e" + import + -m tests.