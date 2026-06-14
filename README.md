# cat-fgs-llm

**This is not another cat-pain detector.** Its headline deliverable is a portable,
power-aware **confound-attribution protocol** for fine-grained animal-affect models —
a one-directional, per-AU audit (FGS-BG-Gap counterfactual + per-AU EBPG
saliency-as-confound-evidence + VLM judge-bias probe) that any future facial-pain-scorer
corpus can run. It ships with a **guarded VLM-as-AU-rater reliability check**
(per-AU VLM-vs-vet quadratic kappa, CI-lower-bound-gated; a result pending an
independent vet anchor), on an explicitly-conceded DINOv2 + CORN engine, with an honest
binary-plus-abstention floor (welfare-asymmetric operating point at pain-recall >= 0.90,
one-sided 95% NPV defer-to-vet curve — cited supporting plumbing, not a headline) that
stands even though the 0-10 layer is reported as inspected-not-validated.

The novelty is the **assembly** plus the per-AU EBPG-as-confound-evidence step plus the
instantiation of equivalence-style audit hygiene — never any individual primitive
(bg-swap, saliency, the one-directional / power-conditioned statistics are all prior
work) and never the kappa CI-lower-bound gate (textbook clinimetrics) or the
rubric-paraphrase guard (published).

**Status of the legs:** the confound protocol is the strong leg — it validates today on
planted positive and negative controls (`src/protocols/standalone_test_corpus.py`). The
kappa reliability check is the guarded second leg: a *protocol with a result pending* —
it cannot report a number until the per-AU vet anchor exists (see Gate 0). The current
dataset is binary pain/no_pain and **cannot** yield 0/1/2 AU ground truth, so until the
anchor is built, the kappa check ships as a runnable protocol, not a finding. Kappa is
not deleted: it is kill-tree insurance — the sole-survivor headline if the confound leg
degrades. The binary-plus-abstention floor is what stands today as the engineering spine;
treat it as the likely v1 ship, not the fallback.

## The engine is conceded plumbing, not claimed novel

The frozen DINOv2 ViT-S/14 + 5 per-AU CORN heads -> 0-10 sum -> 0.39 decision engine
(`src/model/`) is **plumbing, explicitly conceded as not-novel**. Directory names,
module docstrings, artifact tags, and run names never imply the engine is the
contribution. Its outputs feed only `src/wrapper/` (the welfare frame, cited supporting
plumbing) and `src/eval/` (the confound-attribution headline + the guarded kappa check).
The v1 spine is **binary pain/no_pain + wrapper**; the
0-10 layer is built, decoded, and **inspected-not-validated** — it never emits a
validated-claim number, and QWK-vs-VLM is never validation.

**Backbone note:** the `_reg` (register) DINOv2 variants are now the field default
(`dinov2_vits14_reg`) because registers suppress attention artifacts that hurt dense,
localized features — and per-AU FGS scoring is exactly localized (orbital, ear, muzzle
sub-regions). A/B the reg variant before locking ViT-S/14. Expect orbital/ear/head to
be where the Gate 1-B kill-switch fires: a small frozen backbone with no fine-tuning on
a tiny corpus is most likely to miss those AUs, so budget for binary-plus-abstention
being the v1 ship rather than the fallback.

## The portable artifacts (all dataset-agnostic)

1. **FGS-BG-Gap + per-AU EBPG confound-attribution protocol (THE HEADLINE)**
   (`src/eval/confound.py`) — a one-directional, power-conditioned audit ("no confound
   detected at this power") any future facial-pain-scorer corpus can run. Combines a
   background-swap counterfactual, per-AU EBPG saliency-as-confound-evidence, and a VLM
   judge-bias probe. Validated today on planted positive and negative controls
   (`src/protocols/standalone_test_corpus.py`). The contribution is the assembly + the
   per-AU EBPG-as-confound-evidence step + the equivalence-style audit hygiene; the
   underlying primitives and the one-directional / power-conditioned statistics are prior
   work and are cited, not claimed.
2. **VLM-as-AU-rater kappa reliability check (GUARDED, inspected-not-validated)**
   (`src/vlm/`, `src/eval/kappa.py`) — scores any face corpus's per-AU VLM labels against
   a vet anchor; reports 5 quadratic kappa with CI lower bounds. Fires on the CI lower
   bound (Gate 1-B). This is a guarded reliability check ranked below the confound
   protocol, NOT a co-equal contribution; the CI-lower-bound gate is standard
   clinimetrics and the rubric-independence guard is published, so neither is branded as
   novel. **A result requires an independent vet anchor that does not yet exist**; the
   artifact shipped today is the protocol. **Interpretation guard:** a high kappa only
   measures capability if the anchor is independent and the rubric handed to the VLM is
   not the same rubric the vet scored from — otherwise it measures rubric-following, not
   weak-labeling skill. State which one a given run measures. Kept as kill-tree insurance
   (sole-survivor headline if the confound leg degrades).
3. **Welfare-asymmetric decision curve + one-sided-95%-NPV abstention curve (CITED
   SUPPORTING PLUMBING, not a headline)** (`src/wrapper/`) — operating point at fixed
   pain-recall >= 0.90 with the undertreat:overtreat harm ratio swept as a range, plus a
   defer-to-vet boundary with finite-sample lower bounds (the word "guaranteed" is
   banned).

## Gate run-order (strict; each gate blocks downstream)

Each gate writes an immutable artifact; no downstream number is believed until its
prerequisite gate's artifact exists and passes.

```
G0  gate0_power      power calcs + vet-budget integer (no data, no GPU) — blocks all quantitative work
G1  gate1_merge      per-CAT merge vs CAT_ ids (not raw CLIP/pHash); cat-disjoint folds
G2  gate2_confound   one-directional capture-condition audit
G3  gate3_holdout    frozen hashed cat-disjoint hold-out + CI abort
G4  gate4_mps_check  MPS<->CPU logit parity + CORN decode->sum->0.39 (BLOCKING; `make test`)
G5  gate5_nme        alignment / NME / face-pixel-resolution audit
G1-B kappa pilot     run_vlm_labels + eval/kappa (fires on CI lower bound; self-justifies the labeler)
G6  gate6_severity   AU=2 severity-cell collapse decision
```

Run via the Makefile: `make gate0`, `make gate1`, ..., plus `make test` (Gate 4 must
be green before any quantitative target runs). The central orchestrator
(`make gate-orchestrate` / `gate-pipeline` / `gate-e2e-synthetic --synthetic`)
enforces strict order + artifacts + abort (G0 first). Use `make test-portable`
for the deletion-safe protocols surface (zero-torch import isolation + synthetic
arbitrary-AU exerciser).

**Portable protocols import (the citable surface — confound headline + guarded kappa check):**
```python
from src.protocols import (
    per_au_kappa_table, bootstrap_qwk_lb, bg_gap, judge_bias,
    get_au_names, map_df_columns, generic_ordinal_mode, FGS_AU_NAMES,
)
from src.protocols.adapters import apply_au_override
# Standalone / deletion-safe test: python -m src.protocols.standalone_test_corpus
```
**src.protocols standalone quickstart** (see also `src/protocols/README.md:17` for full examples + `python -m src.protocols.standalone_test_corpus` deletion-safe zero-torch proof after `rm -rf` engine; adapters/generic for non-FGS reuse):
```python
from src.protocols import (
    per_au_kappa_table, bootstrap_qwk_lb, bg_gap, judge_bias,
    get_au_names, map_df_columns, generic_ordinal_mode, FGS_AU_NAMES,
)
from src.protocols.adapters import apply_au_override
```
See `src/protocols/` (adapters for generic/non-FGS reuse; standalone corpus
exercises with no engine leak). The confound-attribution headline and the guarded kappa
check are both first-class importable + gate-enforced (see FINAL_DIRECTION.md, paper
sections 04/06 post-sync, Makefile test-portable / gate-e2e-synthetic, and
src/gates/orchestrator.py). DINOv3 prep is staged for an engine successor.

**Kill/pivot:** if orbital/ear/head kappa lower bound falls below the Gate 1-B floor,
drop the 0-10 layer entirely and ship calibrated binary + abstention + the confound
audit; the confound protocol carries the headline alone and the guarded kappa check is
recorded as not-met. The confound protocol is the leg the paper rests on; kappa is the
insurance.

## Environment

uv-managed, Python 3.11 (system 3.9.6 is not used). Local = Apple M4 / MPS, **no
local CUDA**; CUDA is confined to `notebooks/colab_train_rfdetr.ipynb` for detector
training only. All code runs through `uv run`. See `pyproject.toml` and
`configs/*.yaml` (all hyperparameters and seeds live in configs, never hardcoded).
