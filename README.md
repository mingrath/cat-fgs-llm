# cat-fgs-llm

**This is not another cat-pain detector.** Its headline deliverables are two
transportable *methods* no prior feline-pain work produced — a protocol for measuring
whether a frozen VLM can weak-label Feline Grimace Scale action units at human-rater
agreement (per-AU VLM-vs-vet quadratic kappa, CI-lower-bound-gated), and a reusable
capture-condition confound-attribution protocol — shipped on an explicitly-conceded
DINOv2 + CORN engine, with an honest binary-plus-abstention floor (welfare-asymmetric
operating point at pain-recall >= 0.90, one-sided 95% NPV defer-to-vet curve) that
stands even though the 0-10 layer is reported as inspected-not-validated.

**Status of the headline:** the kappa method is a *protocol with a result pending* —
it cannot report a number until the per-AU vet anchor exists (see Gate 0). The current
dataset is binary pain/no_pain and **cannot** yield 0/1/2 AU ground truth, so until the
anchor is built, method #1 ships as a runnable protocol, not a finding. The
binary-plus-abstention floor is what stands today; treat it as the likely v1 ship, not
the fallback.

## The engine is conceded plumbing, not claimed novel

The frozen DINOv2 ViT-S/14 + 5 per-AU CORN heads -> 0-10 sum -> 0.39 decision engine
(`src/model/`) is **plumbing, explicitly conceded as not-novel**. Directory names,
module docstrings, artifact tags, and run names never imply the engine is the
contribution. Its outputs feed only `src/wrapper/` (the welfare frame) and `src/eval/`
(the two portable methods). The v1 spine is **binary pain/no_pain + wrapper**; the
0-10 layer is built, decoded, and **inspected-not-validated** — it never emits a
validated-claim number, and QWK-vs-VLM is never validation.

**Backbone note:** the `_reg` (register) DINOv2 variants are now the field default
(`dinov2_vits14_reg`) because registers suppress attention artifacts that hurt dense,
localized features — and per-AU FGS scoring is exactly localized (orbital, ear, muzzle
sub-regions). A/B the reg variant before locking ViT-S/14. Expect orbital/ear/head to
be where the Gate 1-B kill-switch fires: a small frozen backbone with no fine-tuning on
a tiny corpus is most likely to miss those AUs, so budget for binary-plus-abstention
being the v1 ship rather than the fallback.

## The three portable artifacts (all dataset-agnostic)

1. **VLM-as-AU-rater kappa protocol** (`src/vlm/`, `src/eval/kappa.py`) — scores any
   face corpus's per-AU VLM labels against a vet anchor; reports 5 quadratic kappa with
   CI lower bounds. Fires on the CI lower bound (Gate 1-B). **A result requires an
   independent vet anchor that does not yet exist**; the artifact shipped today is the
   protocol. **Interpretation guard:** a high kappa only measures capability if the
   anchor is independent and the rubric handed to the VLM is not the same rubric the vet
   scored from — otherwise it measures rubric-following, not weak-labeling skill. State
   which one a given run measures.
2. **FGS-BG-Gap + per-AU EBPG confound-attribution protocol** (`src/eval/confound.py`)
   — a one-directional audit ("no confound detected at this power") any future
   facial-pain-scorer corpus can run.
3. **Welfare-asymmetric decision curve + one-sided-95%-NPV abstention curve**
   (`src/wrapper/`) — operating point at fixed pain-recall >= 0.90 with the
   undertreat:overtreat harm ratio swept as a range, plus a defer-to-vet boundary
   with finite-sample lower bounds (the word "guaranteed" is banned).

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
be green before any quantitative target runs).

**Kill/pivot:** if orbital/ear/head kappa lower bound falls below the Gate 1-B floor,
drop the 0-10 layer entirely and ship calibrated binary + abstention + confound audit;
the kappa-as-method and confound protocols still stand.

## Environment

uv-managed, Python 3.11 (system 3.9.6 is not used). Local = Apple M4 / MPS, **no
local CUDA**; CUDA is confined to `notebooks/colab_train_rfdetr.ipynb` for detector
training only. All code runs through `uv run`. See `pyproject.toml` and
`configs/*.yaml` (all hyperparameters and seeds live in configs, never hardcoded).
