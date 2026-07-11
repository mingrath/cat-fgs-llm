# Repo baseline extracts
## pyproject
[project]
name = "cat-fgs-llm"
version = "0.1.0"
description = "Binary-plus-wrapper feline pain decision support; VLM-as-AU-rater kappa + confound-attribution portable methods on a conceded DINOv2+CORN engine."
requires-python = ">=3.11,<3.13"
dependencies = [
  # --- engine (conceded plumbing) ---
  "torch>=2.4",                 # arm64 wheel -> MPS; NO cuda extras locally
  "torchvision>=0.19",
  "transformers>=4.44",         # DINOv2 ViT-S/14 (384-d) + timm fallback
  "timm>=1.0",
  "coral-pytorch>=1.4",         # corn_loss / corn_label_from_logits (5 per-AU heads)
  # --- supervision (VLM weak-label) ---
  "anthropic>=0.40",            # Claude 5-AU rater, Message Batches API, prompt caching
  "pydantic>=2.7",              # structured-output schema (enum [0,1,2])
  # --- wrapper + eval (THE HEADLINE FRAME) ---
  "scikit-learn>=1.5",          # StratifiedGroupKFold, PR-AUC
  "krippendorff>=0.6",          # inter-rater agreement (alongside QWK)
  "netcal>=1.3",                # ClasswiseECE (per-AU); NO binned diagram on the 0-10 sum
  "dcurves>=1.0",               # welfare-asymmetric decision curve (headline artifact)
  "mapie>=1.4,<2",              # risk-control NPV lower-bound (abstention); API pinned in src/wrapper/abstention.py
  "cleanlab>=2.7",              # confident-learning vet-triage
  "imagehash>=4.3",             # near-duplicate detector (NOT cat re-ID — G1)
  "opencv-python-headless>=4.10",
  "numpy>=1.26",
  "scipy>=1.13",
  "pandas>=2.2",
  "pyyaml>=6.0",
  # --- MLOps spine ---
  "wandb>=0.17",
  "roboflow>=1.1",
  "inference-sdk>=0.20",
  "python-dotenv>=1.0",
]

[dependency-groups]
detect = ["rfdetr>=1.6.0", "ultralytics>=8.3"]   # RF-DETR accepts device="mps"; YOLOv11s runner-up
dev    = ["pytest>=8.0", "ruff>=0.6", "pre-commit>=3.7"]

[tool.uv]
default-groups = ["dev"]       # `uv sync` installs dev; detect is opt-in (heavier)

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
markers = [
  "portable: portable protocols / deletion-safe / zero-torch isolation tests (P2)",
  "e2e: end-to-end gate orchestrator + synthetic full seq + wrapper (P1/P7)",
  "gate: gate blocking / order / artifact / power-first (G0-G6)",
  "enforcement: manifests / schema / dedup / power floor / deprecate fail paths (P7)",
  "real_manifests: tests requiring/ exercising committed manifests paths (not pure synthetic)",
]

[tool.ruff]
line-length = 100
## README
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
## Makefile targets
# cat-fgs-llm — thin task runner. One target per gate (strict run-order; each gate
# blocks downstream). Gate 4 (test) must be green before any quantitative target is
# believed. All code runs through `uv run`.
#
# Run order: gate0 -> gate1 -> gate2 -> gate3 -> gate4 (test) -> gate5 -> gate1b -> gate6
# Central enforcement via `make gate-pipeline` / `make gate-e2e-synthetic` or
# `python -m src.gates.orchestrator` (scripts-only delegation; --synthetic for toy/deletion-safe).
# G0 (power) always first — ties all quantitative to pre-registered vet budget.
# FreshPowerG0ManifestsEnforcer: all entry scripts (train_corn etc) now hard SystemExit if no
# data/manifests/power.json or placeholder vet_budget; folds/cache STRICT manifests/ load (no fallback);
# gate0 produces the committed one; dependents/CI fail without. (landed manifests + hard enforcement)

.PHONY: gate0 gate1 gate2 gate3 gate4 gate5 gate6 gate1b test features labels lint checkcites gate-pipeline gate-e2e-synthetic gate-orchestrate

# --- G0: power calcs + vet-budget integer (no data, no GPU; blocks all quantitative work)
gate0:
	uv run python scripts/gate0_power.py

# --- G1: per-CAT merge vs CAT_ ids (not raw CLIP/pHash); cat-disjoint folds
# Hardened (FreshFullGateWiringManifestsEnforcer): depend on gate0 + orch verify for full wiring (real scripts now call G0 guards too).
gate1: gate0
	uv run python -m src.gates.orchestrator --synthetic --portable-only || true  # light verify manifests contract (full: gate-pipeline)
	uv run python scripts/gate1_merge.py

# --- G2: one-directional capture-condition confound audit
gate2:
	uv run python scripts/gate2_confound.py

# --- G3: frozen hashed cat-disjoint hold-out + CI abort
gate3:
	uv run python scripts/gate3_holdout.py

# --- G4: MPS<->CPU logit parity + CORN decode->sum->0.39 unit driver (BLOCKING)
gate4:
	uv run python scripts/gate4_mps_check.py

# --- G5: alignment / NME / face-pixel-resolution audit
gate5:
	uv run python scripts/gate5_nme.py

# --- G6: severity-cell collapse (if AU=2 cells single-digit, collapse high end)
gate6:
	uv run python scripts/gate6_severity.py

# --- G1-B: VLM weak-label kappa pilot (fires on CI lower bound; self-justifies labeler)
gate1b:
	uv run python scripts/run_vlm_labels.py
	uv run python scripts/gate1b_kappa_pilot.py

# --- Gate-4 BLOCKING tests: CORN decode->sum->0.39 + MPS<->CPU parity + leakage guards
# Hardened wiring (per FreshFullGateWiringManifestsEnforcer): all gate-* now conceptually orch + G0 first + verify (real entries have G0 SystemExit tops too).
test: gate0
	uv run pytest tests/

# Portable seam isolation + deletion-safe (P2/P1 fix; zero-torch on protocols + synth gates)
# Add to CI: `make test-portable` (or include in `test`). Exercises src/protocols/ standalone.
# Hardened (FreshDeletionIsolationHardener): full surface + monkeypatch + e2e mocks (schema fail, dedup conflict, power placeholder, manifests missing, wrapper config mandatory) + --synthetic variants.
# CRITICAL PORTABLE CLAIM: "rm -rf src/model src/vlm src/wrapper; python -m src.protocols.standalone..." MUST still pass (guarded restore in CI; proves survives engine removal + reconstruct from md-only).
test-portable:
	uv run python -m src.protocols.standalone_test_corpus
	uv run pytest tests/test_au_order.py -q --tb=line -k "portable or deletion or isolation or au_order"
	# explicit full portable claim test (new monkeypatch sys.modules hide model/vlm/data)
	uv run pytest tests/test_au_order.py::test_portable_protocols_deletion_isolation_monkeypatch -q --tb=line
	# e2e expanded mocks coverage (Fresh wave)
	uv run pytest tests/test_e2e_gates_pipeline.py -q --tb=line -k "schema_fail or dedup_conflict or power_placeholder or manifests_missing or wrapper_config"
	# orchestrator synthetic variants (deletion-safe)
	# Non-blocking dry for local smoke only; CI jobs use strict (no || true) paths per FreshGateOrchE2ECI polish + CI honesty
	# Missing required artifacts now abort outside dry-run; dry-run smokes remain non-blocking for local developer speed.
	uv run python -m src.gates.orchestrator --synthetic --include-wrapper --dry-run || true
	uv run python -m src.gates.orchestrator --synthetic --portable-only --dry-run || true

# CI skeleton note: .github/workflows/ci.yml (absent; add) should run:
#   make lint && make test && make test-portable && make gate4 && (future gate-e2e-synthetic if no keys).
# gate-e2e-synthetic already in Makefile for deletion-safe toy pipeline (hits portable paths).

# --- frozen DINOv2 forward -> data/features/
features:
	uv run python scripts/cache_features.py

# --- Phase B VLM weak-label batch run (Gate 1-B pilot input)
labels:
	uv run python scripts/run_vlm_labels.py

# --- lint
lint:
	uv run ruff check .

# --- checkcites: detect undefined (\cite to a missing key) and unused (orphan bib
# entry) citations in the paper. Operates on the .aux/.bcf left behind by a build, so
# `make paper` (tectonic paper/main.tex) MUST run first; `checkcites paper/main` then
# reads paper/main.aux against references.bib. checkcites ships with TeX Live (install
# via tlmgr: `tlmgr install checkcites`); it is NOT a pip package.
checkcites:
	@command -v checkcites >/dev/null 2>&1 || { echo "checkcites not found — install via TeX Live: 'tlmgr install checkcites'"; exit 1; }
	@test -f paper/main.aux || { echo "paper/main.aux missing — run 'make paper' (tectonic paper/main.tex) first"; exit 1; }
	checkcites --undefined --unused paper/main

# pre-commit skeleton (FreshCI + manifests/power/schema/dedup guards; run after uv sync --dev)
# uv run pre-commit install ; uv run pre-commit run --all-files
# Strict (no ||echo) for CI honesty; exercises check-power-pre-reg + check-manifests-present + vlm/dedup local hooks (MCP local repo patterns credit).
pre-commit:
	# strict (no ||echo); direct python mimic of local custom (power@33/manifests exercised; avoids latent parse issue)
	uv run python -c "import json,sys; p=json.load(open('data/manifests/power.json')); assert 'vet_budget_integer' in p and isinstance(p['vet_budget_integer'],int) and p['vet_budget_integer']>=50,'power vet placeholder or too low'; [print('WARN power note placeholder') if 'placeholder' in str(p.get('note','')).lower() else None]; assert p.get('prevalence_assumed',0)>0,'power placeholder'; [assert f>0 for f in p.get('au_kappa_floors',{}).values()]; print('PASS: power pre-reg (non-placeholder G0) [mimic check-power-pre-reg]')"
	uv run python -c "import json,sys,pathlib; root=pathlib.Path('.'); [[ [p.exists() and p.stat().st_size>10 or (_ for _ in ()).throw(AssertionError('missing: '+m)), json.load(open(p))] for m,p in [(m,root/m)] ] for m in ['data/manifests/power.json','data/manifests/severity.json']]; print('PASS: core manifests present+valid [mimic]')"

# --- Central orchestrator (enforces order, writes run artifacts, aborts on fail; G0 first).
# python -m or make. --synthetic for deletion-safe toy e2e (hits G0-G6 + VLM pilot protocol + wrapper).
gate-orchestrate:
	uv run python -m src.gates.orchestrator

# Full pipeline (real data; requires prior manifests + keys for VLM parts)
gate-pipeline:
	uv run python -m src.gates.orchestrator

# Deletion-safe synthetic e2e (for CI/PR, repro P3/P4, test expansion; no real data/keys)
gate-e2e-synthetic:
	uv run python -m src.gates.orchestrator --synthetic --include-wrapper
## configs
### configs/confound.yaml
# Capture-condition confound-attribution protocol (portable method headline #2).
# ONE-DIRECTIONAL audit: a positive result reads "no confound detected at this power",
# never "CAT_01 is confounded".
direction: one_directional

features:
  fgs_bg_gap: true          # FGS-BG-Gap: foreground-vs-background gap feature (image-side)
  ebpg: true                # per-AU EBPG (energy-based pointing game) attribution (image-side)
  judge_bias: true          # VLM-rater judge-bias axis: |0/1/2 shift| + flip-rate under
                            # named prompt perturbations (position/verbosity/self_enhancement).
                            # Code path: src.eval.confound.judge_bias (consumes baseline +
                            # perturbed rater reruns); vet-free, not gated by the vet anchor.

# Trivial-classifier probe: brightness/blur/aspect (+ optional CLIP) predicting pain.
# NOT a config knob: the scalar feature list is fixed in scripts/gate2_confound.py
# (SCALAR_FEATURES), and CLIP columns ride the --use-clip flag (any column prefixed
# "clip_"). Listed here for documentation only — editing this file changes nothing.
### configs/corn.yaml
# ENGINE config (conceded plumbing, never claimed novel).
# Frozen DINOv2 ViT-S/14 (register variant default) + 5 per-AU CORN heads. Distributional decode is DEFAULT.
num_aus: 5                  # 5 per-AU CORN heads (one per FGS action unit)
K: 3                        # ordinal levels per AU -> {0,1,2}, so K-1 = 2 CORN logits per head

backbone:
  # torch.hub id consumed by src.model.backbone (reg = field default; plain = A/B comparator).
  # _reg registers suppress attention artifacts that hurt dense, localized per-AU FGS
  # features (orbital/ear/muzzle); A/B vs plain dinov2_vits14 before locking.
  # DINOv3: dinov3_vits16 (MCP: same forward_features x_norm_clstoken/patchtokens + get_intermediate_layers
  # for dense localized AU cues; frozen no-FT small data per context7).
  name: dinov2_vits14_reg
  frozen: true
  hidden_dim: 384
  patch_size: 14
  crop_edge: 518            # crop input edge (518 = 37 patches * 14)
  # A/B knob for richer patch per MCP dinov3 (mean_std default; l2 or intermed for orbital/ear localization; dinov3_vits16 primary successor per FINAL A/B + context7 'dense w/o FT' 'oob' for small/medical/facial/ordinal; exact forward x_norm + get_intermed n=list/L2/mean(patch))
  patch_mode: mean_std      # mean_std | l2 | intermed  (updates cache build + load pool support; patch_std richer for localized AU)
  layer: last               # or n=[-4,-3,-2,-1] (list/seq/range) via get_intermediate_layers (MCP n=... for multi-layer dense); dinov3_vits16 A/B primary (compat default v2_reg)
  # dinov3_vits16 promote: backbone load + corn A/B + e2e portable matrix probe (MCP 374 snips + landed contract)
  # Enforce richer (per DINOv3RicherEnforcer): intermed n=list -> mean+std concat in backbone extract + cache; L2 option; uniform no-legacy asserts + A/B hash in cache/train; dinov3_vits16 primary for dense oob frozen small-N imbal ordinal AU (orbital/ear/muzzle) vs priors no FM. Full schema embed + prov cross. 

decode:
  mode: distributional      # DEFAULT: P(rank>k)=cumprod(sigmoid(logits)) -> per-AU pmf -> convolve to 0-10 sum pmf
  # Hard argmax decode is used ONLY for the 0.39 point decision; NO binned reliability
  # diagram on the 11-atom 0-10 sum (RPS-on-sum + per-AU ClasswiseECE only).
  # The 0.39 itself is NOT configurable here: the single definition lives in
  # src/vlm/aggregate.py POINT_DECISION_THRESHOLD (Gate-4-pinned decode contract).

co_teach:
  num_gradual: 10           # epochs to ramp keep-rate 1.0 -> (1 - tau); canonical Han et al. 2018 R(T)
  # tau = estimated VLM noise rate. RUNTIME-OWNED by Gate-1-B: train_corn.py reads
  # est_noise_rate.overall from artifacts/gate1b/kappa_report.json (--tau overrides).
  # Deliberately NOT a config key — a placeholder here silently drops good rows.
  # vet-clean rows (is_vet_clean==1) are NEVER dropped from either selection.

# Head-training knobs (frozen-backbone linear probe, ~120-300 labels, M4/MPS).
# Literal defaults mirror IMPLEMENTATION_PLAN §5.6 table.
train:
  optimizer: adamw
  lr: 1.0e-3                # tiny linear probe; AdamW stable
  weight_decay: 1.0e-2
  epochs: 80               # 60-100; default 80. converges fast on cached feats
  batch_size: 32           # whole train set fits one batch on [N,384] in RAM
  dropout: 0.1             # light reg
  head_init_std: 2.0e-5    # trunc_normal_ std; small-data linear-probe stability
  num_workers: 0           # small data / MPS; deterministic loading
  pool: cls                # cls (default) vs patch_mean; cache holds both
  loss: corn               # corn (default) vs coral ablation
  train_folds: [0, 1, 2]   # cat-disjoint G3 folds used for training
  seeds: [0, 1, 2, 3, 4]   # 5 seeds; report mean +/- bootstrap CI (n is small)
  au_weights: null         # optional [1,1,1.5,1.5,1] to upweight muzzle/whiskers imbalance

# Sibling BINARY pain head (the v1 SPINE). Linear(384,1) + BCEWithLogitsLoss(pos_weight).
# Trained on y_pain, independent of CORN; this is the validated-claim path. The 0-10 CORN
# sum is INSPECTED-NOT-VALIDATED and never emits a validated-claim number.
binary_pain:
  pos_weight: null         # RUNTIME: set to n_neg/n_pos from the train split if null
### configs/crop.yaml
# Face crop & alignment preprocessing (Gate 5). IMPLEMENTATION_PLAN §3.
# All thresholds here are PROPOSED operating values: tuned ONCE on the Gate 5
# audit set then FROZEN before scoring, so the quality gate cannot be re-tuned
# to inflate prevalence (§3.2). Logic reads these; nothing is hardcoded in src.

# --- output crop contract (§3.0, fixed; do not vary per-image) ---
edge: 518                   # 518 = 14*37; divisible by DINOv2 patch size 14
patch: 14                   # DINOv2 ViT-S/14 patch; edge % patch == 0 asserted
# Only legal alternative edges if Gate 5 resolution fails: 518, 672 (14*48), 784 (14*56)
legal_edges: [518, 672, 784]

# --- §3.1 margin EXPANSION (never shrink) ---
expand: 0.135               # 13.5%, midpoint of the 12-15% band (BUILD_PLAN line 125)
height_expand_mult: 1.4     # asymmetric: extra top room for ear tips

# --- §3.2 frontal-pose / quality gate thresholds (route-to-vet) ---
quality_gate:
  yaw_ratio_min: 0.6        # (nose->L-eye)/(nose->R-eye); < min  -> route_vet (~ -25 deg yaw)
  yaw_ratio_max: 1.67       #                              > max  -> route_vet (~ +25 deg yaw)
  roll_abs_max_deg: 25.0    # in-plane tilt; aligner fixes <=25 deg, beyond is a pose problem
  ear_closed_min: 0.12      # eye-aspect-ratio (EAR); < min -> eyes-closed, orbital AU unreadable
  blur_var_min: 60.0        # variance-of-Laplacian; < min -> blur defer (tunable on audit set)
  # Haar coarse fallback localizer (detect-failure -> abstention; NOT the landmark path)
  haar_scale_factor: 1.1
  haar_min_neighbors: 3
  haar_min_size: [48, 48]

haar_cascade: models/cat-face-detector/haarcascade_frontalcatface.xml

# --- §3.4 2-point eye-similarity alignment (only if Gate 5 audit fails) ---
align:
  # canonical eye targets as a fraction of edge: eyes on a horizontal line,
  # interocular = 0.38*edge, centered, slightly high (§3.4).
  left_target:  [0.31, 0.42]
  right_target: [0.69, 0.42]
  interocular_frac: 0.38

# --- §3.3 Gate 5 NME / resolution audit ---
audit:
  n: 50                     # 30-50 CatFLW images
  seed: 5
  nme_pass_max: 0.08        # median NME_A <= 0.08 (<=8% of interocular) to pass
  gap_pass_max: 0.02        # gap(A-B) <= 0.02 to pass with box-as-cropper
  interocular_px_min: 40.0  # median inter-ocular distance >= 40 px AFTER resize to edge
  # TODO(eye-index): the L/R EYE-CENTER indices into the 48-point CatFLW landmark
  # array are NOT yet confirmed against the CatFLW index map (§3.3, §3.6 DoD).
  # The first ~6 points are the eye/canthi cluster; the values below are a
  # documented BEST-GUESS placeholder and MUST be pinned (and recorded into
  # reports/gate5_nme.json) before any NME is computed. Pinning is a gate-exit
  # blocker, not a runtime default to trust silently.
  left_eye_idx: 0           # PLACEHOLDER — pin against CatFLW index map
  right_eye_idx: 1          # PLACEHOLDER — pin against CatFLW index map
  eye_index_pinned: false   # set true only after visual confirmation on the audit set
### configs/detect_rfdetr.yaml
# Phase A detector — RF-DETR-Nano (champion). device="mps" supported; full training on Colab T4.
# Sweep/selection metric is PAIN RECALL, never mAP. Operating point = FIXED pain-recall >= 0.90.
model: RF-DETR-Nano
resolution: 512              # 576 if the T4 holds it at batch 4; must be in {384,512,576,704}
lr: 1.0e-4
lr_encoder: 1.5e-4           # backbone/encoder LR; drop to 7.5e-5 if unstable
epochs: 100                  # expect earlier convergence (early stopping below)
batch_size: 4
grad_accum_steps: 4          # effective batch = 16
weight_decay: 1.0e-4
use_ema: true
early_stopping: true
early_stopping_patience: 10
early_stopping_min_delta: 0.005   # 0.5% mAP; 0.001 is noise on this tiny val
early_stopping_use_ema: true
num_workers: 2
progress_bar: rich
tensorboard: true
seed: 42
# Class imbalance (metadata basis 264/1819, ratio ~2.6): rfdetr's train() exposes
# no per-class loss weight, so OVERSAMPLING (§2.6 rung 2) is the lever — there is
# deliberately no pos_weight knob here (an unwired knob invites a wasted sweep).
# Operating point: fixed pain-recall >= 0.90 (Evangelista anchor, NOT Youden-J/F1),
# hardcoded in src/detect/decision.py select_threshold — not configurable here.

resize: Fit                 # Roboflow export: Fit (reflect/white edges), NOT Stretch (distorts AU geometry)
### configs/detect_yolo.yaml
# Phase A detector — YOLOv11s runner-up. Sweep/selection metric is PAIN RECALL, never mAP.
model: yolo11s
resolution: 512
lr: 1.0e-3                  # SGD/ultralytics default base lr (optimizer=auto sets its own)
epochs: 100
batch_size: 16
patience: 25

# Operating point: fixed pain-recall >= 0.90 (Evangelista anchor, NOT Youden-J/F1),
# hardcoded in src/detect/decision.py select_threshold — not configurable here.

resize: Fit                # Fit (reflect/white edges), NOT Stretch (distorts AU geometry)
### configs/global.yaml
# Global run config — inherited by every gate. Seeds/paths live here, never hardcoded.
seed: 42
deterministic: true          # torch.use_deterministic_algorithms(True, warn_only=True); PYTHONHASHSEED=42
device: null                 # null = auto via src.model.device.DEVICE (mps if available, else cpu). NO local CUDA.
num_workers: 0               # small data; deterministic loading

project: cat-fgs-llm         # W&B project name

paths:
  # DEV ONLY (P3 leak fix): absolute machine path for editable dev runs.
  # In installed package / CI / portable use: omit or set relative/env; protocols/wrapper
  # now prefer importlib.resources + explicit config_path= (deprecate __file__ parents[2]).
  # Never hardcode in src/; consumers must tolerate missing/relative.
  repo_root: /Users/mingrath/ghq/github.com/mingrath/cat-fgs-llm  # GUARDED: dev-only; see src/protocols + wrapper seam
  data: data
  datasets: data/datasets
  features: data/features
  manifests: data/manifests
  external: data/external
  artifacts: artifacts
  figures: artifacts/figures
  tables: artifacts/tables
  model_cards: artifacts/model_cards
  reports: artifacts/reports
### configs/power.yaml
# GATE 0 — power calcs + vet-budget pre-registration (blocks all quantitative work).
#
# PLACEHOLDERS owned by Gate 0 / the clinician (replace before any quantitative claim):
#   - vet_budget_integer : the single committed integer — how many faces the vet scores.
#     ~120 is an UNCONFIRMED placeholder; Gate 0 writes the frozen value. TODO(Gate 0).
#   - min_pain_pos       : minimum pain-positive faces inside the budget (>= 50). TODO(Gate 0).
# Until Gate 0 commits these, no downstream kappa / NPV-LB / 0.39 number is believed.

kappa:
  ci_half_width_max: 0.15  # per-AU quadratic kappa CI half-width target (kappaSize); G0 gate

npv_lb:
  target: 0.90             # NPV lower-bound target (MAPIE / LTT) for the abstention curve
  max_abstention: 0.40     # certified at <= 40% abstention

# The 0.39 point-decision threshold is deliberately NOT a key here: gate0_power.py
# imports the single definition (src.vlm.aggregate.POINT_DECISION_THRESHOLD), so the
# pre-registered power calc can never drift from the operating threshold.

vet_budget_integer: 120    # PLACEHOLDER ~120 — TODO(Gate 0): commit the frozen integer
min_pain_pos: 50           # PLACEHOLDER >= 50 pain-positive faces — TODO(Gate 0)
### configs/splits.yaml
# Cross-validation splits. Group by INDIVIDUAL cat_id, NEVER by clip. (P4 cat vs clip)
# CAT_01 is carved out as a frozen LOIO hold-out; StratifiedGroupKFold runs on the rest.
# P3 config: all knobs here (power ref, multi-strat via group+y). Power-aware sim in folds.py.
cv:
  method: StratifiedGroupKFold
  n_splits: 5
  group: cat_id              # one cat across many clips -> ONE group (P4)
  strata: [y, cat_id]        # multi-strat (image-level pain y + cat_id); per context7 medical best practice
  random_state: 42
  shuffle: true

loio_holdout: CAT_01         # frozen leave-one-individual-out hold-out (carved out before CV)

prevalence_target: 0.127     # target pain prevalence per fold (264/1819 basis)
prevalence_tol: 0.05         # allowed |fold prevalence - target| (§1.6 assert)
min_pos_per_fold: 25         # minimum pain-positive images per fold (recomputed at runtime)

# P3 + G0 power-aware: referenced by src/data/folds for sim/print (small data floors)
power_ref: configs/power.yaml
multi_strat_reporting: true  # always emit distinct_pain_cats + per-fold N (aug copies excluded)
### configs/vlm_fgs.yaml
# VLM 5-AU weak-labeler (Gate 1-B). The VLM emits ONLY the 5 AU atoms in {0,1,2}.
# The 0-10 sum and the 0.39 flag are computed in code, never by the VLM.
model_id: claude-opus-4-8       # placeholder model id; pin exact id at run time
use_batch_api: true             # Message Batches API
enum: [0, 1, 2]                 # per-AU structured-output schema (pydantic enum)

rubric_hash: PLACEHOLDER_RUBRIC_SHA256   # byte-frozen rubric; any change invalidates prompt cache
prompt_caching: true            # log cache_read_input_tokens > 0

# AU order is fixed (the 5 Feline Grimace Scale action units)
au_order:
  - ear
  - orbital
  - muzzle
  - whiskers
  - head

# Gate 1-B fires on the CI LOWER BOUND of each per-AU quadratic kappa, not the point estimate.
# Interpretation guard (see src/eval/kappa.py): a high kappa measures weak-labeling
# capability ONLY if the vet anchor is independent of the rater AND the rubric handed to
# the VLM differs from the rubric the vet scored from; otherwise it measures
# rubric-following. A result is pending an independent per-AU vet anchor that does not yet
# exist (the dataset is binary pain/no_pain), so this ships as a runnable protocol.
# SINGLE SOURCE for the floors: gate0_power.py reads them from HERE and commits them
# into data/manifests/power.json, which gate1b_kappa_pilot.py enforces. Edit here only.
kappa_floors:
  orbital: 0.60
  ear: 0.60
  head: 0.60
  muzzle: 0.40
  whiskers: 0.40
### configs/wrapper.yaml
# THE FRAME — operating point / decision curve / abstention.
#
# PLACEHOLDERS owned by Gate 0 / the clinician (replace before any validated claim):
#   - harm_ratio_sweep.range : undertreat:overtreat ratio is a RANGE; no vet-elicited
#     point ratio exists. The clinician/Gate 0 confirms the endpoints. PLACEHOLDER.
#   - abstention.grid        : abstention-rate grid; confirm against the Gate-0 NPV-LB budget.
#
# Operating point = FIXED pain-recall >= 0.90 (Evangelista anchor), NEVER Youden-J/F1.
# Sens/spec at 0.39 estimated on VET-CONFIRMED labels ONLY (circularity firewall).
# Abstention = one-sided 95% NPV LOWER BOUND. The word "guaranteed" is BANNED.

operating_point:
  metric: pain_recall
  target: 0.90              # fixed pain-recall operating point (Evangelista anchor)
  circularity_firewall: vet_confirmed_only

harm_ratio_sweep:
  # undertreat:overtreat harm ratio swept as a RANGE (welfare-asymmetric decision curve).
  range: [1.0, 10.0]        # PLACEHOLDER endpoints — Gate 0 / clinician to confirm
  num: 19                   # PLACEHOLDER sweep granularity

abstention:
  grid: [0.0, 0.1, 0.2, 0.3, 0.4]   # PLACEHOLDER abstention-rate grid — confirm vs Gate-0 NPV-LB budget
  npv_target: 0.90         # one-sided 95% NPV LOWER BOUND target (not a "guarantee")
  ltt_delta: 0.05          # Learn-Then-Test / conformal error level
