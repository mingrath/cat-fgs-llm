# IMPLEMENTATION_PLAN.md — cat-fgs-llm

**Date:** 2026-06-12
**Status:** Build specification (authoritative for *how*, not *what*)

## Purpose

This document is the end-to-end engineering build spec for cat-fgs-llm: the repo/env scaffold, dataset acquisition and preparation, the binary pain detector (Phase A), face-crop/alignment preprocessing, the VLM weak-labeling + kappa pilot, the frozen-DINOv2 + CORN engine (Phase B), the trustworthiness wrapper, and the gated end-to-end pipeline with its evaluation protocol and build sequence. It exists to make the strategy in `FINAL_DIRECTION.md` executable by a solo developer on Apple M4 / MPS (Colab T4 for detector training only), with every quantitative claim routed through a strict, artifact-emitting gate order.

## Supersedes / relationship

- **`FINAL_DIRECTION.md` is the authoritative strategy** and governs every contested decision (the two portable-method headlines, the conceded engine, the binary-plus-wrapper spine, the inspected-not-validated graded layer, the operating point, the circularity firewall, the gate order, and the kill/pivot rules). Where this document and `FINAL_DIRECTION.md` disagree on *strategy*, `FINAL_DIRECTION.md` wins.
- **This document is the build spec** — it fixes file layout, environment pins, command sequences, code skeletons, and the exact artifacts each gate writes. `BUILD_PLAN.md` carries deltas that are folded in here.
- Nothing in this spec re-litigates strategy. The DINOv2+CORN engine is conceded plumbing throughout; the headline lives in the two portable methods (VLM-as-AU-rater per-AU quadratic kappa, gated on the CI lower bound; and the capture-condition confound-attribution protocol).

---

## Open assumptions to confirm with the user / vet

These are the environment, data, and budget assumptions this build spec rests on. Each must be confirmed (or its placeholder replaced) before the dependent section is executed; several are owned by a downstream gate and are placeholders here by design.

1. **Toolchain in-env (verified this session):** `uv` 0.11.19, arm64 macOS, and system Python 3.9.6 are confirmed present; Python 3.11 is the pinned interpreter via `uv` (system 3.9.6 is never used).
2. **Version floors are floors only.** Exact resolved versions are fixed by `uv.lock` at sync time and must be re-pinned (and `uv.lock` re-committed) if any floor is unavailable on arm64/py311.
3. **RF-DETR device support:** RF-DETR supports `device="mps"` / `accelerator="auto"` (confirmed via current `rfdetr` docs), but full detector training is still routed to Colab T4 for speed; the frozen-DINOv2 + CORN heads run on M4/MPS.
4. **Group key:** the `StratifiedGroupKFold` group key is the per-CAT id (`cat_id`), not raw clip/pHash. The downstream gate spec (G1) owns the exact merge; this spec only fixes the key name in `configs/splits.yaml`.
5. **Placeholders owned by G0 / clinician:** the vet-label budget integer (G0 output) and the harm-ratio sweep RANGE (`wrapper.yaml`) are placeholders here, set by G0 and the clinician respectively.
6. **`.env.example` additions:** `ANTHROPIC_API_KEY` / `WANDB_*` and the fork coordinates are net-new additions (the current file holds only the upstream `lia-k4jkv` coords); actual keys live only in the gitignored `.env`.
7. **Committed manifests:** `data/manifests/` must be un-ignored via `!data/manifests/` (verified absent in the current `.gitignore`) so fold CSVs and the hashed test manifest are committed while bulk data stays ignored.
8. **G0 replaces the vet-anchor placeholder.** The Gate 0 power calc REPLACES the ~120–150 image / ≥50-positive vet-anchor placeholder with the actual committed integer; the number used throughout is the pre-Gate-0 placeholder from `FINAL_DIRECTION`, not a fixed quantity.
9. **Fit re-export version string:** the `Fit`-resize re-export is generated as a NEW Roboflow version (e.g. `v2`) on the fork; the exact version string is assigned at generation time and must be recorded by the section consumers (training + W&B artifact sections). `<FIT_VERSION>` is the placeholder.
10. **Credentials present:** Kaggle credentials (`~/.kaggle/kaggle.json`) and a Roboflow private key scoped to `mingraths-workspace` are available; the key MUST be in `mingraths-workspace` for export/version calls to authorize.
11. **Fork project id:** the fork project id remains `cat-pain-ul7lu-p3rtl` as encoded in `scripts/download_dataset.py` and `.env`; if the actual fork id differs, `ROBOFLOW_PROJECT_FORK` governs.
12. **Optional assets off the critical path:** `facebook/dinov3-vits16` and `cvdl/catfaces` are optional and NOT on the v1 critical path; `dinov2-small` is the committed frozen backbone.
13. **Prevalence basis:** prevalence is reported on the 264/1819 metadata basis (~12.7%) with the 246/1811 live-annotation discrepancy (~12.0%) logged; 1819 is dataset-wide no_pain (FACTCHECK C81), not train-split; neither figure is an individual count.
14. **Fit version exists before export:** a Roboflow version exported with `Fit` resize (not Stretch-to-640) exists or will be generated before export; the placeholder `<FIT_VERSION>` must be filled in.
15. **G1 emits merged group ids:** Gate 1 (per-CAT merge against trusted `CAT_` ids) runs and emits the merged group ids that `build_folds.py` consumes; Phase A does not itself perform re-ID.
16. **G3 leak guard wired:** the Gate 3 hashing + CI-abort harness is wired so any train/sweep run that can read the test manifest aborts; the snippet only asserts disjointness.
17. **RF-DETR `num_classes` derivation:** `num_classes` must equal the class count implied by the COCO file's `category_id` scheme (verified docs: the head is sized from this and can require `max_id+1`, i.e. 3 for a 2-class 1-based file). The recipe derives it from `datamodule.class_names` rather than hard-coding 2; confirm the printed names/count before the first Colab run.
18. **RF-DETR `device` param:** RF-DETR exposes `device` as `'cuda'`/`'cpu'`/`'mps'` (verified in `rfdetr>=1.6.0` training params); the train call omits it to auto-detect CUDA on Colab and passes `device="mps"` only for the M4 smoke run. v1.6 also exposes a plural `devices="auto"`; either works — verify against the installed build.
19. **RF-DETR training knobs:** `lr_encoder`, `grad_accum_steps`, `use_ema`, `early_stopping`, `early_stopping_use_ema`, and the high-level `train(resume=...)` path are all valid in `rfdetr>=1.6.0` (verified). `early_stopping_min_delta` was raised 0.001 → 0.005 to match the documented 0.5%-mAP scale and avoid stopping on val noise.
20. **`pos_weight` basis:** the `pos_weight ≈ 2.6` basis (264/1819) is metadata box counts, not individuals; the live-annotation alternative (~2.7) is acceptable and should be labeled by basis when reported.
21. **Colab VRAM at higher resolution:** Colab T4 is assumed to hold resolution 576 at `batch_size=4`; if it OOMs, fall back to resolution 512 (the default in the recipe) before reducing batch size.

---

## Table of contents

- [0. Overview, deliverables & repo/env scaffold](#0-overview-deliverables--repoenv-scaffold)
  - [0.1 The modal deliverable](#01-the-modal-deliverable-plan-for-this-as-the-default-product)
  - [0.2 The two portable-method headlines](#02-the-two-portable-method-headlines)
  - [0.3 Repo directory layout](#03-repo-directory-layout)
  - [0.4 Python environment (uv-pinned; M4/MPS local)](#04-python-environment-uv-pinned-m4mps-local)
  - [0.5 MLOps spine — W&B + Roboflow](#05-mlops-spine--wb--roboflow)
  - [0.6 Reproducibility checklist](#06-reproducibility-checklist-every-gate-inherits-this)
  - [0.7 Build order baked into the layout](#07-build-order-baked-into-the-layout)
- [1. Datasets — what we need, how to get each, and how to prepare them](#1-datasets--what-we-need-how-to-get-each-and-how-to-prepare-them)
  - [1.0 Directory contract and namespacing](#10-directory-contract-and-namespacing-do-this-before-any-download)
  - [1.1 (a) Forked Roboflow cat-pain — the binary spine](#11-a-forked-roboflow-cat-pain--the-binary-spine)
  - [1.2 (b) The vet anchor — the ONLY graded-label source](#12-b-the-vet-anchor--the-only-graded-label-source)
  - [1.3 (c) Horse-grimace — decode unit-test fixture ONLY](#13-c-horse-grimace--decode-unit-test-fixture-only)
  - [1.4 (d) CatFLW — eye-alignment landmarks ONLY](#14-d-catflw--eye-alignment-landmarks-only)
  - [1.5 (e) Reusable HF / model assets](#15-e-reusable-hf--model-assets)
  - [1.6 Re-split protocol, frozen hold-out, and the gate-bound asserts](#16-re-split-protocol-frozen-hold-out-and-the-gate-bound-asserts)
  - [1.7 Data-provenance datasheet stub](#17-data-provenance-datasheet-stub-provenancedatasheetmd)
- [2. Phase A — binary pain detector (RF-DETR), training recipe & imbalance](#2-phase-a--binary-pain-detector-rf-detr-training-recipe--imbalance)
  - [2.0 Where it trains vs runs (compute split)](#20-where-it-trains-vs-runs-compute-split)
  - [2.1 Architecture decision](#21-architecture-decision-decisive)
  - [2.2 Data export & re-split](#22-data-export--re-split-do-this-first--gates-everything)
  - [2.3 FGS-safe augmentation](#23-fgs-safe-augmentation-light-geometry-no-vertical-flip-no-heavy-color)
  - [2.4 Training recipe — RF-DETR (PRIMARY)](#24-training-recipe--rf-detr-primary)
  - [2.5 Training recipe — YOLOv11s (RUNNER-UP)](#25-training-recipe--yolov11s-runner-up)
  - [2.6 Imbalance ladder](#26-imbalance-ladder-stack-cheapest-first-change-one-lever-per-run-threshold-last)
  - [2.7 Evaluation](#27-evaluation-pr-aucap-primary-never-accuracy-never-roc-auc-as-the-rank-metric)
  - [2.8 The box is a face-crop CANDIDATE — gated behind Gate 5](#28-the-box-is-a-face-crop-candidate--gated-behind-gate-5)
  - [2.9 Phase A definition of done](#29-phase-a-definition-of-done)
- [3. Face crop & alignment preprocessing (Gate 5)](#3-face-crop--alignment-preprocessing-gate-5)
  - [3.0 Inputs, assets, output contract](#30-inputs-assets-output-contract)
  - [3.1 Margin EXPANSION (~12–15%), not shrink](#31-margin-expansion-1215-not-shrink)
  - [3.2 Frontal-pose / quality gate](#32-frontal-pose--quality-gate-route-non-frontal-to-vet)
  - [3.3 Gate 5 audit — NME and resolution](#33-gate-5-audit--nme-inside-the-box-crop-vs-eye-aligned-crop-and-resolution)
  - [3.4 The 2-point eye-similarity alignment fallback](#34-the-2-point-eye-similarity-alignment-fallback)
  - [3.5 Production crop run + manifest](#35-production-crop-run--manifest-after-gate-5-passes)
  - [3.6 Definition of done (Gate 5 exit)](#36-definition-of-done-gate-5-exit)
- [4. VLM weak-labeling pipeline + the kappa pilot (Gate 1-B)](#4-vlm-weak-labeling-pipeline--the-kappa-pilot-gate-1-b)
  - [4.0 Files, env, deps](#40-files-env-deps)
  - [4.1 The forced-tool-use enum schema](#41-the-forced-tool-use-enum-schema-arize-aiphoenix-pattern)
  - [4.2 The rubric system prompt](#42-the-rubric-system-prompt-one-paragraph-per-au-verbatim-evangelista-012)
  - [4.3 In-code aggregation: sum 0–10 and the 0.39 flag](#43-in-code-aggregation-sum-010-and-the-039-flag-never-the-vlm)
  - [4.4 Bulk run: Message Batches API + prompt caching](#44-bulk-run-message-batches-api--prompt-caching-2040-images)
  - [4.5 N≥3 repeated runs → ordinal Krippendorff alpha](#45-n3-repeated-runs---ordinal-krippendorff-alpha-prometheus-eval-pattern)
  - [4.6 Gate 1-B: per-AU quadratic kappa vs vet](#46-gate-1-b-per-au-quadratic-kappa-vs-vet-gated-on-the-ci-lower-bound)
  - [4.7 Circularity firewall](#47-circularity-firewall-state-it-in-code-comments-and-the-paper)
  - [4.8 Anti-benchmark / reporting discipline](#48-anti-benchmark--reporting-discipline)
- [5. Phase B — model architecture & training method (frozen DINOv2 + 5 CORN heads)](#5-phase-b--model-architecture--training-method-frozen-dinov2--5-corn-heads)
  - [5.0 Prerequisites & gate position](#50-prerequisites--gate-position)
  - [5.1 Frozen DINOv2 ViT-S/14 backbone](#51-frozen-dinov2-vit-s14-backbone)
  - [5.2 FEATURE CACHING to disk](#52-feature-caching-to-disk-the-big-m4-win)
  - [5.3 The 5 CORN heads — architecture](#53-the-5-corn-heads--architecture)
  - [5.4 Vendored `corn_loss`](#54-vendored-corn_loss-auditable-40-lines-no-pip-dep)
  - [5.5 Distributional decode](#55-distributional-decode--per-au-pmf--convolve-to-the-010-sum-pmf)
  - [5.6 Training method — co-teaching small-loss](#56-training-method--co-teaching-small-loss-on-vlm-mass--clean-vet-anchor)
  - [5.7 MPS logit parity + Gate-4 decode unit test](#57-mps-logit-parity--gate-4-corn-decodesum039-unit-test-blocking)
  - [5.8 Outputs of Phase B](#58-outputs-of-phase-b-and-what-is--isnt-claimed)
- [6. The trustworthiness wrapper (the headline) — every metric, defined](#6-the-trustworthiness-wrapper-the-headline--every-metric-defined)
  - [6.0 Setup — environment, inputs, files](#60-setup--environment-inputs-files)
  - [6.1 Pillar 1 — VLM-as-AU-rater κ (HEADLINE METHOD #1)](#61-pillar-1--vlm-as-au-rater-κ-headline-method-1)
  - [6.2 Pillar 2 — Ordinal calibration (supporting)](#62-pillar-2--ordinal-calibration-supporting-the-010-layer--inspected-not-validated)
  - [6.3 Pillar 3 — Welfare-asymmetric DECISION CURVE (HEADLINE artifact)](#63-pillar-3--welfare-asymmetric-decision-curve-headline-wrapper-artifact)
  - [6.4 Pillar 4 — Defer-to-vet ABSTENTION](#64-pillar-4--defer-to-vet-abstention-one-sided-95-npv-lower-bound)
  - [6.5 Pillar 5 — Confound-attribution PROTOCOL (HEADLINE METHOD #2)](#65-pillar-5--confound-attribution-protocol-headline-method-2-portable)
  - [6.6 Run order & kill-criteria crosswalk](#66-run-order--kill-criteria-crosswalk)
- [7. Gates, evaluation protocol, end-to-end pipeline & build sequence](#7-gates-evaluation-protocol-end-to-end-pipeline--build-sequence)
  - [7.0 Repository layout & conventions](#70-repository-layout--conventions)
  - [7.1 Gates G0–G6 + G1-B — ordered checklist](#71-gates-g0g6--g1-b--ordered-checklist)
  - [7.2 Full evaluation protocol](#72-full-evaluation-protocol)
  - [7.3 End-to-end pipeline diagram](#73-end-to-end-pipeline-diagram)
  - [7.4 Risk-first build sequence / timeline](#74-risk-first-build-sequence--timeline-solo-dev-m4)
  - [7.5 Release artifacts + datasheet](#75-release-artifacts--datasheet)
  - [7.6 Modal-product statement](#76-modal-product-statement)
- [Definition of done for v1](#definition-of-done-for-v1)

---

## 0. Overview, deliverables & repo/env scaffold

This section is the engineering ground floor: what the repo ships, how the directories map to the build sequence, the exact `uv`-pinned Python environment (M4/MPS local, CUDA confined to Colab), the W&B + Roboflow MLOps spine, and a reproducibility contract every downstream gate inherits. It fixes *what to build and lay out*; strategy is settled in `FINAL_DIRECTION.md` and not re-litigated here.

**One framing rule that governs every file in this repo:** the DINOv2+CORN engine is **plumbing, conceded, never claimed novel**. The headline lives in two *portable methods* (0.2). Directory names, module docstrings, artifact tags, and W&B run names must never imply the engine is the contribution.

### 0.1 The modal deliverable (plan for this as the default product)

Lay out the repo for the **modal outcome**, not the optimistic one. The shippable product is a **binary-plus-wrapper** vertical with six named components. The abstract must hold with component 6 dropped, and the word **"graded" is struck from every validated-claim sentence**:

| # | Component | What it is | Where it lives | Validated? |
|---|---|---|---|---|
| 1 | **Calibrated binary pain decision** | RF-DETR/YOLO detector → per-image pain decision at a **fixed pain-recall ≥0.90** operating point (Evangelista anchor), **never** Youden-J/F1 | `src/detect/`, `src/wrapper/operating_point.py` | YES — vet-confirmed labels only (circularity firewall) |
| 2 | **κ-as-method** | Per-AU VLM-vs-vet quadratic weighted kappa with **CI lower bound**; the *protocol* is the deliverable, CAT_01 numbers instantiate it. QWK-vs-VLM is **never** validation of the decision | `src/vlm/`, `src/eval/kappa.py` | YES (headline #1) |
| 3 | **Confound-attribution protocol** | FGS-BG-Gap + per-AU EBPG, **one-directional** ("no confound detected at this power") | `src/eval/confound.py` | YES (headline #2) |
| 4 | **Welfare-asymmetric decision curve** | dcurves net-benefit, sweeping undertreat:overtreat as a **range** (no vet-elicited point ratio) — the **headline wrapper artifact, not ECE** | `src/wrapper/decision_curve.py` | YES |
| 5 | **LB-abstention curve** | One-sided **95% NPV lower bound** at each abstention rate; the word **"guaranteed" is banned**. Demoted to exploratory if the G0 power budget for the NPV-LB is not met | `src/wrapper/abstention.py` | YES (exploratory if G0 budget unmet) |
| 6 | **Inspected graded-CORN** | 5 per-AU CORN heads → 0–10 sum → distributional pmf; shipped **inspected-not-validated** | `src/model/`, `src/eval/distributional.py` | **NO — never a validated claim** |

Component 6 is gated, labeled, and walled out of every validated results table. Its outputs feed only `src/eval/distributional.py` for inspection, never a headline number.

### 0.2 The two portable-method headlines

These are the only claimed contributions; both are **dataset-agnostic** and ship as released, runnable code:

1. **VLM-as-AU-rater κ protocol + result** (`src/vlm/` + `src/eval/kappa.py`): scores any face corpus's per-AU VLM labels against a vet anchor, reports 5 quadratic κ with **CI lower bounds**. Forward-looking method finding ("can a frozen VLM weak-label feline FGS AUs at human-rater agreement"), **not** an audit of CAT_01's specific labels. Fires on the **CI lower bound** (Gate 1-B). The labeler choice is **self-justified by this pilot**; no external weak-label citation is invoked to justify it.
2. **Capture-condition confound-attribution protocol** (`src/eval/confound.py`): FGS-BG-Gap + per-AU EBPG, a **one-directional** audit any future facial-pain-scorer corpus can run. Deliverable is the *protocol*, not "CAT_01 is confounded."

Calibration / RPS / ClasswiseECE / MAPIE are **supporting evidence**, never headlines. There is **no binned reliability diagram on the 11-atom 0–10 sum** anywhere in the repo (distributional path uses RPS-on-sum + per-AU ClasswiseECE only).

### 0.3 Repo directory layout

```text
cat-fgs-llm/
├── README.md                      # one-paragraph thesis + "engine is conceded plumbing" disclaimer
├── FINAL_DIRECTION.md             # authoritative (exists)
├── BUILD_PLAN.md  DATA_DECISION.md  HF_SOLUTION.md  GITHUB_MINE.md  FACTCHECK.md  (exist)
├── pyproject.toml                 # uv-managed; [project] + [tool.uv] + [dependency-groups]
├── uv.lock                        # committed; the reproducibility anchor
├── .python-version                # "3.11"  (uv reads this; NOT system 3.9.6)
├── .env.example                   # exists — Roboflow coords; add ANTHROPIC/WANDB + fork coords
├── .gitignore                     # exists — ignores datasets/ data/ models/ wandb/ runs/; add !data/manifests/
├── Makefile                       # thin task runner: make gate0 / gate1 / features / labels ...
│
├── configs/                       # ALL hyperparameters/seeds live here, never hardcoded
│   ├── global.yaml                # seed=42, deterministic=true, device=mps, paths
│   ├── splits.yaml                # StratifiedGroupKFold(5), group=cat_id, random_state=42
│   ├── detect_rfdetr.yaml         # RF-DETR-Nano @512, lr/encoder-lr, pain-recall>=0.90 target
│   ├── detect_yolo.yaml           # YOLOv11s runner-up
│   ├── vlm_fgs.yaml               # model id, batch API, cached rubric hash, enum [0,1,2]
│   ├── corn.yaml                  # 5 AUs, frozen DINOv2 ViT-S/14, distributional decode
│   ├── wrapper.yaml               # operating point (pain-recall>=0.90), harm-ratio sweep RANGE, abstention grid
│   ├── confound.yaml              # FGS-BG-Gap + EBPG features, one-directional thresholds
│   └── power.yaml                 # G0: kappa N / NPV-LB(MAPIE) / 0.39-CI vet-budget integer
│
├── data/                          # gitignored; manifests committed under data/manifests/
│   ├── datasets/                  # Roboflow fork export; horse-grimace = DECODE SCAFFOLD only  (gitignored)
│   ├── features/                  # cached frozen DINOv2 features .npy/.pt  (gitignored)
│   ├── manifests/                 # COMMITTED: fold CSVs, hashed test ids, cat_id map, vet-budget
│   └── external/                  # CatFLW (alignment only) / Zhang-archive, OUTSIDE datasets/  (gitignored)
│
├── src/
│   ├── data/                      # parse cat_id, StratifiedGroupKFold, leakage asserts (G1/G3)
│   ├── detect/                    # Phase A: RF-DETR/YOLO train + per-image pain decision
│   ├── crop/                      # box→margin-expand→eye-align (G5), face crop for Phase B
│   ├── vlm/                       # Claude 5-AU weak-labeler (Batches API, cached rubric, enum schema)
│   ├── model/                     # frozen DINOv2 ViT-S/14 + 5 CORN heads (ENGINE — conceded plumbing)
│   ├── wrapper/                   # operating_point / decision_curve / abstention (THE FRAME)
│   └── eval/                      # kappa / confound / distributional (RPS+ClasswiseECE) / bootstrap CI
│
├── scripts/                       # thin CLI entrypoints (argparse), one per gate/stage
│   ├── download_dataset.py        # exists — Roboflow fork v1 export
│   ├── gate0_power.py             # power calcs + vet-budget integer  (no data, no GPU)
│   ├── gate1_merge.py             # per-CAT merge: validate vs CAT_ ids, not raw CLIP/pHash
│   ├── gate2_confound.py          # one-directional capture-condition audit
│   ├── gate3_holdout.py           # frozen hashed cat-disjoint hold-out + CI abort
│   ├── gate4_mps_check.py         # MPS↔CPU logit parity + CORN decode→sum→0.39 unit driver
│   ├── gate5_nme.py               # alignment / NME / face-pixel-resolution audit
│   ├── cache_features.py          # frozen DINOv2 forward → data/features/
│   ├── run_vlm_labels.py          # Phase B weak-label batch run (Gate 1-B pilot input)
│   └── train_corn.py              # 5 CORN heads on cached features
│
├── tests/                         # pytest; Gate 4 lives here and is BLOCKING in CI
│   ├── test_corn_decode.py        # synthetic logits → 0/1/2 → 0–10 sum → 0.39 (BLOCKING)
│   ├── test_group_split.py        # one cat across clips → ONE group; folds cat-disjoint
│   ├── test_mps_parity.py         # DINOv2 MPS vs CPU logit parity
│   └── test_no_test_leak.py       # CI abort: no train/sweep can read hashed test manifest
│
├── notebooks/                     # Colab T4 ONLY (CUDA-confined): detector training
│   └── colab_train_rfdetr.ipynb
│
└── artifacts/                     # gitignored run outputs; figures/tables/model cards
    ├── figures/   tables/   model_cards/   reports/
```

`data/`, `models/`, `datasets/`, `wandb/`, `runs/`, `outputs/` are already in `.gitignore`. **Committed exceptions** (reproducibility-critical small text): `data/manifests/` (fold CSVs, hashed test ids, `cat_id` map, vet-budget integer), `uv.lock`, `configs/`. Add `!data/manifests/` to `.gitignore` (verified absent today).

### 0.4 Python environment (uv-pinned; M4/MPS local)

System Python is **3.9.6 — do not use it**. `uv` is **0.11.19** (verified in-env). Pin Python 3.11 (best torch+MPS wheel support, broad lib compat).

```bash
# from repo root
uv python install 3.11
uv init --no-readme --python 3.11        # writes pyproject.toml + .python-version
uv sync                                   # creates .venv, resolves, writes uv.lock
```

`pyproject.toml` core dependencies (local M4 group has **no CUDA**; PyTorch arm64 wheels ship MPS):

```toml
[project]
name = "cat-fgs-llm"
requires-python = ">=3.11,<3.13"
dependencies = [
  # --- engine (conceded plumbing) ---
  "torch>=2.4",                 # arm64 wheel → MPS; NO cuda extras locally
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
  "mapie>=0.9",                 # LTT / conformal NPV lower-bound (abstention)
  "cleanlab>=2.7",              # confident-learning vet-triage
  "imagehash>=4.3",             # near-duplicate detector (NOT cat re-ID — G1)
  "opencv-python-headless>=4.10",
  "numpy>=1.26", "scipy>=1.13", "pandas>=2.2", "pyyaml>=6.0",
  # --- MLOps spine ---
  "wandb>=0.17",
  "roboflow>=1.1", "inference-sdk>=0.20",
  "python-dotenv>=1.0",
]

[dependency-groups]
detect = ["rfdetr>=1.6.0", "ultralytics>=8.3"]   # RF-DETR accepts device="mps"; YOLOv11s runner-up
dev    = ["pytest>=8.0", "ruff>=0.6"]

[tool.uv]
default-groups = ["dev"]       # `uv sync` installs dev; detect is opt-in (heavier)
```

```bash
uv sync --group detect        # add RF-DETR / ultralytics when starting Phase A
uv run python scripts/gate4_mps_check.py   # all code runs via `uv run`
```

> **Version floors are floors.** Exact resolved versions are fixed by `uv.lock` at sync time. If any floor (`mapie>=0.9`, `dcurves>=1.0`, `netcal>=1.3`, `coral-pytorch>=1.4`, `rfdetr>=1.6.0`) proves unavailable on arm64/py311, re-pin and re-commit `uv.lock`.

**CUDA confinement.** `bitsandbytes` 4-bit on macOS arm64 is impractically slow → **not in the local env at all**. Local VLM use is **hosted-API inference-only** (no local VLM training). If a local fine-tune is ever needed, use `mlx-vlm` (`uv add mlx-vlm`), never bnb. **`bitsandbytes` and any `--extra-index-url .../cu121` torch pin live ONLY in `notebooks/colab_train_rfdetr.ipynb`** (a `!pip install` cell), never in `pyproject.toml`. RF-DETR *can* train on MPS (`device="mps"` is a supported parameter; `accelerator="auto"` falls GPU→MPS→CPU), but the M4 is too slow for full detector training, so **detector training is run on Colab T4**; the frozen-DINOv2 forward + 5 CORN heads run locally on MPS.

```python
# src/model/device.py — single source of truth for device
import torch
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"   # never assume cuda locally
```

### 0.5 MLOps spine — W&B + Roboflow

**Roboflow = immutable dataset system-of-record. W&B = runs / sweeps / Artifacts.** Mirror every Roboflow version id into a W&B Artifact.

- **Roboflow:** forked binary set `mingraths-workspace/cat-pain-ul7lu-p3rtl` v1 (added to `.env` as `ROBOFLOW_PROJECT_FORK`). Export with **`Fit (reflect/white edges)`, NOT Stretch-to-640** (stretch distorts AU geometry). Re-generate a `Fit` version before any geometry-sensitive work. Key stays in gitignored `.env` only; never pasted in chat.
- **W&B:** every run tagged with **`git_sha` + `roboflow_version`**; Colab runs use `resume="allow"` for preemption. Sweep `metric` = **pain recall**, never mAP. Champion locked via `sweep.best_run().config`.

```python
import subprocess, wandb
git_sha = subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
run = wandb.init(
    project="cat-fgs-llm",
    config={"seed": 42, "git_sha": git_sha,
            "roboflow_version": "cat-pain-ul7lu-p3rtl:v1",
            "deterministic": True, "device": "mps"},
    tags=[f"git:{git_sha}", "roboflow:v1", "gate4-passed"],
    resume="allow",
)
art = wandb.Artifact("roboflow-cat-pain", type="dataset",
                     metadata={"roboflow_version": "cat-pain-ul7lu-p3rtl:v1", "resize": "Fit"})
run.log_artifact(art)
```

`.env.example` additions (do not commit `.env`): `ANTHROPIC_API_KEY=`, `WANDB_API_KEY=`, `WANDB_ENTITY=`, `ROBOFLOW_WORKSPACE_OWNED=mingraths-workspace`, `ROBOFLOW_PROJECT_FORK=cat-pain-ul7lu-p3rtl`. (Current `.env.example` holds only the upstream `lia-k4jkv/cat-pain-ul7lu` coords; the fork coords and the two new keys are net-new.)

### 0.6 Reproducibility checklist (every gate inherits this)

| Item | Implementation | Where |
|---|---|---|
| **Single seed = 42** | one `seed_everything()` (python/numpy/torch/mps); never re-seed ad-hoc | `configs/global.yaml`, `src/data/seed.py` |
| **Deterministic mode** | `torch.use_deterministic_algorithms(True, warn_only=True)`; `PYTHONHASHSEED=42`; `num_workers=0` on small data | `src/data/seed.py` |
| **`git_sha` + `roboflow_version` on every run** | injected into W&B `config` + `tags` (above) | all `scripts/*.py` |
| **Locked env** | `uv.lock` + `.python-version` committed; runs go through `uv run` | repo root |
| **Frozen, hashed test manifest** | `data/manifests/test_ids.sha256`; CI aborts any train/sweep that can read it (G3) | `tests/test_no_test_leak.py` |
| **Gate 4 unit tests BLOCKING in CI** | CORN decode→sum→0.39 + MPS↔CPU parity must pass before any metric is believed | `tests/`, `Makefile` |
| **Augmented copies never in reported N** | always print **distinct-pain-CAT denominator** + Clopper-Pearson/bootstrap CI; never "we beat 77/79/95%" | `src/eval/bootstrap.py` |
| **Cached-rubric hash pinned** | VLM rubric byte-frozen; log `cache_read_input_tokens>0`; any byte change invalidates cache | `configs/vlm_fgs.yaml`, `src/vlm/` |
| **Sens/spec firewall** | 0.39 sens/spec estimated ONLY on vet-confirmed labels; QWK-vs-VLM never enters validation | `src/wrapper/operating_point.py` |

```python
# src/data/seed.py
import os, random, numpy as np, torch
def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
```

### 0.7 Build order baked into the layout

The directory/script layout encodes the **strict gate run-order** (each gate blocks downstream): **G0** `gate0_power.py` (power calcs + vet-budget integer; blocks all quantitative work) → **G1** `gate1_merge.py` (per-CAT merge vs `CAT_` ids, not raw CLIP/pHash) → **G2** `gate2_confound.py` (one-directional capture-condition audit) → **G3** `gate3_holdout.py` (frozen hashed cat-disjoint hold-out + CI abort) → **G4** `gate4_mps_check.py` (MPS↔CPU logit parity + CORN decode→sum→0.39, **BLOCKING**) → **G5** `gate5_nme.py` (alignment / NME) → **G1-B** κ pilot (`run_vlm_labels.py` + `eval/kappa.py`, fires on **CI lower bound**, self-justifies the labeler) → **G6** severity-cell collapse (if AU=2 cells single-digit, collapse high end).

A `Makefile` target per gate makes the order executable: `make gate0`, `make gate1`, …, plus `make test` (Gate 4 must be green before any quantitative target runs). Nothing in `src/model/` (the conceded engine) produces a claimed-novel artifact — its outputs feed `src/wrapper/` (the frame) and `src/eval/` (the two portable methods). **Kill/pivot:** if orbital/ear/head κ-LB falls below the G1-B floor, drop graded entirely and ship calibrated binary + abstention + confound audit; κ-as-method and the confound protocol still stand.

---

## 1. Datasets — what we need, how to get each, and how to prepare them

> **Read first (constraints that govern every line below).** The DINOv2+CORN **engine is conceded plumbing** — no dataset choice here is sold as novel. The **v1 spine is binary pain/no_pain**; the 0–10 layer is built but ships **inspected-not-validated** (the word "graded" never appears in a validated-claim sentence). The **only graded-label source is the vet anchor** (§1.2); every other set on this page is *scaffolding, preprocessing, or unit-test fixture* and **never produces a headline number**. The operating point is **fixed pain-recall ≥0.90** (Evangelista anchor — not Youden-J/F1) and abstention is a **one-sided 95% NPV lower-bound** curve; those are downstream of this section but they dictate that the **test set must be cat-disjoint, hashed, and frozen (Gate 3)** and that **augmented copies never enter any reported N**. Run order is gated: **G0 (power/budget) → G1 (per-CAT merge) → G2 (confound audit) → G3 (frozen hold-out) → G4 (MPS correctness) → G5 (alignment/NME) → G1-B (κ pilot) → G6 (severity collapse)**. Datasets must be in place to feed each gate in that order.

### 1.0 Directory contract and namespacing (do this before any download)

Every source lives in its own namespace so a CC0 / CC BY-NC / horse / mouse image can **never** leak into a cat-pain loader. The pain export tree is the **only** thing `datasets/cat-pain/` ever contains.

```bash
ROOT=/Users/mingrath/ghq/github.com/mingrath/cat-fgs-llm
cd "$ROOT"
mkdir -p datasets/cat-pain        # (a) forked Roboflow binary — the spine
mkdir -p datasets/vet-anchor      # (b) the ONLY graded-label source (§1.2)
mkdir -p datasets/horse-grimace   # (c) decode→sum→threshold UNIT-TEST fixture only
mkdir -p datasets/catflw          # (d) landmarks for eye-alignment ONLY
mkdir -p models/dinov2-small      # (e) frozen backbone
mkdir -p models/cat-face-detector # (e) Haar cascade fallback cropper
mkdir -p datasets/_archive        # Zhang-2008 fallback aligner — OUTSIDE datasets/cat-pain, never a loader root
mkdir -p splits provenance        # fold CSVs, hashed test manifest, datasheet
```

Add to `.gitignore` (already present in repo): `datasets/`, `models/`, `.env`. **Licenses must not co-mingle** — keep a one-line license stamp file in each source dir (see §1.7 datasheet).

| Source | Dir | Role in pipeline | License | Carries pain/AU labels? |
|---|---|---|---|---|
| (a) Forked Roboflow cat-pain | `datasets/cat-pain/` | **Binary spine** (train detector + crop source) | CC BY 4.0 | Binary pain/no_pain boxes only |
| (b) Vet anchor (n FIXED BY G0) | `datasets/vet-anchor/` | **Only graded source** (κ pilot + 0.39 validation) | derived from (a) | per-AU 0/1/2 (vet-confirmed) |
| (c) Horse-grimace | `datasets/horse-grimace/` | **Unit-test** decode→sum→threshold (Gate 4) | CC BY 4.0 | 3/5 AUs 0/1/2 (HORSE) |
| (d) CatFLW | `datasets/catflw/` | **Eye-alignment / NME audit** (Gate 5) | CC BY-NC 4.0 | 48 landmarks, no pain |
| (e) facebook/dinov2-small | `models/dinov2-small/` | Frozen encoder (conceded plumbing) | Apache-2.0 | n/a |
| (e) Haar `frontalcatface` | `models/cat-face-detector/` | Fallback cropper / abstention signal | MIT | n/a |
| Zhang-2008 archive (fallback) | `datasets/_archive/` | Fallback 2-pt eye-aligner **only if CatFLW blocked** | CC0 (Kaggle) | 9 coarse landmarks, no pain |

---

### 1.1 (a) Forked Roboflow cat-pain — the binary spine

**Purpose.** The one in-domain corpus. Trains the binary pain detector (Phase A) and supplies the **tight head crops** that feed the frozen DINOv2 encoder (Phase B). ~2040 records / ~1547 distinct images, ~12–13% prevalence. This is the corpus the headline methods are *instantiated on*; it is **not** itself a contribution.

**License.** CC BY 4.0 (project metadata, FACTCHECK confirmed). Releasable — but keep it walled from the CC BY-NC CatFLW images so a derived release stays CC BY.

**What labels it carries.** Binary `pain` / `no_pain` bounding boxes, nothing else. **Zero** per-AU / 0–10 / FGS labels. The negative class is an *unknown mixture* (possibly sedated / post-recovery cats) and must be documented as such.

**Load-bearing facts (FACTCHECK-corrected — do not regress these):**
- **264** = a *pain-box* metadata count, **NOT** an individual count. Distinct pain individuals are not computable without re-ID. `CAT_01` (84 clips) is the **lone known camera id**. Live-annotation sum gives 246/1811 (~12.0%); metadata gives 264/1819 (~12.7%). **Pin every exact derived figure to the metadata basis 264/1819 (~12.7% prevalence, ~6.9:1 imbalance)** and log the 246/1811 discrepancy; **1819 is a dataset-wide no_pain count, not train-split** (FACTCHECK C81).
- **Leaky shipped split:** **191 of 336 source clips (56.8%)** appear in both train and valid (full-2040 manifest figure; the older 201/337 was a 76%-sample extrapolation — retired). Discard the shipped split.
- **493 exact-duplicate filenames** (2040 records / **1547 distinct**). Must be de-duplicated on export so a duplicate cannot straddle the split.
- **135 of 336 clips are class-mixed** → you cannot collapse a clip to a single label; stratify at **image level with grouping**.
- v1 export uses **Stretch-to-640** which horizontally inflates ~4:3 faces and **distorts AU geometry** — must re-export with `Fit`.

#### Step 1 — Fork into `mingraths-workspace`

The source is `lia-k4jkv/cat-pain-ul7lu` (Universe). The fork target is `mingraths-workspace/cat-pain-ul7lu-p3rtl` (already referenced in `scripts/download_dataset.py`; `ROBOFLOW_PROJECT_FORK` in `.env` governs if the actual id differs). `.env` is seeded with the *source* coords; the fork coords come from `ROBOFLOW_WORKSPACE_OWNED` / `ROBOFLOW_PROJECT_FORK`.

```bash
# .env (copy from .env.example; NEVER commit)
ROBOFLOW_API_KEY=<your_private_key>        # app.roboflow.com/settings/api
ROBOFLOW_WORKSPACE=lia-k4jkv               # source (for reference)
ROBOFLOW_PROJECT=cat-pain-ul7lu
ROBOFLOW_WORKSPACE_OWNED=mingraths-workspace
ROBOFLOW_PROJECT_FORK=cat-pain-ul7lu-p3rtl
ROBOFLOW_VERSION=1
```

Fork via the Roboflow MCP tool `projects_fork` (source workspace `lia-k4jkv`, project `cat-pain-ul7lu`, into `mingraths-workspace`) **[Claude]**, or in the UI **[You]**: open `universe.roboflow.com/lia-k4jkv/cat-pain-ul7lu` → **Fork Project** → destination `mingraths-workspace`. Confirm with `projects_get` that `cat-pain-ul7lu-p3rtl` exists and reports ~2040 source images.

> **Note on key scope (MEMORY):** the API key is scoped to `mingraths-workspace`; you must fork into that workspace before any export/version call will authorize.

#### Step 2 — Generate a NEW version with `Fit`, NOT Stretch

The shipped v1 is unusable for geometry (Stretch). Generate a fresh immutable version with **Fit (reflect edges)** resize and **no augmentation baked into the export** (augment in the training loop, never in the stored val/test data — augmented copies must never enter a reported N).

Via MCP `versions_generate` on `cat-pain-ul7lu-p3rtl` **[Claude]**, with a preprocessing config equivalent to:

```jsonc
// version generate settings — the load-bearing bits
{
  "preprocessing": {
    "resize": { "width": 640, "height": 640, "format": "Fit (reflect edges)" }  // NOT "Stretch to"
  },
  "augmentation": {}   // EMPTY at export. No stored augmentation. Augment in-loop only.
}
```

If the UI is easier **[You]**: `app.roboflow.com/mingraths-workspace/cat-pain-ul7lu-p3rtl/generate` → Resize → **Fit (reflect edges)** 640×640 → Augmentation: **None** → Generate. Record the resulting version string (e.g. `v2`) — it goes in the datasheet and in every W&B artifact (the exact string is assigned at generation time).

> **Why reflect/white, not stretch:** the orbital/muzzle/whiskers AUs are *geometry-sensitive*; horizontal stretching of 4:3 faces inflates box aspect ratios and corrupts exactly the signal the CORN heads must read. Reflect padding preserves aspect; white padding is the acceptable alternative if reflect introduces mirror artifacts at the face edge.

#### Step 3 — Export (COCO for parsing; YOLO for detector training)

```bash
# COCO export — needed for the filename→group_id parse and the duplicate audit
python scripts/download_dataset.py --format coco --version 2 \
  --location "$ROOT/datasets/cat-pain/coco"

# YOLO export — for the RF-DETR / YOLOv11 detector training (Colab T4)
python scripts/download_dataset.py --format yolov8 --version 2 \
  --location "$ROOT/datasets/cat-pain/yolo"
```

(`scripts/download_dataset.py` already reads the fork coords from `.env`; pass `--version 2` to pull the new `Fit` version.)

#### Step 4 — Parse `group_id` from filenames (the regex)

Strip the Roboflow suffix `_png.rf.<hash>.jpg` to the stem, then map to a group. **Key on the filename stem ONLY — never a parent folder named `CAT`** (that neutralizes the phantom Zhang-archive `CAT_00..06` folder collision).

```python
import re
# Roboflow suffix: "<stem>_png.rf.<32hex>.jpg"  → recover <stem>.png first.
SUFFIX = re.compile(r"_png\.rf\.[0-9a-f]+\.jpg$", re.IGNORECASE)
CAM    = re.compile(r"^CAT_(\d{2})_(\d{8})_(\d{3})\.png$")   # CAT_01_00000100_007.png
PLAIN  = re.compile(r"^(\d{8})_(\d{3})\.png$")                # 00000100_007.png

def group_id(fname: str) -> str:
    stem = SUFFIX.sub(".png", fname)            # -> "<stem>.png"
    m = CAM.match(stem)
    if m:
        cam, clip, _frame = m.groups()
        return f"CAT{cam}_{clip}"               # namespaced camera clip
    m = PLAIN.match(stem)
    if m:
        clip, _frame = m.groups()
        return f"P_{clip}"                       # plain clip
    raise ValueError(f"unparseable filename: {fname!r}")
```

**Unit assertions (must pass before any split is built):**
- `group_id("CAT_01_00000100_007_png.rf.<h>.jpg") == "CAT01_00000100"`
- `group_id("00000100_007_png.rf.<h>.jpg") == "P_00000100"`
- **Clip `00000100` → TWO distinct groups** (`CAT01_00000100` vs `P_00000100`) — the lone numeric collision; assert `len({the two}) == 2`.
- All 2040 names parse (594 CAT-prefixed, 1446 plain per FACTCHECK C32).

#### Step 5 — Duplicate + integrity audit (493 exact-dup filenames)

```python
# After export: collapse the 493 exact-duplicate filenames to 1547 distinct records
# BEFORE splitting, so a duplicate image cannot land in both train and test.
import hashlib, collections, pathlib
seen = {}
for p in pathlib.Path("datasets/cat-pain/coco").rglob("*.jpg"):
    h = hashlib.md5(p.read_bytes()).hexdigest()
    seen.setdefault(h, []).append(p)          # pixel-identical dedup
# assert: distinct pixel-hashes ~= 1547 ; log the dup map to provenance/dup_map.json
```

Treat near-duplicates (different hash, same scene) with `imagehash` Hamming ≤10 **only as a near-dup detector** to keep a near-dup pair from straddling the split — **never** as cat re-ID (CLIP/pHash carry no individual-identity signal here; see Gate 1).

**How it enters the pipeline:** YOLO export → RF-DETR-Nano/Small (or YOLOv11s fallback) binary detector on Colab T4, focal/class-weighted loss with `pos_weight ≈ sqrt(1819/264) ≈ 2.6` (**metadata-based**, per FACTCHECK C41), oversample the 264 pain boxes; the detector's tight head boxes become the crop source for the frozen DINOv2 encoder. Always print the **distinct-pain-cat denominator + Clopper-Pearson/bootstrap CIs**; augmented copies never counted. Decision-support triage framing only — never an autonomous analgesia trigger.

---

### 1.2 (b) The vet anchor — the ONLY graded-label source

**Purpose.** The single source of trustworthy per-AU 0/1/2 labels. It powers the **headline κ measurement** (VLM-as-AU-rater vs vet, Gate 1-B), the **0.39 sensitivity/specificity** estimate (validated on these labels ONLY — the circularity firewall), and the abstention NPV lower bound. **It is not downloaded — it is produced.** Size is *not* a guess: it is the **single integer fixed by the Gate 0 power calc** before any vet hour is spent.

**Target (pre-Gate-0 placeholder, to be replaced by the power calc):** ~**120–150 confirmed images, ≥50 pain-positive**. This is the n that the κ CI half-width (≤0.15 via `kappaSize`), the MAPIE/LTT NPV-band calc, and the 0.39-CI reportability calc must all certify. If Gate 0 says the budget can't certify a useful NPV band, the abstention curve ships **exploratory** (the word "guaranteed" is banned regardless).

**License.** Derived from the CC BY 4.0 Roboflow images; the *labels* are project-original. Releasable as the open weak-labeled-AU artifact + vet corrections.

**What labels it carries.** 5 AUs (ear, orbital, muzzle, whiskers, head), each **0/1/2** (0=absent; 1=moderate-or-uncertain; 2=obvious — Evangelista wording), vet-confirmed. Sum 0–10 and the 0.39 ratio are computed **in code**, never elicited from the labeler.

**How it is produced (the review loop):**

1. **VLM pre-fill (structured output).** Run Claude over the crops with a forced tool-use schema: one tool, 5 enum properties each `[0,1,2]`, `additionalProperties:false`, **rationale-before-score**, optional per-AU confidence/abstain. Strict structured outputs strip min/max → use enums (FACTCHECK C48/C49). Profile / eyes-closed / occluded crops route straight to the vet. Use the Message Batches API (50% off) with the **frozen** Evangelista rubric in a cached system block (any byte change invalidates the cache).
2. **Triage by value-per-hour.** Send the vet the cases that buy the most: pain-positive + near-threshold (sum 3–5/10), the weakest AUs (muzzle/whiskers), the most-diagnostic orbital, `cleanlab` confident-learning flags, and multi-run VLM disagreements.
3. **Vet accept/correct in a review UI.** Prefilled accept/correct (not from-scratch labeling) clears the κ-CI floor in ~4–5 vet hours. This is the **binding cost** of the project — money and compute are not the constraint.
4. **The κ is computed row-paired with `weights="quadratic"`** (nominal `labels=[0,1,2]` alone is a bug to avoid). Gate 1-B fires on the **CI lower bound**: orbital/ear/head ≥0.6; muzzle/whiskers 0.4–0.6 acceptable-with-caveat. (The labeler choice is self-justified by this pilot — **no external citation** is invoked.)

> **Firewall (non-negotiable):** sens/spec at 0.39 is estimated **only** on these vet-confirmed labels. **QWK-vs-VLM is never validation** — it measures the model re-learning the VLM heuristic. The κ-vs-vet number is the *method finding*, not validation of the 0.39 decision.

**How it enters the pipeline:** these are the held-out **clean anchor** — never used as training mass under co-teaching, always the evaluation/validation reference. Feeds Gate 1-B (κ), Gate 6 (severity-cell count → collapse if AU=2 cells single-digit), and the final fixed-recall operating-point + abstention curves.

---

### 1.3 (c) Horse-grimace — decode unit-test fixture ONLY

**Purpose.** The **only** open AU-graded grimace set in our 0/1/2 shape. Used **exclusively** to unit-validate the `CORN-decode → per-AU 0–2 → 0–10 sum → 0.39-threshold` path on *real* graded labels (Gate 4), and to optionally warm-start 3/5 heads (ear/orbital/muzzle). **No horse number appears in the abstract, results, or any transfer table** — methods/appendix sentence only: *"we unit-validated the decode→sum→threshold path on 5-horse genuine 0/1/2 labels."*

**License.** CC BY 4.0 (HF, FACTCHECK C17). **Caveat baked in:** only **5 unique horses (M1–M5)**, **3 of 5 AUs** (no whiskers, no head). Grouped by horse id = ~5 subjects → a coarse ordinal-direction prior, **never calibrated transfer**.

**Acquisition:**

```bash
hf download oliveirabruno01/openfarm-horse-grimace-region \
  --repo-type dataset --local-dir "$ROOT/datasets/horse-grimace"
```

**What labels it carries.** per-region ordinal 0/1/2 over {ears, orbital/eye, mouth-chin-nostrils} + derived binary. ~945 train / 279 test (heldout-balanced).

**How it enters the pipeline:** cache frozen DINOv2 features for horse crops → fit 3 CORN heads grouped by the 5 horse ids → assert the decode path returns the known sums on the genuine labels (this *is* Gate 4's CORN-decode test, run against real data after the synthetic-logit unit test passes). Then the cat heads are a button-press, not a 2-week build.

> Optional companion (also fixture-only, **never** a headline): `git clone https://github.com/mytalbot/MGS_data datasets/mgs` gives open per-AU **mouse** ordinal labels *including a whisker AU* (scores only, no images) — the only cross-species relief for our whisker blind spot. Same rule: unit/scaffold use, no reported number.

---

### 1.4 (d) CatFLW — eye-alignment landmarks ONLY

**Purpose.** Gold landmarks for the **Gate 5 NME / eye-alignment audit** — measuring whether the detector box crop is geometrically clean enough to feed CORN, and supplying a 2-point eye-similarity transform if not. **NOT a graded anchor** (it is landmarks + bbox; using it as a graded FGS anchor is a category error — explicitly cut from the critical path).

**License.** **CC BY-NC 4.0** (FACTCHECK C13) — **non-commercial**. This *blocks commercial deployment* of any model derived from it and **must not leak into a CC BY release**. Keep `datasets/catflw/` walled; the aligner trained on it inherits NC.

**What labels it carries.** ~2079 cat faces, **48 CatFACS-aligned landmarks** + bounding box. No pain, no AU scores.

**Acquisition (Kaggle):**

```bash
# Requires ~/.kaggle/kaggle.json (Kaggle → Account → Create New API Token)
kaggle datasets download -d georgemartvel/catflw -p "$ROOT/datasets/catflw" \
  && unzip -q "$ROOT/datasets/catflw/catflw.zip" -d "$ROOT/datasets/catflw"
```

Format reference (per-image JSON `{labels:(48,2), bounding_boxes}`): `github.com/martvelge/CatFLW`.

**How it enters the pipeline:** Gate 5 only. On 30–50 project crops, measure CatFLW-landmark **NME inside the RF-DETR crop vs a manual eye-aligned crop**, plus median face-pixel resolution after resize to the DINOv2 /14 grid. If NME is high → add a **2-point eye-similarity alignment** step before any CORN head trains (misaligned crops make "calibration" measure crop quality, not pain). CatFLW dominates the Zhang archive on every axis; use the archive aligner only if CatFLW access stalls.

---

### 1.5 (e) Reusable HF / model assets

**facebook/dinov2-small — the frozen encoder (conceded plumbing).**

```bash
hf download facebook/dinov2-small --local-dir "$ROOT/models/dinov2-small"
```

License Apache-2.0. ViT-S/14, 384-d CLS at token index 0, **patch size 14 → crops must be divisible by 14** (e.g. 224, 518). Freeze idiom: `Dinov2Model.from_pretrained(...).requires_grad_(False).eval()`; read `config.hidden_size` (384), derive patch from `embeddings.patch_embeddings.projection.stride[0]` (don't hardcode). **Cache pooled features to disk once** (`only_feat` pass) then train the 5 CORN heads off the cache — no backbone forward per epoch (the big M4/MPS win). The backbone is **frozen**; do not domain-adapt it on the Zhang archive (real risk of degrading features for ~zero gain). dinov2-small is the committed v1 backbone; `facebook/dinov3-vits16` is a gated optional upgrade, not on the critical path.

**Cat-face Haar cascade — fallback cropper / abstention signal.**

```bash
hf download d-v-18/cat-face-detector --local-dir "$ROOT/models/cat-face-detector"
# → haarcascade_frontalcatface.xml  (MIT)
```

Coarse `frontalcatface` → `detectMultiScale` → pad ~0.3 → clamp → crop. Use **only** as a zero-training pre-crop fallback when the RF-DETR box is missing, and **log a detect-failure as an abstention signal** — do **not** make it the landmark pipeline (Steagall trap). (`cvdl/catfaces`, 29,842 unlabeled cat faces, is an optional SSL/aug pool — not needed for v1.)

---

### 1.6 Re-split protocol, frozen hold-out, and the gate-bound asserts

This is the keystone correction (the single highest-risk step in the project). Replace the shipped split entirely.

#### Gate 1 — per-CAT merge (validate against `CAT_` ids)

Collapse clips toward true individuals **before** building folds, validating the merge against the **trusted `CAT_` filename ids**, not CLIP/pHash (which are duplicate detectors, not re-ID, and collapse 98% of pain pairs into one blob). **Group by individual (`cat_id`) for CV — never by clip.** Letting one individual's clips straddle folds is same-individual leakage — the exact failure that inflates accuracy 30–55% under record- vs subject-level splitting (PMC8604922); **subject-exclusive / leave-one-animal-out is the named standard in this subdomain** (Feighelstein et al. 2022/2024; RSNA CV guide: "partition at the patient level, not the examination level"). The one dominant individual `CAT_01` (~84 clips / ~605 imgs ≈ 30% of the corpus) *would* fuse into a single degenerate CV fold under naive per-CAT grouping (`StratifiedGroupKFold` just degrades to `GroupKFold` for a dominant group), so **carve `CAT_01` out as a frozen leave-one-individual-out (LOIO) hold-out** and run `StratifiedGroupKFold(groups=cat_id)` over the remaining ~335 individuals only — leak-safe *and* balanced. The per-CAT merge is the CV **group key**, not merely a disjointness assert. *(Fallback: if you must keep `CAT_01` inside CV, group by `cat_id` and accept the degenerate fold with honest wide bootstrap CIs — still leak-safe, just imbalanced.)*

#### The splitter (shared by Phase A and Phase B)

```python
from sklearn.model_selection import StratifiedGroupKFold
# 1) carve the dominant individual out as a frozen leave-one-individual-out (LOIO) hold-out
loio_mask = (cat_id == "CAT_01")                 # ~605 imgs; reported leak-safe via LOIO, NEVER in CV
X_cv, y_cv, g_cv = X[~loio_mask], y[~loio_mask], cat_id[~loio_mask]
# 2) group by INDIVIDUAL (cat_id), image-level y (135/336 clips are class-mixed -> cannot collapse to a label)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for tr, va in sgkf.split(X_cv, y_cv, groups=g_cv):
    assert set(g_cv[tr]).isdisjoint(set(g_cv[va]))               # G1 disjointness (per individual)
    assert "CAT_01" not in set(g_cv[va])                         # dominant individual never enters a CV fold
    pf = y_cv[va].mean()
    assert abs(pf - 0.127) <= 0.05 and y_cv[va].sum() >= 25      # re-verify floors after removing CAT_01
```

Per-fold pain counts must be **recomputed** after removing `CAT_01` and grouping by `cat_id` (the prior `[39, 36, 27, 33, 38]` was computed under the disqualified per-clip grouping). Re-verify the `pf≈0.127` and `≥25 pain imgs` floors hold per fold and that no individual straddles folds. Pool **out-of-fold** predictions; report **PR-AUC + pain recall at the fixed ≥0.90-recall operating point** with bootstrap 95% CIs and the CI width stated up front; report `CAT_01` performance **separately** via the LOIO hold-out. Do **all** preprocessing / threshold tuning / calibration **inside each fold**.

#### Gate 2 — capture-condition confound audit (one-directional, runs before any spend)

Train a *trivial* classifier on brightness / blur / box-aspect / CLIP-embedding to predict pain. If it beats chance, pain is entangled with acquisition context. **Report "no confound *detected at this power*," never "no confound"** (well-powered to detect, underpowered to rule out). The deliverable is the **transportable FGS-BG-Gap + per-AU EBPG protocol**, not "CAT_01 is confounded." CPU, no vet, no GPU — the cheapest kill/reframe; runs *before* any training or vet hour.

#### Gate 3 — frozen, hashed, cat-disjoint hold-out + CI-abort

```python
import json, hashlib, pathlib
# Build the boundary-rich, true-~13%-prevalence test set from merged (cat-disjoint) groups.
test_groups = sorted(selected_test_group_ids)             # cat-disjoint from all train/val folds
manifest = {"version": "cat-pain-ul7lu-p3rtl@v2",
            "test_groups": test_groups,
            "fold_csv_sha256": hashlib.sha256(pathlib.Path("splits/folds.csv").read_bytes()).hexdigest()}
blob = json.dumps(manifest, sort_keys=True).encode()
manifest_hash = hashlib.sha256(blob).hexdigest()
pathlib.Path("splits/test_manifest.json").write_bytes(blob)
pathlib.Path("splits/test_manifest.sha256").write_text(manifest_hash)
# CI gate: ABORT any train/sweep/threshold run that can read splits/test_manifest.json.
# (structurally prevents the post-hoc goalpost move)
```

The test set is **frozen and never touched during selection**. A CI check fails the build if a training/tuning script imports or opens the test manifest.

#### Gate 4 — MPS compute-correctness (BLOCKING, before any scoring)

Before any weak-labeling or scoring run: (a) MPS-vs-CPU DINOv2 logit parity; (b) 1-epoch coral-pytorch CORN smoke test on MPS; (c) **synthetic CORN-decode → 0–2 → 0–10 sum → 0.39 unit test** against known values (unit-test against the coral-pytorch doctest `tensor([1,3])`). Keep `probas = cumprod(sigmoid(logits))` (per-level cumulative P(rank>k)) for the distributional path — **do not decode straight to a hard label**; the `>0.5` hard threshold is point-estimate-only (the 0.39 decision). A leak-proof PR-AUC from silently-wrong MPS logits is worthless.

#### Gate 5 — alignment/NME (uses CatFLW, §1.4); Gate 1-B — κ pilot (uses vet anchor, §1.2); Gate 6 — severity-cell collapse

Gate 5 settles eye-alignment; Gate 1-B fires the κ CI-lower-bound GO/NO-GO; Gate 6 counts AU=2 cells in the anchor and **collapses the high-end scale** (merge AU 1+2 / report only painful-vs-not) if single-digit — a pre-registered structural decision, not a caveat. **Distributional CORN** path downstream: per-AU pmf → convolve to pmf over the 0–10 sum → **RPS-on-sum + per-AU ClasswiseECE**; argmax-decode only for the 0.39 point decision; **no binned reliability diagram on the 11-atom sum**.

> **Kill/pivot (carry from FINAL_DIRECTION).** If orbital/ear/head κ-LB falls below floor at Gate 1-B → drop the 0–10 layer entirely and ship **calibrated BINARY + abstention + confound audit**; the κ-as-method finding and the wrapper artifacts still stand.

---

### 1.7 Data-provenance datasheet stub (`provenance/datasheet.md`)

Fill this as a checklist; it is the released artifact's honesty layer.

```markdown
# Datasheet — cat-fgs-llm data provenance

## Motivation & composition
- Spine corpus: forked Roboflow `mingraths-workspace/cat-pain-ul7lu-p3rtl` @ v<FIT_VERSION>
  (Fit-reflect 640, NO stored augmentation). Upstream provenance = Zhang/Sun/Tang 2008
  Flickr cat-head family (NOT Finka 2019 — drop the Finka leakage warning).
- N: 2040 records / 1547 distinct (493 exact-dup filenames removed); 336 source clips.
- Labels: binary pain/no_pain BOXES only. 264 = pain-BOX metadata count (NOT individuals;
  re-ID not run). CAT_01 = 84 clips = lone known camera id. Prevalence ~12.7% (264/1819 basis;
  live-annotation sum 246/1811 = 12.0% — discrepancy logged). 1819 = dataset-wide no_pain, NOT train-only.
- Negative class = UNKNOWN MIXTURE (possible sedated/post-recovery). Pain labels are
  UNVALIDATED pseudo-labels on generic Flickr photos until the Gate-1-B κ pilot + Gate-2
  confound audit pass.

## Known leakage & how handled
- Shipped split leaks 191/336 clips (56.8%) → DISCARDED. Re-split: StratifiedGroupKFold(5),
  group=clip stem, image-level y (135/336 clips class-mixed). Clip 00000100 → 2 distinct groups
  (asserted). Cat-disjoint hashed test set frozen (Gate 3, sha256 = <HASH>).

## Graded labels
- Source = VET ANCHOR ONLY (n FIXED BY GATE 0 POWER CALC; pre-G0 placeholder ~120–150 imgs,
  ≥50 pain-pos). VLM pre-fill → vet accept/correct. 5 AUs × {0,1,2}. Sum/0.39 computed in code.
- "graded" is INSPECTED-NOT-VALIDATED; never a validated-claim word. QWK-vs-VLM never reported
  as validation (circularity firewall). sens/spec @0.39 on vet labels ONLY. Operating point =
  fixed pain-recall ≥0.90 (NOT Youden-J/F1). Abstention = one-sided 95% NPV LOWER BOUND;
  "guaranteed" is BANNED.

## External sources & licenses (kept separate — do not co-mingle)
- Roboflow cat-pain: CC BY 4.0 (releasable).
- Vet-anchor labels: project-original (releasable).
- Horse-grimace (oliveirabruno01/openfarm-horse-grimace-region): CC BY 4.0. UNIT-TEST ONLY,
  5 horses, 3/5 AUs. NO number in abstract/results/transfer table.
- CatFLW (georgemartvel/catflw): CC BY-NC 4.0 — NON-COMMERCIAL; aligner inherits NC; do NOT
  leak into a CC BY release. Alignment/NME ONLY, not a graded anchor.
- facebook/dinov2-small: Apache-2.0 (frozen, conceded plumbing).
- Haar frontalcatface (d-v-18/cat-face-detector): MIT. Fallback crop / abstention signal.
- Zhang-2008 archive: CC0 (Kaggle). FALLBACK aligner only; namespaced OUTSIDE datasets/cat-pain.

## Anti-benchmark discipline
- Augmented copies NEVER enter any reported N. Always print distinct-pain-cat denominator +
  Clopper-Pearson/bootstrap CIs. Never "we beat 77/79/95%". Decision-support triage framing
  only; never an autonomous analgesia trigger.

## Engine concession
- DINOv2+CORN engine = plumbing, explicitly NOT claimed novel.
```

---

### Solo-dev checklist (run top to bottom)

- [ ] **G0 first:** run the three power calcs (`kappaSize` κ-CI half-width ≤0.15; MAPIE/LTT NPV band; 0.39-CI reportability) → commit the **single vet-budget integer**. Blocks everything quantitative.
- [ ] Fork `lia-k4jkv/cat-pain-ul7lu` → `mingraths-workspace/cat-pain-ul7lu-p3rtl`; verify with `projects_get`.
- [ ] Generate **Fit (reflect/white) 640**, augmentation EMPTY; record version string.
- [ ] Export COCO + YOLO via `scripts/download_dataset.py --version <new>`.
- [ ] Parse `group_id`; pass the 4 unit asserts (incl. `00000100` → 2 groups).
- [ ] Dedup 493 filenames → 1547 distinct; log `provenance/dup_map.json`.
- [ ] `hf download` dinov2-small, horse-grimace, cat-face-detector; `kaggle datasets download` catflw.
- [ ] **G1** per-CAT merge vs `CAT_` ids; **G2** confound audit (one-directional); **G3** frozen hashed cat-disjoint test + CI-abort.
- [ ] **G4** MPS logit parity + synthetic CORN decode→sum→0.39 unit test (BLOCKING); then horse-grimace decode unit test.
- [ ] **G5** CatFLW NME/resolution audit (add 2-pt eye-align if needed).
- [ ] Produce the **vet anchor** (VLM pre-fill → vet accept/correct); **G1-B** κ CI-lower-bound GO/NO-GO; **G6** AU=2 cell count → collapse if single-digit.
- [ ] Fill `provenance/datasheet.md`.

---

## 2. Phase A — binary pain detector (RF-DETR), training recipe & imbalance

> **Scope & framing.** Phase A is the **binary spine** of v1 (pain / no_pain) and the only Phase that produces a *validated* number under the firewall: detector P/R and the fixed-recall operating point are reported on **vet-confirmable binary labels**, never on VLM-derived AU labels. The detector is also the **face-crop CANDIDATE** for Phase B, but the box is a candidate **only** — it does not feed CORN until **Gate 5 (alignment/NME)** passes (§2.8). The DINOv2 backbone inside RF-DETR is **plumbing, never claimed novel**. Nothing here runs until **Gate 1 (per-CAT merge)** and **Gate 3 (frozen hashed cat-disjoint hold-out + CI-abort)** have produced the fold CSV. **Gate 4 (MPS compute-correctness)** is not required for Phase A *training* (Colab T4) but IS required before any MPS-side inference number is believed.

### 2.0 Where it trains vs runs (compute split)

| Stage | Machine | Why |
|---|---|---|
| Dataset export, re-split, copy-paste oversampling, fold CSV build | **M4 (local, CPU)** | Deterministic data prep; keeps the hashed-split logic local and auditable. |
| **Detector training** (RF-DETR-Nano/Small, 5 folds) | **Colab T4 (CUDA)** | T4 gives ~16-effective-batch in reasonable wall-clock. MPS training works as of `rfdetr>=1.6.0` but is slow for a 5-fold sweep. |
| Smoke test / 1-epoch correctness / local inference for the FIND loop | **M4 (MPS, `rfdetr>=1.6.0`)** | Forward passes and a 1-epoch sanity run are MPS-friendly; use for the active-learning scoring pass (§6) so Colab credits are spent only on training. |
| YOLOv11s runner-up (if RF-DETR stalls) | **Colab T4** primary, **M4/MPS** fallback | Mature CLI, fast iteration. |

**Pin the environment first (both machines).** RF-DETR MPS support lands in **v1.6.0** — pin at or above it.

```bash
# M4 local (uv-managed venv; MPS inference + data prep only)
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install "rfdetr>=1.6.0" supervision scikit-learn imagehash \
               roboflow python-dotenv torchmetrics pandas pillow
python -c "import torch; print('mps', torch.backends.mps.is_available())"  # must print: mps True
```
```bash
# Colab T4 (training) — first cell
!pip -q install "rfdetr>=1.6.0" supervision scikit-learn torchmetrics
import torch; assert torch.cuda.is_available(), "Switch runtime → T4 GPU"
```

### 2.1 Architecture decision (decisive)

- **PRIMARY: RF-DETR-Nano** (`RFDETRNano`), step up to **RF-DETR-Small** (`RFDETRSmall`) only if Nano's pain-recall plateaus below target. Few-epoch convergence on small custom data; native MPS as of v1.6.0.
- **RUNNER-UP: YOLOv11s** (`yolo11s.pt`) — faster iteration, mature CLI, documented weighted-loss recipe; the fallback if RF-DETR training is unstable on this tiny, imbalanced set.
- **Detection-only** (not crop-then-classify): the detector doubles as the Phase B face-crop candidate, so a separate localizer is wasted work. The per-image binary pain decision is the **max-confidence pain box vs. the tuned threshold** (§2.6).

### 2.2 Data export & re-split (do this FIRST — gates everything)

The shipped Roboflow split is **leaky** (191/336 source clips, 56.8%, in both train+valid). Discard it. Re-export and re-split locally.

**Export geometry (load-bearing):** export with **`Fit (reflect/white edges)`, NOT `Stretch-to-640`** — Stretch distorts ~4:3 images and warps orbital/muzzle/whisker geometry. Export **COCO** (RF-DETR ingests `_annotations.coco.json` per split folder).

```bash
# M4: export the FORKED project, COCO, Fit resize (set via Roboflow version generate, not here)
python scripts/download_dataset.py --format coco --version <FIT_VERSION>
# -> datasets/  with train/ valid/ test/ each holding _annotations.coco.json
```

**Parse the group id from filenames** (the cat/clip is the CV group, never the image):

| Pattern | Regex | group_id |
|---|---|---|
| CAT-prefixed | `^CAT_(\d{2})_(\d{8})_(\d{3})\.png$` | `CAT{cam}_{clip}` |
| plain | `^(\d{8})_(\d{3})\.png$` | `P_{clip}` |

Namespace `CAT_`-prefixed clips separately (one numeric collision exists). **Then collapse clips → true individuals per Gate 1** (validated against the trusted `CAT_` ids — CLIP/pHash are duplicate detectors, not re-ID). The fold CSV is built on the **merged per-CAT groups**, not raw clips.

```python
# build_folds.py (M4, CPU) — consumes the Gate-1 per-CAT merge, emits the Gate-3 hashed split
from sklearn.model_selection import StratifiedGroupKFold
# carve the dominant individual CAT_01 out as a frozen LOIO hold-out (see Gate-1 block above) — never in CV
m = (cat_id != "CAT_01"); Xc, yc, gc = X[m], y[m], cat_id[m]
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
# Xc=image ids, yc=image-level pain(1)/no_pain(0), groups=merged CAT id (per INDIVIDUAL, not clip)
for fold,(tr,va) in enumerate(skf.split(Xc, yc, groups=gc)):
    assert set(gc[tr]).isdisjoint(set(gc[va]))                   # Gate 1/3 disjointness guard
    assert abs(yc[va].mean() - 0.127) < 0.05 and yc[va].sum() >= 25 # ~13% prevalence, >=25 pos (recompute)
# write folds.csv (image_id, fold, y, cat_id); sha256 it; CI aborts any train run that can read test ids
```

- Stratify at **image level with grouping** (135/336 clips are class-mixed — a clip cannot collapse to one label).
- **De-dup guard:** 493 exact-duplicate filenames (2040 records / 1547 distinct) plus near-dups — run an `imagehash` Hamming≤10 / CLIP cosine>0.6 union-find **as a near-duplicate detector only** and assert no near-dup pair straddles train/val. (This is NOT the per-cat merge — that is Gate 1.)
- Expected per-fold pain count ≈ 45 at k=5; the clean valid pain count is small, so **always report mean ± SD across the 5 cat-grouped folds**, never a single split.

### 2.3 FGS-safe augmentation (light geometry, no vertical flip, no heavy color)

Subtle orbital/muzzle/whisker cues are the whole signal — destroying them poisons the Phase B crop. **Allowed:** horizontal flip, small translate/scale, ≤10° rotation, modest saturation/value jitter, mosaic early then disabled for final epochs. **Banned:** `flipud`, large rotations, heavy color distortion, aggressive mixup/copy-paste *blends*.

| Knob | Value | Note |
|---|---|---|
| horizontal flip | `0.5` | safe (face is L/R symmetric) |
| vertical flip (`flipud`) | **0.0** | banned — inverts grimace geometry |
| translate | `0.1` | |
| scale | `0.3–0.5` | |
| rotation (`degrees`) | `≤10` | larger destroys ear/whisker angle cues |
| `hsv_s` / `hsv_v` | modest | no heavy color |
| `hsv_h` | ~0 | coat-color is a confound, don't amplify |
| mosaic | `1.0` early, `close_mosaic=10–15` | final epochs see the real distribution |

RF-DETR applies its own train-time augmentation pipeline; keep it at defaults and rely on the COCO `Fit` export for geometry fidelity (do NOT add Roboflow-side flipud or strong color steps). For YOLOv11s the knobs above map 1:1 to `yolo train` args.

### 2.4 Training recipe — RF-DETR (PRIMARY)

Start from the **auto-downloaded COCO-pretrained backbone** (`cat` is COCO id 15, so transfer is *expected* — present it as expected, not benchmarked). Effective batch 16 on T4 via `batch_size=4, grad_accum_steps=4`. Lower LR on the encoder/backbone than the head.

> **`num_classes` footgun — verify before the first run.** RF-DETR sizes its classification head from `num_classes`, and the value must cover the **max COCO `category_id` present** (RF-DETR reserves the high index for background; a 2-category file can require `num_classes=3` depending on whether ids are 0- or 1-based). Do **not** hard-code `2` blindly. After the datamodule is built, assert the head matches the data:
> ```python
> # confirm class count from the actual COCO file before training
> dm.setup("fit"); print(dm.class_names)   # expect ['no_pain','pain'] (sorted)
> NUM_CLASSES = len(dm.class_names)         # let the data set it; override only if the head errors
> ```

```python
# train_rfdetr_fold.py  (Colab T4)  — run once per fold k in {0..4}
from rfdetr import RFDETRNano   # swap RFDETRSmall only if Nano recall plateaus

model = RFDETRNano(num_classes=NUM_CLASSES)   # NUM_CLASSES from the COCO file (see footgun box); COCO weights auto-download
model.train(
    dataset_dir=f"./datasets/fold{K}",        # train/ valid/ test/ with _annotations.coco.json
    resolution=512,                           # valid set: 384/512/576/704 (divisible by patch_size*num_windows); 576 if VRAM allows
    epochs=100,                               # expect earlier convergence
    batch_size=4,
    grad_accum_steps=4,                       # effective batch = 16
    lr=1e-4,
    lr_encoder=1.5e-4,                        # backbone/encoder LR (low-ish; drop to 7.5e-5 if unstable)
    weight_decay=1e-4,
    use_ema=True,
    early_stopping=True,
    early_stopping_patience=10,
    early_stopping_min_delta=0.005,           # require 0.5% mAP gain (docs default-scale; 0.001 is noise on this tiny val)
    early_stopping_use_ema=True,
    num_workers=2,
    output_dir=f"./out/fold{K}",
    progress_bar="rich",
    tensorboard=True,
    seed=42,
    # device omitted -> auto-detects CUDA on Colab; pass device="mps" for the M4 smoke run below
)
```

- **Resolution:** `512` default; `576` if the T4 holds it at batch 4 — higher res preserves facial AU pixels for the Phase B crop. Must stay in the divisible-valid set (`384/512/576/704`).
- **EMA on** (`use_ema=True`) and gate early-stop on the EMA model (`early_stopping_use_ema=True`).
- **Resume / 2-phase:** the active-learning loop (§6) fine-tunes **from the previous fold checkpoint**, not COCO, once a champion exists: `model.train(..., resume="./out/fold{K}/checkpoint.pth")`. The high-level `train(resume=...)` path accepts both legacy `.pth` and Lightning `.ckpt` (converted automatically).
- **MPS smoke run (M4):** the *same* call with `device="mps"`, `epochs=1`, and a 50-image subset confirms the install before burning Colab credits; it is also the Phase-A half of the Gate-4 MPS-correctness check.

### 2.5 Training recipe — YOLOv11s (RUNNER-UP)

```bash
# Colab T4 or M4/MPS fallback. ALWAYS fine-tune from the COCO .pt — never random init.
yolo detect train model=yolo11s.pt data=fold{K}.yaml imgsz=640 batch=16 epochs=120 \
     patience=25 optimizer=auto close_mosaic=10 \
     fliplr=0.5 flipud=0.0 degrees=10 translate=0.1 scale=0.4 \
     hsv_h=0.0 hsv_s=0.4 hsv_v=0.3 device=0   # device=mps on M4
```

### 2.6 Imbalance ladder (stack cheapest-first; change ONE lever per run; threshold LAST)

Prevalence ~12–13%, imbalance ~6.9:1 (metadata basis 264 pain / 1819 no_pain *boxes* — these are **box counts, not individuals**; live annotation sum gives 12.0% / 7.4:1). Handle with **data + loss, not architecture**, and **never move the threshold until the loss/data ladder is exhausted.**

| Rung | Lever | Setting | Fires when |
|---|---|---|---|
| 1 | **Class weight / `pos_weight`** | `pos_weight ≈ sqrt(1819/264) ≈ 2.6` (metadata-based; cap ~3). YOLO: `cls_pw` 0.25→1.0 (inverse-freq). RF-DETR: per-class weight on the classification term **if the pinned version exposes it; if not, fall through to rung 2 as the baseline lever and note it.** | always (baseline) |
| 2 | **Oversample + object-level copy-paste** | Duplicate the pain images; **manually** paste pain face-crops into no_pain frames as a preprocessing step (box datasets can't use Ultralytics seg-only `copy_paste`). Oversampling is the strongest detection-imbalance lever. | if rung-1 pain-recall < target |
| 3 | **Focal loss** | `FocalLoss(gamma=1.5, alpha=0.25)`. | if rungs 1–2 insufficient |
| 4 | **Threshold tuning (LAST)** | sweep confidence on the **val PR curve**; pick the **highest threshold still meeting pain-recall ≥0.90** inside train folds (nested) — raise the bar only as far as the recall floor allows, which is what makes the specificity it buys (reported with bootstrap CIs) a meaningful number. | only after 1–3 |

**Operating point is fixed pain-recall ≥0.90 (Evangelista anchor), NOT Youden-J / F1** — a symmetric-cost knee is the wrong loss for a welfare instrument where undertreatment ≫ overtreatment. The chosen threshold and its **cross-fold variance** are reported explicitly; the ≥0.90 gate is justified against Evangelista's 90.7% reference sensitivity, not against prior automated work.

**Reported-N discipline:** augmented/oversampled/copy-paste copies **never enter any reported N**. Print the **distinct-pain-CAT denominator** (from the Gate-1 merge) with Clopper-Pearson / bootstrap CIs alongside every rate.

### 2.7 Evaluation (PR-AUC/AP primary; never accuracy, never ROC-AUC as the rank metric)

- **Primary:** **PR-AUC / Average Precision** with pain = positive (the right metric under heavy imbalance). Derive the per-image decision (max-conf pain box vs. tuned threshold) and score *that* with PR-AUC + pain-recall.
- **Per-class:** precision / recall / **F1** for pain and no_pain, confusion matrix, specificity.
- **Localization only (NOT a quality verdict):** `torchmetrics.MeanAveragePrecision` → `map`, `map_50`, `map_75`, `map_per_class`. These judge box placement for the Phase B crop, not pain skill.
- **Go/no-go metric:** **pain-class recall ≥0.90** on the frozen cat-disjoint test set.
- **Reporting:** **mean ± SD across the 5 cat-grouped folds**, ideally bootstrap 95% CIs. Never rank on accuracy or ROC-AUC. Never write "we beat 77/79/95%" — those are anti-benchmarks.

```python
# eval_fold.py (M4 or Colab) — per-image binary decision scored with PR metrics
from torchmetrics import MeanAveragePrecision
from sklearn.metrics import precision_recall_curve, auc
# 1) localization: torchmetrics mAP over predicted vs GT boxes -> map_50, map_per_class
# 2) decision: per image, s = max pain-box confidence (0 if none); sweep threshold on val PR curve
prec, rec, thr = precision_recall_curve(y_true_img, s_img)    # pain=1
pr_auc = auc(rec, prec)
# pick highest thr still meeting rec >= 0.90 (inside train folds), then freeze and apply to the held-out test fold
```

### 2.8 The box is a face-crop CANDIDATE — gated behind Gate 5

The detector box is **verified** to be a tight, centered, frontal head crop (median area ~0.24–0.28 of frame), which is *why* it is a plausible Phase B cropper. But **box-as-AU-cropper is unvalidated alignment, a RISK not an advance**, until **Gate 5** passes: on 30–50 project images, measure CatFLW-landmark **NME inside the RF-DETR crop vs. a manual eye-aligned crop**, plus the **median face-pixel resolution after resize to the DINOv2/14 grid** (crops must be **divisible by 14**). If NME is high or resolution too low, insert a **2-point eye-similarity alignment** step (and/or a higher-patch ViT) before any CORN head trains. Apply a **~12–15% margin EXPANSION** to the box before cropping (not a shrink) so ear tips / whiskers aren't clipped. Until Gate 5 passes, Phase A ships as a **standalone calibrated binary detector**; the crop hand-off to Phase B is conditional.

### 2.9 Phase A definition of done

1. Fold CSV built on the **Gate-1 per-CAT merge**, hashed, CI-abort armed (Gate 3); no individual straddles a fold.
2. RF-DETR-Nano trained 5 folds from COCO on Colab T4 with the §2.4 recipe; `num_classes` confirmed against `datamodule.class_names`.
3. Imbalance ladder climbed cheapest-first; threshold set LAST at **pain-recall ≥0.90** inside train folds.
4. **PR-AUC/AP + per-class P/R/F1 reported as mean ± SD across 5 cat-grouped folds**, with the distinct-pain-CAT denominator and bootstrap CIs; no augmented copy in any N; no accuracy/ROC headline.
5. Gate-4 MPS logit parity confirmed before any MPS inference number is trusted; Gate-5 NME/resolution audit decides whether the box feeds Phase B.

---

## 3. Face crop & alignment preprocessing (Gate 5)

This section specifies the **detect → quality-gate → expand → align → resize → encode** pipeline that produces the exact crop tensor the frozen DINOv2 ViT-S/14 engine consumes, plus the **Gate 5 NME/resolution audit** that decides whether the cheap detector-box crop is good enough or whether the 2-point eye-similarity aligner must be switched on. The DINOv2+CORN engine downstream is conceded plumbing; this stage exists only to stop us measuring crop quality and mistaking it for calibration. **Gate 5 is BLOCKING:** no CORN head is trained and no vet hour is spent until it passes (FINAL_DIRECTION Gate order; BUILD_PLAN line 60, §0 Gate 5).

> **Run-order note.** Gate 5 runs after Gate 4 (MPS compute-correctness) and before Gate 1-B (the κ pilot) and Phase B feature caching. The 30–50-image audit is the *only* part that must complete before any scoring; the production cropper it certifies is then frozen and applied to all ~2,040 images. The audit runs CPU-only on the M4; no Colab T4 needed (T4 is for detector training only).

### 3.0 Inputs, assets, output contract

| Asset | Path (already on disk) | Role in this stage |
|---|---|---|
| Forked Roboflow crops/boxes | `datasets/train/{images,labels}`, `datasets/valid/{images,labels}` (YOLO txt: `cls cx cy w h`, normalized) | Source images + candidate face boxes (Phase A detector output is the same geometry) |
| CatFLW landmarks | `datasets/catflw/CatFLW dataset/{images,labels}` — labels are JSON `{"labels": [[x,y]×48], "bounding_boxes":[...]}` in **pixel** coords (CC BY-NC 4.0; never enters a released model artifact) | NME ground-truth + eye landmarks for the audit and the aligner |
| Haar `frontalcatface` | `models/cat-face-detector/haarcascade_frontalcatface.xml` (`hf download d-v-18/cat-face-detector`) | Coarse fallback localizer + **detect-failure → abstention** signal only (never the landmark path; Steagall-trap avoidance) |
| Cat-landmark aligner ref | `ref/cat-landmark-align` (`github.com/chelsea23311/Cat-Face-Landmark-Detection`, interocular-NME + PCK code) | Template for the 2-point eye-similarity fallback (GITHUB_MINE §3b) |

**Output crop spec (the contract the encoder consumes — fixed, do not vary per-image):**

| Field | Value | Reason |
|---|---|---|
| Spatial size | **518 × 518** px | `518 = 14 × 37`; divisible by patch size 14 (FACTCHECK C26), DINOv2 ViT-S/14 native eval res, integer 37×37 token grid |
| Channels / color | **RGB**, sRGB, 8-bit decoded then float | DINOv2 expects RGB; never feed BGR (OpenCV default — convert with `cv2.cvtColor(..., COLOR_BGR2RGB)` before save) |
| Pad strategy | **letterbox to square BEFORE resize**, replicate-edge pad (`cv2.BORDER_REPLICATE`), then single resize to 518 | Preserves aspect ratio so ear tips/whiskers are not stretched; replicate avoids a black border the backbone reads as a feature |
| Interpolation | `cv2.INTER_AREA` (downscale) / `INTER_CUBIC` (upscale) | Quality on small faces |
| Normalization | ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]` | DINOv2 hub default; applied in the dataloader, not here |
| Dtype on disk | uint8 PNG (`crops/<image_id>.png`) + a `crop_manifest.parquet` row | Re-decode deterministically; features cached separately |

The square-then-resize keeps the grid divisible-by-14 at any crop size — the only knob is the final edge (518). If a higher-resolution probe is ever needed (Gate 5 resolution fail), the only legal alternatives are **`518`, `672 = 14×48`, or `784 = 14×56`**; never an arbitrary edge. A higher edge raises VRAM and forward cost; 518 is the default and 672 is the first step only if the resolution check below fails.

### 3.1 Margin EXPANSION (~12–15%), not shrink

FGS reads **ear tips** (ear AU) and **whisker** splay — both sit *outside* a tight face box. The forked boxes are tight head crops at ~0.27–0.30 of frame area (FACTCHECK box-geometry inspection; BUILD_PLAN §1/§2), so we **expand** the box before cropping.

```python
# scripts/crop_pipeline.py  — expand_box()
EXPAND = 0.135  # 13.5%, midpoint of the 12-15% band (BUILD_PLAN line 125)

def expand_box(cx, cy, w, h, img_w, img_h, expand=EXPAND):
    # cx,cy,w,h are normalized YOLO; return pixel xyxy, clamped, ASYMMETRIC top bias
    w2, h2 = w * (1 + expand), h * (1 + expand * 1.4)   # extra top room for ear tips
    x1 = (cx - w2 / 2) * img_w; x2 = (cx + w2 / 2) * img_w
    y1 = (cy - h2 / 2) * img_h; y2 = (cy + h2 / 2) * img_h
    # clamp; record clip flags so a clipped ear/whisker can route to vet
    clipped = (x1 < 0) or (y1 < 0) or (x2 > img_w) or (y2 > img_h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    return (round(x1), round(y1), round(x2), round(y2)), clipped
```

- **Asymmetric**: 1.4× the expansion on height (ear tips are above the head box; whiskers are roughly symmetric horizontally so the plain `expand` on width suffices). The 1.4× is a proposed default — confirm on the §3.6 worked examples that ear tips are retained without pulling in excessive background.
- `clipped=True` (box already at frame edge, so expansion can't recover the ear/whisker) is logged into the manifest and is one **abstention signal** (route to vet) — a clipped grimace AU cannot be scored honestly.
- Do **not** shrink. A shrunk box that drops ear tips silently zeros the ear AU.

### 3.2 Frontal-pose / quality gate (route non-frontal to vet)

FGS assumes a near-frontal view. Profile / tilted / occluded / eyes-closed faces are **not scored** — they are routed to the vet queue and never enter the CORN feature cache or any reported denominator. This is a deterministic, CPU-only filter applied per expanded crop.

| Check | Method (no training) | Threshold → route-to-vet |
|---|---|---|
| Frontal / yaw | Eye-pair from CatFLW-style landmarks (or Haar `eyepair`); ratio of (nose-to-left-eye)/(nose-to-right-eye) | ratio < 0.6 or > 1.67 (≈ ±25° yaw) |
| In-plane tilt | angle of the inter-ocular line | abs(roll) > 25° (the aligner can fix ≤25°; beyond is a pose problem, not alignment) |
| Eyes closed | eye-aspect-ratio (EAR) from upper/lower lid landmarks | EAR < 0.12 (orbital AU unreadable) |
| Occlusion / blur | variance-of-Laplacian on the crop | var < 60 (tunable on the audit set) → blur defer |
| Detector miss | Haar `frontalcatface` `detectMultiScale` returns 0 boxes on the frame | **detect-failure → abstention** (BUILD_PLAN P2.2 face-crop) |

```python
# Haar coarse fallback localizer + detect-failure abstention (NOT the landmark path)
import cv2
haar = cv2.CascadeClassifier("models/cat-face-detector/haarcascade_frontalcatface.xml")
def haar_present(gray):
    faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(48, 48))
    return len(faces) > 0   # False -> abstention flag 'haar_miss'
```

Every gate decision is written to `crop_manifest.parquet` columns `route_vet:bool, reasons:list[str]`. All thresholds above are proposed operating values; they are tuned **once** on the audit set and then **frozen** before scoring so the gate cannot be re-tuned to inflate prevalence.

### 3.3 Gate 5 audit — NME inside the box-crop vs eye-aligned crop, and resolution

The audit answers two yes/no questions on **30–50 images** (FINAL_DIRECTION Gate 5; BUILD_PLAN line 60). Use CatFLW images that also carry a `CAT_` id (e.g. `datasets/catflw/CatFLW dataset/images/CAT_01_*.png`) so the audit set overlaps our corpus's morphology; supplement with our own Roboflow crops to which CatFLW landmarks have been transferred via the Martvel landmark detector **if** that detector is available (if not, the audit runs on CatFLW images only — see assumptions).

**Metric — interocular-normalized NME (the standard cat-landmark metric, `ref/cat-landmark-align`):**

```
NME = (1/N) Σ_i  ||p_i_pred − p_i_gt||_2  /  d_interocular
```

Here `p_gt` are the 48 CatFLW landmarks (JSON `labels`), `d_interocular` = distance between the two eye-center landmarks. **The exact L/R eye-center indices must be pinned against the CatFLW index map before this script is run** (the first ~6 points are the eye/canthi cluster, but the precise centers are not yet confirmed — see assumptions). We do **not** train a landmark detector for the audit — we evaluate **landmark stability under the two crop transforms**: how much the *same* CatFLW ground-truth landmarks, re-expressed in each candidate crop's coordinate frame, drift relative to a canonical eye-aligned layout. Concretely:

1. **Crop A (box-as-cropper):** expand the CatFLW bbox by 13.5%, letterbox→518. Map the 48 GT landmarks into this frame.
2. **Crop B (eye-aligned reference):** 2-point similarity transform putting the two eye centers on a fixed canonical line (see §3.4), then 518. Map the 48 GT landmarks into this frame.
3. Report **NME_A** (landmark spread within the box-crop frame vs the canonical layout) and the **NME_A − NME_B gap**.

| Audit output | Pass condition | Fail action |
|---|---|---|
| Median NME_A | ≤ **0.08** (≤8% of interocular) AND gap(A−B) ≤ 0.02 | Switch on the 2-point aligner (§3.4) as the **production** crop step |
| Median face-pixel resolution | inter-ocular distance ≥ **40 px** *after* resize to the 518 grid (≈ face spans ≥ a 10×10-token region) | If < 40 px median, bump probe edge to 672 (`14×48`); if still low, flag low-res images to vet |
| Worst-decile NME_A | report (no hard gate) | informs the per-image abstention threshold |

The 0.08 NME and 40-px floor are **proposed** operating values calibrated to the Martvel context (below), not pre-registered; confirm or adjust them against the actual audit output before freezing.

```bash
# Gate 5 driver — CPU/M4, ~2 min on 50 imgs
python scripts/gate5_nme_audit.py \
  --catflw "datasets/catflw/CatFLW dataset" \
  --n 50 --seed 5 \
  --crop-edge 518 --expand 0.135 \
  --out reports/gate5_nme.json reports/gate5_nme.csv
# Emits: median_nme_box, median_nme_aligned, gap, median_interocular_px_after_resize,
#        worst_decile_nme, decision in {"box_ok","need_aligner","need_higher_res"}
```

**Why the gate matters / kill linkage:** Martvel 2024 reports automated cat-landmark NME of **9–26% by morphology** (FACTCHECK C27) and a ~7-point pain-accuracy cost (0.73→0.66, Table 6) from automated vs manual landmarks (FACTCHECK C28). If NME_A lands in that range, the box-crop is feeding the backbone a misaligned face and any downstream "calibration" number is measuring crop jitter — hence the alignment fallback is mandatory, not optional. Gate 5 failing is **BLOCKING** (FINAL_DIRECTION §7).

### 3.4 The 2-point eye-similarity alignment fallback

Switched on only if §3.3 fails. It is a closed-form similarity transform (rotate + uniform scale + translate) from the two detected eye centers to two **canonical** target points — no learned warp, no per-image optimization, MPS-irrelevant (pure NumPy/OpenCV on CPU).

```python
# scripts/align_eyes.py
import numpy as np, cv2
OUT = 518
# canonical eye targets: eyes on a horizontal line, interocular = 38% of edge, centered, slightly high
LEFT_T  = np.float32([OUT * 0.31, OUT * 0.42])
RIGHT_T = np.float32([OUT * 0.69, OUT * 0.42])

def align_by_eyes(img, left_eye, right_eye):           # eyes in pixel coords of the expanded crop
    src = np.float32([left_eye, right_eye])
    dst = np.float32([LEFT_T, RIGHT_T])
    M, _ = cv2.estimateAffinePartial2D(src, dst)        # similarity (rot+scale+trans), no shear
    return cv2.warpAffine(img, M, (OUT, OUT),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
```

- **Eye source priority:** (1) CatFLW / Martvel-detector eye centers when available; (2) the local Zhang-2008 archive 2 eye landmarks **only** as a fallback if CatFLW access stalls (DATA_DECISION §4 — CatFLW dominates; do not pre-build the archive aligner before the audit says it's needed). Cite `ref/cat-landmark-align` for the interocular-normalization + PCK/NME implementation we copy.
- Canonical interocular = 0.38·edge keeps ears/whiskers inside the 518 frame after rotation (replicate border on the corners that rotate out).
- The aligned crop **also** passes through §3.2's tilt check first — the aligner corrects ≤25° roll; >25° is a pose problem and routes to vet rather than being force-aligned.
- If the aligner is enabled, re-run the §3.3 audit on it to confirm median NME drops below 0.08 before freezing it as the production cropper.

### 3.5 Production crop run + manifest (after Gate 5 passes)

Once the gate decides (`box_ok` or `need_aligner`), freeze the chosen transform and crop all ~2,040 images once. Crops and a manifest are the only artifacts that flow downstream; **augmented copies never enter any reported N** (FINAL_DIRECTION anti-benchmark).

```bash
python scripts/crop_pipeline.py \
  --images datasets/train/images datasets/valid/images \
  --boxes  datasets/train/labels datasets/valid/labels \
  --mode "$(jq -r .decision reports/gate5_nme.json)" \
  --expand 0.135 --edge 518 \
  --haar models/cat-face-detector/haarcascade_frontalcatface.xml \
  --out-crops crops/ --out-manifest crop_manifest.parquet
# --mode is box_ok | need_aligner (need_higher_res must be resolved to one of these first)
```

`crop_manifest.parquet` columns (one row per source image):

| column | meaning |
|---|---|
| `image_id`, `clip_id`, `cat_id` | provenance; `cat_id` feeds Gate 1 per-CAT merge / cat-disjoint folds |
| `box_xyxy`, `expand`, `clipped` | crop geometry + ear/whisker clip flag |
| `align_mode`, `interocular_px` | `box`/`eye_aligned`; post-resize face resolution |
| `route_vet`, `reasons` | pose/blur/eyes-closed/haar_miss/clipped abstention reasons |
| `crop_path` | `crops/<image_id>.png`, 518×518 RGB |

Rows with `route_vet=True` are **excluded from feature caching and from every sens/spec/κ denominator** and join the vet queue. This makes "detector/quality failure → defer-to-vet" an explicit, counted abstention channel that the §F one-sided NPV-lower-bound abstention curve later consumes — not a silent drop.

### 3.6 Definition of done (Gate 5 exit)

- [ ] `reports/gate5_nme.json` written; `decision` recorded; median NME_A ≤ 0.08 **or** aligner enabled and re-audited ≤ 0.08.
- [ ] Median interocular ≥ 40 px after resize to 518 (else probe edge bumped to a `×14` value and re-checked).
- [ ] L/R eye-center CatFLW indices pinned and recorded in the audit JSON before any NME is computed.
- [ ] Pose/quality gate thresholds frozen; `route_vet` reasons enumerated.
- [ ] Output crops verified **518×518, RGB, divisible-by-14** by an assert in `crop_pipeline.py` (`assert edge % 14 == 0 and img.shape == (518, 518, 3)`).
- [ ] `crop_manifest.parquet` produced for all images; augmented copies excluded.
- [ ] One worked example image saved at each stage (raw → expanded → aligned → 518) into `reports/gate5_examples/` for the datasheet.

---

## 4. VLM weak-labeling pipeline + the kappa pilot (Gate 1-B)

This section builds the **supervision layer** and the **headline-#1 measurement**. Two outputs ship from here:

1. A reproducible **VLM weak-labeler** that emits 5 per-AU ordinal scores (`0/1/2`) per cat face via a forced-tool-use enum schema, run cheaply over the ~2040 forked-Roboflow faces with the Message Batches API + prompt caching, plus an `N>=3`-run **ordinal Krippendorff alpha** self-consistency column (vet-free).
2. The **Gate 1-B pilot**: a per-AU **quadratic-weighted kappa vs. the vet anchor**, **gated on the bootstrap CI lower bound**. This pilot *self-justifies* the labeler — we select the VLM by measured per-AU kappa, and we **do not** cite the unretrievable Sci Rep 2025 "only Claude acceptable" claim (FACTCHECK C54 / FINAL_DIRECTION §3).

> **Engine is conceded — plumbing only, never novel.** Nothing here trains or claims the DINOv2+CORN engine; the engine is conceded as plumbing throughout. The *measurement* (per-AU VLM-vs-vet quadratic kappa) is the portable method; the structured-output call is borrowed plumbing (Arize-ai/phoenix forced-tool pattern, GITHUB_MINE Pass 2).

> **The VLM NEVER emits the sum or the decision.** It emits 5 atoms in `{0,1,2}`. The 0–10 sum and the `>=0.39` analgesia flag are computed **in code** (§4.3, §4.6). This keeps the engine's decode path (Gate 4 unit test) the single source of truth for the threshold and prevents the labeler from leaking a clinical decision it was never validated to make.

**Run-order placement.** Gate 1-B fires **after Gate 0** (the vet-budget integer + power calcs must exist first — the pilot's `n` and the per-AU kappa floors are read from Gate 0, not hardcoded here) and runs alongside the §3.4 Monte-Carlo. The bulk ~2040-image weak-labeling run (§4.4) is *blocked* until Gate 1-B returns GO on orbital/ear/head; on NO-GO we pivot to the binary-plus-wrapper fallback and the kappa numbers ship as a method finding anyway.

### 4.0 Files, env, deps

```
scripts/
  vlm/
    schema.py            # forced-tool input_schema + Pydantic mirror (parse-failure-free)
    rubric.py            # system-prompt rubric, one paragraph per AU (verbatim Evangelista 0/1/2)
    call.py              # single-image forced-tool call (sync; used by pilot + smoke)
    batch_submit.py      # Message Batches API submit over a manifest, prompt-cached system block
    batch_collect.py     # poll + download .jsonl results, parse tool_use blocks -> parquet
    consistency.py       # N>=3 repeated runs -> ordinal Krippendorff alpha per AU
    aggregate.py         # in-code sum 0-10 + 0.39 flag (NEVER from the VLM)
  pilot/
    sample_pilot.py      # draw pilot imgs (Gate-0 n, >=50 pos), cat-disjoint from frozen test (Gate 3)
    review_export.py     # pre-fill vet review UI with VLM scores+rationales, triage-ordered
    kappa_gate.py        # per-AU quadratic kappa + bootstrap CI LOWER BOUND -> GO/NO-GO
data/
  weak_labels/           # parquet of per-image 5-AU scores, rationale, confidence, abstain
  pilot/                 # pilot manifest, vet CSV, kappa report JSON
```

```bash
# scripts/vlm/requirements.txt (pin)
anthropic>=0.40           # Messages + Batches + prompt caching
pydantic>=2.7             # schema mirror for in-code validation of tool input
krippendorff>=0.8.0       # ordinal alpha (prometheus-eval pattern)
scikit-learn>=1.4         # cohen_kappa_score(weights='quadratic')  (m-rewardbench pattern)
numpy pandas pyarrow      # manifest + parquet
pillow                    # base64 image encode at fixed long-edge
tenacity                  # retry/backoff on transient 429/529
```

All of §4 is **hosted-API only** (no local VLM weights): it runs identically from the M4/MPS box or a Colab T4 — the only compute on-machine is the Pillow re-encode and the bootstrap. No CUDA is required anywhere in this section.

```bash
# .env  (gitignored — already have .env.example + .gitignore in repo)
ANTHROPIC_API_KEY=sk-ant-...
VLM_MODEL=claude-opus-4-8          # candidate under test in Gate 1-B; swap to compare
VLM_MODEL_ALT=claude-sonnet-4-...  # second candidate; pilot selects by per-AU kappa
```

> **Model selection is empirical, not cited.** Run the pilot for each candidate `VLM_MODEL` and pick the one with the higher per-AU kappa **CI lower bounds**. Record both candidates' per-AU kappa tables in the datasheet. The labeler is self-justified by the pilot; the phantom Sci Rep 2025 citation is dropped (FACTCHECK C54).

---

### 4.1 The forced-tool-use enum schema (Arize-ai/phoenix pattern)

One tool, forced via `tool_choice={"type":"tool","name":...}`, `disable_parallel_tool_use=true` so the model emits **exactly one** tool-use block. Each of the 5 AUs is an `enum:[0,1,2]` (Claude strict structured outputs **strip numeric min/max** — FACTCHECK C48/C49 — so `enum` is the only way to pin the ordinal levels). `additionalProperties:false`, every field `required`. Rationale is emitted **before** the score (forces the model to reason then commit), plus per-AU `confidence` and `abstain`. **No `sum`, no `decision`, no `pain` field exists in the schema** — those are uncomputable by the VLM by construction.

```python
# scripts/vlm/schema.py
AU_NAMES = ["ear", "orbital", "muzzle", "whiskers", "head"]  # Evangelista 5-AU order

def _au_property():
    # rationale BEFORE score; confidence + abstain alongside. Order matters: JSON-schema
    # property order is the generation order under tool use.
    return {
        "type": "object",
        "properties": {
            "rationale":  {"type": "string",
                           "description": "1-2 sentences citing the visible feature, BEFORE scoring."},
            "score":      {"type": "integer", "enum": [0, 1, 2]},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "abstain":    {"type": "boolean",
                           "description": "true if occluded/blurred/out-of-frame/non-frontal for THIS AU."},
        },
        "required": ["rationale", "score", "confidence", "abstain"],
        "additionalProperties": False,
    }

FGS_TOOL = {
    "name": "record_fgs_action_units",
    "description": ("Record the Feline Grimace Scale action-unit scores for one cat face. "
                    "Score EACH of the 5 AUs independently on the 0/1/2 ordinal scale. "
                    "Do NOT compute a total or any pain decision."),
    "input_schema": {
        "type": "object",
        "properties": {**{au: _au_property() for au in AU_NAMES},
                       "image_quality": {"type": "string",
                                         "enum": ["frontal_clear", "partial", "unusable"]}},
        "required": AU_NAMES + ["image_quality"],
        "additionalProperties": False,
    },
}

TOOL_CHOICE = {"type": "tool", "name": "record_fgs_action_units",
               "disable_parallel_tool_use": True}
```

```python
# Pydantic mirror — validates tool input in-code; logs validation failures as datasheet provenance
# (dsRAG pattern). Parse-failure-free by design, but we still assert the contract.
from pydantic import BaseModel
from typing import Literal

class AU(BaseModel):
    rationale: str
    score: Literal[0, 1, 2]
    confidence: Literal["low", "medium", "high"]
    abstain: bool

class FGSResult(BaseModel):
    ear: AU; orbital: AU; muzzle: AU; whiskers: AU; head: AU
    image_quality: Literal["frontal_clear", "partial", "unusable"]
```

**Single-image call** (used by the pilot and the Gate-4-adjacent smoke test):

```python
# scripts/vlm/call.py
import base64, os, io
from anthropic import Anthropic
from PIL import Image
from .schema import FGS_TOOL, TOOL_CHOICE, FGSResult
from .rubric import SYSTEM_BLOCKS   # see §4.2 (cache-controlled)

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def encode_face(path, long_edge=896):           # face crops are small; cap long edge for cost
    im = Image.open(path).convert("RGB")
    im.thumbnail((long_edge, long_edge))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()

def score_face(path, model=None, temperature=0.0, seed_tag=""):
    model = model or os.environ["VLM_MODEL"]
    msg = client.messages.create(
        model=model,
        max_tokens=1200,
        temperature=temperature,
        system=SYSTEM_BLOCKS,                    # cached rubric block(s)
        tools=[FGS_TOOL],
        tool_choice=TOOL_CHOICE,                 # forced; exactly one tool_use block
        messages=[{"role": "user", "content": [
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/jpeg", "data": encode_face(path)}},
            {"type": "text", "text": f"Score this cat face. tag={seed_tag}"},
        ]}],
    )
    block = next(b for b in msg.content if b.type == "tool_use")
    return FGSResult(**block.input)              # raises on contract violation -> log + abstain-route
```

---

### 4.2 The rubric system prompt (one paragraph per AU, verbatim Evangelista 0/1/2)

The rubric lives in a **cache-controlled system block** so the ~2040-image run pays for it once. Use the **verbatim Evangelista descriptors** (FACTCHECK C50: `0=absent; 1=moderate OR uncertain; 2=marked/obvious`), ground AU anatomy in CatFACS (the manual from GITHUB_MINE §4), and instruct **default-ambiguity-to-1** with low confidence + `abstain` on occlusion/blur/non-frontal. The block carries `cache_control: {"type":"ephemeral"}` (5m default TTL; bump to `"1h"` for a long batch window).

```python
# scripts/vlm/rubric.py  (abridged — the 5 AU paragraphs are verbatim Evangelista; keep full text in file)
RUBRIC = """You are scoring the Feline Grimace Scale (FGS, Evangelista et al. 2019) on ONE cat face.
Score 5 action units, each on a 0/1/2 ordinal scale, where 0 = action unit ABSENT,
1 = action unit MODERATELY present OR you are UNCERTAIN, 2 = action unit MARKEDLY/OBVIOUSLY present.
Write the rationale BEFORE the score, citing the specific visible feature. Score each AU INDEPENDENTLY.

EAR POSITION: 0 = ears facing forward; 1 = ears slightly pulled apart or moderately rotated;
2 = ears flattened and rotated outwards. ...
ORBITAL TIGHTENING: 0 = eyes opened; 1 = eyes partially opened OR eye squinting beginning;
2 = eyes squinted/closed. ...
MUZZLE TENSION: 0 = relaxed, round muzzle; 1 = mild tension/oval; 2 = tense, elliptical muzzle. ...
WHISKERS CHANGE: 0 = loose and curved whiskers; 1 = slight straightening/forward;
2 = straight and moving forward. ...
HEAD POSITION: 0 = head above shoulder line; 1 = head aligned with shoulder line;
2 = head below shoulder line OR tilted down. ...

RULES:
- When a feature is genuinely ambiguous, DEFAULT THE SCORE TO 1 and set confidence='low'.
- If an AU region is occluded, blurred, out-of-frame, or the face is NON-FRONTAL, set abstain=true
  for THAT AU (still emit a best-guess score, but it will be down-weighted/routed to vet).
- Set image_quality='unusable' only if the whole face cannot be assessed.
- You may NOT output a total score or any pain/treatment decision. Output only the 5 AU records."""

SYSTEM_BLOCKS = [{
    "type": "text",
    "text": RUBRIC,
    "cache_control": {"type": "ephemeral"},   # use {"type":"ephemeral","ttl":"1h"} for long batches
}]
```

> **Frontal/quality gate is in the schema, not a separate model.** `image_quality` + per-AU `abstain` route non-frontal/occluded crops to the vet (FGS assumes a frontal view). Apply the ~12–15% box **margin expansion** at crop time (BUILD_PLAN §3.1) so ear tips / whiskers are not clipped before the VLM ever sees them — that is upstream of this section but is the precondition for trusting `abstain`.

---

### 4.3 In-code aggregation: sum 0–10 and the 0.39 flag (NEVER the VLM)

```python
# scripts/vlm/aggregate.py
AU_NAMES = ["ear", "orbital", "muzzle", "whiskers", "head"]

def fgs_sum(result_dict):                 # result_dict[au]['score'] in {0,1,2}
    return sum(result_dict[au]["score"] for au in AU_NAMES)   # 0..10

def analgesia_flag(s):                    # ratio = sum/10; clinical cut at >= 0.39 (~4/10)
    return (s / 10.0) >= 0.39

def any_abstain(result_dict):             # for routing/triage, not a clinical output
    return any(result_dict[au]["abstain"] for au in AU_NAMES) \
        or result_dict["image_quality"] != "frontal_clear"
```

This is the **same decode contract** the engine's Gate-4 unit test pins (`decode -> per-AU 0–2 -> sum -> 0.39`). The VLM path and the CORN path must produce identical sums from identical 5-atom inputs; reuse this exact function in both so there is one threshold definition in the repo. The 0.39 flag here is **triage decision-support only**, never an autonomous analgesia trigger, and the sum it rides on is an **inspected-not-validated** quantity (the word "graded" is struck from every validated-claim sentence).

---

### 4.4 Bulk run: Message Batches API + prompt caching (~2040 images)

The full-corpus weak-labeling (~50% cheaper, async, 24h window) is **blocked until Gate 1-B GO**. Build one batch request per image; the cached rubric system block is shared across all requests.

```python
# scripts/vlm/batch_submit.py
import os, json, pandas as pd
from anthropic import Anthropic
from .schema import FGS_TOOL, TOOL_CHOICE
from .rubric import SYSTEM_BLOCKS
from .call import encode_face

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = os.environ["VLM_MODEL"]

def build_requests(manifest_csv, n_runs=1):
    df = pd.read_csv(manifest_csv)        # columns: image_id, path  (cat-disjoint folds already assigned)
    reqs = []
    for _, r in df.iterrows():
        for k in range(n_runs):           # n_runs>=3 only for the self-consistency subset (§4.5)
            reqs.append({
                "custom_id": f"{r.image_id}__run{k}",
                "params": {
                    "model": MODEL,
                    "max_tokens": 1200,
                    "temperature": 0.0 if n_runs == 1 else 1.0,   # spread for alpha
                    "system": SYSTEM_BLOCKS,                       # cache_control rides here
                    "tools": [FGS_TOOL],
                    "tool_choice": TOOL_CHOICE,
                    "messages": [{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64",
                         "media_type": "image/jpeg", "data": encode_face(r.path)}},
                        {"type": "text", "text": f"Score this cat face. tag={r.image_id}__run{k}"},
                    ]}],
                },
            })
    return reqs

def submit(manifest_csv, n_runs=1):
    batch = client.messages.batches.create(requests=build_requests(manifest_csv, n_runs))
    print("batch.id =", batch.id, "status =", batch.processing_status)
    return batch.id
```

```python
# scripts/vlm/batch_collect.py  — poll, then stream the .jsonl results into parquet
import os, time, json, pandas as pd
from anthropic import Anthropic
from .schema import FGSResult
from .aggregate import fgs_sum, analgesia_flag, any_abstain

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def wait_and_collect(batch_id, out_parquet, poll_s=60):
    while True:
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":          # SDK terminal state is "ended"
            break
        print(b.request_counts); time.sleep(poll_s)
    rows = []
    for entry in client.messages.batches.results(batch_id):     # SDK streams the .jsonl by custom_id
        cid = entry.custom_id
        if entry.result.type != "succeeded":
            rows.append({"custom_id": cid, "error": entry.result.type}); continue
        msg = entry.result.message
        tu = next((c for c in msg.content if c.type == "tool_use"), None)
        if tu is None:
            rows.append({"custom_id": cid, "error": "no_tool_use"}); continue
        try:
            res = FGSResult(**tu.input)
        except Exception as e:
            rows.append({"custom_id": cid, "error": f"schema:{e}"}); continue   # datasheet provenance
        d = res.model_dump()
        s = fgs_sum(d)
        rows.append({
            "custom_id": cid,
            **{f"{au}_score": d[au]["score"] for au in AU_NAMES},
            **{f"{au}_abstain": d[au]["abstain"] for au in AU_NAMES},
            "image_quality": d["image_quality"],
            "fgs_sum": s, "analgesia_flag": analgesia_flag(s), "any_abstain": any_abstain(d),
        })
    pd.DataFrame(rows).to_parquet(out_parquet)
```

> `AU_NAMES` must be imported into `batch_collect.py` (from `.schema`) so the per-AU column build is keyed on the fixed 5-AU order, not on dict iteration.

**Cost/throughput notes (hosted-API only — no local VLM):** Batches give ~50% off and run async overnight; prompt caching makes the rubric ~free after the first cache write. ~2040 single-run images at small face-crop resolution is a few dollars. The `N>=3` self-consistency runs (§4.5) are restricted to a subset (e.g. the pilot images + active-learning candidates), not the whole corpus, to keep cost bounded. Identical behaviour from M4 or Colab T4 since no model runs locally.

---

### 4.5 N>=3 repeated runs -> ordinal Krippendorff alpha (prometheus-eval pattern)

A **vet-free** reliability axis: run each image `N>=3` times at `temperature=1.0` (varied seed tag) and compute **ordinal** Krippendorff alpha per AU over the (n_runs x n_items) matrix. This is (a) a cheap pre-screen *before* spending vet budget, (b) an **abstention signal** (low-alpha AUs route to vet), and (c) it complements — never replaces — the vs-vet kappa.

```python
# scripts/vlm/consistency.py
import numpy as np, pandas as pd, krippendorff
AU_NAMES = ["ear", "orbital", "muzzle", "whiskers", "head"]

def alpha_per_au(parquet_path):
    df = pd.read_parquet(parquet_path)                      # custom_id = "{image_id}__run{k}"
    df[["image_id", "run"]] = df["custom_id"].str.split("__run", expand=True)
    out = {}
    for au in AU_NAMES:
        # reliability_data: rows = runs (coders), cols = items (images); values in {0,1,2}
        wide = df.pivot(index="run", columns="image_id", values=f"{au}_score")
        out[au] = krippendorff.alpha(reliability_data=wide.values.astype(float),
                                     level_of_measurement="ordinal")
    return out   # e.g. {'ear':0.81,'orbital':0.88,'muzzle':0.52,'whiskers':0.47,'head':0.74}
```

Report alpha per AU alongside the kappa table. Low alpha on muzzle/whiskers is *expected* (these are the weakest AUs even for human experts, ICC 0.55–0.67) and feeds both the abstention curve and the per-AU confidence gating. Alpha is a self-consistency signal only — it is **never** offered as validation of the 0.39 flag (see §4.7).

---

### 4.6 Gate 1-B: per-AU quadratic kappa vs vet, gated on the CI LOWER BOUND

This is **headline-#1**. Sample size and per-AU kappa floors are **read from the Gate-0 output** (`data/gate0/power.json`) — the values below are FINAL_DIRECTION/BUILD_PLAN defaults that Gate 0 overwrites; do not hardcode them. Target is a Gate-0-sized pilot (default ~120 images, **>=50 pain-positive**), **cat-disjoint from the frozen Gate-3 hold-out** so the pilot never touches held-out individuals.

**4.6.1 Sample the pilot.**

```python
# scripts/pilot/sample_pilot.py  (sketch)
# - read pilot n + per-AU floors from data/gate0/power.json  (Gate 0 supplies both)
# - load Gate-1 per-CAT merge + Gate-3 frozen test CAT_ ids
# - exclude any image whose CAT_ id is in the test set (cat-disjoint, validated on CAT_ ids not raw CLIP/pHash)
# - stratify to >=50 VLM-pain-positive (fgs_sum/10 >= 0.39) + near-threshold (sum 3-5) + negatives
# - cap per-CAT count so no single individual (e.g. CAT_01, 84 clips) dominates the pilot
# - write data/pilot/pilot_manifest.csv  (image_id, path, cat_id, vlm_sum, fold)
```

**4.6.2 Vet review UI — triage order (value-per-hour).** Pre-fill the review tool (Roboflow review, or CVAT/Label Studio) with the 5 VLM scores **and rationales**; the vet **accepts/corrects flagged AUs only**, never re-scores from a blank slate. Triage order (BUILD_PLAN §3.2):

| Priority | What to review | Why |
|---|---|---|
| 1 | VLM pain-positive / near-threshold (`fgs_sum` 3–5/10) + the rare positive class | The 0.39 boundary and the scarce-positive cells drive every downstream CI |
| 2 | Low-reliability AUs **muzzle & whiskers** (ICC 0.55–0.67) and the most-diagnostic **orbital tightening** | These decide whether the graded artifact survives inspection or we pivot to binary |
| 3 | Low-alpha (§4.5) + cleanlab-flagged likely errors + multi-VLM disagreements | Confident-learning send-to-vet queue; highest expected label correction per minute |

Only **vet-confirmed** rows enter the kappa computation and (later) training.

**4.6.3 Compute per-AU quadratic kappa + bootstrap CI lower bound (m-rewardbench row-paired pattern + `weights='quadratic'`).** The m-rewardbench code is **nominal** (`labels=[0,1,2]` only fixes the class set); we **add `weights='quadratic'`** to make it ordinal — that addition is the contribution-grade number.

```python
# scripts/pilot/kappa_gate.py
import json, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
AU_NAMES = ["ear", "orbital", "muzzle", "whiskers", "head"]

def qwk(y_vlm, y_vet):
    return cohen_kappa_score(y_vlm, y_vet, labels=[0, 1, 2], weights="quadratic")  # ORDINAL

def bootstrap_qwk_lb(y_vlm, y_vet, n_boot=5000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed); n = len(y_vlm); stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try: stats.append(qwk(y_vlm[idx], y_vet[idx]))
        except ValueError: continue                       # degenerate resample (one class) -> skip
    lo = float(np.nanpercentile(stats, 100 * alpha))      # one-sided 95% LOWER BOUND
    return qwk(y_vlm, y_vet), lo

def load_floors(power_json="data/gate0/power.json"):
    # Gate-0 supplies per-AU floors + pilot n; defaults below are FINAL_DIRECTION placeholders ONLY.
    defaults = {"orbital": 0.60, "ear": 0.60, "head": 0.60,        # gate on LB >= 0.60
                "muzzle": 0.40, "whiskers": 0.40}                  # 0.40-0.60 = acceptable-with-caveat
    try:    return json.load(open(power_json))["au_kappa_floors"]
    except (FileNotFoundError, KeyError): return defaults

def run_gate(merged_csv, out_json):
    df = pd.read_csv(merged_csv)                           # one row/image: {au}_vlm, {au}_vet
    floors = load_floors()
    report, decision = {}, {}
    for au in AU_NAMES:
        k, lb = bootstrap_qwk_lb(df[f"{au}_vlm"].to_numpy(), df[f"{au}_vet"].to_numpy())
        report[au] = {"kappa": k, "ci_lb": lb, "floor": floors[au],
                      "pass": lb >= floors[au]}            # GATE ON THE LOWER BOUND, not the point
    # GO requires orbital/ear/head LB >= floor; muzzle/whiskers below 0.40 => caveat or drop that head
    decision["graded_go"] = all(report[au]["pass"] for au in ["orbital", "ear", "head"])
    decision["muzzle_whiskers"] = {au: ("ok" if report[au]["ci_lb"] >= floors[au] else "drop_head")
                                   for au in ["muzzle", "whiskers"]}
    json.dump({"per_au": report, "decision": decision}, open(out_json, "w"), indent=2)
    return decision
```

**Gate rule (strict):** **GATE ON THE CI LOWER BOUND, NOT THE POINT ESTIMATE** — at the pilot `n`, a 0.60 floor is statistically indistinguishable from a true 0.47, so a point-estimate gate is not a gate (FINAL_DIRECTION §3).

| AU | Floor (gate on CI lower bound) | On failure |
|---|---|---|
| orbital | LB >= 0.60 | **Drop graded entirely** -> binary-plus-wrapper fallback (kappa-as-method + confound + welfare curve still stand) |
| ear | LB >= 0.60 | same |
| head | LB >= 0.60 | same |
| muzzle | 0.40–0.60 acceptable-with-caveat; LB < 0.40 -> drop this head | proceed on surviving AUs; state how dropping a head shifts the reachable 0–10 range and whether 0.39 (~4/10) is even reachable |
| whiskers | 0.40–0.60 acceptable-with-caveat; LB < 0.40 -> drop this head | same |

---

### 4.7 Circularity firewall (state it in code comments and the paper)

- **QWK-vs-VLM is NEVER validation.** The per-AU kappa here measures *VLM-vs-vet agreement* — a forward-looking method finding ("can a frozen VLM weak-label feline FGS AUs at human-rater agreement"). It is **not** evidence that the 0.39 flag is correct.
- A QWK computed against *VLM-derived* labels would measure only the model re-learning the VLM's heuristic — **forbidden as a validation metric.**
- **sens/spec at 0.39 are estimated ONLY on vet-confirmed labels** (the §3.3 firewall), never on weak labels, and the confidence intervals are Clopper-Pearson/bootstrap.
- **Graded-sum QWK is an internal inspection metric only** — the artifact ships inspected-not-validated; the word "graded" is struck from every validated-claim sentence, and no binned reliability diagram is drawn on the 0–10 sum (the sum's calibration lives on the distributional RPS/ClasswiseECE path, not here).
- The **labeler is self-justified by this pilot** ("we selected the VLM by measured per-AU kappa on the vet anchor"); **do NOT cite Sci Rep 2025** (FACTCHECK C54).

### 4.8 Anti-benchmark / reporting discipline

Always print the **distinct-pain-CAT denominator** next to every kappa (the pilot's positives may concentrate in few individuals, e.g. CAT_01); report **bootstrap CI lower bounds**, never bare point kappas; **augmented copies never enter the pilot N**. Never frame any number as "we beat 77/79/95%", and never carry a horse-grimace figure as a headline — horse AUs are decode scaffolding only. Frame every output as decision-support triage, never an autonomous analgesia trigger. On NO-GO, the honest paper is binary-plus-wrapper + kappa-as-method (a mediocre-but-first per-AU kappa is still a publishable measurement) + the confound protocol + the welfare decision curve + the one-sided NPV lower-bound abstention curve (the word "guaranteed" is banned).

---

## 5. Phase B — model architecture & training method (frozen DINOv2 + 5 CORN heads)

> **Scope of this section.** This is the *engine* — frozen DINOv2 ViT-S/14 → 5 per-AU CORN ordinal heads → per-AU 0/1/2 → sum 0–10 → 0.39 point decision. Per FINAL_DIRECTION §A, **this engine is conceded as plumbing and is NEVER claimed as novel**; it exists only to carry the two portable-method headlines (VLM-as-AU-rater per-AU κ; confound-attribution protocol) and the binary-plus-wrapper spine. **The binary pain head + abstention is the v1 spine; the 0–10 CORN sum ships INSPECTED-NOT-VALIDATED** — the word "graded" is struck from every validated-claim sentence, and no sum QWK is ever reported as validation. This section specifies exactly what to *build* and *run* on an Apple M4 (MPS, no CUDA); the calibration/abstention/decision-curve wrapper is Phase C and is only hooked here.

### 5.0 Prerequisites & gate position

This section runs **only after** these gates clear (strict order from FINAL_DIRECTION §6). Do **not** train a head before Gate 4 passes.

| Gate | Must be green before §5 | Where in §5 |
|---|---|---|
| **G0** power-calc + vet-budget integer committed | yes (blocks all quantitative) | budget feeds §5.6 label split |
| **G1** per-CAT merge (fold by `CAT_` id) | yes | §5.6 cat-disjoint training |
| **G3** frozen hashed cat-disjoint hold-out + CI-abort | yes | §5.6 / §5.8 read fold CSV, never test manifest |
| **G4** MPS compute-correctness (logit parity + CORN-decode→sum→0.39 unit test) | **BLOCKING — built IN §5.5/§5.7** | §5.5 (parity), §5.7 (decode unit test) |
| **G5** alignment / NME (crop quality) | yes | §5.2 feeds the crop into the cache |

Engine work order inside §5: **(1)** install + freeze backbone (§5.1) → **(2)** cache features to disk (§5.2) → **(3)** build 5-head CORN module + vendored loss (§5.3–5.4) → **(4)** distributional decode (§5.5) → **(5)** MPS logit parity + decode unit test (§5.7, **G4**, BLOCKING) → **(6)** horse 1-epoch smoke + train heads on the cat label split (§5.6) → only then Phase C wrapper.

### 5.1 Frozen DINOv2 ViT-S/14 backbone

**Choice (conceded plumbing):** `dinov2_vits14` — hidden size **384**, patch 14, forward-only. Frozen because the supervised set is ~120–300 vet/VLM labels; full fine-tune at this n is not viable. Only the ~5 light heads train.

> **MPS/Python caveat (BLOCKING precheck):** the project venv is **Python 3.9.6**. Confirm `torch>=2.2` with a working MPS backend installs *and* `torch.backends.mps.is_available()` returns `True` on this interpreter before proceeding. If the wheel resolves but MPS is unavailable, bump the venv to 3.11+ (assumption flagged below) — do not silently fall back to CPU for training-correctness checks.

Install into the project venv:

```bash
# from project root /Users/mingrath/ghq/github.com/mingrath/cat-fgs-llm
.venv/bin/pip install "torch>=2.2" torchvision pillow numpy scikit-learn tqdm
.venv/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
# coral-pytorch is installed ONCE only to cross-check the vendored decode in the Gate-4 test (§5.7),
# then dropped as a runtime dep:
.venv/bin/pip install coral-pytorch
```

**Freeze idiom** (pattern from `chandar-lab/semantic-wm`, GITHUB_MINE P2.2). Derive patch size and hidden size from the model — do **not** hardcode 14/384 in load-bearing code; the constants in tables are for the reader.

`engine/backbone.py`:
```python
import torch, torch.nn as nn

def load_frozen_dinov2(device: str = "mps"):
    # torch.hub ViT-S/14; no registers variant for the v1 spine
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    m.requires_grad_(False)      # freeze ALL params
    m.eval()                     # disable dropout/BN-update; deterministic features
    m.to(device)
    hidden = m.embed_dim         # 384 for ViT-S; assert below
    assert hidden == 384, f"expected 384, got {hidden}"
    return m, hidden

@torch.inference_mode()
def extract(m, x):               # x: [B,3,H,W] already normalized + resized to /14 grid
    out = m.forward_features(x)  # dict
    cls   = out["x_norm_clstoken"]      # [B,384]   (CLS, == index 0 of the token seq)
    patch = out["x_norm_patchtokens"]   # [B,N,384] (patch tokens; HF-equiv of [:,1:,:])
    return cls, patch
```

> **CLS-vs-patch note for the cache (GITHUB_MINE P2.2 `Semi-supervised-learning`):** the HF `Dinov2Model` returns CLS at sequence index 0, so patch tokens are `last_hidden_state[:, 1:, :]`. The `torch.hub` API above exposes them already split. **Default pooling for the heads = CLS (`x_norm_clstoken`).** Cache mean-pooled patch tokens **as a second column** (`patch.mean(1)`) so the §5.6 A/B (CLS vs mean-patch) is a config flag, not a re-cache.

**Input transform** (DINOv2 ImageNet stats; crop comes from the Phase-A box after the G5 alignment step):

```python
from torchvision import transforms
IMNET_MEAN, IMNET_STD = (0.485,0.456,0.406), (0.229,0.224,0.225)
# 518 = 37*14 → 37x37 patch grid; divisible-by-14 is mandatory
preprocess = transforms.Compose([
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(IMNET_MEAN, IMNET_STD),
])
```

### 5.2 FEATURE CACHING to disk (the big M4 win)

**Principle (GITHUB_MINE P2.2, `microsoft/Semi-supervised-learning` `only_feat`/`only_fc` split):** the backbone is frozen, so its output for a given crop never changes. Run **one** pass over every crop, write pooled features to disk, then train the heads off the cache with **zero backbone forwards per epoch**. On M4 this turns each head epoch from minutes into milliseconds and makes the §5.6 sweep cheap.

`engine/cache_features.py` (run once per crop set: cat split, and separately the horse warm-start set):

```python
import torch, numpy as np
from pathlib import Path
from PIL import Image
from backbone import load_frozen_dinov2, extract, preprocess

ROOT = Path("/Users/mingrath/ghq/github.com/mingrath/cat-fgs-llm")
def build_cache(manifest_csv, out_npz, device="mps"):
    # manifest_csv columns: img_path, cat_id, fold, y_ear,y_orbital,y_muzzle,y_whiskers,y_head, y_pain, is_vet_clean
    m, hidden = load_frozen_dinov2(device)
    rows, cls_feats, patch_feats = [], [], []
    import csv
    with open(manifest_csv) as f:
        for r in csv.DictReader(f):
            img = preprocess(Image.open(r["img_path"]).convert("RGB"))[None].to(device)
            cls, patch = extract(m, img)
            cls_feats.append(cls.squeeze(0).cpu().numpy())
            patch_feats.append(patch.mean(1).squeeze(0).cpu().numpy())  # mean-pool patches
            rows.append(r)
    np.savez_compressed(
        out_npz,
        cls=np.stack(cls_feats).astype(np.float32),       # [N,384]
        patch_mean=np.stack(patch_feats).astype(np.float32),
        img_path=[r["img_path"] for r in rows],
        cat_id=[r["cat_id"] for r in rows],
        fold=[r["fold"] for r in rows],
        # labels: -1 sentinel where VLM/vet did not score this AU
        y=np.array([[int(r[f"y_{au}"]) for au in
                     ("ear","orbital","muzzle","whiskers","head")] for r in rows], dtype=np.int64),
        y_pain=np.array([int(r["y_pain"]) for r in rows], dtype=np.int64),
        is_vet_clean=np.array([int(r.get("is_vet_clean", 0)) for r in rows], dtype=np.int64),
    )
    print(f"cached {len(rows)} crops -> {out_npz}")
```

Cache files (G3-hashed; the trainer reads `fold`, never a raw test manifest):
```
artifacts/cache/cat_features.npz        # ~120–300 cat crops, CLS + mean-patch + 5-AU labels + y_pain
artifacts/cache/horse_features.npz      # 5-horse genuine 0/1/2 warm-start (decode scaffolding ONLY)
```

> **Datasheet line:** record DINOv2 commit hash, `preprocess` params, device, and torch version in `artifacts/cache/PROVENANCE.json` — the cache is an input to every reported number.

### 5.3 The 5 CORN heads — architecture

CORN over CORAL/softmax: rank-consistency without CORAL's shared-bias capacity restriction (BUILD_PLAN §3, [CORN arXiv 2111.08851]). **K=3 levels per AU ⇒ K−1 = 2 logits per AU.** Build the **wide single-head (B)** — it gives the cleanest `torch.split` plumbing from `mueller-franzes/odelia_breast_mri`'s `CornLossMulti`:

| Layout | Shape | Notes |
|---|---|---|
| (A) `nn.ModuleList` of 5 | 5× `Linear(384, 2)` | matches `qde-ordinality` `RobertaOrdinalHead` pattern; explicit per-AU |
| **(B) one wide head (BUILD this)** | `Linear(384, 10)` then `torch.split(out, 2, dim=1)` | Σ(K_au−1)=5×2=10; one matmul; per-AU chunks fed to per-AU `corn_loss` |

The 5 AUs (FACTCHECK: Min-aggregation for whiskers AND head): **ear, orbital, muzzle, whiskers, head**.

`engine/heads.py`:
```python
import torch, torch.nn as nn

AUS = ("ear","orbital","muzzle","whiskers","head")
K = 3                              # levels 0/1/2 per AU
N_LOGITS = len(AUS) * (K-1)        # = 10

class CornMultiHead(nn.Module):
    def __init__(self, in_dim=384, n_aus=5, k=3, p_drop=0.1):
        super().__init__()
        self.k = k; self.n_aus = n_aus
        self.drop = nn.Dropout(p_drop)        # light reg on tiny linear probe
        self.proj = nn.Linear(in_dim, n_aus*(k-1))   # 384 -> 10
        # trunc_normal_ init for stable small-data linear-probe
        # (BenediktAlkin/vtab1k-pytorch, GITHUB_MINE P2.2)
        nn.init.trunc_normal_(self.proj.weight, std=2e-5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feat):                  # feat: [B,384] (CLS or mean-patch)
        logits = self.proj(self.drop(feat))   # [B,10]
        return list(torch.split(logits, self.k-1, dim=1))  # 5 × [B,2]
```

The wrapper's **binary pain head** (the v1 spine) is a sibling `Linear(384, 1)` trained with `BCEWithLogitsLoss(pos_weight=...)` on `y_pain`; it shares the cache, is independent of CORN, and is what the calibration/abstention/decision-curve wrapper (Phase C) actually operates on. The 0.39 sum is the *inspected* path.

### 5.4 Vendored `corn_loss` (auditable, ~40 lines, no pip dep)

Vendor the loss so the datasheet shows exactly what trained the heads (GITHUB_MINE P2.2: inline from `ludwig-ai/ludwig` `corn.py`, torch+F only, MIT). Cross-check it once against `coral_pytorch.losses.corn_loss` in the unit test (§5.7), then drop the dep.

`engine/corn.py`:
```python
import torch
import torch.nn.functional as F

def corn_loss(logits, y, num_classes):
    """Conditional ordinal (CORN) loss for ONE AU.
    logits: [B, num_classes-1]   y: [B] in {0,...,num_classes-1}
    Shi, Cao & Raschka 2021. Vendored (MIT, ludwig-ai/ludwig)."""
    sets = []
    for i in range(num_classes - 1):
        label_mask = (y > i - 1)                 # samples still "in play" at rank i
        label_tensor = (y[label_mask] > i).to(torch.int64)
        if label_mask.sum() == 0:
            continue
        sets.append((label_mask, label_tensor))
    losses = 0.0
    n = 0
    for i, (mask, lab) in enumerate(sets):
        pred = logits[mask, i]                    # conditional logit for rank i
        loss = -torch.sum(
            F.logsigmoid(pred) * lab + (F.logsigmoid(pred) - pred) * (1 - lab)
        )
        losses = losses + loss
        n += mask.sum().item()
    return losses / max(n, 1)

def corn_label_from_logits(logits):
    """Hard decode for the 0.39 POINT decision ONLY.
    logits: [B, num_classes-1] -> labels [B]. predict = sum_k( cumprod(sigmoid)[k] > 0.5 )."""
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)         # P(y>0), P(y>0 & y>1), ...
    return torch.sum(probas > 0.5, dim=1)

def corn_cumprobs(logits):
    """SOFT path (DEFAULT, FINAL_DIRECTION §E.1). Returns cumulative P(rank>k) per level.
    DO NOT hard-decode for calibration — these soft probs feed pmf/RPS/ECE."""
    return torch.cumprod(torch.sigmoid(logits), dim=1)   # [B, num_classes-1]
```

**Multi-head loss** (`CornLossMulti` plumbing): sum the 5 per-AU `corn_loss`, skipping AUs with the `-1` sentinel (an AU the VLM/vet did not score on that image):

```python
def multi_corn_loss(logits_list, y, num_classes=3, au_weights=None):
    # logits_list: 5 × [B,2]; y: [B,5] with -1 = not scored
    total = 0.0
    for a in range(len(logits_list)):
        mask = y[:, a] >= 0
        if mask.sum() == 0:
            continue
        l = corn_loss(logits_list[a][mask], y[mask, a], num_classes)
        w = 1.0 if au_weights is None else au_weights[a]
        total = total + w * l
    return total
```

> Optionally upweight `muzzle`/`whiskers` (`au_weights=[1,1,1.5,1.5,1]`) for imbalance — these are the AUs FINAL_DIRECTION §7 flags as κ-collapse risks; the KILL/PIVOT path (§5.8) governs what happens if they collapse.

### 5.5 Distributional decode — per-AU pmf → convolve to the 0–10 sum pmf

**This is the DEFAULT path (FINAL_DIRECTION §E.1).** Keep soft cumulative `P(rank>k)=cumprod(sigmoid(logits))`; build each AU's pmf over {0,1,2}; **convolve the 5 pmfs into one pmf over the 0–10 sum**. From that distribution: **RPS-on-the-sum (one scalar, bootstrap CI) + per-AU ClasswiseECE**. `argmax`/hard-decode is reserved for the **0.39 POINT decision only**. **No binned reliability diagram on the 11-atom sum** (degenerate at ~11 atoms; per-bin SE ±0.18–0.26).

`engine/decode.py`:
```python
import numpy as np, torch

def au_pmf_from_cumprobs(cum):          # cum: [B,2] = [P(y>0), P(y>0 & y>1)]
    p_gt0, p_gt1 = cum[:,0], cum[:,1]
    p0 = 1 - p_gt0
    p1 = p_gt0 - p_gt1                   # = P(y>0) - P(y>1)
    p2 = p_gt1
    pmf = torch.stack([p0, p1, p2], dim=1)          # [B,3]
    return torch.clamp(pmf, min=0)      # guard tiny negatives from float error

def sum_pmf(au_pmfs):                    # au_pmfs: list of 5 × [B,3] (numpy)
    # convolve 5 pmfs -> pmf over 0..10 (length 11)
    B = au_pmfs[0].shape[0]
    out = np.zeros((B, 11))
    for b in range(B):
        acc = np.array([1.0])
        for a in range(5):
            acc = np.convolve(acc, au_pmfs[a][b])
        out[b] = acc / acc.sum()         # renormalize
    return out                           # [B,11], sums to 1 over the 0..10 sum
```

The 0.39 point decision (sum threshold ≈ 4/10):
```python
def point_sum(logits_list):              # hard decode for the decision ONLY
    labels = [corn_label_from_logits(l) for l in logits_list]   # 5 × [B]
    s = torch.stack(labels, dim=1).sum(1)        # [B] in 0..10
    return s, (s.float()/10.0 >= 0.39)           # painful flag
```

> The operating point is **fixed pain-recall ≥0.90** (Evangelista anchor), selected inside train folds, **NOT** Youden-J/F1. That selection, the welfare-asymmetric decision curve (undertreat:overtreat swept as a *range*), and the one-sided-95%-NPV abstention curve are **Phase C**, fit on **vet-confirmed labels only** (circularity firewall: sum-QWK vs VLM is never validation).

### 5.6 Training method — co-teaching small-loss on VLM mass + clean vet anchor

**Data composition (FINAL_DIRECTION §6 budget; G0 integer):** the VLM-weak-labeled mass (all scored crops) + the vet-confirmed clean anchor (~120–300, ≥50 pain-positive). Train **NOT confirmed-only** (too few) **and NOT naive-all** (bakes in the VLM under-estimation bias) — use **co-teaching / small-loss selection** (`bhanML/Co-teaching`, BUILD_PLAN §3): two heads, each selects the small-loss subset for the other; vet-clean rows (`is_vet_clean==1`) are **never** dropped from either selection.

**Folds:** cat-disjoint, read from the G1 per-`CAT_` merge + G3 frozen fold CSV. No individual straddles a fold. Augmented copies never enter a reported N.

`engine/train_heads.py` (cache-only loop — no backbone in the inner loop):
```python
import numpy as np, torch
from torch.utils.data import TensorDataset, DataLoader
from heads import CornMultiHead
from corn import multi_corn_loss

def load_split(npz, train_folds=(0,1,2), pool="cls"):
    d = np.load(npz, allow_pickle=True)
    feat = d[pool]                                      # "cls" or "patch_mean" -> [N,384]
    fold = d["fold"].astype(int); y = d["y"]; clean = d["is_vet_clean"]
    tr = np.isin(fold, train_folds)
    return (torch.tensor(feat[tr]), torch.tensor(y[tr]), torch.tensor(clean[tr]))

def train(npz, device="mps", epochs=80, pool="cls", seed=0):
    torch.manual_seed(seed)
    Xtr, ytr, clean = load_split(npz, pool=pool)
    Xtr = Xtr.to(device)
    dl = DataLoader(TensorDataset(Xtr, ytr.to(device), clean.to(device)),
                    batch_size=32, shuffle=True, num_workers=0)  # num_workers=0 on small data / MPS
    # two heads for co-teaching
    netA = CornMultiHead().to(device); netB = CornMultiHead().to(device)
    optA = torch.optim.AdamW(netA.parameters(), lr=1e-3, weight_decay=1e-2)
    optB = torch.optim.AdamW(netB.parameters(), lr=1e-3, weight_decay=1e-2)
    tau = est_noise_rate            # 1 - tau = floor keep-rate ~ estimated VLM mislabel rate (Gate 1-B); NOT a fixed 0.5
    for ep in range(epochs):
        # canonical Co-teaching R(T) (Han et al. 2018): keep-rate ramps 1.0 -> (1-tau) over num_gradual=10 epochs
        keep = 1.0 - tau*min(1.0, ep/10)                # inclusive -> selective (memorization effect)
        for xb, yb, cb in dl:
            la = netA(xb); lb = netB(xb)
            # per-sample loss for selection (sum over scored AUs)
            with torch.no_grad():
                pa = torch.stack([_persample(la[a], yb[:,a]) for a in range(5)]).sum(0)
                pb = torch.stack([_persample(lb[a], yb[:,a]) for a in range(5)]).sum(0)
                k = int(keep*len(xb))
                # always keep vet-clean rows; pick small-loss among the rest
                selA = _select(pb, cb, k)   # B selects for A (small loss by B)
                selB = _select(pa, cb, k)
            optA.zero_grad(); multi_corn_loss([l[selA] for l in la], yb[selA]).backward(); optA.step()
            optB.zero_grad(); multi_corn_loss([l[selB] for l in lb], yb[selB]).backward(); optB.step()
    return netA   # report netA; netB is the teaching partner
```
(`_persample` = per-row `corn_loss` with `reduction='none'`, summed over scored AUs; `_select` returns indices of vet-clean ∪ smallest-loss-non-clean up to k. Keep these ~15 lines in the same file.)

**Hyperparameters — frozen-backbone linear probe, ~120–300 labels, M4/MPS:**

| Knob | Value | Rationale |
|---|---|---|
| optimizer | `AdamW(lr=1e-3, weight_decay=1e-2)` | tiny linear probe; AdamW stable |
| epochs | 60–100 (default 80) | converges fast on cached feats; early-stop on val multi-corn loss |
| batch size | 32 (whole train set fits one batch — fine) | features are [N,384] in RAM |
| dropout | 0.1 | light reg |
| head init | `trunc_normal_(std=2e-5)` | small-data linear-probe stability |
| co-teach keep-rate | **1.0→(1−τ) ramp** over 10 epochs (canonical Han et al. R(T), inclusive→selective); τ = est. VLM noise rate; vet-clean always kept | small-loss on noisy VLM mass |
| pooling A/B | `cls` (default) vs `patch_mean` | config flag; cache holds both |
| loss ablation | `corn` (default) vs CORAL | one-flag switch (`life2vec` enum pattern) |
| seeds | 5 seeds, report mean ± bootstrap CI | n is small; single seed is noise |

The whole sweep (pooling × loss × 5 seeds) runs in **minutes** on M4 because the backbone never re-runs.

### 5.7 MPS logit parity + Gate-4 CORN-decode→sum→0.39 unit test (BLOCKING)

**Two jobs, both before any cat number is believed:**

**(a) Horse 1-epoch smoke (decode SCAFFOLDING ONLY — never a headline number, FINAL_DIRECTION §7 cut-table).** The 5-horse genuine 0/1/2 labels (3/5 AUs populated; the other 2 carry the `-1` sentinel so `multi_corn_loss` skips them) exercise the *exact* `cache → CornMultiHead → multi_corn_loss → decode → sum` path on real ordinal labels. Run `cache_features.py` on the horse crops → `train_heads.py` for 1 epoch on MPS → confirm loss decreases and `corn_label_from_logits` produces in-range 0/1/2. **No horse number enters abstract/results/transfer table** — a methods/appendix sentence only ("we unit-validated the decode→sum→threshold path on 5-horse genuine 0/1/2 labels").

**(b) Gate-4 synthetic CORN-decode→sum→0.39 unit test (BLOCKING).** This is *the* Gate 4 artifact. It pins the decode against **values verified by hand** so a silently-wrong MPS path cannot ship. The decode rule is `predict = sum_k( cumprod(sigmoid(logits))[k] > 0.5 )`; the anchors below are computed for the **actual K=3 (2-logit) head shape**, not a borrowed K=5 example.

`tests/test_gate4_decode.py`:
```python
import torch, numpy as np
from engine.corn import corn_label_from_logits, corn_cumprobs
from engine.decode import au_pmf_from_cumprobs, sum_pmf, point_sum

def test_corn_decode_anchor_k3():
    # K=3 (2 logits/AU). cumprod-of-sigmoid > 0.5 decode, verified by hand:
    #   [ 9,  9] -> sig~1,1   -> cum 1,1   -> 2
    #   [ 9, -9] -> sig~1,0   -> cum 1,0   -> 1
    #   [-9, -9] -> sig~0,0   -> cum 0,0   -> 0
    logits = torch.tensor([[ 9.,  9.],
                           [ 9., -9.],
                           [-9., -9.]])
    out = corn_label_from_logits(logits)
    assert out.tolist() == [2, 1, 0], out.tolist()

def test_corn_decode_monotone_in_logits():
    # rank consistency: raising any logit cannot lower the decoded level
    base = torch.tensor([[0.3, -0.2]])
    up   = base + torch.tensor([[0.0, 2.0]])
    assert corn_label_from_logits(up).item() >= corn_label_from_logits(base).item()

def test_vendored_matches_coral_pytorch():
    # one-time cross-check that the vendored decode == coral-pytorch reference,
    # then the dep is dropped. Skips cleanly if coral-pytorch is not installed.
    coral = __import__("importlib").util.find_spec("coral_pytorch")
    if coral is None:
        import pytest; pytest.skip("coral-pytorch not installed")
    from coral_pytorch.dataset import corn_label_from_logits as ref
    x = torch.randn(16, 2)
    assert torch.equal(corn_label_from_logits(x), ref(x))

def test_pmf_sums_to_one():
    cum = corn_cumprobs(torch.randn(8,2))            # [B,2] for K=3
    pmf = au_pmf_from_cumprobs(cum).numpy()
    assert np.allclose(pmf.sum(1), 1.0, atol=1e-5)
    assert (pmf >= -1e-6).all()                      # no negative atoms

def test_sum_pmf_is_distribution_over_0_10():
    cums = [corn_cumprobs(torch.randn(4,2)) for _ in range(5)]
    pmfs = [au_pmf_from_cumprobs(c).numpy() for c in cums]
    S = sum_pmf(pmfs)                                 # [4,11]
    assert S.shape == (4,11)
    assert np.allclose(S.sum(1), 1.0, atol=1e-5)

def test_point_decision_at_039():
    hi = [torch.tensor([[ 9.,  9.]]) for _ in range(5)]   # decodes to 2 each -> sum 10
    lo = [torch.tensor([[-9., -9.]]) for _ in range(5)]   # decodes to 0 each -> sum 0
    s_hi, f_hi = point_sum(hi); s_lo, f_lo = point_sum(lo)
    assert s_hi.item() == 10 and bool(f_hi.item()) is True
    assert s_lo.item() == 0  and bool(f_lo.item()) is False
    # boundary: ratio>=0.39 means sum>=3.9 -> sum 4 painful, sum 3 not
    assert (4/10) >= 0.39 and (3/10) < 0.39
```

`tests/test_gate4_mps_parity.py` (logit parity; runs the §5.1 backbone):
```python
import torch
from engine.backbone import load_frozen_dinov2, extract, preprocess
from PIL import Image

CROP = "datasets/<fill-in-a-real-post-G5-aligned-crop>.png"  # MUST be a real aligned crop

def test_mps_cpu_logit_parity():
    img = preprocess(Image.open(CROP).convert("RGB"))[None]
    m_cpu, _ = load_frozen_dinov2("cpu");  c_cpu,_ = extract(m_cpu, img)
    m_mps, _ = load_frozen_dinov2("mps");  c_mps,_ = extract(m_mps, img.to("mps"))
    torch.testing.assert_close(c_cpu, c_mps.cpu(), rtol=1e-3, atol=1e-3)
```

Run the gate:
```bash
.venv/bin/python -m pytest tests/test_gate4_decode.py tests/test_gate4_mps_parity.py -q
# ALL must pass before any §5.6 training output is trusted. A red Gate 4 BLOCKS Phase B.
```

> **If the vendored decode and `coral_pytorch` disagree** (`test_vendored_matches_coral_pytorch`), the vendoring is wrong — fix `corn.py` before training; do not "fix" the expected value to match a broken decode.

### 5.8 Outputs of Phase B (and what is / isn't claimed)

| Artifact | Path | Claim status |
|---|---|---|
| Cached features | `artifacts/cache/cat_features.npz`, `horse_features.npz` | input only |
| Trained CORN heads (5 seeds) | `artifacts/heads/corn_{pool}_{seed}.pt` | engine — **conceded plumbing** |
| Binary pain head | `artifacts/heads/pain_binary_{seed}.pt` | **v1 spine** (feeds Phase C wrapper) |
| Per-AU soft cumprobs on held-out | `artifacts/scores/cumprobs.npz` | feeds Phase C calibration/abstention |
| 0–10 sum pmf (convolved) | `artifacts/scores/sum_pmf.npz` | **inspected-not-validated** |
| Gate-4 test report | `artifacts/gates/gate4.txt` | BLOCKING pass/fail |

**Explicit claim discipline carried out of §5:**
- The **binary pain decision + abstention** is the validated spine; sens/spec at 0.39 are estimated in Phase C **only on vet-confirmed labels** (circularity firewall).
- The **0–10 sum is INSPECTED-NOT-VALIDATED** — its QWK-vs-VLM is *never* validation; it is reported qualitatively with the circularity stated. The abstract holds with this output dropped.
- The **engine is never claimed novel.** A "swap-the-backbone + extra metrics" reviewer finds nothing to kill here, by design.
- **KILL/PIVOT (FINAL_DIRECTION §7):** if orbital/ear/head κ-CI-lower-bound (Gate 1-B, which fires on the CI lower bound, not the point estimate) falls below the pre-registered floor → drop the 0–10 layer entirely, ship the calibrated binary + abstention + confound audit; the 5-head module still trains and is shipped as the inspected artifact, but no graded claim is made.

---

## 6. The trustworthiness wrapper (the headline) — every metric, defined

> **Read this first.** The engine (frozen DINOv2 ViT-S/14 → 5 per-AU CORN heads → 0/1/2 → sum 0–10 → 0.39 decision) is **conceded plumbing** — never claimed novel. *This* section is the contribution. The v1 spine is **binary-plus-wrapper**; the 0–10 layer is consumed here only for **inspected-not-validated** artifacts (RPS, ClasswiseECE, EBPG). Every validated number in pillars 1, 3, 4 rides on **vet-confirmed labels only** (circularity firewall); QWK-vs-VLM is *never* validation. All pillars run **post-hoc on cached held-out scores** — no backbone forward, M4/MPS-friendly, mostly CPU/sklearn/numpy. Each pillar ends with the **kill criterion** it feeds.

### 6.0 Setup — environment, inputs, files

One pinned venv for the whole wrapper. Run from repo root.

```bash
# pip deps (NOT git) — GITHUB_MINE §3 "Exact next commands"
pip install torch-uncertainty cleanlab probmetrics netcal relplot dcurves \
            mapie krippendorff coral-pytorch scikit-learn statsmodels \
            segment-anything opencv-python
# reference repos to clone (read/copy, not pip) — GITHUB_MINE §3
git clone https://github.com/MSKCC-Epi-Bio/dcurves           ./ref/dcurves
git clone https://github.com/scikit-learn-contrib/MAPIE      ./ref/mapie
git clone https://github.com/facebookresearch/reliable_vqa   ./ref/reliable_vqa
git clone https://github.com/MadryLab/backgrounds_challenge  ./ref/bg-challenge
git clone https://github.com/visinf/beyond-accuracy          ./ref/beyond-accuracy
git clone https://github.com/haofanwang/Score-CAM            ./ref/score-cam
git clone https://github.com/ENSTA-U2IS-AI/torch-uncertainty ./ref/torch-uncertainty
git clone https://github.com/EFS-OpenSource/calibration-framework ./ref/netcal
git clone https://github.com/apple/ml-calibration            ./ref/relplot
git clone https://github.com/cleanlab/cleanlab               ./ref/cleanlab
```

> **API-drift guard (do before coding any pillar).** Three import surfaces below changed across versions and MUST be pinned against the *installed* wheel, not memory: MAPIE risk control (§6.4: the controller class, `method='ltt'`, the `negative_predictive_value` risk function, and the `.fit`/`.predict(alpha=, delta=)` signature), netcal `ClasswiseECE` (§6.2), and torch-uncertainty selective-classification metrics (§6.4: `AURC`, `CovAt5%Risk`, `RiskAt80%Cov` are torchmetrics-style `.update`/`.compute` classes, *not* one-shot functions). Confirm each via Context7 or the cloned `./ref/` example before writing the script.

**Inputs the wrapper consumes** (all produced upstream, cached to disk so no backbone forward re-runs):

| file | shape / contents | produced by |
|---|---|---|
| `cache/holdout_scores.parquet` | per-image: `cat_id`, `vet_pain∈{0,1}` (vet-confirmed only), `p_pain∈[0,1]` (calibrated binary), 5×`au_logits` (each `K-1=2` CORN logits), `capture_cond` (clinic/ward/home string), `path` | Gate-3 hold-out scoring |
| `cache/au_pmf.npz` | per-image: `pmf_au[5,3]` (per-AU pmf over {0,1,2}); `pmf_sum[11]` (convolved pmf over 0–10) | §6.2 distributional decode |
| `cache/vlm_vs_vet.parquet` | per-image × per-AU: `vlm_au∈{0,1,2}`, `vet_au∈{0,1,2}` (Gate 1-B pilot, vet-anchored) | Gate 1-B pilot |
| `cache/landmarks.parquet` | per-image: 48 CatFLW landmark `(x,y)` → 5 AU ROI boxes | Martvel Colab / Gate-5 |
| `frozen/test_ids.sha256` | hashed cat-disjoint hold-out manifest (CI-abort if read pre-lock) | Gate 3 |

All wrapper scripts live in `scripts/wrapper/`. Decode helpers (the soft-prob path) live in `scripts/engine/corn_decode.py` and are **the same** functions Gate-4's unit test pins (`probas = cumprod(sigmoid(logits))`, never the `>0.5` hard label except for the 0.39 point decision).

---

### 6.1 Pillar 1 — VLM-as-AU-rater κ (HEADLINE METHOD #1)

**Claim:** *a frozen VLM can weak-label feline FGS action units at human-rater agreement* — a forward-looking, portable method result, not an audit of CAT_01's labels. Per-AU **quadratic-weighted Cohen κ** of VLM-vs-vet, with a **bootstrap CI**, **reported and gated on the LOWER BOUND**.

**Metric & library.** `sklearn.metrics.cohen_kappa_score(weights="quadratic", labels=[0,1,2])`. The `labels=` arg fixes the class set; `weights="quadratic"` is what makes it **ordinal** — this is the `m-rewardbench` bug-to-avoid (their code is nominal). Bootstrap the CI ourselves (sklearn gives none): paired resample of *images* (not AU-rows) with `n_boot=10000`, percentile 2.5/97.5.

```python
# scripts/wrapper/p1_vlm_vs_vet_kappa.py
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
AUS = ["ear","orbital","muzzle","whiskers","head"]
df = pd.read_parquet("cache/vlm_vs_vet.parquet")
def qwk(v, p): return cohen_kappa_score(v, p, weights="quadratic", labels=[0,1,2])
rng, B = np.random.default_rng(0), 10000
out = {}
for au in AUS:
    d = df[df.au == au]; v, p = d.vet_au.values, d.vlm_au.values
    n = len(d); idx = rng.integers(0, n, size=(B, n))
    boots = np.array([qwk(v[i], p[i]) for i in idx])
    out[au] = dict(kappa=qwk(v, p), lo=np.percentile(boots,2.5),
                   hi=np.percentile(boots,97.5), n=n)
pd.DataFrame(out).T.to_csv("results/p1_kappa.csv")
```

Also emit, vet-free, the **ordinal Krippendorff α** over N≥3 repeated VLM runs (varied temp/seed) per AU — `krippendorff.alpha(reliability_data, level_of_measurement="ordinal")` (GITHUB_MINE P2.1 #4, `prometheus-eval/eval/consistency.py` pattern). This is a self-consistency pre-screen that feeds abstention (low α → defer) and **does not** touch the vet budget.

**Self-justify the labeler:** the pilot *selects* the VLM ("we chose the VLM by measured per-AU κ on the anchor"). **Do NOT cite Sci Rep 2025** (FACTCHECK C54, phantom citation) — the κ table is the justification.

**Gate / kill (Gate 1-B, fires on CI LOWER BOUND):**
- Pre-registered, kappaSize-power-backed floors: orbital/ear/head **LB ≥ 0.60**; muzzle/whiskers **LB 0.40–0.60** (caveat band).
- **orbital OR ear OR head LB < floor → PIVOT to binary spine** (drop the 0–10 layer entirely). The κ-as-method headline *still stands* — a mediocre but honest first measurement is publishable.
- orbital/ear/head pass, muzzle/whiskers fail → proceed on surviving AUs; state how dropping 2/5 heads shifts the reachable 0–10 range and whether 0.39 (≈4/10) is reachable.

---

### 6.2 Pillar 2 — Ordinal calibration (supporting; the 0–10 layer = inspected-not-validated)

Two metrics, both on the **distributional CORN path**. **No binned reliability diagram / binned ECE on the 11-atom sum** (degenerate at ~11 atoms, per-bin SE ±0.18–0.26).

**(a) Decode to distributions** — the soft path, identical to Gate-4's pinned helper:

```python
# scripts/engine/corn_decode.py
import torch, numpy as np
def au_pmf(logits2):                      # logits2: [...,2]  (K-1 for K=3)
    pgt = torch.sigmoid(logits2)          # P(rank>0), P(rank>1)
    cum = torch.cumprod(pgt, dim=-1)      # CORN: P(y>k)=∏ sigmoid
    p0 = 1 - cum[...,0]
    p1 = cum[...,0] - cum[...,1]
    p2 = cum[...,1]
    return torch.stack([p0,p1,p2], -1)    # pmf over {0,1,2}
def sum_pmf(pmf5):                         # pmf5: [5,3] -> pmf over 0..10
    acc = np.array([1.0])
    for k in range(5): acc = np.convolve(acc, pmf5[k])
    return acc                             # len 11, sums to 1
```

**(b) RPS-on-the-0–10-sum** (single scalar, bootstrap CI). RPS is the ordinal-aware Brier; CORN gives the CDF for free (GITHUB_MINE P2.2 `gluonts.rps`). For obs sum label `y`:

```python
# RPS = Σ_k (CDF_pred(k) − CDF_obs(k))^2 ; obs CDF is a step at y
def rps_sum(pmf_sum, y):
    cdf_p = np.cumsum(pmf_sum); cdf_o = (np.arange(11) >= y).astype(float)
    return float(((cdf_p - cdf_o)**2).sum())
# bootstrap over IMAGES (cat-grouped) for the CI on mean RPS
```
> The summed-CORN QWK is an **internal inspection metric only** — never a validated-instrument claim. The word "graded" stays struck from every validated sentence.

**(c) Per-AU ClasswiseECE** via **netcal** (`EFS-OpenSource/calibration-framework`) on each AU's 3-class pmf — makes miscalibration **attributable to a specific AU**:

```python
from netcal.metrics import ClasswiseECE   # per-AU, 3 classes, bins=5 (small n)
cwece = {au: ClasswiseECE(bins=5).measure(pmf_au[au], vet_au[au]) for au in AUS}
```
Cross-check with **SmoothECE** (`apple/ml-calibration` relplot) because the FGS set is small and binned ECE is unstable at low n.

**(d) Temperature-scale per head (LBFGS).** Fit on train folds only, one scalar `T` per CORN head, NLL closure (`gpleiss/temperature_scaling` skeleton; or `dholzmueller/probmetrics` for the maintained fit). Report NLL+ClasswiseECE **before/after** per AU.

```python
# scripts/wrapper/p2_temp_scale.py  (per-head scalar T; train folds only)
import torch
def fit_T(logits, labels):                # logits:[N,2] CORN; labels:[N] in {0,1,2}
    T = torch.ones(1, requires_grad=True); opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
    def closure():
        opt.zero_grad()
        loss = corn_nll(logits / T, labels)   # CORN NLL, NOT softmax CE
        loss.backward(); return loss
    opt.step(closure); return T.detach()
```

**Kill link:** calibration is **hygiene/supporting**, not a gate. But ClasswiseECE that is pathological on orbital/ear/head corroborates a Gate 1-B pivot. Calibration numbers are reported only for AUs that survive Gate 1-B.

---

### 6.3 Pillar 3 — Welfare-asymmetric DECISION CURVE (HEADLINE wrapper artifact)

**This, not ECE, is the headline wrapper artifact.** Net-benefit decision-curve analysis (Vickers) via **`MSKCC-Epi-Bio/dcurves`**, with the **undertreat:overtreat harm ratio swept as a RANGE** across the threshold-probability axis (we have **no** vet-elicited ratio, so we never pin one).

**Operating point first.** The clinical cutoff is **fixed pain-recall ≥ 0.90** (Evangelista anchor, sens 90.7%) selected **inside train folds (nested)** — **NOT** Youden-J/F1 (a symmetric-cost knee is the wrong loss for a welfare instrument). Report the **specificity it buys** with bootstrap CIs and the cutoff's variance across folds. Benchmark against the validated FGS point (AUC 0.94 / sens 90.7% / spec 86.6%) **as context only** — **never** "we beat 77/79/95%".

```python
# scripts/wrapper/p3_decision_curve.py
import pandas as pd, numpy as np
from dcurves import dca, plot_graphs
df = pd.read_parquet("cache/holdout_scores.parquet")   # vet_pain, p_pain (calibrated)
# threshold-probability axis IS the harm-ratio sweep: pt = overtreat/(under+over)
# undertreatment >> overtreatment  =>  emphasise the LOW-pt region (pt 0.05..0.40)
res = dca(data=df, outcome="vet_pain", modelnames=["p_pain"],
          thresholds=np.arange(0.01, 0.50, 0.01))      # low-pt welfare region
plot_graphs(res)                                       # net-benefit vs treat-all/treat-none
```

The figure shows model net-benefit beating both **treat-all** and **treat-none** across the welfare-relevant low-`pt` band; the fixed-recall point is annotated on it. Bootstrap (cat-grouped) the net-benefit curve for a CI ribbon. Decision-support triage framing only — output is *"grimace consistent with pain, X/10; recommend vet assessment,"* **never** an autonomous analgesia trigger.

**Kill link:** if Gate 0(b) fails (no budget certifies a useful abstention band), the decision curve **becomes the primary clinical deliverable** (abstention demoted to exploratory) — so this pillar is the floor that survives every downstream null.

---

### 6.4 Pillar 4 — Defer-to-vet ABSTENTION (one-sided 95% NPV LOWER BOUND)

**The word "guaranteed" is BANNED.** Report a **one-sided 95% NPV lower-bound curve**: NPV (with a one-sided 95% LB) at each abstention rate, shown to clear ≥0.90 only at honestly-reported abstention rates.

**(a) Risk-controlled band via MAPIE Learn-then-Test** (`scikit-learn-contrib/MAPIE`, the `risk_control` tutorial pattern, GITHUB_MINE P2.1 #1). Two thresholds around the **fixed 0.39 point** (band selector, NOT a replacement): `p < λ₁ → no-pain`, `p > λ₂ → pain`, middle → **defer-to-vet**. Control the **`negative_predictive_value`** risk function (missed pain = dominant cost) — one of MAPIE's built-in binary risk functions alongside `abstention_rate`. LTT gives a **distribution-free finite-sample LOWER BOUND**: say "lower bound," never "guarantee." The `delta` arg is the LTT confidence (one-sided 95% → `delta=0.05`); `alpha` is the tolerated risk (1 − target NPV).

> **API is version-sensitive — confirm before coding.** MAPIE's risk-control surface has changed across releases. The verified pattern in the installed-version tutorial is the `method='ltt'` controller fit on a calibration split and queried with `alpha`/`delta`, *not* a `BinaryClassificationController(metric_control=..., target_level=..., alpha=...).calibrate(...)` call. Pin the exact class name and signature against `./ref/mapie` / Context7 first; the skeleton below mirrors the tutorial shape.

```python
# scripts/wrapper/p4_abstention.py  (LTT risk control; pin class/signature to installed MAPIE)
import numpy as np, pandas as pd
from mapie.risk_control import PrecisionRecallController  # verify name vs installed wheel
df = pd.read_parquet("cache/holdout_scores.parquet")      # vet_pain, p_pain
# fit on the calibration split learned inside train folds:
ctrl = PrecisionRecallController(estimator=clf, method="ltt",
                                 metric_control="negative_predictive_value")
ctrl.fit(X_cal, y_cal)                                     # learns valid thresholds (lambdas)
# alpha = 1 - target_NPV ; delta = 0.05 -> one-sided 95% LB
_, y_band = ctrl.predict(X_test, alpha=0.10, delta=0.05)
# sweep alpha -> trace (abstention_rate, NPV_LB) curve; the valid band brackets 0.39
```

**(b) Learned-selector head (the delta over max-prob).** Adopt `facebookresearch/reliable_vqa`: a tiny MLP over `[DINOv2 CLS feat ⊕ 5 concatenated CORN logits]` predicting P(correct-vs-vet), instead of raw max-prob. **Max-prob is the baseline so the learned selector is the reported delta.** (v1.x add-on; if time-boxed out, max-prob + the MAPIE LTT band is the shippable minimum and the selector is future work.)

**(c) Risk–coverage AURC.** Report **AURC** (and AUGRC) via `ENSTA-U2IS-AI/torch-uncertainty` selective-classification metrics; frame as SAC (Galil ICLR2023, `IdoGalil/benchmarking-uncertainty-estimation-performance`). These are torchmetrics-style classes (`.update` then `.compute`) — the coverage-at-risk metrics are named `CovAt5%Risk` / `RiskAt80%Cov`, not a generic `CovAtxRisk`. Bootstrap-CI them ourselves (their code does not).

```python
from torch_uncertainty.metrics import AURC          # torchmetrics-style: update -> compute
m = AURC()
m.update(confidence, (pred != vet_pain))            # probs/confidence + error indicator
aurc = m.compute()                                  # max-prob vs learned selector
```

**Kill link (Gate 0):** the **LTT/MAPIE sample-size power calc runs BEFORE any vet spend.** If the budget can't certify NPV≥0.90 at ≤40% abstention, the curve is **exploratory-only** and "guaranteed" never appears — relocate the deliverable to the §6.3 decision curve. **Not a kill.**

---

### 6.5 Pillar 5 — Confound-attribution PROTOCOL (HEADLINE METHOD #2, portable)

**One-directional by construction:** well-powered to *detect* confounding, underpowered to rule it out. Every sentence says **"no confound detected at this power,"** never "ruled out." The deliverable is the **reusable protocol**, not "CAT_01 is confounded."

**(a) FGS-BG-Gap — background counterfactual** (`MadryLab/backgrounds_challenge` + `visinf/beyond-accuracy` BG-Gap, GITHUB_MINE P2.1 #2). Mask the cat face (no training: DINOv2 attention-rollout, or SAM `segment-anything`, or grabcut), composite the face onto (i) same-FGS and (ii) **swapped clinic** backgrounds; measure the score shift.

```python
# scripts/wrapper/p5_bg_gap.py  — mixed_rand / only_bg_t compositing
# FGS-BG-Gap = mean(|score_swapped - score_orig|) over 0-10 sum, per AU
# pain-flip rate = fraction whose 0.39 binary decision flips on bg swap
```
Report **FGS-BG-Gap** = mean 0–10 shift **+ pain-flip rate, per AU**, stratified by `capture_cond`. A large gap on a specific AU = that head is reading the cage/clinic.

**(b) Per-AU EBPG — energy-based pointing game** (`haofanwang/Score-CAM/energyPointGame.py`, GITHUB_MINE P2.1 #3). Per-AU saliency (attention-rollout or Score-CAM on the frozen backbone, 1–2 passes); ROI from **CatFLW 48-landmark** AU boxes (`cache/landmarks.parquet`); report `energy_in_ROI / energy_whole` per AU, stratified by capture condition. Answers *"does the ear head fire on the ear?"* — per-AU localization faithfulness no prior cat-pain work reports. If landmark coverage is partial, EBPG is reported only on the covered subset (state the denominator).

```python
# scripts/wrapper/p5_ebpg.py
def ebpg(sal_map, roi_mask):                  # both HxW, sal>=0
    return float((sal_map * roi_mask).sum() / (sal_map.sum() + 1e-8))
```

**Gate / kill (Gate 2, one-directional):**
- Detects confounding as a **transportable protocol → PROCEED and CO-HEADLINE** (publishable even if everything downstream is null).
- Fires damning but yields **only** "CAT_01 is confounded" (non-portable) → demote to a threat-to-validity paragraph; the κ method (§6.1) carries the headline alone. If Gate 1-B *also* fails, the honest paper is a confound/audit note.

---

### 6.6 Run order & kill-criteria crosswalk

Wrapper pillars run **after** Gates 0–6 in this order; each blocks downstream.

| Pillar | Gate it feeds | Fires kill when | On kill |
|---|---|---|---|
| §6.1 κ | **Gate 1-B** (CI LOWER BOUND) | orbital/ear/head LB < 0.60 | PIVOT to binary spine; κ-as-method still stands |
| §6.2 calibration | — (supporting) | pathological ClasswiseECE on core AUs | corroborates 1-B pivot |
| §6.3 decision curve | Gate 0(b) fallback | n can't certify abstention | becomes PRIMARY clinical deliverable |
| §6.4 abstention | **Gate 0(b)** (power calc pre-vet) | budget can't certify NPV≥0.90 @≤40% | exploratory-only; "guaranteed" never printed |
| §6.5 confound | **Gate 2** (one-directional) | non-portable damning result | demote to threat-to-validity |

**Anti-benchmark invariants for every table:** always print the **distinct-pain-CAT denominator** + **Clopper-Pearson / bootstrap CIs**; augmented copies **never** enter reported N; triage framing only. **Modal product** = binary-plus-wrapper + κ-as-method + confound-protocol + welfare decision curve + LB-abstention, with the 0–10 CORN layer as the inspected-not-validated artifact.

---

## 7. Gates, evaluation protocol, end-to-end pipeline & build sequence

This section ties the engine, supervision, and wrapper into one runnable program for a **solo dev on Apple M4 / MPS (no local CUDA), with Colab T4 for detector training and one vet** as the sole annotator. It is deliberately mechanical: run the gates **in order**, each gate writes an immutable artifact, and **no downstream number is believed until its prerequisite gate's artifact exists and passes**. The engine (frozen DINOv2 ViT-S/14 → 5 CORN heads → 0–10 sum → 0.39 decision) is **plumbing, explicitly conceded as not-novel**; the headline deliverables are the two portable methods (VLM-as-AU-rater per-AU κ; capture-condition confound-attribution protocol) plus the welfare wrapper. The **binary-plus-wrapper path is the v1 spine**; the graded 0–10 layer is built, decoded, and **inspected-not-validated** — the word "graded" never appears in a validated-claim sentence.

### 7.0 Repository layout & conventions

All paths absolute under the repo root `/Users/mingrath/ghq/github.com/mingrath/cat-fgs-llm`. Gates write to `artifacts/gateN/` and are content-hashed; a gate is "passed" iff its `PASS` sentinel file exists.

```
cat-fgs-llm/
├── .env                         # ROBOFLOW_API_KEY, ANTHROPIC_API_KEY (gitignored)
├── scripts/
│   ├── download_dataset.py      # EXISTS — Roboflow fork v1 → datasets/
│   ├── g0_power.R               # kappaSize + NPV/0.39 power calcs (canonical)
│   ├── g0_power.py              # python fallback (statsmodels/numpy)
│   ├── g1_percat_merge.py       # clip→CAT_ merge, fold CSV
│   ├── g2_confound_audit.py     # FGS-BG-Gap + per-AU EBPG protocol
│   ├── g3_freeze_holdout.py     # hash test manifest, write CI guard
│   ├── g4_mps_correctness.py    # logit parity + CORN-decode unit test
│   ├── g5_alignment_nme.py      # CatFLW NME inside crop
│   ├── g1b_vlm_kappa_pilot.py   # VLM weak-label + vet-κ + CI-LB gate
│   ├── g6_severity_cells.py     # AU=2 cell count → collapse decision
│   ├── feature_cache.py         # frozen DINOv2 ViT-S/14 → .pt cache
│   ├── train_corn.py            # 5 CORN heads on cached features
│   ├── decode_distributional.py # cumprod→pmf→convolve→RPS+ClasswiseECE
│   └── wrapper_metrics.py       # fixed-recall sens/spec, dcurve, NPV-LB
├── configs/
│   ├── prereg.yaml              # Gate-0 frozen budget integers + floors + G5 thresholds
│   └── folds.yaml               # StratifiedGroupKFold params (seed=42)
├── artifacts/
│   ├── gate0/ … gate6/          # one dir per gate, with PASS sentinel
│   ├── folds/folds.csv          # cat-grouped CV assignment (Gate 1)
│   └── holdout/test_ids.sha256  # frozen hashed test manifest (Gate 3)
└── datasets/                    # Roboflow export (gitignored)
```

Today only `scripts/download_dataset.py` and `.env.example` exist; every other file/dir above is a **proposed convention** that Sections 1–6 and the implementation must adopt or rename consistently. `.env.example` already ships `ROBOFLOW_*`; add `ANTHROPIC_API_KEY` to it for Gate 1-B.

Pin the env with `uv`:
```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install "rfdetr>=1.6.0" ultralytics torch torchvision \
  "anthropic>=0.40" pydantic scikit-learn torchmetrics coral-pytorch \
  imagehash open_clip_torch cleanlab dcurves statsmodels python-dotenv \
  wandb pandas numpy scipy
# torch/torchvision resolve to the MPS-capable macOS wheels (no CUDA extras on M4).
# R for the weighted-kappa power calc that has no good python equivalent:
#   install.packages(c("kappaSize"))
```

---

### 7.1 Gates G0–G6 + G1-B — ordered checklist

Run order is **strict and total**: `G0 → G1 → G2 → G3 → G4 → G5 → G1-B → G6`. (G1-B sits after G5 because it needs aligned crops to weak-label; G0–G3 are CPU/zero-data and can be drafted in a single sitting.) Each row: **exit criterion** (the artifact + numeric test that flips PASS), **compute**, and the **kill/pivot** decision drawn from FINAL_DIRECTION §6–7.

| Gate | Goal & exit criterion (writes `artifacts/gateN/PASS`) | Compute | Kill / pivot decision |
|---|---|---|---|
| **G0 — Power calcs + vet-budget pre-registration** (blocks ALL quantitative) | Commit the **single integer** vet-label budget to `configs/prereg.yaml` and run 3 zero-data calcs: **(a)** faces for per-AU weighted-κ CI half-width ≤0.15 (`kappaSize` weighted-κ, 3 levels; canonical in `g0_power.R`); **(b)** faces for an LTT/MAPIE-certified NPV≥0.90 band at ≤40% abstention; **(c)** whether the 0.39 sens/spec CI is reportable at budgeted n. Exit: budget integer + κ floors + ρ-band decision all written and committed. | None (CPU, ~1 afternoon) | **(b) fails** → abstention is **exploratory only**; "guaranteed" never appears; the deliverable relocates to the welfare decision curve. **Not a kill.** **(a) fails** → the magnitude claim (§A) is demoted on paper now. |
| **G1 — Per-CAT merge** (keystone) | Parse `group_id` from filenames, collapse clips to **true individuals validated against `CAT_` IDs** (CLIP/pHash are duplicate detectors, **not** re-ID). Exit: `artifacts/folds/folds.csv` written; assert **no individual straddles any fold**; **distinct-pain-cat denominator printed** and stored. | CPU | If merge cannot be validated against `CAT_` IDs → fall back to per-clip grouping, label it as such, flag re-ID as a known limitation (no kill). |
| **G2 — Capture-condition confound audit** (one-directional, portable) | Train a **trivial** classifier (brightness/blur/box-aspect/CLIP-embedding) to predict pain; quantify **FGS-BG-Gap + per-AU EBPG**. Exit: result logged as **"no confound detected at this power,"** never "no confound." | CPU, no vet | **Detects confounding as a transportable protocol** → PROCEED; this co-headlines even if everything downstream is null. **Only yields "CAT_01 is confounded" (non-portable)** → demote to threat-to-validity; κ-method must carry the headline alone. If **G1-B also fails** → honest paper is a confound/audit note; pre-register that venue now. |
| **G3 — Frozen hashed cat-disjoint hold-out + CI-abort** | Build a boundary-rich, true-~13%-prevalence test set from merged groups; commit fold CSV + test ids, **hash them** (`test_ids.sha256`), install a CI guard that **aborts any train/sweep/threshold run that can read the test manifest**. Exit: hash committed + guard green on a deliberate violation test. | CPU | Hard blocker: no scoring until the CI guard demonstrably aborts a leaking run. |
| **G4 — MPS compute-correctness** (BLOCKING) | **(a)** MPS-vs-CPU DINOv2 logit parity (`max|Δ| < 1e-3` after fp32 cast); **(b)** 1-epoch coral-pytorch CORN smoke test on MPS (loss finite, decreasing); **(c)** synthetic **CORN-decode → per-AU 0–2 → 0–10 sum → 0.39** unit test vs known values. Exit: all three assert-pass. | M4 / MPS | **BLOCKING** — a leak-proof PR-AUC from silently-wrong MPS logits is worthless. No number believed until green. |
| **G5 — Alignment / NME gate** (before Phase B) | On 30–50 project images: CatFLW-landmark **NME inside the RF-DETR crop vs eye-aligned crop**, and median face-pixel resolution after resize to the ViT/14 grid. Exit: NME + resolution both clear the **thresholds pre-set in `prereg.yaml`**; else add a **2-point eye-similarity alignment** and re-measure. | M4 | BLOCKING for Phase B: misaligned crops make "calibration" measure crop quality, not pain. |
| **G1-B — Weak-label reliability κ pilot** (self-justifies labeler; fires on **CI LOWER BOUND**) | ~120-image pilot (≥50 positive, per G0 budget): VLM-vs-vet **per-AU quadratic κ with 95% CI**; the pilot **selects the VLM**. Exit GO iff **κ CI-LB** clears pre-registered power-backed floors: **orbital/ear/head ≥ 0.60**; **muzzle/whiskers 0.40–0.60 acceptable-with-caveat**. No Sci Rep 2025 citation; the labeler is justified by this pilot alone. | Hosted-API inference + CPU | **κ-LB below floor on orbital/ear/head** → **drop graded entirely**; ship calibrated **binary + abstention + confound audit** (κ-as-method + wrapper still stand). **PIVOT to binary spine, not a kill.** **Passes orbital/ear/head, fails muzzle/whiskers** → proceed on surviving AUs; state how dropping 2/5 heads shifts the achievable 0–10 range and whether 0.39 (≈4/10) is even reachable. |
| **G6 — Severity-cell collapse** (after the anchor; pure counting) | Count vet-confirmed **AU=2 (severe)** cells per AU. Exit: if any AU's =2 cell is **single-digit → collapse the high end** (merge AU 1+2, or report only painful/not-above-threshold). **Pre-committed structural decision, not a caveat.** | CPU | Expected to **fire** given base rate → collapse; **do NOT print a per-cell severe sens/calibration number** (its CI spans ~[0.35,0.97] at single-digit n). |

**Cross-cutting kill triggers (FINAL_DIRECTION §7):**
- §3.4 Monte-Carlo **GO under ρ=0 but NO-GO under ρ=0.3** → declare the gate **fragile**, say so, default to the binary fallback.
- **G4 or G5 fail** → BLOCKING; fix before any metric is believed.

**Modal product the build plans for:** binary-plus-wrapper + κ-as-method + confound-protocol + welfare decision curve + LB-abstention, with graded-CORN shipped as an inspected-not-validated artifact.

---

### 7.2 Full evaluation protocol

#### 7.2.1 Cross-validation & hold-out (the splitter is shared by both phases)

```python
from sklearn.model_selection import StratifiedGroupKFold
# dominant individual CAT_01 is held out as a frozen LOIO partition (see §1 Gate 1) — excluded from CV here
m = (cat_ids != "CAT_01")
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
# groups = per-CAT id from Gate 1 (per INDIVIDUAL — NOT per-image, NOT raw clip);
# y = pain/no_pain at image level (135/336 clips are class-mixed).
for tr, va in skf.split(X=images[m], y=labels[m], groups=cat_ids[m]):
    assert set(cat_ids[m][tr]).isdisjoint(set(cat_ids[m][va]))   # G1 guarantee (per individual)
    assert "CAT_01" not in set(cat_ids[m][va])                  # never in a CV fold
```

- **Frozen hashed cat-disjoint hold-out (G3)** is carved out *before* CV and never touched during selection. Every preprocessing / threshold / calibration step is fit **inside each fold only** (nested). A CI guard aborts any run that can read `artifacts/holdout/test_ids.sha256`.
- **CI-abort discipline:** the test manifest is hashed; the hold-out evaluation script verifies the hash and refuses to run if the working tree's fold CSV was mutated after freeze. This structurally defeats post-hoc goalpost-moving.
- **Reporting unit:** always print the **distinct-pain-CAT denominator** alongside every rate. **Augmented copies never enter any reported N.** Headline CV is the multi-frame, per-cat-grouped CV with honest wide CIs + per-clip inference aggregation; a one-frame-per-cat secondary clean hold-out is reported **only if** the merged distinct-pain-cat count stays >50.

#### 7.2.2 Phase A metrics (binary detector — the spine)

| Metric | Library call | Role |
|---|---|---|
| **PR-AUC / AP** (pain=positive) | `sklearn.metrics.average_precision_score` on per-image max-conf pain score | **Primary**, ranks models |
| Per-class P/R/F1, specificity, confusion matrix | `sklearn.metrics.classification_report` | reporting |
| `map`, `map_50`, `map_75`, `map_per_class` | `torchmetrics.detection.MeanAveragePrecision` | **localization only** (never the headline) |
| **Pain-class recall** | from confusion matrix | **go/no-go** lever |

Derive the per-image pain decision (max-conf pain box vs the tuned threshold) and score **that** with PR-AUC/recall. **Never rank on accuracy or ROC-AUC.** Report **mean ± SD across the 5 cat-grouped folds**, plus bootstrap 95% CIs.

#### 7.2.3 Phase B metrics (engine output + wrapper — graded path inspected-not-validated)

1. **Per-AU VLM-vs-vet Quadratic Weighted Kappa** (headline #1, the only gated Phase-B number). `cohen_kappa_score(y_vet, y_vlm, weights='quadratic')` per AU; **bootstrap the CI and gate/report on the CI LOWER BOUND** (G1-B). MAE, adjacent-accuracy (|pred−true|≤1), and per-AU 3-class macro-F1 are reported as **secondary/descriptive only**, never as the gate.
2. **Distributional calibration of the 0–10 sum** (the DEFAULT path):
   ```python
   # per-AU soft CORN probabilities → pmf over {0,1,2}
   p_gt = torch.cumprod(torch.sigmoid(logits_au), dim=-1)   # P(rank>k)
   pmf_au = to_pmf(p_gt)                                    # [N,3]
   # convolve 5 per-AU pmfs into one pmf over the 0–10 sum:
   pmf_sum = reduce(np.convolve, [pmf_au_i for i in 5_aus]) # [N,11]
   ```
   Report **RPS-on-the-sum** (single scalar, bootstrap CI) + **per-AU ClasswiseECE**. **Do NOT print a binned reliability diagram / binned ECE on the 11-atom sum** — degenerate at ~11 atoms (per-bin SE ±0.18–0.26). `argmax`-decode is used **only** for the 0.39 point decision. This whole block is labeled **inspection-only**.
3. **Clinical decision (binary spine, CIRCULARITY FIREWALL):** sens/spec at the 0.39 cutoff estimated **ONLY on vet-confirmed labels**; **QWK-vs-VLM is never validation**. Operating point selected **inside train folds at fixed pain-recall ≥0.90** (Evangelista anchor), **not Youden-J/F1**; report the specificity it buys with bootstrap CIs and the cutoff's fold-to-fold variance. Evangelista's published operating point (sens ≈0.91 at AUC ≈0.94) is cited **only to source the fixed-recall target**, reported as prior-art context — **not** a head-to-head comparison and **never** a "we beat 77/79/95%" claim.
4. **Welfare-asymmetric decision curve (HEADLINE wrapper artifact, not ECE).** `dcurves.dca`; sweep **undertreat:overtreat as a RANGE** across the threshold-probability axis (no vet-elicited ratio exists).
5. **Defer-to-vet abstention:** **one-sided 95% NPV LOWER-BOUND curve** at each abstention rate; the word **"guaranteed" is BANNED**.

**Always:** mean ± SD across folds + bootstrap 95% CIs (Clopper-Pearson for proportions); distinct-pain-CAT denominator printed; the graded-sum and its QWK labeled internal/inspection-only.

---

### 7.3 End-to-end pipeline diagram

```
                          configs/prereg.yaml  (G0: budget integer, κ floors, ρ band, G5 thresholds)
                                   │ (gates every quantitative claim downstream)
                                   ▼
 raw Roboflow fork ──[export Fit, NOT Stretch; coco]──► datasets/  (~2040 imgs, ~13% prev)
 (cat-pain fork v1)                                        │
                                                           ▼
                                 ┌──────────── G1: per-CAT merge (validate vs CAT_ ids) ───────────┐
                                 │  filename → group_id → collapse clips → true individuals          │
                                 │  artifacts/folds/folds.csv  (no cat straddles a fold)             │
                                 └───────────────────────────────┬──────────────────────────────────┘
                                                                 ▼
                         G2: confound audit (FGS-BG-Gap + per-AU EBPG, one-directional, portable)
                                                                 │  (PROCEED / reframe / kill)
                                                                 ▼
                         G3: freeze + hash cat-disjoint hold-out  →  artifacts/holdout/test_ids.sha256
                                                                 │       (CI-abort guard armed)
                                                                 ▼
   RF-DETR-Nano detector (Colab T4 train; M4 infer) ──► candidate face boxes
                                                                 │
                                                  G5: NME/resolution gate (eye-align if high)
                                                                 ▼
                                            crop / align face  ──► aligned 518×518 crops
                                                                 │
            ┌────────────────────────── G1-B: VLM weak-label (N≥3 votes/AU) ──────────────────────────┐
            │  Claude Message Batches, cached rubric, enum[0,1,2] schema → 5 AU pseudo-labels          │
            │            │                                                                              │
            │            └──► vet ANCHOR (~120 imgs, ≥50 pos) ──► per-AU quadratic κ + CI-LOWER-BOUND   │
            │                  (GO iff orbital/ear/head κ-LB ≥0.60; muzzle/whiskers 0.40–0.60)          │
            └───────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                         │ (GO → graded built; NO-GO → binary spine only)
                                                         ▼
   G4: MPS correctness (logit parity + CORN decode→sum→0.39 unit test; horse decode = scaffolding only)
                                                         │                                  ──► frozen DINOv2 ViT-S/14
                                              feature cache  (artifacts/features/*.pt)
                                                         ▼
                                    5 CORN ordinal heads (Linear(feat,2) × 5 → 10 logits)
                                                         │
                          distributional decode: cumprod(σ) → per-AU pmf → convolve → pmf(0–10)
                                                         │
                G6: severity-cell count → collapse high end if AU=2 cells single-digit
                                                         ▼
        wrapper metrics:  RPS-on-sum + per-AU ClasswiseECE │ fixed-recall≥0.90 sens/spec (vet-only)
                          welfare-asymmetric decision curve │ one-sided-95%-NPV abstention curve
                                                         ▼
   ARTIFACTS:  weak-label AU annotations · frozen-backbone+CORN weights · hashed test split ·
               κ/confound/wrapper report · datasheet (graded layer = inspected-not-validated)
```

---

### 7.4 Risk-first build sequence / timeline (solo dev, M4)

Ordered so the **cheapest project-killers fire first** and the **vet's time is spent exactly once, late, against a frozen budget**. "Effort" is calendar-effort for one person and **excludes vet scheduling latency** (the real critical-path risk); ∥ marks what runs in parallel with the prior line; 🐾 marks the only steps that need the vet.

| # | Step | Effort | Compute | Parallel? | Needs vet? |
|---|---|---|---|---|---|
| 1 | **G0** power calcs + freeze `configs/prereg.yaml` (budget integer, κ floors, ρ band, G5 thresholds) | ~½ day | CPU | — | no (but commits the vet budget) |
| 2 | **G1** per-CAT merge → `folds.csv`; print distinct-pain-cat denominator | ~1 day | CPU | — | no |
| 3 | **G2** confound audit (FGS-BG-Gap + per-AU EBPG); publishes either way | ~1 day | CPU | ∥ with #2 | no |
| 4 | **G3** freeze + hash hold-out; arm CI-abort guard | ~½ day | CPU | — | no |
| 5 | **G4** MPS correctness (logit parity + CORN unit test on synthetic; horse decode as scaffolding only, never a reported number) | ~1 day | M4/MPS | ∥ with #3 | no |
| 6 | **RF-DETR-Nano** detector train (binary spine) on Colab T4; infer locally | ~2 days | T4 train / M4 infer | partly ∥ | no |
| 7 | **G5** NME/resolution gate on 30–50 imgs; add eye-align if needed | ~1 day | M4 | after #6 | no |
| 8 | VLM weak-labeler: Claude schema + cached rubric, Message Batches over the corpus | ~1–2 days | hosted API | ∥ with #6–7 | no |
| 9 | **G1-B** vet κ pilot 🐾 (~120 imgs, ≥50 pos) → per-AU κ CI-LB → GO/PIVOT | ~1 day build + **vet sitting(s)** | API + CPU | — | **yes** 🐾 |
| 10 | feature cache (frozen DINOv2) + train 5 CORN heads (only if GO) | ~1 day | M4 | — | no |
| 11 | distributional decode + **G6** severity-cell count → collapse if needed | ~½ day | CPU | — | no |
| 12 | wrapper: fixed-recall sens/spec (vet-only), decision curve, NPV-LB abstention | ~1–2 days | CPU | — | uses #9 labels |
| 13 | hold-out evaluation (CI-abort verifies hash), bootstrap CIs, datasheet, release | ~1–2 days | M4/CPU | — | no |

**Critical-path note:** #1→#5 are all CPU/M4 and several are parallelizable, so the **kill-or-reframe decision (G2)** and the **silent-correctness gate (G4)** both clear before any vet time or Colab spend. The **single vet engagement (#9)** is the only unbreakable dependency and the only step that can pivot the whole spine to binary. Steps #6/#8 run in parallel with the gate work. If G1-B PIVOTs to binary, steps #10–11 collapse and #12–13 run on the binary spine only. If the G0 vet-budget integer delivers fewer than 50 pain-positive confirmed labels, steps #9–12 inherit the **exploratory-only** abstention framing.

---

### 7.5 Release artifacts + datasheet

**Released artifacts (all dataset-agnostic where the method is the deliverable):**
1. **Weak-labeled AU annotations** (VLM 5-AU 0/1/2 + rationale + confidence), with provenance flags marking vet-confirmed vs VLM-only rows.
2. **Frozen-backbone + CORN head weights** + the decode→sum→0.39 reference implementation; engine explicitly stated **not novel**.
3. **Frozen, hashed, cat-disjoint test split** (`test_ids.sha256` + fold CSV) so any re-run is provably leak-free.
4. **VLM-as-AU-rater κ protocol + result** — code that scores any face corpus's per-AU VLM labels against a vet anchor and reports 5 quadratic κ with CI lower bounds.
5. **FGS-BG-Gap + per-AU EBPG confound-attribution protocol** — one-directional audit any future facial-pain corpus can run.
6. **Welfare-asymmetric decision-curve + one-sided-95%-NPV abstention-curve** code.
7. **Minimal HF Space demo** as a delivery vehicle (not the contribution).

**Datasheet (must document):**
- Pseudo-label provenance (VLM model + version, rubric, vote count N≥3, vet-confirmed subset).
- **Confound audit result**, phrased "no confound **detected at this power**."
- Single-source data named as the **top threat-to-validity**; **negative class documented as an unknown mixture** (possibly sedated/post-op).
- Morphology / pose / coat-color coverage gaps; distinct-pain-CAT denominator.
- Explicit statement: **no clean nociception ground truth** exists; the **graded 0–10 layer is inspected-not-validated**; horse-grimace appears as decode scaffolding only (methods/appendix sentence, never a headline number).
- Decision-support triage framing: "grimace consistent with pain, X/10; recommend vet assessment" — **never an autonomous analgesia trigger**.

---

### 7.6 Modal-product statement

> **The deliverable is a binary-pain decision-support instrument with a trustworthiness wrapper — calibrated at a fixed pain-recall ≥0.90 operating point, with a welfare-asymmetric decision curve and a one-sided-95%-NPV defer-to-vet abstention curve — accompanied by two portable methods (a per-AU VLM-as-AU-rater quadratic-κ protocol gated on the CI lower bound, and a capture-condition confound-attribution protocol), shipped on an explicitly-conceded DINOv2+CORN engine, with the graded 0–10 FGS layer included as an inspected-not-validated artifact.** This product is solo-deliverable on M4/MPS with one vet, stands even if the graded layer is dropped, and is not another cat-pain detector.

---

## Definition of done for v1

v1 is **done** when the modal product ships end-to-end and survives its own gates: Gates G0→G6 + G1-B have each written their `PASS` sentinel (or recorded the pre-registered pivot), and the released bundle contains (1) the calibrated binary pain decision at a fixed pain-recall ≥0.90 operating point with sens/spec estimated on vet-confirmed labels only, (2) the per-AU VLM-as-AU-rater quadratic-κ protocol + result reported on its CI lower bound, (3) the one-directional capture-condition confound-attribution protocol, (4) the welfare-asymmetric decision curve, and (5) the one-sided 95% NPV lower-bound abstention curve (or its documented exploratory demotion) — with the graded 0–10 CORN layer present strictly as an inspected-not-validated artifact, the frozen hashed cat-disjoint test split and `uv.lock` committed, every reported N carrying its distinct-pain-CAT denominator and bootstrap/Clopper-Pearson CIs, and the datasheet filed. The bar is explicitly met **even if Gate 1-B pivots to the binary spine**: the abstract must hold with the word "graded" struck and component 6 dropped, and the engine is never claimed novel.
