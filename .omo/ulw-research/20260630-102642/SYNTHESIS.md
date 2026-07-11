# Ultraresearch Synthesis: New cat FGS pain detector pipeline improvement

Workers: 8 · Waves: 3 · Sources: 20+ · Verifications: 3 local command artifacts

## Executive summary

If building a new cat FGS pain detector from this project, the highest-value improvement is **not** to market a direct VLM scorer or simply replace the pipeline with landmark-XGBoost. The worthwhile improvement is a **vet-anchor-first, foundation-feature A/B upgrade**: keep the existing gated binary-plus-wrapper spine, then empirically compare `dinov2_vits14_reg` against the repo’s staged `dinov3_vits16` path and one simple strong baseline such as ConvNeXt/SigLIP2, while adding a hard **expert-vet AU anchor** and **threshold-safety gate** before any graded-FGS claim. This fits the repo because the current code already supports DINOv3 plumbing and a strict 0.39 decision contract, but lacks empirical proof and independent AU labels. [Local: `verify-local-contracts.md`; S10; S11]

The newest external evidence does **not** overturn the repo’s caution. Up to June 30, 2026, no verified 2026 primary cat-pain/FGS method paper surfaced in this search. The latest verified method-adjacent work is 2025: a DINO/ViT segment-level XAI framework reporting strong cat-pain classifier performance but not a dedicated FGS detector, and a chatbot/VLM FGS paper showing direct chatbot scoring remains clinically unsafe. The strongest direct automated FGS paper remains the 2023 smartphone/landmark/geometric pipeline; the strongest fully automated temporal pain pipeline is the 2024 video-landmark model. [S3; S5; S7; S8]

So the pipeline improvement worth doing is: **DINOv3/DINOv2-reg measured A/B + vet-anchored weak-label reliability + calibration/abstention + optional temporal/crop stabilization**, not “VLM diagnoses pain.” The output should remain **decision-support triage**, not owner-facing diagnosis, until subgroup, breed, ocular-pain, and threshold-safety validation pass. [S1; S8; local `claim-ledger.md`]

## Findings by theme

### 1. Current repo state: strong protocol shell, not validated clinical detector

The repo positions itself as a protocol/prototype: portable confound-attribution, guarded VLM-as-AU-rater κ check pending independent vet anchor, and a binary-plus-abstention spine. It explicitly says the current dataset is binary pain/no_pain and cannot yield 0/1/2 AU ground truth. The engine is DINOv2+CORN plumbing, not the contribution. [Local: `README.md:3-47`, `README.md:60-76`, `wave-1-codebase-docs-decisions.md`]

Executable pipeline: ingest/folds → detector/crop/quality gate → frozen backbone cache → CORN + binary head → VLM labels/κ → wrapper/confound/eval gates. Local tests confirm the shared 0.39 threshold, backbone variant contract, and CORN decode tests pass. [Local: `wave-1-codebase-executable-pipeline.md`; `verify-targeted-tests.md`]

### 2. Most recent technique signal: DINO-family features and temporal landmarks, with caveats

The field’s arc is: manual landmarks → automated landmarks → smartphone FGS → raw-video temporal landmarks → segment/XAI with DINO/ViT. The 2025 segment-based framework is the newest verified method-adjacent signal supporting DINO/ViT features for cat pain, but it is not a purpose-built clinical FGS detector. [S7]

DINOv3 is current as a vision foundation model with strong dense features and official Hugging Face/Meta/GitHub support. The repo already includes a `dinov3_vits16` branch, but local verification shows it remains an A/B candidate behind a 384-d cache contract, not a proven default. [S10; S11; Local: `verify-local-contracts.md`, `wave-2-repo-dinov3-fit.md`]

### 3. What not to do: direct VLM scoring as product

A 2025 FGS chatbot study tested ChatGPT, Claude AI, Gemini, and Perplexity against an expert rater. Most chatbots had poor agreement; Claude had the best retest bias, but limits of agreement still spanned the 0.39 analgesia threshold. This supports using VLMs only as candidate weak labelers behind local vet κ gates, not as direct clinical scorers. [S8]

### 4. Dataset reality: public geometry assets exist; public graded pain labels do not

CatFLW and cat head datasets are useful for geometry/landmarks/pretraining. They are not pain labels. The main automated cat pain/FGS papers keep datasets request-only or unavailable; no public downloadable graded cat-FGS dataset was found across Kaggle/HF/GitHub/Zenodo/figshare/Dataverse searches. [S9; S15; `wave-1-datasets-oss.md`]

### 5. Detector/crop front-end: improve only if crop quality blocks downstream

RF-DETR and YOLO are both plausible. YOLO is simpler and mature; RF-DETR is modern and already in this repo. SAM2 is useful for masks/video/crop stabilization, but overkill for still-image bbox classification. Since the current bottleneck is labels and validation, detector changes are second priority unless Gate 5/NME/crop quality fails. [S12; S13; `wave-1-oss-architecture.md`]

## Codebase findings

- `src/model/backbone.py` supports `dinov2_vits14_reg`, `dinov2_vits14`, `dinov3_vits16`; default remains `dinov2_vits14_reg`. [Local: `wave-2-repo-dinov3-fit.md`]
- `src/model/cache_features.py` and downstream loaders enforce 384-d cache schema with `variant`, `layer`, and `patch_mode`. [Local: `wave-2-repo-dinov3-fit.md`]
- `src/vlm/aggregate.py` and `src/model/decode.py` share the 0.39 point-decision threshold. [Local: `verify-local-contracts.md`; `verify-targeted-tests.md`]
- `configs/corn.yaml` already carries backbone A/B knobs. [Local: `wave-2-repo-dinov3-fit.md`]
- `README.md` bans validated graded claims until an independent vet anchor exists. [Local: `local-line-refs.md`]

## Sources ranked

1. [S1] Evangelista et al. 2019 FGS validation — official clinical target and 0.39 threshold: https://www.nature.com/articles/s41598-019-55693-8
2. [S8] Ngai et al. 2025 chatbot/VLM FGS agreement — negative/cautionary VLM evidence: https://www.nature.com/articles/s41598-025-27404-z
3. [S3] 2023 smartphone automated FGS — strongest direct automated FGS method: https://www.nature.com/articles/s41598-023-49031-2
4. [S5] 2024 video landmark cat pain — strongest temporal automation evidence: https://www.nature.com/articles/s41598-024-78406-2
5. [S7] 2025 segment/XAI DINO-ViT framework — newest foundation-feature evidence: https://www.nature.com/articles/s41598-025-96634-y
6. [S10/S11] DINOv3 official GitHub/Meta pages — current foundation backbone: https://github.com/facebookresearch/dinov3 and https://ai.meta.com/research/dinov3/
7. [S9/S15] CatFLW — geometry asset, not pain labels: https://arxiv.org/abs/2310.09793 and https://www.kaggle.com/datasets/georgemartvel/catflw

## Verified claims

- Local repo supports DINOv3 candidate path, DINOv2-reg default, shared 0.39 threshold, and 384-d cache contract. Verdict: CONFIRMED by `verify-local-contracts.md`.
- Local backbone/threshold/decode tests pass. Verdict: CONFIRMED by `verify-targeted-tests.md` (`12 passed`).
- Portable protocol suite and synthetic orchestrator dry-runs pass. Verdict: CONFIRMED by `verify-portable-suite.md`.
- High-risk non-code claims are listed in `claim-ledger.md`; one claim is intentionally unresolved and phrased as “no verified 2026 method surfaced,” not “none exists.”

## Contradictions and resolution

- Prior landmark/geometric methods have stronger direct FGS evidence than foundation backbones; however, their datasets are request-only/closed and their landmark path does not match this repo’s current architecture. Resolution: do not claim superiority; run a measured A/B and keep the protocol framing. [S3; S9; local]
- VLMs are attractive for weak labels but unsafe as direct scorers. Resolution: use VLM only behind independent vet κ and Bland–Altman/threshold gates. [S8]
- DINOv3 is current, but local code support is not empirical proof. Resolution: treat DINOv3 as an alternate backbone experiment, not default. [S10; local]

## Gaps

- No independent vet AU anchor in the repo.
- No public downloadable graded cat-FGS pain dataset found.
- No verified 2026 primary method paper surfaced in this search; this is not an absolute nonexistence claim. See `recency-search-log.md`.
- No local empirical A/B result for DINOv2-reg vs DINOv3/SigLIP2/ConvNeXt.
- Subgroup robustness, especially brachycephalic/ocular-pain/e-collar cases, remains unvalidated.

## Deployment/product constraints

The recommended improvement is not owner-app launch. Product posture remains decision-support triage. Direct VLM scoring is unsafe because chatbot limits of agreement can cross the 0.39 threshold; CatFLW-derived assets may add non-commercial licensing constraints; poor crops, e-collars, ocular pain, and brachycephalic morphology require defer/subgroup gates. See `wave-3-deployment-product-constraints.md`.

## Methods matrix closure

`closure-optional-leads.md` closes material optional leads: RF-DETR/YOLO are front-end A/B choices only if crop quality fails; validation design requires expert AU agreement plus Bland–Altman/threshold-safety/subgroup gates; allowed model-card wording is prototype/decision-support, while validated diagnosis/owner-app claims are forbidden.

## Recommended pipeline upgrade

1. **Keep v1 spine**: binary pain/no_pain + wrapper + abstention; no validated graded claim.
2. **Run a frozen-feature A/B**: `dinov2_vits14_reg` vs `dinov3_vits16` vs one simple baseline (`ConvNeXt` or `SigLIP2`) using same cat-disjoint folds, same cache provenance, same threshold metrics.
3. **Add/execute vet-anchor Gate 1-B**: expert AU labels; report per-AU κ CI lower bounds and Bland–Altman LoA around 0.39; VLM labels are training aids only if they pass.
4. **Use DINOv3 only if it wins** on PR-AUC/recall, threshold safety, calibration, and confound audit; otherwise keep DINOv2-reg.
5. **Add temporal/crop stabilization only if data warrants it**: SAM2/video smoothing for video or poor crops; otherwise not the first ROI.
6. **Report as research prototype** until external cohort/subgroup validation passes.

## Expansion trace

- Wave 1: codebase executable pipeline, codebase docs, literature, datasets, OSS architecture, skeptic risk.
- Wave 2: DINOv3 repo fit; FGS/VLM/vet-anchor expansion.
- Wave 3/closures: malformed EXPAND tails closed; deployment/product constraints added; RF-DETR-vs-YOLO, validation design, model-card wording, and methods matrix closed; recency claim downgraded with search log; unrelated dirty repo diff excluded in `worktree-scope.md`.
- Convergence: no unchecked core lead changes the recommendation; optional leads are either closed or explicitly report extensions.
