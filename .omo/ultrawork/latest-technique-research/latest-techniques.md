# Recent techniques/repos usable for cat-FGS pain detector

Date: 2026-06-30
Mode: ultrawork + ultimate-browsing route
Scope: recent papers/repos/community techniques for improving current `cat-fgs-llm` pipeline without claiming clinical readiness.

## Project stage inferred from repo

Current stage: research prototype / protocol shell.

Current spine remains correct:
- binary pain/no-pain classifier + FGS wrapper
- abstention/defer policy
- portable confound-attribution protocol
- DINOv2/DINOv3/ConvNeXt/SigLIP-style frozen-feature A/B direction
- no direct VLM/LLM medical product claim

Main gap: not model novelty. Main gap is reliable expert-vet AU labels and validation.

## Ranked usable upgrades

### 1. Vet-anchor-first active-learning label loop

Use now.

Technique:
- collect small expert-vet AU anchor set first, not more model code first
- use model entropy + feature-space diversity + LLM/VLM disagreement only to select samples for labeling
- audit label noise with cleanlab/FiftyOne-style data-centric workflow
- annotate with Label Studio or equivalent

Why it matters:
- Current biggest blocker is no independent expert-vet AU 0/1/2 anchor.
- A stronger backbone cannot prove FGS validity if labels are weak.
- Best near-term gain is choosing the next 80-200 images/videos for expert review, especially uncertain cases and subgroup/confound cases.

Repos:
- cleanlab: https://github.com/cleanlab/cleanlab
- FiftyOne: https://github.com/voxel51/fiftyone
- Label Studio: https://github.com/HumanSignal/label-studio

Integration:
- add `sample_for_vet_review.py`: rank images by uncertainty, embedding distance, crop/quality failures, subgroup metadata, and AU disagreement.
- output a Label Studio task JSON and a locked audit CSV.

### 2. DINOv3 frozen feature A/B

Use after label anchor exists.

Technique:
- replace/compare `dinov2_vits14_reg` with DINOv3 ViT-S/16 and ConvNeXt-Tiny/Small frozen embeddings
- test CLS/pooler, patch mean/std, and optional localized patch pools for eye/ear/muzzle regions
- keep current binary+wrapper architecture; do not fine-tune first

Why it matters:
- DINOv3 is a recent vision foundation model family with dense feature focus.
- 2025 animal affective-computing paper supports DINO-pretrained ViT + Grad-CAM++ as strong for cat pain explainability/performance.

Sources:
- DINOv3 repo: https://github.com/facebookresearch/dinov3
- DINOv3 blog: https://ai.meta.com/blog/dinov3-self-supervised-vision-model/
- Segment-based explainability paper: https://www.nature.com/articles/s41598-025-96634-y

Integration:
- add extractor adapter only; keep classifier/evaluation unchanged.
- acceptance: LOCO/cat-group split, AU-kappa gate, calibration/abstention metrics improve vs DINOv2 baseline.

### 3. Segment-based explainability instead of generic heatmaps

Use with DINOv3/DINOv2 A/B.

Technique:
- segment-level attribution over biologically meaningful parts: eyes/orbit, ears, muzzle/mouth, whisker pad, head pose
- prefer Grad-CAM++/power transform style comparison where compatible
- score whether attribution lands on FGS-relevant segments, not background

Why:
- Generic saliency can look convincing but fail biologically.
- Recent animal affective computing paper explicitly ranks segment importance and compares backbones/heatmaps.

Integration:
- add `explain_segments.py` to produce per-segment attribution table.
- add gate: fail if top attribution repeatedly lands on background/collar/cage/human hand.

### 4. Cat landmarks / video quality lane

Use if dataset includes video or if crop/quality failures dominate.

Technique:
- use automated cat facial landmarks and video-quality metrics to detect occlusion, bad face angle, landmark failure, and low-confidence frames
- aggregate temporal evidence only when enough valid frames exist

Sources:
- 2024 Scientific Reports video cat pain pipeline: https://www.nature.com/articles/s41598-024-78406-2
- Cat facial landmarks paper / CatFLW: https://arxiv.org/abs/2310.09793

Integration:
- add a quality gate before classifier: `face_visible`, `landmark_nme`, `frame_valid_ratio`, `temporal_jitter`.
- for videos, classify only valid frame windows and abstain on deficiency.

### 5. RF-DETR / SAM2 for crop and mask quality, not pain scoring

Use only if current detection/crop front-end is weak.

Technique:
- RF-DETR for cat/face detection experiments
- SAM2 for segmentation/masking/crop stabilization, especially video or cluttered backgrounds

Sources:
- RF-DETR repo: https://github.com/roboflow/rf-detr
- SAM2 repo: https://github.com/facebookresearch/sam2

Integration:
- compare against existing cropper on crop recall, false face crop rate, landmark NME, and downstream abstention rate.
- do not replace classifier until front-end metrics show clear bottleneck.

### 6. SuperAnimal / DeepLabCut-style pose priors

Use as optional geometry prior, not first move.

Technique:
- use animal pose foundation models to stabilize face/head pose, body context, and video adaptation
- likely more useful for video, head orientation, and frame filtering than direct FGS scoring

Source:
- SuperAnimal paper: https://www.nature.com/articles/s41467-024-48792-2

Integration:
- test whether pose/head orientation features improve abstention and subgroup robustness.

### 7. VLM/chatbot FGS scoring as negative control only

Do not use direct VLM/LLM scoring as product path.

Technique:
- use chatbot/VLM FGS predictions only as weak disagreement signal for active learning
- never as expert substitute or final clinical score

Why:
- 2025 Scientific Reports chatbot study found mostly poor agreement and unsafe limits around analgesia threshold.

Source:
- https://www.nature.com/articles/s41598-025-27404-z

### 8. MAPIE/conformal abstention/risk control

Use as safety layer around the existing classifier.

Technique:
- calibrate prediction sets and defer thresholds on held-out vet-labeled calibration split
- report coverage, selective risk, sensitivity at fixed deferral rate, and subgroup risk

Source:
- https://github.com/scikit-learn-contrib/MAPIE

Integration:
- keep as wrapper after binary model; no dependency on specific backbone.

## Recommended next build sequence

1. Freeze current baseline and run current metrics.
2. Build active-learning export for expert-vet AU anchor.
3. Label the anchor set: AU 0/1/2 + cannot-score + crop/quality notes.
4. Run DINOv2 vs DINOv3 ViT-S/16 vs DINOv3 ConvNeXt-Tiny/Small vs SigLIP2 embeddings.
5. Add segment-level explainability gate.
6. Add conformal abstention/risk-control report.
7. Only then test RF-DETR/SAM2/landmarks if crop/video quality is proven bottleneck.

## Decision

Most recent useful technique is not a single paper model. It is:

**data-centric vet-anchor active learning + DINOv3 frozen feature A/B + segment-level explainability + conformal abstention.**

This makes the project worth improving because it targets the real bottleneck: reliable labels, biological validity, calibration, and abstention.

## Verification notes

Sources were checked through live web search/open on 2026-06-30. Ultimate-browsing escalation beyond normal browsing was not needed because target pages were reachable.
