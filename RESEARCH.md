> **⚠️ SUPERSEDED where it conflicts with `FINAL_DIRECTION.md` (authoritative) and the code.**
> Early-exploration notes: the QWK-primary framing and the convnext_tiny backbone here are
> NOT what shipped. The engine is frozen DINOv2 ViT-S/14 + CORN heads; QWK-vs-VLM is a
> labeler-agreement method finding, NEVER validation. Trust `FINAL_DIRECTION.md` + the code.

# Cat FGS Pain Scoring — Research & Plan

## Goal
Predict a cat's pain from a photo. Ultimate target: the **Feline Grimace Scale (FGS)** — a graded 0–10 score. The available Roboflow dataset only supports a **binary** pain/no-pain model, so the project is two-phase.

## The Feline Grimace Scale (FGS)
5 facial **action units (AUs)**, each scored **0 / 1 / 2**, summed to **0–10** and normalized to a 0–1 ratio:
1. Ear position
2. Orbital tightening
3. Muzzle tension
4. Whiskers position/change
5. Head position

**Clinical decision threshold: ratio > 0.39 (≈ 4/10) → administer analgesia** (sensitivity 90.7%, specificity 86.6%). Validation: inter-rater ICC 0.89; vs Glasgow CMPS-Feline ρ = 0.86.
Source: Evangelista et al., *Sci Rep* 9:19128 (2019) — https://pmc.ncbi.nlm.nih.gov/articles/PMC6911058/

## The dataset (`lia-k4jkv/cat-pain-ul7lu`, forked to `mingraths-workspace/cat-pain-ul7lu-p3rtl`)
- **Object detection**, 2 classes: `pain` / `no_pain` (bounding boxes on cats).
- ~2,040 images (train 1,631 / valid 409 / test 0). License CC BY 4.0.
- **Severe imbalance: no_pain 1,819 vs pain 264 (~87/13).**
- ⚠️ Contains **no FGS / per-AU / 0–10 labels** — only binary. It cannot train a graded FGS scorer.

## Prior art (what works for cat pain from faces)
- **Steagall/Feighelstein et al. 2023** (closest to our goal): landmark CNN (ShuffleNetV2) → 35 geometric descriptors → XGBoost. **Binary pain 95.5% acc, AUROC 0.97**; FGS regression MSE 0.0096. Smartphone-deployable. https://pmc.ncbi.nlm.nih.gov/articles/PMC10703818/
- **Feighelstein et al. 2022**: ResNet50 (73.6%) ≈ landmark-geometry MLP (72.4%) — landmarks carry the signal and are interpretable. https://pmc.ncbi.nlm.nih.gov/articles/PMC9187730/
- **Martvel et al. 2024**: YOLOv8 face → 48-landmark ensemble → autoencoder → XGBoost over video; temporal dynamics help. https://pmc.ncbi.nlm.nih.gov/articles/PMC11564822/
- Zero-shot chatbots/VLMs **underestimate FGS** and are not clinically reliable. https://www.nature.com/articles/s41598-025-27404-z

## Best-practice modeling recipe
- **Preprocessing is highest-leverage:** detect cat face → align by eyes → crop 224×224. Face alignment measurably improved every published model.
- **Backbone (transfer learning):** `timm` — `convnext_tiny.fb_in22k_ft_in1k` (default) or frozen `vit_small_patch14_dinov2` linear probe (fast strong baseline).
- **Target framing (for FGS, Phase B):** 5 multi-task **ordinal (CORN)** heads (each 0/1/2) → sum to 0–10 → apply 0.39 threshold. Interpretable per-AU. (`coral-pytorch`, CORN > CORAL.)
- **Augmentation (Albumentations):** h-flip, mild affine/rotation ±15°, brightness/contrast, light noise/JPEG. **Avoid** heavy CoarseDropout, strong blur, vertical flip, grayscale — they destroy subtle AU cues.
- **Small/imbalanced data:** stratified **5-fold CV grouped by individual cat** (no cat in train+val — #1 leakage trap), class-weight/focal loss, freeze→unfreeze, discriminative LRs, EMA, TTA.
- **Training:** AdamW, cosine LR + warmup, mixed precision, label smoothing 0.1, early stopping on val QWK.
- **Evaluation (ordinal, not plain accuracy):** Quadratic Weighted Kappa (primary), MAE, per-AU F1/confusion, adjacent accuracy, calibration; AUROC/sensitivity/specificity at the 0.39 threshold (tune for high sensitivity).

## VLM ("LLM") angle
Fine-tuning a VLM is unlikely to beat a CV backbone on small data and is more data-hungry. Best role: **zero/few-shot weak labeler** to bootstrap FGS labels (then human review), and a baseline to beat. If pursued: PaliGemma 2 3B / Qwen2.5-VL 7B via TRL QLoRA, freeze vision encoder, constrained JSON decoding for the 5-AU rubric.

## Plan
- **Phase A (solo):** binary pain/no-pain detector from the forked dataset + face detect/align/crop pipeline. Working baseline.
- **Phase B (human-in-loop):** VLM pre-scores 5 AUs → vet/vet-tech reviews/corrects in Roboflow → train multi-task ordinal 0–10 FGS model + active-learning loop.

## Environment
- Roboflow key in `.env` → workspace `mingraths-workspace`. Forked project: `cat-pain-ul7lu-p3rtl`, version 1 (auto-orient + resize 640; aug: rotate 14°, brightness ±21%, h-flip; 2× images).
- Local: Apple M4 (MPS, no CUDA), system python 3.9 / no ML libs → prefer Roboflow hosted training or Colab GPU.
