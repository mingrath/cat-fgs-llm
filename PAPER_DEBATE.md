# Decision Briefing: Two Cat-Pain Papers vs. Our FGS-LLM Project

## 1. What each paper is about

**Feighelstein et al. 2023 (Scientific Reports) — "Explainable automated pain recognition in cats."**
Binary pain/no-pain classification from cat faces on a deliberately heterogeneous (multi-breed, multi-sex, mixed-condition) clinical sample. It pits a transfer-learned ResNet50 (raw pixels) against a 48-landmark geometric pipeline feeding Random Forest/MLP, with pain ground truth from the validated Glasgow CMPS-feline scale (≥5) plus an independent clinical reason. The landmark+RF approach won (≈77% vs ≈65%), reversing the authors' prior homogeneous-data result and arguing geometry beats data-hungry CNNs when data is small and noisy. A second contribution is explainability (occlusion + GradCAM + feature importance) converging on the mouth/muzzle as most important and ears as least important for the machine.

**Namboonlue 2023 (MSc thesis, Srinakharinwirot Univ.) — "Feline Feelings Unleashed."**
An end-to-end deep-learning binary pain classifier on a self-collected Thai/Asian cat dataset, motivated by claimed morphological differences from European cats. It compares EfficientNetB7 vs ResNet50V2 via two-stage transfer learning with a large hyperparameter grid, Grad-CAM explainability, and clinician-assigned FGS-style 0–4 labels binarized to pain. The "best" model reports 79% accuracy / 90% recall — but on a final test set of ~10 real images (≈5 per class) with no cross-validation, image-level (not per-cat) splitting, documented overfitting, and internally inconsistent metric tables.

## 2. Good / Bad / Ugly

### Feighelstein 2023
| | |
|---|---|
| **Good** | One frame per cat + subject-disjoint 10-fold CV makes individual leakage structurally impossible (gold-standard template). Validated two-source label (CMPS≥5 AND clinical reason). Inter-annotator ICC2 reliability reported. Balanced 42/42 so accuracy is interpretable. Geometry+RF beating ResNet50 is the strongest external prior for our frozen-backbone bet. |
| **Bad** | No confidence intervals, no significance test — with ~8 test cats/fold the 77-vs-65 gap is ≈1 cat and likely not significant. No F1/AUC. Labels curated for separability (CMPS=4 and "≥5-without-reason" cats deleted). Single clinic, one capture rig; no external validation. Both pipelines depend on **manual** landmark alignment, so even the "DL" arm is not landmark-free. |
| **Ugly** | "Ears least important" is plausibly a camera-angle/mixed-breed **artifact** and contradicts validated human FGS (Evangelista: ears are reliable, high-weight). "No-pain" mixes post-recovery check-ups → context/condition confound subject-CV cannot catch. Label is partly **circular** (CMPS includes facial/demeanor items, then graded against a face model). |

### Namboonlue 2023
| | |
|---|---|
| **Good** | Augmentation applied **after** the split to prevent augmented-copy leakage (correct). Full compute disclosure (Colab/A100/TF, named grids). Published its failures honestly (Grad-CAM, the "Smudge" meme-cat false positive at 0.847). EfficientNetB7 > ResNet50V2 as a weak architectural prior. |
| **Bad** | Single fixed split, no CV, no seeds, no CIs. Number of distinct cats never reported; image-level random split → probable subject leakage. Class balance by discarding majority data. Post-hoc moved the pain threshold (dropped score-1) **after** Experiment 1 underperformed. |
| **Ugly** | Headline 79%/90% rests on ~5 real pain images dressed up as "100 augmented" — a 90% recall on n≈5 has a 95% CI of roughly 55–100%. Internally inconsistent metric tables (precision=1.0 nearly everywhere; F1 disagrees with its own precision/recall; abstract's "best=SGD" contradicts Exp-1's RMSProp). Severe documented overfitting (train acc ~1.0 / test ~0.80) used to report headline numbers. **The single most dangerous, least trustworthy number in this literature.** |

## 3. Points of consensus (all or most experts)

1. **Both headline accuracies (77%, 79%) are anti-benchmarks, not targets.** Inflated by tiny test N, separability curation, undersampled balance, single-clinic data, (thesis) subject leakage and a likely metric bug. We must never write "we beat 77%/79%."
2. **Per-cat (not per-clip) grouping is the keystone leakage fix.** Our own plan flags CAT_01 spanning clips 100-184, so clip_id under-counts individuals. Until same-cat clips are merged into true individuals and disjointness asserted on **merged** groups, every metric we produce is provisional.
3. **Augmented images never enter any reported N.** Report distinct-cat denominators and 95% CIs (Clopper-Pearson/bootstrap).
4. **Frozen self-supervised backbone + shallow per-AU heads over full fine-tuning** — justified primarily by the thesis's overfitting evidence and data-hunger prior, not by Feighelstein's underpowered gap.
5. **Keep all 5 CORN heads equal-capacity; do NOT down-weight the ear AU** on the strength of the papers' "ears least important" XAI — it is likely an artifact and contradicts validated human FGS.
6. **Label provenance/reliability is the deepest hole.** Neither paper reported inter-rater reliability of its own pain labels. Our ~120–150-image, ≥50-pain-positive two-rater per-AU kappa pilot is the FIRST reliability measurement in this research line and must be a GO/NO-GO gate.
7. **Keep true ~13% prevalence; judge on PR-AUC + pain recall (Phase A) and QWK + sens/spec at the calibrated cutoff (Phase B), never accuracy.**
8. **Freeze the 0.39 cutoff and calibrate only inside train folds.** Adopt Feighelstein's two-source rule: a "pain" label needs grimace AND a charted clinical reason.

## 4. Live disagreements — and my rulings

**(a) Is Feighelstein's one-frame-per-cat CV the "north star" we should approximate for our final headline?**
*Dispute:* ML-rigor/CV critics say yes (gold standard). Red-teamer and dataset critic warn it would shrink our pain hold-out to <50 distinct pain cats, producing an underpowered, false-precision headline of exactly the kind we mock.
**RULING — Split the difference, dataset critic wins on the headline.** Use one-frame-per-cat as the *structural ideal* for leakage but **report the multi-frame, per-cat-grouped CV with honest wide CIs plus per-clip inference aggregation** as the headline. Only adopt a one-frame-per-cat *secondary* clean hold-out if the merged distinct-pain-cat count stays large enough (>50) to keep CIs usable. Always print the distinct-cat denominator.

**(b) Does "geometry beats DL" justify a landmark-first Phase B?**
*Dispute:* CV critic notes both pipelines ate manual-landmark-aligned faces, so the gap conflates representation with alignment; ML critic adds it is underpowered noise (≈1 cat/fold). Vet critic warns landmark geometry shifts by skull type and could systematically mis-score by morphology.
**RULING — The gap is underpowered AND confounded; do NOT make geometry primary.** Our frozen-DINOv2 + per-AU-CORN choice stands on overfitting risk at 264 pain faces, not on the 77-vs-65 number. Keep the landmark/geometric track as a **parallel interpretability lane gated behind an NME sanity-check**, never the primary scorer. Settle the backbone empirically with our own bake-off, not by inheriting their conclusion.

**(c) Is the automatic RF-DETR box good-enough alignment for AU geometry?**
*Dispute:* Plan treats detector-box-as-face-cropper as an advance; CV/dataset/red-team critics call it an UNVALIDATED load-bearing extrapolation (CatFLW NME cited at 9–26%).
**RULING — It is a RISK, not an advance, until measured.** Gate Phase B on a 30–50-image NME audit inside detector crops vs. manual eye-aligned crops, plus a face-pixel-resolution check after resize to the DINOv2/14 grid. Add a 2-point eye-similarity alignment step if NME is high.

**(d) Can we recover per-image sedation/clinical metadata?**
*Dispute:* Vet critic wants it as a gating control (a sedated-but-painful cat is a silently inverted label that CV cannot catch). Dataset critic calls it largely unrecoverable for scraped Roboflow data.
**RULING — Attempt recovery, but do NOT block on metadata archaeology.** Where recoverable, store it as first-class fields; where not, **document the negative class as an unknown mixture in the model card** and design a boundary-rich, true-prevalence hold-out. The vet critic's underlying point stands: subject-disjoint CV does not fix state-confound, so this is a documented limitation, not a solved problem.

**(e) What actually breaks label circularity?**
*Dispute:* Red-teamer/ML critic want a non-face anchor (charted clinical reason / analgesia-given). Vet critic warns analgesia administration is itself confounded by protocol/dose tables — not a clean nociception label. ML critic notes the kappa study fixes noise but NOT circularity.
**RULING — Kappa is necessary-but-not-sufficient; the non-face anchor is the circularity-breaker, with caveats stated.** Validate the 0.39 decision against the best available non-face anchor on vet-confirmed rows only, while explicitly acknowledging in the model card that **no clean nociception ground truth exists** in this dataset. This is, per the red-teamer's reversal, the single most existential risk: a perfectly-computed QWK against VLM-derived labels could be measuring nothing but the model re-learning the VLM's FGS heuristic.

**(f) Is the deployment/compute (MPS, decode path) half of reproducibility being neglected?**
*Dispute:* Reproducibility skeptic argues the debate over-indexed on the data half and ignored the compute half — a leakage-proof PR-AUC from silently-wrong MPS logits is as worthless as the thesis's 79%.
**RULING — Sustained; promote compute checks from "open risks" to BLOCKING pre-checks.** MPS-vs-CPU logit parity for frozen DINOv2, a 1-epoch CORN smoke test, and a unit-tested CORN-decode→sum→0.39 path must pass before any weak-labeling or scoring run.

## 5. What these papers tell us to do (and avoid)

**Do (proven to work / sound discipline):**
- One-frame-per-cat / subject-disjoint CV as the leakage ideal (Feighelstein).
- Two-source pain rule: grimace AND charted clinical reason (Feighelstein).
- Report inter-annotator reliability before trusting labels (Feighelstein ICC2 → our per-AU kappa).
- Augment strictly **after**/inside the split (both papers).
- Prefer feature/frozen methods over full CNN fine-tuning in the few-hundred-label regime (both, via win + overfitting).
- Publish failures and run a negative-control "Smudge test" suite (thesis).
- Pin compute environment fully (thesis disclosure).

**Avoid (failure modes to design against):**
- Counting augmented copies toward test N (thesis).
- Image-level / per-clip-only splits when individuals < clips (thesis; our CAT_01).
- Deleting borderline cases (CMPS=4, pain-score=1) to inflate separability — our 0.39 cutoff lives in exactly that deleted band (both).
- Undersampling the majority to a balanced test, then reporting balanced accuracy (both).
- Post-hoc moving the label/threshold after seeing test performance (thesis Exp1→Exp2).
- Trusting a paper whose own metric tables are internally inconsistent (thesis → uncitable, not a "loose prior").
- Down-weighting the ear AU on artifactual XAI (Feighelstein).
- Reporting metrics from an overfit checkpoint with no CIs (both).

## 6. THE FINAL PLAN for our project

A strict **run-order gate**: no number is believed until all prerequisite steps pass; anything produced earlier is labeled provisional.

1. **Per-CAT merge (keystone, blocks everything).** Run a same-cat union-find merge to collapse clips into true individuals BEFORE building folds. **Validate the merge against the trusted CAT_ filename ids** rather than blindly trusting CLIP cosine>0.6 / pHash Hamming≤10 (CLIP/pHash are duplicate detectors, not cat re-ID — tune the threshold or use a proper re-ID embedding so it does not produce false splits/merges). Assert no individual straddles any fold.

2. **Capture-condition confound audit (co-gate).** Train a trivial classifier on brightness/blur/box-aspect/CLIP-embedding to predict pain. If it beats chance, pain is entangled with acquisition context (the post-recovery / stressed-at-vet trap) — log as a blocker. Per-cat disjointness does NOT catch context-shortcut leakage.

3. **Frozen, hashed, cat-disjoint hold-out.** Build a boundary-rich, true-~13%-prevalence test set from merged groups. Commit the fold CSV + test group-ids, hash them, and enforce a **CI gate** that aborts any train/sweep/threshold run able to read the test manifest. Prevents the thesis's post-hoc goalpost-move structurally.

4. **Compute-correctness pre-checks (blocking).** MPS-vs-CPU logit parity for frozen DINOv2; 1-epoch coral-pytorch CORN smoke test on MPS; **unit-test the CORN-decode → per-AU 0–2 → 0–10 sum → 0.39-threshold path on synthetic logits/labels** against known values. Note the bitsandbytes-4bit-QLoRA CUDA-only blocker on macOS arm64 (route VLM weak-labeling to Colab/T4).

5. **Phase A — binary detector as triage/localizer only.** RF-DETR (DINOv2 backbone) or YOLO, fine-tuned from COCO weights (never from scratch). Keep real 6.9:1 imbalance; handle via class-weighted/focal loss + copy-paste oversampling **inside folds only**. Rank on **PR-AUC + pain recall** at a frozen threshold, plus an explicit specificity/FPR metric and the "Smudge" negative-control suite. Report mean±SD across 5 grouped folds with bootstrap 95% CIs and a distinct-cats denominator. Binary "pain" is a triage flag, never the clinical endpoint.

6. **Alignment/resolution gate before Phase B.** On 30–50 project images: measure CatFLW-landmark NME inside the RF-DETR crop vs. manual eye-aligned crop, and median face-pixel resolution after resize to the DINOv2/14 grid. If NME high or resolution too low, add a 2-point eye-similarity alignment step and/or a higher-patch ViT before any CORN head is trained.

7. **Phase B labeling — kappa pilot as GO/NO-GO.** VLM weak-labels 5 AUs → vet review. Run the ~120–150-image, **≥50 distinct pain-positive cats**, two-rater pilot reporting per-AU **quadratic-weighted kappa with 95% CIs**. Pre-register GO/NO-GO thresholds (orbital/ear/head ≥0.6; muzzle/whiskers 0.4–0.6). Run the Monte-Carlo noise-propagation pre-check first to confirm the 0.39 decision is even estimable at 13% prevalence. Default ambiguous AU scores to 1; keep and **oversample** near-threshold cases (do not delete them as both papers did).

8. **Phase B model — frozen DINOv2 ViT-S + 5 equal-capacity CORN ordinal heads.** All heads equal capacity (ear AU preserved). Run an architecture bake-off the papers never did — frozen DINOv2 linear/CORN probe vs. landmark→geometric-feature→XGBoost (interpretability lane, NME-gated) vs. RF-DETR binary head — all on identical cat-grouped folds. Log train-vs-val gap as an overfitting monitor; never report from an overfit checkpoint.

9. **Break circularity at validation.** Estimate sens/spec at the frozen 0.39 cutoff on vet-confirmed labels only, anchored against an **independent non-face signal** (charted clinical reason / analgesia outcome) where available — never report a QWK against VLM-derived labels as "validation." Calibrate the cutoff inside train folds only (nested), and report its variance across folds.

10. **Clinical success criterion.** The only meaningful target is the validated **Evangelista FGS operating point (AUC 0.94, sens 90.7%, spec 86.6%)** on vet-confirmed labels — NOT the curated 77%/79%. Phase A target: pain recall ≥0.90 at fixed specificity.

11. **Inference robustness.** Aggregate per-clip predictions (per-AU Mode for ear/orbital/muzzle/head, Min for whiskers, per Feighelstein Sci Rep 2023) rather than scoring isolated frames — a free temporal-robustness gain neither paper attempted.

12. **Model card framing.** Ship as **decision-support triage** ("grimace consistent with pain, X/10; recommend vet assessment and rescore after analgesia"), never an autonomous analgesia trigger. Document the negative class as an unknown mixture (possible sedated/recovered cats), the morphology/pose/coat-color coverage gaps, and the absence of a clean nociception label.

## 7. Open questions the papers could not answer

1. **Does a static frontal phone image carry enough signal to resolve 0 vs 1 vs 2 *per AU*?** Both papers are binary; neither validates 3-level ordinal granularity. Open until our kappa pilot + bake-off.
2. **Is an automatic detector box good-enough alignment for AU geometry on multi-source phone frames?** Neither paper tested anything but manual landmark alignment.
3. **How does performance vary by skull type / coat color?** Both papers excluded brachycephalic/dark/calico/Sphynx cats; FGS landmark geometry shifts by morphology and our breed mix is unknown.
4. **Can pain be separated from other negative affective states (nausea, fear, dyspnea)?** Both flag the confound; neither solves it. A facial detector firing on a frightened cat is a welfare hazard.
5. **What is the true distinct-cat (and distinct-pain-cat) count after merging?** Decides every CI width and whether a one-frame-per-cat hold-out is viable — unknown until step 1 runs.
6. **Does our frozen DINOv2 + CORN path produce correct, identical numbers on M4/MPS vs CPU vs T4?** No prior work touched non-CUDA deployment; zero reproducibility cover.
7. **Is the machine's "mouth-dominant, ear-weak" weighting biological or acquisition artifact, and does it hold on our frontal-gated crops?** A finding to measure per-AU, not inherit.