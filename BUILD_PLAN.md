# Cat FGS Pain Detector — Consolidated Build Plan

> **⚑ SUPERSEDED IN PART by [FINAL_DIRECTION.md](FINAL_DIRECTION.md) (2026-06-12, direction debate).** Where the two conflict, FINAL_DIRECTION wins. The load-bearing reframings, applied inline below: (A) **headline = two PORTABLE METHODS** (VLM-as-AU-rater κ; capture-condition confound-attribution protocol); the **DINOv2+CORN engine is conceded as plumbing, NOT claimed novel.** (B) **binary-plus-wrapper is the v1 spine**; **graded is built-but-NOT-VALIDATED** (the word "graded" is struck from every validated-claim sentence). (C) **gold-set acquisition is CUT** from the critical path. (D) operating point = **fixed pain-recall ≥0.90, not Youden-J/F1**; the **welfare-asymmetric decision curve is the headline wrapper artifact, not ECE.** (E) CORN ships the **distributional path** (RPS-on-sum + per-AU ClasswiseECE); **no binned reliability diagram on the 11-atom sum**; the empirical 5×5 error-correlation matrix is replaced by a **2-point ρ=0/0.3 sensitivity band.** (F) **delete "guaranteed"** on abstention → **one-sided 95% NPV lower-bound** curve. New **Gate 0** (power calcs + vet-budget pre-registration, blocks everything quantitative) and **Gate 6** (severity-cell collapse) added below; **Gate 1-B now fires on the κ CI lower bound**; the **confound audit is one-directional** ("no confound *detected at this power*").

This plan consolidates research across reference implementations, detection training, class imbalance, face pipeline, ordinal modeling, VLM weak-labeling, evaluation, MLOps, and verified live inspection of the forked Roboflow dataset (`mingraths-workspace/cat-pain-ul7lu-p3rtl` v1) and its annotation geometry. Two findings from live dataset inspection are load-bearing and override generic guidance: (1) the shipped train/valid split is **leaky** (191/336 source clips, 56.8%, appear in both splits — full-manifest figure; the earlier 201/337 was a 76%-sample extrapolation), and (2) the existing pain/no_pain boxes are already **tight head/face crops** usable directly as Phase B crop source — **but see Gate 5: box-as-cropper is an unvalidated risk, not a settled advance, until the NME audit passes.**

> **Reconciled with [PAPER_DEBATE.md](PAPER_DEBATE.md).** The two-paper debate (Feighelstein 2023, Namboonlue 2023) added a strict **run-order gate** discipline now captured in **Section 0**. Headline framing rule: the curated **77% / 79%** accuracies are *anti-benchmarks, never targets* — we never write "we beat 77/79%." The only real target is the validated **Evangelista FGS operating point (AUC 0.94, sens 90.7%, spec 86.6%)** on vet-confirmed labels.

> **Reframed by [GAP_ANALYSIS.md](GAP_ANALYSIS.md) (8-paper gap analysis).** This is the current top-level direction and **supersedes the "accuracy-first" reading of §1–§4 where they conflict.** See **Section 0.5** for the thesis and the replication traps to avoid. Also reconciled with [DATA_DECISION.md](DATA_DECISION.md) and [HF_SOLUTION.md](HF_SOLUTION.md).

> **Fact-checked against primary sources ([FACTCHECK.md](FACTCHECK.md), 87 claims).** Novelty thesis **survives** (zero counter-examples to the 3 pillars). Load-bearing corrections applied below: (a) the **95.5% / per-AU aggregation** result is **Steagall 2023** (`s41598-023-49031-2`), NOT Feighelstein 2023 (`s41598-023-35846-6` = the 77/65% binary paper); (b) leakage = **191/336 clips (56.8%)**, not 201/337; (c) train pain boxes = **414** (~202 source), and 1819 no_pain is **dataset-wide**, not train-split; (d) head-position aggregation is **Min**, not Mode. **Still UNVERIFIED — treat as hypotheses, do not assert:** the Sci Rep 2025 "VLMs underestimate FGS" claim that justifies Claude-as-labeler (paper not yet retrieved); "DINOv2-frozen > landmark-XGBoost" (extrapolation — must test in Gate 4/5); any "distinct pain individuals" count (needs re-ID — `264` is a **box count**; `CAT_01` = 84 clips is the lone known individual).

---

## 0.5. Project Thesis & What We're Actually Building (the decision)

**The headline deliverable is TRUSTWORTHINESS at the 0.39 decision point, NOT accuracy.** The field has validated the human FGS + 0.39 threshold, automated *binary* pain many times, automated *graded* FGS exactly once **for the FGS per-AU 0/1/2→0–10 structure** (Steagall 2023 — closed data, hand-crafted geometry + XGBoost; the adjacent "Feline SentiNet" 2023 does 5-category pain grading but not the FGS AU structure), and shipped **zero** calibration / confidence intervals / open artifacts / measured training-label reliability. That last clause is the unowned white space we take.

> **HEADLINE REFRAME (FINAL_DIRECTION §A/§B).** The two **portable methods** are the headline — (i) **VLM-as-AU-rater κ vs vet** (can a frozen VLM weak-label FGS AUs at human-rater agreement) and (ii) the **capture-condition confound-attribution protocol** (FGS-BG-Gap + per-AU EBPG, reusable on the next dataset). The **engine below is conceded as plumbing — explicitly NOT claimed novel** (single-source data kills any non-portable claim). The **v1 spine is binary-plus-wrapper**; the graded layer ships **inspected-not-validated**. Read the layers below through that lens.

**What we build (one vertical, three layers):**
1. **Engine (CONCEDED as plumbing — not a novelty claim)** — frozen **DINOv2 ViT-S + 5 rank-consistent CORN ordinal heads** → per-AU 0/1/2 → 0–10 → 0.39 decision. A standard frozen-backbone + ordinal-head assembly; we state plainly in the paper that the engine is not claimed as novel (a "swap-the-backbone + extra metrics" reviewer filing must find nothing to kill here).
2. **Supervision** — **VLM weak-labels the 5 AUs** → small vet calibration anchor → active learning. The only path from our binary/web data to graded FGS without a closed expert corpus. *(Never attempted in this field.)*
3. **Validity wrapper (THE HEADLINE — genuinely unowned):** VLM-as-rater per-AU **κ vs vet**; **calibration** (reliability diagram, Brier/ECE); **pain-recall @ fixed sensitivity with bootstrap 95% CIs, half-width stated first**; **decision-curve** under undertreatment≫overtreatment; **defer-to-vet abstention** (accuracy-vs-coverage) as a model OUTPUT; **capture-condition confound audit**. The 2026 COSMIN review (Lee & Steagall, JVIM) covers **only human-rater instruments** and flags measurement-error / reliability / validation / interpretability as underreported — **calibration, CIs, and automated-scorer reliability fall outside its scope entirely**, which is exactly the white space we take. *(Do not claim the review "named calibration/CIs" or "explicitly excluded automated scorers" — it does neither; AI scoring is simply out of its topic scope.)*
4. **Artifact** — released weak-labeled AU annotations + frozen-backbone+CORN code/weights + frozen clip-grouped test split + a **datasheet** documenting pseudo-label provenance and the confound audit. **Not** a "trustworthy-labels leaderboard" (we distrust our own labels; CatFLW is CC BY-NC).

**Why feasible on our stack:** the backbone is frozen (only light CORN heads train on M4/Colab); the entire wrapper is post-hoc analysis on held-out scores — **no FGS gold corpus, no second cohort, no video, no large compute** required.

**Survives failure (the insurance):** if VLM-vs-vet κ collapses on muzzle/whiskers (where human reliability is also lowest), fall back to a **calibrated binary decision + abstention** — the validity-wrapper contribution still stands. There is no project-killing null result.

### Replication traps — what we explicitly do NOT do
1. **Not** Steagall's landmark → geometric-descriptor → XGBoost graded pipeline (done; needs 8 real-time vet raters we lack).
2. **Not** Martvel's video/temporal pipeline (we have no video / timestamps / paired frames).
3. **Not** the binary-accuracy leaderboard (77/79/95% are saturated anti-benchmarks; dishonest on our confounded labels).
4. **No** external-cohort validation claim (one confounded Flickr/CAT_01 source — name single-source as the top threat-to-validity, don't pretend to fill it).
5. **No** responsiveness / longitudinal within-cat rescoring (needs paired pre/post-analgesia timestamped frames we don't have).
6. **No** multi-rater model-vs-human ICC/Bland-Altman (needs a rater panel; we have at most one vet).
7. **No** headline cross-species horse transfer (5 horses, 3/5 AUs) — **pipeline-validation scaffolding only**.
8. **No** shipped mobile app or multimodal posture/audio as the thesis (a minimal HF Space demo is a delivery vehicle, not the contribution).

### First actions (gate-ordered)
0. **Gate 0 — Power calcs + vet-budget pre-registration FIRST (FINAL_DIRECTION §6; blocks everything quantitative, ~1 afternoon, zero data, no GPU).** Write the *single integer* (how many faces the vet scores, per-AU, in how many sittings) and run three zero-data power calcs: (a) faces for per-AU κ CI half-width ≤0.15 (kappaSize); (b) faces for LTT/MAPIE-certified NPV≥0.90 at ≤40% abstention; (c) whether the 0.39 CI is reportable at the budgeted n. **This precedes the actions below** — every quantitative claim draws on the same ~50-positive vet account, so the budget is the project's single point of failure.
1. **Gate 2 confound audit** — trivial brightness/blur/box-aspect/CLIP classifier on the existing binary Roboflow labels. Cheapest step; decides whether the corpus is salvageable and frames the project's honesty. **One-directional: report "no confound *detected at this power*," never "no confound."** Frame the deliverable as a *portable protocol*, not "CAT_01 is confounded." *(Requires no vet, no GPU.)*
2. **VLM 5-AU weak-labeler + ~120-image vet-κ pilot (Gate 1-B below)** — the per-AU reliability number every downstream claim rides on; the pilot itself **self-justifies the labeler** (we select the VLM by measured per-AU κ on the anchor).
3. **Frozen DINOv2 + CORN validated on the horse 0/1/2 labels first** (Gate 4 decode→sum→threshold sanity) before touching cat AUs.

> **Gate 1-B (weak-label reliability, GO/NO-GO):** a ~120-image (≥50 pain-positive) VLM-vs-vet per-AU **quadratic κ with 95% CI** pilot, run **before** scaling any labeling. **GO fires on the κ CI LOWER BOUND, not the point estimate** (FINAL_DIRECTION §3) — a point-estimate gate is not a gate at this n. Pre-registered, power-backed thresholds: orbital/ear/head κ-LB ≥ 0.6; muzzle/whiskers 0.4–0.6 acceptable-with-caveat. Below floor on an AU → report it honestly and/or drop to the binary fallback. This sits alongside the §3.4 low-prevalence Monte-Carlo go/no-go (now a 2-point ρ band).

---

## 0. Blocking Gates (strict run-order — no number is believed until its prerequisites pass)

Anything produced before its gate passes is labeled **provisional**. Gates run in order; a failing gate blocks everything downstream.

0. **Gate 0 — Power calcs + vet-budget pre-registration (NEW, FINAL_DIRECTION §6; blocks everything quantitative).** Before any vet spend: commit the *single integer* vet-label budget and run three zero-data power calcs — (a) faces for per-AU κ CI half-width ≤0.15 ([kappaSize](https://cran.r-project.org/web/packages/kappaSize/kappaSize.pdf)); (b) faces for an LTT/MAPIE-certified NPV≥0.90 abstention band at ≤40% abstention; (c) whether the 0.39 sens/spec CI is even reportable at the budgeted n. **Exit:** budget integer committed; if (a)/(b) fail, the headline magnitude claim (§A) and the abstention "guarantee" (§F) are demoted *on paper now*. Compute: none. *Rationale: every quantitative claim (κ, 0.39, NPV, correlation, severity cells) draws on the same ~50-positive vet account — this is the project's single point of failure.*
1. **Gate 1 — Per-CAT merge (keystone, blocks everything).** Collapse clips into true individuals *before* building folds. **Validate the merge against the trusted `CAT_` filename IDs — do NOT blindly trust `CLIP cosine>0.6` / `pHash Hamming≤10`.** CLIP/pHash are **duplicate detectors, not cat re-ID**; tune the threshold against the `CAT_` ground truth or swap in a proper re-ID embedding so the merge does not produce false splits/merges. Assert no individual straddles any fold. (Debate ruling 6.1; supersedes the CLIP/phash language in §4 and §7.)
2. **Gate 2 — Capture-condition confound audit (co-gate; CHANGED — now a portable protocol + one-directional).** Train a *trivial* classifier on brightness / blur / box-aspect / CLIP-embedding to predict pain. If it beats chance, pain is entangled with acquisition context (the post-recovery / stressed-at-vet trap) → **log as a blocker.** Per-cat disjointness does NOT catch context-shortcut leakage. **One-directional (FINAL_DIRECTION §3):** well-powered to *detect* confounding, underpowered to *rule it out* — report "no confound **detected at this power**," never "no confound." Frame the headline deliverable as the **transportable FGS-BG-Gap + per-AU EBPG protocol** (reusable on the next corpus), not "CAT_01 is confounded." (Debate ruling 6.2.)
3. **Gate 3 — Frozen, hashed, cat-disjoint hold-out + CI abort.** Build the boundary-rich, true-~13%-prevalence test set from merged groups. **Commit the fold CSV + test group-ids, hash them, and enforce a CI gate that aborts any train / sweep / threshold run able to read the test manifest.** This structurally prevents the thesis's post-hoc goalpost move. (Debate ruling 6.3.)
4. **Gate 4 — Compute-correctness pre-checks (BLOCKING, promoted from "risk").** Before any weak-labeling or scoring run: (a) MPS-vs-CPU logit parity for frozen DINOv2; (b) 1-epoch coral-pytorch CORN smoke test on MPS; (c) **unit-test the CORN-decode → per-AU 0–2 → 0–10 sum → 0.39-threshold path on synthetic logits/labels against known values.** A leak-proof PR-AUC from silently-wrong MPS logits is as worthless as the thesis's 79%. (Debate ruling 6.4 / (f).)
5. **Gate 5 — Alignment/resolution gate before Phase B.** On 30–50 project images: measure CatFLW-landmark **NME inside the RF-DETR crop vs. a manual eye-aligned crop**, and median **face-pixel resolution after resize to the DINOv2/14 grid**. The detector-box-as-cropper is a **RISK, not an advance, until measured.** If NME is high or resolution too low, add a **2-point eye-similarity alignment** step and/or a higher-patch ViT before any CORN head is trained. (Debate rulings 6.3/6.6; supersedes the "boxes are directly reusable" framing in §1 / §2.)
6. **Gate 6 — Severity-cell count gate (NEW, FINAL_DIRECTION §3-G; runs after the vet anchor; pure counting).** Pre-committed structural decision, **not a caveat**: count vet-confirmed **AU=2 (severe)** cells per AU. **If any AU's =2 cell is single-digit → collapse the high end** (merge AU 1+2, or report only painful/not-above-threshold) — do NOT print a per-cell severe sens/calibration number. *Rationale: an AU=2 sensitivity CI spans ~[0.35, 0.97] at single-digit n — no information — and a caveat leaves a misleading anchoring number in a table. This gate is expected to fire given the base rate.* Compute: CPU.

---

## 1. Executive Summary

- **Phase A = object detection, RF-DETR-Nano primary** (YOLOv11s runner-up), fine-tuned from COCO on the existing 2-class boxes. Keep it as a detector — the box is the *candidate* face localizer/cropper feeding Phase B. Verified: boxes are tight frontal head crops covering ~27–30% of frame, centered ([box geometry inspection](https://pmc.ncbi.nlm.nih.gov/articles/PMC6911058/)). **But the box-as-AU-cropper is unvalidated alignment — gated behind the Gate 5 NME/resolution audit, not assumed good.** Add a 2-point eye-similarity alignment step if NME is high.
- **Re-split before trusting any metric.** The shipped Roboflow split leaks the same cat/clip across train and valid. Group by source-clip id parsed from filenames (`{8-digit clip}_{3-digit frame}.png`) and use `StratifiedGroupKFold(n_splits=5)`. This is the single highest-risk correction in the project.
- **Handle the ~6.9:1 imbalance with data + loss, not architecture:** oversample/copy-paste the 264 pain faces, class-weighted/focal loss, and **threshold tuned for high pain recall (≥0.90)** on the validation PR curve. Judge on PR-AUC and per-class pain recall, never accuracy or raw mAP.
- **Phase B = VLM weak-label (5 AUs, each 0/1/2) → vet review → frozen-DINOv2-ViT-S + 5 CORN ordinal heads**, summed to 0–10, ratio = sum/10, thresholded at >0.39 for the analgesia decision. Frozen backbone beats landmark-XGBoost and full fine-tuning in the few-hundred-label regime ([DINOv2](https://arxiv.org/html/2304.07193v2)).
- **Compute split:** RF-DETR (≥v1.6.0) and frozen DINOv2 + CORN run locally on M4/MPS; do real detector training on Colab T4. **bitsandbytes 4-bit on macOS arm64 is CPU/alpha-Metal supported but impractically slow** (not strictly CPU-only — corrected per FACTCHECK C78) — so use `mlx-vlm` for *speed*, or (preferred for v1) hosted-API inference-only weak-labeling. Colab-T4 still needs CUDA bitsandbytes for bnb-4bit.

---

## 2. Phase A — Binary Pain / No-Pain Detector

### Decisions (decisive)
- **Detection vs crop-then-classify → DETECTION.** The detector is also your Phase B face-cropper *candidate*; a redundant localizer is wasted work. The shipped boxes are verified full-face head crops, so detection output is **reusable downstream only if Gate 5 (NME/resolution audit) passes** — treat "directly reusable" as a hypothesis to measure, not a settled fact. Crop-then-classify is a later optimization only.
- **Roboflow-hosted vs Colab → Colab/local with the `rfdetr` package (PRIMARY); Roboflow-hosted is the runner-up** for a zero-setup v1 smoke run. Choose local because you need custom imbalance handling (per-class loss weights, manual copy-paste oversampling, threshold tuning) that hosted training does not expose. Use Roboflow only as the dataset system-of-record and for its evaluation/Workflow tooling.
- **Architecture → RF-DETR-Nano/Small (PRIMARY), YOLOv11s (RUNNER-UP).** RF-DETR uses a DINOv2 backbone, converges in fewer epochs on small custom data, and trains natively on MPS as of [v1.6.0](https://github.com/roboflow/rf-detr/releases). YOLOv11s is the faster-to-iterate fallback with a mature CLI and a documented [weighted-loss recipe](https://docs.ultralytics.com/guides/custom-trainer) for imbalance.

### Pipeline

**1. Data + re-split (do this first).**
- Export the dataset **with `Fit (reflect/white edges)`, NOT `Stretch-to-640`** — v1 stretches ~4:3 images, horizontally inflating box aspect ratios and distorting facial AU geometry.
- Parse `group_id` from filenames: `^CAT_(\d{2})_(\d{8})_(\d{3})\.png` → `CAT{cam}_{clip}`, else `^(\d{8})_(\d{3})\.png` → `P_{clip}`. Namespace `CAT_`-prefixed clips separately (1 numeric collision exists).
- Discard the shipped split. Build folds with `sklearn StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, `groups=group_id`, `y=pain/no_pain`. Stratify at **image level with grouping** (135/336 clips are class-mixed, so you cannot collapse a clip to one label).
- Assert `set(train_groups).isdisjoint(set(val_groups))` and each val fold pain-fraction within ±0.05 of 0.127 with ≥25 pain images. Expected per-fold pain count ~45 at k=5.

**2. Preprocessing / augmentation (FGS-safe, light geometry).**
- `fliplr=0.5`, `translate=0.1`, `scale=0.3–0.5`, `degrees<=10`, modest `hsv_s`/`hsv_v`. **Avoid** `flipud`, large rotations, heavy color distortion, aggressive mixup/copy_paste blends — subtle orbital/muzzle/whisker cues are easily destroyed.
- `mosaic=1.0` early, `close_mosaic=10–15` to disable for final epochs so final-epoch stats reflect the real distribution.

**3. Model + training recipe.**

RF-DETR (PRIMARY, Colab T4 or M4 MPS, `rfdetr>=1.6.0`):
- `RFDETRNano` or `RFDETRSmall`, resolution **512 or 576** (must be divisible by patch×windows; valid: 384/512/576/704).
- Effective batch 16 on T4: `batch_size=4`, `grad_accum_steps=4`.
- `lr=1e-4`, `lr_encoder=1.5e-4` (lower backbone LR), `weight_decay=1e-4`.
- `epochs=100` (expect earlier convergence), `use_ema=True`, `early_stopping=True`, `early_stopping_patience=10`, `early_stopping_min_delta=0.001`.
- Source: [RF-DETR training parameters](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/).

YOLOv11s (RUNNER-UP, fastest iteration):
- `yolo detect train model=yolo11s.pt data=data.yaml imgsz=640 batch=16 epochs=120 patience=25 optimizer=auto close_mosaic=10`.
- Always fine-tune from the COCO `.pt` (cat is a COCO class → strong transfer); never random init.

**4. Imbalance handling (stack cheapest-first, change one lever at a time).**
- (1) **Class weights:** YOLO `cls_pw` (start 0.25–0.5; full inverse-freq = 1.0); or subclass `v8DetectionLoss` with `BCEWithLogitsLoss(pos_weight≈sqrt(1819/264)≈2.6–3)` per the [custom-trainer guide](https://docs.ultralytics.com/guides/custom-trainer).
- (2) **Oversampling + copy-paste:** duplicate the 264 pain images and implement **object-level copy-paste** (paste pain face crops into other images) as a preprocessing step — box-only datasets need manual copy-paste since Ultralytics `copy_paste` is segmentation-only. [Oversampling is the strongest detection-imbalance lever](https://ceur-ws.org/Vol-3628/paper14.pdf).
- (3) **Focal loss** only if pain recall stays low: `FocalLoss(gamma=1.5, alpha=0.25)` ([Ultralytics loss](https://docs.ultralytics.com/reference/utils/loss)).
- (4) **Threshold tuning** (last): sweep conf on the val PR curve, pick the highest threshold still meeting **pain recall ≥0.90** (false negatives are the clinically costly error; the bar rises only as far as the recall floor allows). Report the chosen threshold explicitly.
- Treat the **414** train "pain" boxes (~202 source + augmentation) as a **box count, not individuals** (distinct cats need re-ID); the clean **valid split (53 pain / 356 no_pain) is unaugmented** — budget review and de-dup against it.

**5. Evaluation.** Per-class precision/recall/F1, confusion matrix, **PR-AUC / Average Precision** as primary, plus `mAP@50` and `mAP@[.5:.95]` for localization only. Track **pain-class recall** as the go/no-go metric. Always report mean ± SD across the 5 cat-grouped folds.

---

## 3. Phase B — Graded 0–10 FGS

Treat Phase B as a measurement-error problem under low prevalence: even a labeler with 95% specificity yields low *observed* sensitivity (the worked example is **10% prevalence → ~53% sensitivity**; our dataset is ~12–13%) ([Chavoshi et al. 2025](https://arxiv.org/abs/2506.07273)). So **never estimate the >0.39 decision on weak labels — only on vet-confirmed ones**, and prioritize VLM specificity.

### 3.1 VLM weak-labeling
- One structured-output call per cropped face. With Claude: `client.messages.parse()` + Pydantic, or strict tool-use `input_schema`. Schema pins **each of 5 AUs to `enum [0,1,2]`** (NOT min/max — Claude structured outputs ignore numeric bounds), `additionalProperties:false`, all required, plus per-AU `rationale` (emitted *before* score) and `confidence`/`abstain`. **Compute the 0–10 sum and the analgesia decision in code, never let the VLM emit them.**
- Rubric in the system prompt: one paragraph per AU (ear, orbital, muzzle, whiskers, head) with verbatim Evangelista 0/1/2 descriptors (0=absent, 1=moderate/partial/uncertain, 2=marked/obvious; [Evangelista 2019](https://www.nature.com/articles/s41598-019-55693-8)). Instruct: default ambiguity to **1**, set low confidence on occlusion/blur/out-of-frame.
- **Frontal-pose / quality gate before scoring** — route profile/tilt/occluded/eyes-closed crops to the vet (FGS assumes frontal view). Apply ~12–15% margin expansion to the box (not a shrink) to avoid clipping ear tips/whiskers.
- Run via the **Message Batches API** (50% cost, async) with the rubric in a **cached system block** (prompt caching) across all ~2040 images.
- **Labeler selection is SELF-JUSTIFIED by the Gate-1-B κ pilot, not by citation (FINAL_DIRECTION §3 / FACTCHECK C54).** Do **NOT** cite the unretrievable Sci Rep 2025 "VLMs underestimate FGS / only Claude acceptable" claim (`s41598-025-27404-z` — paper not retrieved, claim unconfirmed) as the reason for choosing Claude. Instead: run the candidate VLM(s) through the per-AU vet-κ pilot and state "we selected the labeler by measured per-AU κ on the anchor." If a systematic per-AU offset *is* observed empirically in the pilot, model it as a per-AU offset and debias — as a measured fact, not a cited expectation.
- Reusable templates: [`Amitr16/catmd` `src/ai/fgs.ts`](https://github.com/Amitr16/catmd) prompt+schema; AU definitions grounded in [CatFACS](https://github.com/AnimalFACS/AnimalFACS).

### 3.2 Human (vet) review — triage by value-per-hour
- Pre-fill a review UI (Roboflow review, or CVAT/Label Studio) with the 5 VLM scores + rationales; vet **accepts/corrects flagged AUs only**, doesn't re-score from scratch. Only vet-confirmed rows enter training.
- Review priority: **(1)** VLM pain-positive / near-threshold (summed 3–5/10) and the rare positive class; **(2)** low-reliability AUs **muzzle & whiskers** (ICC 0.55–0.67) and the most-diagnostic **orbital tightening**; **(3)** [cleanlab](https://github.com/cleanlab/cleanlab) confident-learning-flagged likely errors and multi-VLM disagreements.
- **Budget:** ~120–150 vet-confirmed images (stratified, ≥50 pain-positive) is both the per-AU kappa 95%-CI floor ([kappaSize](https://cran.r-project.org/web/packages/kappaSize/kappaSize.pdf)) and the practical floor to train 5 frozen-backbone CORN heads. Scale to ~300–500 confirmed in B1.

### 3.3 Multi-task ordinal model
- **Backbone:** frozen **DINOv2 ViT-S/14** (forward-only, MPS-friendly), fed a fixed-size crop from the Phase A box. Frozen features + shallow heads is the best-supported low-data choice; **do NOT make the landmark→XGBoost pipeline the v1 path** — CatFLW-pretrained landmark NME is 9–26% by morphology on field photos and automated landmarks cost ~7pts of pain accuracy ([Frontiers 2024](https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2024.1442634/full)). Keep landmarks as a parallel interpretability track only after an NME sanity-check on ~30–50 project images.
- **Heads:** 5 independent **CORN** ordinal heads, one per AU. Each = `Linear(feat_dim, num_classes-1)` = `Linear(feat_dim, 2)` (3 levels). Total output 10 logits. CORN over CORAL/softmax for rank consistency without CORAL's capacity restriction ([CORN](https://arxiv.org/abs/2111.08851)).
- **Loss:** per AU `corn_loss(logits_au, y_au, num_classes=3)` (zero-indexed labels), summed across the 5 heads (optionally per-AU weighted for imbalance and to upweight muzzle/whiskers). Train with **co-teaching / small-loss selection** ([Han et al.](https://arxiv.org/abs/1804.06872)) on the VLM-labeled mass + the vet-confirmed set as a clean anchor — not confirmed-only, not naive-all (which bakes in the VLM under-estimation bias).
- **Decode (distributional path is now the DEFAULT — FINAL_DIRECTION §3-E.1).** Keep the soft CORN probabilities `P(rank>k) = cumprod(sigmoid(logits))` per AU → form each AU's pmf over {0,1,2} → **convolve the 5 per-AU pmfs into a single pmf over the 0–10 sum.** From that distribution report **RPS-on-the-sum (one scalar, bootstrap CI) + per-AU ClasswiseECE.** **Relegate `corn_label_from_logits` argmax-decode to the 0.39 POINT decision only.** **Do NOT print a binned reliability diagram on the 0–10 sum** — at ~11 atoms it is degenerate (per-bin SE ±0.18–0.26); that re-enters only at ~24+ samples/atom (n in the high hundreds).
- **Threshold / operating point (CHANGED — fixed high-sensitivity, FINAL_DIRECTION §3-D).** Clinical decision `painful = ratio >= 0.39` (≈4/10). **Select the cutoff inside train folds only (nested) at pain-recall ≥0.90** (the Evangelista anchor, sens 90.7%) — **NOT** Youden-J/F1 (a symmetric-cost knee is the wrong loss for a welfare instrument where undertreatment ≫ overtreatment). Report the specificity that high-sensitivity cutoff buys, with bootstrap CIs, and the cutoff's variance across folds. **The wrapper's HEADLINE artifact is the harm-ratio-weighted decision curve** (sweep undertreat:overtreat as a *range* across the dcurves threshold-probability axis — we have no vet-elicited ratio), **not ECE.** Benchmark against the validated FGS operating point (AUC 0.94, sens 90.7%, spec 86.6%; [Evangelista 2019](https://www.nature.com/articles/s41598-019-55693-8)).
- **Break circularity at validation (existential risk).** A QWK computed against VLM-derived labels measures *nothing but the model re-learning the VLM's heuristic* — **never report it as validation.** Estimate sens/spec at the 0.39 cutoff on **vet-confirmed labels only**, anchored against an **independent non-face signal** (charted clinical reason / analgesia outcome) where available. Apply the **two-source pain rule** (Feighelstein): a "pain" label needs grimace **AND** a charted clinical reason. The model card must state plainly that **no clean nociception ground truth exists in this dataset** (debate ruling 6.9).
- Reference scaffold: [`marroyol/comp3000-marie`](https://github.com/marroyol/comp3000-marie) (runnable cat-FGS pipeline); ordinal code from [coral-pytorch](https://github.com/Raschka-research-group/coral-pytorch); DeepMGS per-AU pattern from [`BenjaminCorvera/MGS-pipeline-distribution`](https://github.com/BenjaminCorvera/MGS-pipeline-distribution).

### 3.4 Noise-propagation pre-check (run before scaling labels)
~30-line NumPy Monte Carlo: build a 3×3 per-AU confusion matrix reproducing target quadratic kappa, draw 5 noisy AUs/image, sum, threshold at 4/10, 5,000 trials → read MAE and sens/spec at ~13% prevalence ([method, Chavoshi et al. 2025](https://arxiv.org/abs/2506.07273); the worked example is 10% prevalence). Seed muzzle/whiskers kappa at 0.4–0.6, ear/orbital/head ≤0.85.
- **Cross-AU error correlation: 2-point ρ SENSITIVITY BAND, not an empirical 5×5 matrix (CHANGED — FINAL_DIRECTION §3-E.2).** Do **NOT** fit an empirical 10-off-diagonal cross-AU error-correlation matrix from the anchor: at n≈50 each entry's CI is ~[−0.4, +0.6] (it launders noise as data-driven rigor and may not even be positive-definite). Instead run the Monte-Carlo under **ρ=0 AND a pinned ρ=0.3 (uniform)**, and **report GO only if the decision holds under BOTH.** A pinned ρ is honest; a noisily-fit matrix is false rigor.
- **GO** if orbital/ear/head kappa ≥0.6, simulated summed-MAE ≤1.0/10, and simulated threshold sensitivity ≥0.80 **under both ρ=0 and ρ=0.3.** If GO under ρ=0 but NO-GO under ρ=0.3, declare the gate **fragile**, say so, and default to the binary fallback rather than claim a GO the correlation could flip. **NO-GO / pivot** to the binary fallback if after ~500 confirmed labels vet-only threshold sensitivity stays <0.70.

---

## 4. Evaluation & Cross-Validation Protocol

- **Splitter (shared by both phases):** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, `groups=clip_id` (the individual cat/clip), `y=pain/no_pain`. The group is the **cat/source-clip, not the image**. Verified necessary: the shipped split leaks **191/336 clips (56.8%)** across train+valid ([scikit-learn CV](https://scikit-learn.org/stable/modules/cross_validation.html)).
- **Leakage guards (assert in code):** train/val group sets disjoint; perceptual-hash / CLIP-embedding near-duplicate check that duplicates don't straddle the split (Roboflow search API returns a 768-dim CLIP vector per image — no local download needed). **Use cosine>0.6 / `imagehash` Hamming≤10 union-find ONLY as a near-duplicate detector — NOT as the per-cat merge.** Per **Gate 1**, the same-cat merge must be validated against the trusted `CAT_` filename IDs (CLIP/pHash are duplicate detectors, not cat re-ID, and will false-split/merge individuals); tune the threshold against `CAT_` ground truth or use a proper re-ID embedding.
- **Hold out one cat-disjoint test set** never touched during selection. Do all preprocessing / threshold tuning / calibration **inside each fold**.
- **Phase A metrics:** PR-AUC / AP (primary, pain=positive), per-class P/R/F1, confusion matrix, specificity; localization via `torchmetrics MeanAveragePrecision` (`map`, `map_50`, `map_75`, `map_per_class`). Derive the per-image pain decision (max-conf pain box vs tuned threshold) and score *that* with PR-AUC/recall. Never rank on accuracy or ROC-AUC.
- **Phase B metrics (CHANGED — FINAL_DIRECTION §3-E/§3-F).** Per-AU VLM-vs-vet **Quadratic Weighted Kappa with 95% CI** (the headline-#1 measurement; gate/report on the **CI lower bound**), MAE, adjacent-accuracy (|pred−true|≤1), per-AU macro-F1 (3-class). Calibration of the 0–10 sum uses the **distributional path**: **RPS-on-the-sum (bootstrap CI) + per-AU ClasswiseECE**, fit only on train folds. **Do NOT print a binned reliability diagram / binned ECE on the 11-atom sum** (degenerate at this granularity). The clinical decision: AUROC + sensitivity/specificity at the **fixed pain-recall ≥0.90** cutoff vs the FGS reference (sens 90.7% / spec 86.6%), plus the **harm-ratio-weighted decision curve** (headline wrapper artifact) and the **defer-to-vet abstention curve reported as NPV with a one-sided 95% lower bound** at each abstention rate (the word "guaranteed" is **banned** — see §F / Gate 0). *Graded-sum QWK is an internal/inspection metric only; it is never reported as a validated graded-instrument claim.*
- **Report mean ± SD across folds** (ideally bootstrap 95% CIs), never a single split.

---

## 5. Reference Implementations to Reuse

| Repo / Dataset | URL | How to use |
|---|---|---|
| marroyol/comp3000-marie | https://github.com/marroyol/comp3000-marie | **Best single Phase B scaffold** — runnable cat-FGS pipeline (landmark regression, FGS geometric features, control-vs-painful classifier, Gradio demo). Fork as interpretability/landmark track. |
| Amitr16/catmd (`src/ai/fgs.ts`) | https://github.com/Amitr16/catmd | VLM FGS prompt + strict JSON schema (5 AUs 0/1/2, rationale, confidence, quality gate). Adapt to Claude tool-use `input_schema`. |
| Raschka coral-pytorch | https://github.com/Raschka-research-group/coral-pytorch | `corn_loss`, `corn_label_from_logits` for the 5 per-AU ordinal heads. Has a Lightning CORN tutorial; use `num_workers=0` on small data. |
| facebookresearch/dinov2 | https://github.com/facebookresearch/dinov2 | Frozen ViT-S/14 features (torch.hub), the Phase B v1 backbone. Forward-only → MPS-friendly. |
| roboflow/rf-detr | https://github.com/roboflow/rf-detr | Phase A primary detector. Pin `rfdetr>=1.6.0` for native MPS training. |
| ultralytics/ultralytics | https://github.com/ultralytics/ultralytics | Phase A runner-up (YOLOv11s) + [`WeightedDetectionLoss` custom-trainer](https://docs.ultralytics.com/guides/custom-trainer) for per-class pain weighting. |
| martvelge/CatFLW | https://github.com/martvelge/CatFLW | 2079 cat faces, 48 CatFACS-aligned landmarks + bbox (Kaggle `georgemartvel/catflw`). Source for any landmark detector / eye-alignment; **dataset only, no project labels**. |
| BenjaminCorvera/MGS-pipeline-distribution | https://github.com/BenjaminCorvera/MGS-pipeline-distribution | DeepMGS per-AU 0–2 grimace pattern — closest precedent for the Phase B multi-task ordinal head. |
| cleanlab/cleanlab | https://github.com/cleanlab/cleanlab | `find_label_issues` to triage which VLM-labeled images go to the vet first; estimate VLM noise rates. |
| Blaizzy/mlx-vlm | https://github.com/Blaizzy/mlx-vlm | M4-native LoRA/QLoRA fine-tune of Qwen2.5-VL — the substitute for the CUDA-only bitsandbytes path. |
| AnimalFACS/AnimalFACS | https://github.com/AnimalFACS/AnimalFACS | Authoritative CatFACS AU definitions for vet labeling guidelines + VLM prompt grounding. |
| anshereina/Pawthos_Website | https://github.com/anshereina/Pawthos_Website | `fgsScoring.utils.ts` composite→severity bucketing (0–2 none / 3–5 moderate / 6–10 severe) and >4 analgesia threshold for the interpretation layer. |
| **Steagall et al. 2023** (Sci Rep, `s41598-023-49031-2`) | https://www.nature.com/articles/s41598-023-49031-2 | Validated two-stage FGS method (37-landmark→geometric→XGBoost, **95.5% acc**, MSE 0.0096) + per-AU aggregation guidance (**Mode for ear/orbital/muzzle; Min for whiskers AND head**). |
| Feighelstein et al. 2023 (Sci Rep, `s41598-023-35846-6`) | https://www.nature.com/articles/s41598-023-35846-6 | *Separate* paper — heterogeneous-cohort **binary** pain, landmark 77% vs DL 65%; XAI (mouth>eyes>ears). Do NOT attribute the 95.5%/aggregation to this. |

---

## 6. Iterative Subagent-Orchestration Improvement Loop

Spine: **Weights & Biases** (runs, sweeps, Artifacts) + **Roboflow** as immutable dataset system-of-record. Mirror each Roboflow version id into a W&B Artifact; tag every run with `git_sha` + `roboflow_version`. Pin env (uv/requirements), seeds, deterministic mode. W&B over MLflow for a solo M4/Colab workflow (hosted artifacts/sweeps, zero infra).

Run discrete, logged, idempotent rounds keyed by round number (resumable via `wandb` `resume='allow'` for Colab preemption):

1. **FIND** — current champion scores the unlabeled/holdout pool; emit a ranked low-confidence list. Image-level uncertainty = **Max** (best mAP/sample trade-off) or **Sum** aggregation of per-box (1−confidence) ([NVIDIA active learning](https://arxiv.org/pdf/2004.04699)); for Phase B prioritize images whose predicted FGS ratio sits near **0.39** (decision-boundary sampling). Preferentially mine predicted-pain + hard background false-positives.
2. **PUSH** — top-N (label budget) into Roboflow via the **Dataset Upload workflow block** gated by a `ContinueIf` confidence filter; predictions become pre-annotations for fast vet correction ([Roboflow CV skills](https://github.com/roboflow/computer-vision-skills)).
3. **VERSION** — generate a new immutable Roboflow version (`Fit` resize, defined aug); log it as a W&B dataset Artifact with the Roboflow version string in metadata.
4. **TRAIN** — fine-tune **from the previous checkpoint** (not COCO once decent); run a W&B **bayes sweep** whose `metric` is **pain recall** (or balanced F1), not mAP. Lock the champion via `sweep.best_run().config`.
5. **EVALUATE** — on the **fixed cat-disjoint hold-out test set** (never touched by active learning); log mAP@50, per-class P/R, pain recall, confusion matrix.
6. **ERROR-ANALYZE** — Roboflow confusion matrix / per-class metrics / vector explorer to decide what to mine next.
7. Repeat.

**Stopping criteria (stop when ANY fires):**
- (a) Pain recall on the frozen test set plateaus (<~1% absolute gain over 2 consecutive rounds);
- (b) Target reached — **pain recall ≥0.90** (Phase A) and acceptable vet-agreement on the 0.39 threshold (Phase B);
- (c) Label budget / vet time exhausted;
- (d) Newly mined low-confidence images drop below a count/uncertainty floor (model confident on the remaining pool).

Record the deciding metric and decision in W&B each round.

---

## 7. Risks & Open Decisions

- **[VERIFIED, must-fix] Leaky shipped split.** 191/336 clips (56.8%, full-manifest) span train+valid. Re-split by clip-id before any metric is believed. *Open:* one 8-digit clip = one recording, but a single cat (`CAT_01_`) spans clips 100–184 — true individuals are fewer than 336 and are **NOT computable without re-ID** (treat 264 as a pain-box count, not an individual count). Collapse same-cat clips for a fully per-*cat* (not just per-clip) leakage-safe split — **but validate the merge against the trusted `CAT_` filename IDs, not raw CLIP/phash (Gate 1).** CLIP/phash are duplicate detectors, not cat re-ID.
- **[VERIFIED] Re-export without Stretch.** v1 uses Stretch-to-640 on ~4:3 images, distorting AU geometry. Re-generate a version with `Fit` before any geometry-sensitive work.
- **Low-prevalence sensitivity collapse (Phase B).** ~13% prevalence makes the 0.39 decision fragile to VLM specificity errors. Validate sens/spec only on vet-confirmed labels; run the Monte Carlo go/no-go before scaling labels.
- **VLM reliability is unmeasured per-AU.** No published per-AU VLM kappa exists; muzzle/whiskers are the weakest AUs even for human experts (ICC 0.55–0.67). Your ~120-image pilot kappa study *is* the first measurement and a project contribution. Decide per-AU whether to trust, gate by confidence, or drop muzzle/whiskers.
- **CUDA blocker.** bitsandbytes 4-bit QLoRA is CPU-only on macOS arm64. Decision: **v1 = hosted-API inference-only weak-labeling** (no local VLM training); if local fine-tuning is later needed, use `mlx-vlm`, not bnb. Colab T4 can do 16-bit LoRA but not bnb-4bit without CUDA bnb.
- **MPS silent-correctness bugs → now BLOCKING (Gate 4), not just a risk.** Spot-check DINOv2/ViT logits MPS-vs-CPU, run a 1-epoch coral-pytorch smoke test on MPS, and unit-test the CORN-decode→sum→0.39 path on synthetic logits *before* any weak-labeling or scoring run. (Promoted per debate ruling (f): a leak-proof PR-AUC from wrong MPS logits is worthless.)
- **Open: detection vs two-stage classifier for Phase A.** Staying detection-only for now; revisit a dedicated binary classifier head only if Phase A localization is solved and you want higher binary accuracy.
- **[RESOLVED] Clip-level figures recomputed on the FULL 2040-image manifest:** **336 clips, 138 pain-bearing, 135 mixed, 191 leaked (56.8%)**. The prior 76%-sample extrapolation (337/146/142/201) is retired. Also note **493 exact-duplicate filenames** (2040 records / 1547 distinct) to handle on export.
- **Open: Roboflow plan gating.** Confirm the workspace tier permits the Dataset Upload workflow block and model-evaluation tooling; some are paid-gated. If hosted inference doesn't expose per-box confidences for the FIND step, run inference via `inference-sdk` to get raw per-detection scores.
- **[FRAMING, non-negotiable] Anti-benchmark + model-card discipline (debate consensus 1, ruling 6.12).** Never write "we beat 77/79%" — those are anti-benchmarks inflated by tiny test N, separability curation, and (thesis) likely subject leakage + a metric bug. Always print the **distinct-cat (and distinct-pain-cat) denominator** and **95% CIs** (Clopper-Pearson / bootstrap); augmented copies never enter any reported N. Ship as **decision-support triage** ("grimace consistent with pain, X/10; recommend vet assessment and rescore after analgesia"), never an autonomous analgesia trigger. Document the **negative class as an unknown mixture** (possible sedated / post-recovery cats that subject-disjoint CV cannot catch), plus morphology / pose / coat-color coverage gaps.
- **Open: headline CV choice (debate ruling 6.a).** Report the **multi-frame, per-cat-grouped CV with honest wide CIs + per-clip inference aggregation** as the headline. Adopt a one-frame-per-cat *secondary* clean hold-out **only if** the merged distinct-pain-cat count stays >50 (else it produces an underpowered, false-precision headline of exactly the kind we mock). Unknown until Gate 1 runs.
- **Inference robustness (free win neither paper attempted).** Aggregate per-clip predictions — per-AU **Mode** for ear/orbital/muzzle, **Min** for whiskers AND head ([Steagall 2023](https://www.nature.com/articles/s41598-023-49031-2)) — rather than scoring isolated frames.