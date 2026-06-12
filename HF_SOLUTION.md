# Feline Grimace Scale (FGS) Data Gap — Decision Briefing

## 1. Did Hugging Face have anything?

**Verdict: NO.** Hugging Face contains **zero** open datasets with graded per-action-unit feline FGS labels (ear/orbital/muzzle/whiskers/head each 0/1/2) or even binary cat-pain facial labels. Five independent search strategies (cat-pain-direct, cross-species, cat-face-landmark, model-hub, spaces-and-adjacent) all converge: every direct `cat pain` / `feline grimace` / `FGS` / `CatFACS` query returns 0 hits, and every published feline pain dataset (Steagall, Evangelista, Finka/TiHo) remains request-only off-platform. The gap is **not closable from HF**.

What HF *does* offer splits cleanly into one cross-species label set and a set of reusable preprocessing models — none of which give us cat FGS labels for free.

### Real graded-pain label sets (cat: none; cross-species: one)

| id | what it gives us | labels | license |
|---|---|---|---|
| `oliveirabruno01/openfarm-horse-grimace-region` | **Only** open AU-graded grimace set in our exact shape. ~6,070 rows (945 train / 279 test heldout-balanced + 3,930/916 raw). **HORSE not cat**; covers 3 of our 5 AUs (no whiskers/head). For curriculum pretraining + pipeline validation. Mirror of Mendeley DOI 10.17632/t8rtzcgwxm.3. **Caveat: only 5 unique horses (M1–M5)** — grouped by subject this is ~5 subjects, a coarse ordinal-direction prior, not calibrated transfer. | per-region ordinal 0/1/2 over {ears, orbital/eye, mouth-chin-nostrils} + derived binary | cc-by-4.0 |

> No cat equivalent exists. `KAITANG2003/cat-emotion-dataset` = LLM-generated affect (not clinical, do not use). `openfarm-dog-pain` = clinical TEXT. `openfarm-ungulate-valence` = AUDIO. `harisansarkhan/CatFaceLandmarks` = EMPTY repo. All confirmed dead-ends.

### Reusable preprocessing / model assets (detect → align → crop → encode; **no pain labels**)

| id | what it gives us | role | license |
|---|---|---|---|
| `facebook/dinov3-vits16-pretrain-lvd1689m` | Strongest small SSL backbone; drop-in upgrade to planned DINOv2 ViT-S | Frozen Phase-B encoder under 5 CORN heads | (HF gated/research) |
| `facebook/dinov2-small` / `timm/vit_small_patch14_dinov2.lvd142m` | Original-plan ViT-S (prefer the `_reg` register variant as the field default for localized per-AU features; A/B before locking ViT-S/14); transformers-native (`pooler_output` 384-d) or timm | Frozen encoder fallback; MPS-friendly | apache-2.0 |
| `mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped` | Real `.pt` weights (Faster-RCNN/SSDLite + HRNet/RTMPose) zero-shot quadruped detect+head pose | Detect → crop → head ROI | (DLC zoo) |
| `AlexEMG/DeepLabCutModelZoo-cat` | Only real cat keypoint weights on HF (ResNet-50, DLC runtime) | Cat face/body keypoints for ROI | (DLC zoo) |
| `d-v-18/cat-face-detector` | Stock OpenCV `haarcascade_frontalcatface.xml` | Dependency-light face-crop detector | mit |
| `cvdl/catfaces` | 29,842 raw cat-face RGB images, no labels | Unlabeled pool for SSL/aug/bootstrap | mit |
| `Qwen/Qwen2.5-VL-3B/7B-Instruct` | Open VLM for silver weak-labeling / AU sanity | (secondary to Claude weak-labeler) | apache-2.0 |

**No pretrained grimace/FGS/FACS ordinal head exists for any species** — that head must be trained by us.

## 2. The gap, restated

We still lack the one thing the project is built on: **trustworthy, vet-grade graded per-AU FGS labels** — 5 facial action units (ear, orbital, muzzle, whiskers, head) each scored 0/1/2, summed to 0–10, thresholded at the analgesia ratio **>0.39**.

Why no search closes it:
- **No open cat dataset carries these labels** (HF = 0 hits; published sets request-only).
- **The horse set covers only 3 of 5 AUs** — whiskers and head have **no** cross-species source anywhere. These two AUs are the project's structural blind spot under *every* approach.
- **What we currently hold is a binary Roboflow set** (~260 pain / 1,819 no_pain) with **no per-AU labels**, and per `DATA_DECISION.md §1` those labels are likely **unvalidated Flickr pseudo-labels capturing flat-faced/grumpy breed morphology and capture context, not nociception**. Any method that learns *from* this base (pseudo-labeling, synthetic generation, naive transfer) inherits and amplifies that confound.
- **AU=2 (severe) cells are single-digit** even after labeling everything — so the clinically critical high-pain end (exactly where the 0.39 decision lives) is statistically unsupportable without acquiring more real severe examples.

The gap is therefore a **data-acquisition-and-labeling problem, not a modeling/tuning problem**. No clever architecture manufactures clinical ground truth that does not exist.

## 3. Solution options compared

| approach | yields real FGS labels? | cost | payoff | verdict |
|---|---|---|---|---|
| **1. VLM weak-label (Claude Opus 4.8, 5-AU enum 0/1/2) + cleanlab/boundary triage → vet review → frozen DINOv2 + 5 CORN heads + active learning** | **Silver only**, not gold; trust comes from the vet layer it feeds | ~$12 (single batched+cached Opus pass over ~2,040 crops); $0 GPU; **binding cost = ~4–5 vet hrs** to clear the 120–150 kappa-CI floor | Complete 5-AU silver layer over all images **this week**, zero external permission; concentrates scarce vet hours on boundary/disagreement cases (prefill accept/correct); first per-AU VLM-vs-vet kappa protocol for cats (result pending the independent vet anchor) | **pursue-now (critical path / spine)** |
| **5. Data acquisition: email Steagall (1,188-img per-AU) + Evangelista (110) + Zamansky/Martvel; CatFLW today; vet-clinic later** | **YES — the only route to vet-grade gold**, covers all 5 AUs incl. whiskers/head | ~$0 + 1–2 hrs for emails; CatFLW free; vet-clinic = 20–60 vet hrs + ethics | Highest single-action EV (the email). But 30–50% share odds, often partial, weeks–months latency, outside our control; CatFLW/request sets are **CC BY-NC** (non-commercial) | **pursue-now but PARALLEL — never gates the timeline** |
| **2/4. Cross-species (horse) curriculum + SSL/few-shot, frozen DINOv2 + CORN** *(same engine, merged)* | **No** — label-efficiency engine, not a label source; covers 3/5 AUs | ~$0, **zero vet hrs**, ~1 day | De-risks the entire DINOv2+5-CORN+sum+0.39 pipeline on **real** graded data before any cat is scored; warm-starts 3/5 heads | **pursue-now as scaffolding** (promote from "later"); merge into spine |
| **4 (pseudo-label leg specifically)** | No | low | Stretches each vet label via active learning | **gated** — only after Gate-2 confound audit; under co-teaching with vet anchor; **never on the unaudited Flickr base** (amplifies confound) |
| **3. Synthetic / generative (diffusion img2img + ControlNet/IP-Adapter / LoRA / 3D rig)** | **No — circular** (label = generation condition); cannot mint truth we lack | <$20 compute + **8–20 vet hrs** to seed+QC | Low. AU-entanglement (can't move 1 AU to calibrated 0/1/2), sim2real gap, **amplifies the breed-morphology confound**, whiskers/head unanchored; consumes the same scarce vet hours it claims to save | **pursue-later, auxiliary only** — AU=2 minority augmentation + CORN monotonicity stress-test, **walled out of val/test and the 0.39 numbers** |

**Resolved disagreement:** Approaches 1, 2, and 4 are not rivals — they are **one stack** (Claude/VLM front-end → vet arbitration → frozen DINOv2 + 5 CORN back-end, warm-started on horse data). Approach 5 is **upstream of all of them** (the eventual gate on every clinical claim) but cannot drive a timeline we control. Approach 3 is the only genuine trap.

## 4. THE RECOMMENDED PATH

A single integrated pipeline, sequenced so nothing blocks on anything outside our control.

**Spine:** Claude Opus 4.8 structured-output weak-labeling → cleanlab + decision-boundary triage → vet accept/correct → frozen DINOv2/v3 ViT-S (register variant `dinov2_vits14_reg` as the field default — registers suppress attention artifacts that hurt the localized per-AU features; A/B the reg variant before locking ViT-S/14, plain stays backwards-compatible) + 5 CORN ordinal heads, with code-side sum→0.39 and an active-learning loop.
**Scaffolding (parallel, zero vet hrs):** horse-grimace curriculum warm-starts 3/5 heads and validates the whole pipeline on real graded data first.
**Insurance (parallel, never gating):** Steagall/Evangelista/Zamansky data-request emails + CatFLW download — the only route to true gold and to all-5-AU coverage.

### Sequenced plan

**Day 0 — in one sitting (no dependencies between these):**
1. **Send the data-request emails.** One-page non-commercial academic-use proposal + signed DUA/MTA offer + co-authorship commitment to **P.V. Steagall** (City University of Hong Kong, via felinegrimacescale.com team), requesting the **1,188-image per-AU FGS set + Evangelista 110-image set**; parallel note to **Zamansky/Martvel** (Univ. Haifa, Finka/TiHo). Budget for non-response; this never gates the timeline.
2. **Download CatFLW** (Kaggle `georgemartvel/catflw`, free) for the Gate-5 NME/eye-alignment audit — needed regardless of any reply.
3. **Clear Gate 4 (BUILD_PLAN §0):** MPS-vs-CPU DINOv2 logit-parity test **and** the CORN-decode → per-AU 0–2 → 0–10 sum → 0.39-threshold **unit test on synthetic logits**. This is the very first code keystroke. A leak-proof metric from wrong MPS logits is worthless.

**Day 1–2 — scaffolding (zero vet hours):**
4. Cache frozen `dinov2-small` (or `dinov3-vits16`) features for the **horse-grimace set**, fit the 3 transferable CORN heads (ear/orbital/muzzle), **grouped by horse id (only 5 horses)** to avoid leakage. This proves the end-to-end pipeline on real graded 0/1/2 data and warm-starts 3/5 heads. Training cat data later becomes a button-press, not a 2-week build.
5. Run the **Gate-5 NME/resolution audit** on 30–50 CatFLW/Roboflow crops; settle whether eye-alignment is even needed.

**Day 3+ — only after Gates 4–5 pass:**
6. **Run the ~$12 single Opus 4.8 pass** over ~2,040 crops via the **Message Batches API (50% off)** with the Evangelista rubric in a **cached system block** (verify `cache_read_input_tokens > 0` first; freeze the rubric — any byte change invalidates the cache). Each AU pinned to a tool-use `input_schema` enum `[0,1,2]`, `additionalProperties:false`, **rationale-before-score**, confidence/abstain. **The 0–10 sum and the 0.39 decision are computed in code, never emitted by the VLM.** Profile/occluded/eyes-closed crops route straight to the vet.
7. **Triage for vet review by value-per-hour:** pain-positive + near-threshold (sum 3–5/10), the weakest AUs (muzzle/whiskers, human ICC 0.55–0.67) and the most-diagnostic orbital, cleanlab confident-learning flags, and multi-VLM disagreements. **~4–5 vet hrs** of prefilled accept/correct clears the 120–150 kappa-CI floor (≥50 pain-positive).
8. **Train 5 CORN heads** on the frozen backbone with **co-teaching / small-loss selection** on the VLM mass + the vet-confirmed clean anchor; model VLM under-estimation as a **per-AU offset and debias it**. Pseudo-labeling on the Roboflow base is **gated behind the Gate-2 confound audit** and only as noisy mass under co-teaching — never fed unaudited.
9. **Validate sens/spec at 0.39 on vet-confirmed labels ONLY**, anchored where possible to an independent non-face clinical signal (BUILD_PLAN §3.3). **Never report QWK against VLM labels** (circularity). Run the §3.4 low-prevalence Monte Carlo go/no-go **before** scaling; **NO-GO / pivot to landmark-AU** if after ~500 confirmed labels vet-only threshold sensitivity stays <0.70.
10. **Active-learning loop:** champion mines images whose predicted FGS ratio sits near 0.39 → vet corrects → retrain. Fold in any Steagall/Evangelista gold the moment it arrives (re-init cat heads from horse heads where regions match).

### How this updates BUILD_PLAN.md / DATA_DECISION.md

- **BUILD_PLAN.md §3:** confirm the spine (VLM → vet → frozen DINOv2 + 5 CORN). **Add an explicit Step 0 cross-species curriculum stage** (horse warm-start of ear/orbital/muzzle, grouped by 5 horse ids) ahead of the VLM pass — promote cross-species transfer from "later" to "now / scaffolding."
- **BUILD_PLAN.md §0 Gates:** make Gate 4 (MPS logit parity + CORN-decode→sum→0.39 unit test) and Gate 5 (NME/resolution) **hard blockers before any scoring run or vet hour is spent**. Add **Gate 2 confound audit as a prerequisite for any pseudo-labeling**.
- **BUILD_PLAN.md §3.3–3.4:** codify "validate the 0.39 decision on vet-confirmed labels only; never report QWK vs VLM labels; run the prevalence Monte Carlo go/no-go before scaling." Add an explicit **per-AU under-estimation offset / debias** step before training.
- **DATA_DECISION.md §4–5.5:** record the HF verdict (no cat FGS data exists; horse set is the only cross-species asset, 5 horses, 3/5 AUs) and the **parallel-but-never-gating** status of the Steagall/Zamansky requests and CatFLW. Note **whiskers + head as the un-transferable AUs** whose only ground-truth source is acquired gold or owned vet-clinic collection.
- **DATA_DECISION.md §1:** reaffirm that the Roboflow binary base is a **confounded pseudo-label set** and must be treated as noisy mass under co-teaching, never as a clean anchor.

## 5. What to reuse from HF right now — exact next commands

```bash
# --- Backbone (frozen Phase-B encoder) + horse curriculum data ---
hf download facebook/dinov2-small --local-dir ./models/dinov2-small
# optional upgrade (gated; accept terms on the model page first):
# hf download facebook/dinov3-vits16-pretrain-lvd1689m --local-dir ./models/dinov3-vits16

hf download oliveirabruno01/openfarm-horse-grimace-region \
  --repo-type dataset --local-dir ./datasets/horse-grimace

# --- Detect / crop assets ---
hf download d-v-18/cat-face-detector --local-dir ./models/cat-face-detector            # haarcascade_frontalcatface.xml
hf download mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped --local-dir ./models/superanimal-quadruped
# AlexEMG/DeepLabCutModelZoo-cat — only if you adopt the DLC runtime for cat keypoints

# --- Unlabeled cat-face pool (SSL / augmentation / bootstrap) ---
hf download cvdl/catfaces --repo-type dataset --local-dir ./datasets/catfaces

# --- CatFLW (off-HF; gold landmarks for align/crop + Gate-5 NME audit) ---
# Kaggle CC BY-NC:
kaggle datasets download -d georgemartvel/catflw -p ./datasets/catflw && \
  unzip -q ./datasets/catflw/catflw.zip -d ./datasets/catflw

# --- Python deps for the back-end ---
pip install "coral-pytorch" cleanlab "transformers>=5.2" torch  # CornLoss / corn_label_from_logits; MPS auto-detected
```
**Do NOT** attempt to use `harisansarkhan/CatFaceLandmarks` (empty repo) or `KAITANG2003/cat-emotion-dataset` (synthetic LLM affect labels) as ground truth.

## 6. Open risks

- **Whiskers + head AUs are un-transferable** — no cross-species or synthetic source exists; they are data-starved under every approach until real gold (Steagall) or owned vet-clinic data arrives. Confidence-gate or drop them if the pilot shows them untrustworthy.
- **Low-prevalence sensitivity collapse:** at ~13% prevalence a 95%-specificity labeler yields only ~53% observed sensitivity (Wu 2025). Run the Monte Carlo go/no-go **before** scaling; the 0.39 decision is fragile to VLM specificity error.
- **VLM systematic under-estimation of FGS** (Sci Rep 2025) bakes bias into the severe end where 0.39 lives — must be modeled as a per-AU offset and debiased, not trained naive-all.
- **Confound inheritance:** the Roboflow base likely encodes breed-morphology, not pain (DATA_DECISION §1). Pseudo-labeling/synthetic generation on it amplifies the confound — gate behind the confound audit; validate only on vet-confirmed labels.
- **Circularity trap:** reporting QWK vs VLM labels measures the model re-learning the VLM heuristic. Forbidden — vet-confirmed labels only.
- **Vet availability is the true bottleneck** (project is solo; MEMORY: needs a reviewer). The whole spine stalls without ~4–5 confirmed vet hours; money and compute are not the constraint.
- **Acquisition is a coin-flip and slow:** Steagall is gated behind their commercial app (30–50% share, often partial/landmark-only, weeks–months or silence). Never on the critical path.
- **MPS silent-correctness (Gate 4)** and **prompt-cache invalidation** (any rubric byte change wipes the batch cache) — both cheap to guard, expensive if skipped.
- **AU=2 severe-cell sparsity** caps per-AU calibration at the clinically critical end regardless of label quality — only liftable by acquiring real severe examples.
- **License ceiling:** CatFLW and likely all request sets are **CC BY-NC** — fine for research/portfolio, fatal for any commercial product.

**Single most important first keystroke:** the Gate-4 CORN-decode→sum→0.39 unit test. **Single highest-EV first action (same sitting):** the Steagall data-request email.

Relevant files: `/Users/mingrath/ghq/github.com/mingrath/cat-fgs-llm/BUILD_PLAN.md`, `/Users/mingrath/ghq/github.com/mingrath/cat-fgs-llm/DATA_DECISION.md`, `/Users/mingrath/ghq/github.com/mingrath/cat-fgs-llm/RESEARCH.md`.