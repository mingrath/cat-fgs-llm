> **⚠️ SUPERSEDED where it conflicts with `FINAL_DIRECTION.md` (authoritative) and the code.**
> In particular this doc's "group by CLIP, never collapse the individual" guidance is
> REVERSED in the shipped pipeline: CV is grouped by `cat_id` (see `src/data/folds.py`,
> `scripts/gate2_confound.py`). Trust the code + `FINAL_DIRECTION.md` over this briefing.

# DECISION BRIEFING: Is Our Data Enough, and Should We Use the Local Cat Archive?

## 1. Is our current dataset enough?

**Phase A (binary pain detector): YES to TRAIN, NOT YET to make a clinical recall claim — but the picture is materially better than feared.**

The ground-truth count is **260 unique pain frames** (not 264 — that is a box-instance count; one image is class-mixed) out of **2040 unique source images** (train 1631 / valid 409; the shipped "3262 train" is 2x augmentation, not real data). 260 pain-positives clears the hard floors for transfer learning: the ~100-positive minimum and the ~10-events-per-effective-parameter EPV rule. With a frozen/transfer backbone, focal/class-weighted loss, and oversampling of the 260 pain faces, this is a textbook data-starved-but-trainable regime.

**The distinct-pain-cat reality (the real sample size) is far healthier than the early "low-tens (15–40)" guess.** Filename parsing (BUILD_PLAN regex, 260/260 matched) yields **173 distinct pain source-clips** and a defensible **~132–173 distinct pain individuals** — 4–10x the placeholder. Only **one** CAT_xx camera (CAT_01) exists in the entire dataset. The much-discussed "CLIP cosine > 0.6 connected-components" merge is **unusable** (CLIP carries no individual-identity signal here; it collapses 98% of pain pairs into one blob) — do not use it; if true per-individual dedup is ever needed, do it on downloaded pixels with imagehash or a cat-face re-ID model.

What this buys us: `StratifiedGroupKFold(5)` **grouped by CLIP** gives pain-clips per fold of [39, 36, 27, 33, 38] — **no fold at 0 or 1**, so subject-exclusive CV is safe and validatable. (Grouping by collapsed individual is a trap: it fuses CAT_01 into one 605-image super-group and produces a degenerate fold. **Group by clip, not by collapsed cat.**)

The honest limit: the *shipped* 409-image valid split holds only ~50 pain images and is leaky. A pain-recall point estimate off ~50 positives carries a 95% CI half-width of ~±0.13 — not clinically credible. **Fix: replace the single shipped split with repeated cat/clip-grouped k-fold CV, pool out-of-fold predictions, and report PR-AUC + pain recall with bootstrap 95% CIs and the CI width stated up front.**

**Phase B (5-AU ordinal 0–10 FGS): NO — currently impossible, and underpowered even after labeling.** The dataset has **zero** per-AU / 0–10 / FGS labels (binary boxes only), so current Phase B statistical power is literally **0**. Even if all 260 pain images were vet-labeled across 5 AUs, the **severe (level-2) cells** of each AU would fall to single-digits–low-tens — far below any per-class floor — making per-AU QWK/MAE unstable and the summed-score calibration untrustworthy exactly at the 0.39 (4/10) decision boundary. Phase B is a **data-acquisition and labeling problem, not a tuning problem.**

> ⚠️ **Bigger-than-leakage validity threat surfaced during investigation:** our pain/no_pain labels are almost certainly **unvalidated subjective pseudo-labels on generic Flickr pet photos** (see §2), not clinically grounded surgical pain. The binary signal may be capturing flat-faced/grumpy breed morphology and acquisition context rather than analgesia-relevant pain. A **label-reliability pilot and capture-condition confound audit must be GO/NO-GO gates** before any clinical claim.

---

## 2. What is the local archive, really?

The local archive is the **Zhang/Sun/Tang ECCV 2008 "Cat Head Detection" dataset** (redistributed as Kaggle `crawford/cat-dataset`), confidence VERY HIGH. On disk it shows 19,994 `.jpg` files, but the `cats/` subfolder is a **byte-identical duplicate** — the real count is **9,997 unique** full-scene Flickr cat photos (whole animals in natural scenes, ~500px long edge, NOT pre-cropped faces), each paired with a `.jpg.cat` sidecar holding **9 coarse facial landmarks** (1 left eye, 1 right eye, 1 mouth, 2×3 ear points). It has **no bounding boxes, no pain/no_pain class, no per-AU FGS labels, no 0–10 scores** — zero pain signal. License is CC0 on the Kaggle redistribution (unverified locally; underlying Flickr per-image rights may apply). It is purely a **face-detection / landmark** asset. **Notably, this same image family is the upstream parent of our Roboflow set** (identical 8-digit Flickr IDs + CAT_xx fingerprint), confirming our "pain" labels are pseudo-labels layered on this generic corpus.

---

## 3. Should we use the archive — and exactly how?

**Decision: Largely SKIP it. Pursue at most ONE narrow, conditional use; it is dominated by CatFLW on every dimension that matters.**

Of five candidate uses, four are dead on their merits:

- **Cat-face detector — DROP.** BUILD_PLAN already verified RF-DETR boxes are tight frontal head crops (~27–30% of frame): "no separate face detector is needed." The archive has no boxes anyway.
- **SSL / DINOv2 domain-adaptation pretraining — DROP.** 10k uncropped Flickr cats cannot meaningfully shift DINOv2's pretraining; the backbone is deliberately **frozen**. Real risk of degrading features for ~zero gain.
- **Hard-negative mining — DROP.** Out-of-domain Flickr cats with no boxes are distribution shift, not useful hard negatives; the in-domain no_pain set (1819) + Smudge negative-control suite already cover this.
- **The one viable use — the test-time EYE-ALIGNMENT helper (BUILD_PLAN items 204/253).** The flagged NME risk needs only a **2-point eye-similarity transform**, and the archive's 2 eye landmarks on ~10k cats are plentiful for that. **BUT** use this **only as a fallback if CatFLW is blocked** — CatFLW's 48 CatFACS-aligned landmarks dominate the archive's 9 coarse, non-FACS points (1 mouth point, ear tips) for any FGS geometry. Realistically: **CatFLW wins, the archive contributes nothing, and that is an acceptable, honest outcome.** Do not pre-build the aligner before the 30–50-image NME audit says it is needed.

**CAT_xx-overlap leakage — explicitly a FALSE ALARM.** The archive's `CAT_00..CAT_06` are *parent folders* bucketing unrelated Flickr cats; our `CAT_01` is an *in-filename camera id*. They never collide **as long as the grouping key is derived from filename stems only** (strip `_png.rf.<hash>.jpg`), never from a parent path. Our export tree is just `train|valid/{images,labels}` with **no CAT_xx directories**, so cross-archive phantom-merge is structurally impossible. The only real internal collision is clip `00000100`, which appears in both `CAT_01_00000100_*` and bare `00000100_*` forms; the BUILD_PLAN regex correctly maps these to **distinct** keys (`CAT01_00000100` vs `P_00000100`). Add a unit assertion that `00000100` yields two groups. **Keep the archive namespaced and stored outside the `datasets/` export path so it can never enter pain loaders.**

---

## 4. External datasets to add

**No open dataset closes the pain or per-AU FGS gap.** Every pain/FGS-labeled corpus is **request-only** (email the authors). Everything openly downloadable is **landmark-only with zero pain signal**. Verified directly via the authenticated Roboflow REST API (bypasses the Cloudflare 403) and PMC full-text + supplements.

### Tier 1 — PAIN / FGS-labeled (high value, all request-only)

| Name | URL | Pain/FGS-labeled? | Size | License | How to get it | Why |
|---|---|---|---|---|---|---|
| **Steagall / DeepMGS FGS corpus** | pmc.ncbi.nlm.nih.gov/articles/PMC10703818/ | **YES — 1188 imgs with true 5-AU 0/1/2 FGS** | 3447 imgs (1188 FGS-scored), 37 landmarks | Request-only; withheld pending commercial app | Email **P.V. Steagall** (Montreal), academic non-commercial framing | The single richest match to our exact Phase B target. Hardest to get. |
| **Evangelista 2019 FGS validation** | pmc.ncbi.nlm.nih.gov/articles/PMC6911058/ | **YES — 110 imgs scored 0–2 on all 5 AUs** | 110 imgs / 55 cats | Request-only | Email P.V. Steagall (same group) | Canonical gold-standard graded FGS; tiny but exact. |
| **Feighelstein / Finka + TiHo pain sets** | nature.com/articles/s41598-024-78406-2 | YES but **composite (MCPS/CMPS), not per-AU** | Finka ~26–29 cats; TiHo 72 videos | Request-only | Email **A. Zamansky / G. Martvel** (Haifa/TiHo) | Extra binary pain-positives + video path; NOT per-AU supervision. |

### Tier 2 — Landmark-only (open, for preprocessing only — ZERO pain signal)

| Name | URL | Pain/FGS-labeled? | Size | License | How to get it | Why |
|---|---|---|---|---|---|---|
| **CatFLW** (Cat Facial Landmarks in the Wild) | kaggle.com/datasets/georgemartvel/catflw | No | ~2079 imgs, **48 CatFACS-aligned landmarks** + bbox | **CC BY-NC 4.0** | Open Kaggle/GitHub download, no request | **Best open asset** for DETECT+LANDMARK+EYE-ALIGN+CROP + NME validation. Strict upgrade over the archive. |
| **Zhang 2008 / crawford cat-dataset** (our archive) | kaggle.com/datasets/crawford/cat-dataset | No | 9997 unique, 9 landmarks | CC0 (Kaggle) | Already in hand | Fallback eye-aligner only; inferior to CatFLW. Dedupe (it is 2x-duplicated; one CAT folder has corrupt `.cat`). |

**Not worth pursuing:** Roboflow Universe emotion/expression/"sick" proxies (Cat Moods Scanner, Cat Emotions, Cat Expression Detection) — coarse, smaller than our 260, no AU structure; `icu-egjok/pain-paitents` is **human** ICU. CatFACS is a free coding *manual*, not a dataset (use it as the AU rubric for vet review).

---

## 5. THE DECISION (numbered, tied to BUILD_PLAN.md)

1. **Phase A pain training — use ONLY our Roboflow set (260 pain / 1819 no_pain).** Train the binary RF-DETR/YOLO detector with focal/class-weighted loss + oversample/copy-paste of the 260 pain faces (BUILD_PLAN lines 11, 51, 251). Do **not** add external cats to pain training.
2. **Re-split FIRST: `StratifiedGroupKFold(5)` grouped by CLIP** (group_id from filename stem per BUILD_PLAN line 28). Pool out-of-fold predictions; report **PR-AUC + pain recall with bootstrap 95% CIs** and the CI width. Report distinct-pain-individual count (**~132–173**) as the true sample size, not "264/2040." Add a unit assertion that clip `00000100` → two distinct groups.
3. **Face-alignment / NME validation — adopt CatFLW (48 landmarks, CC BY-NC).** Download now; use it to train/validate the DETECT+LANDMARK+EYE-ALIGN+CROP stage (BUILD_PLAN lines 78, 110). Run the **30–50-image NME audit** (PAPER_DEBATE 204/253) comparing RF-DETR crops vs eye-aligned crops *before* building any aligner.
4. **Local archive — fallback only.** Build the 2-point eye-aligner from it **only if** CatFLW access/license stalls **and** the NME audit shows alignment is needed. Otherwise do nothing with it. Keep it namespaced and outside `datasets/`.
5. **Acquire next (parallel, do not gate the timeline):** email **Steagall** (1188-img + 110-img FGS sets) and **Zamansky/Martvel** (Finka/TiHo). Budget for non-response, especially Steagall (withheld for a commercial app).
6. **We must label the per-AU FGS gap ourselves — there is no shortcut.** Proceed with the Phase B plan: **VLM weak-label 5 AUs → vet review** (CatFACS manual as the rubric), prioritizing acquisition of **more distinct pain cats and especially severe AU=2 examples**. Power target: ~400+ pain-positive evaluation cases (~2x today) to pin the 0.39-threshold sensitivity/specificity to a ±0.10 CI.
7. **Gate Phase A clinical claims behind a label-reliability pilot + capture-condition confound audit** (our labels are pseudo-labels on generic Flickr photos). Update CLAUDE.md / RESEARCH.md / BUILD_PLAN.md: **provenance = Zhang 2008, NOT Finka 2019** — drop the Finka cross-dataset leakage warning entirely.

---

## 6. Risks & leakage guardrails when combining sources

- **Group by clip, never by collapsed individual.** Collapsing CAT_01 creates a 605-image super-group → one degenerate fold. Clip-grouping keeps every fold at 27–39 pain clips.
- **Never use CLIP-cosine for individual dedup** — no identity signal; any count it produces is a threshold artifact. Use imagehash / face re-ID on pixels if true per-cat dedup is ever required.
- **Grouping key = filename stem only.** Strip `_png.rf.<hash>.jpg`; never key on a parent folder named `CAT`. This neutralizes the lone `00000100` collision and makes archive CAT_xx folder collision impossible.
- **Provenance = Zhang 2008, not Finka.** If we later obtain CatFLW/Finka/Steagall, they share **no** cats with our Roboflow set → no Finka-overlap leakage. But each acquired set has only 26–84 cats — **regroup any merge by individual cat** or recall/PR-AUC will inflate.
- **License tracking:** our set CC BY 4.0; archive CC0; **CatFLW CC BY-NC 4.0 (blocks commercial deployment of any derived model)**; all request-only pain sets have no stated data license. Keep per-source licenses separate; do not let CC0/NC images leak into a CC BY release.
- **Augmentation ≠ subjects.** 2x augmentation expands frames, adds zero pain individuals; it stabilizes training but does not widen the generalization CI. Always count unique frames (1631 train), not augmented (3262).
- **Label-validity confound is the top risk** — bigger than any leakage question. Treat the binary pain signal as unvalidated until the reliability pilot passes.