# FACT-CHECK REPORT — Cat FGS Pain Detector BUILD_PLAN.md

The plan's **novelty thesis survives**, but six load-bearing factual claims are wrong (mostly mis-attributed citations and sample-extrapolated dataset numbers), one compute claim is outdated, and three load-bearing claims that underpin the engine/labeler choice were never verified against their source.

---

## 1. Scorecard

| Verdict | Total | Load-bearing |
|---|---|---|
| **Confirmed** | 38 | 24 |
| **Partially-true** | 18 | 10 |
| **Outdated** | 1 | 1 |
| **Refuted** | 6 | 5 |
| **Unverifiable / asserted-not-verified** | 12 | 7 |
| **Totals (87 claims)** | 87 | ~47 load-bearing |

Headline: **5 refuted load-bearing** (C73, C12, C70/C01, C39, C81 — plus root-cause C80), **1 outdated load-bearing** (C78), **and the novelty spine holds** with two required rescopings (C06, C12).

---

## 2. Confirmed solid (rely on as-is)

**Clinical spine (Evangelista 2019, s41598-019-55693-8; Steagall 2023, s41598-023-49031-2):**
- **C04** — FGS operating point **AUC 0.94, sens 90.7%, spec 86.6%** (Evangelista 2019 Fig 7).
- **C05** — analgesia threshold **>0.39/1.0 (~4/10)** (Evangelista abstract; COSMIN 2026 Table 5).
- **C08 / C09** — 5 AUs (ear, orbital, muzzle, whiskers, head), each 0/1/2, sum 0–10.
- **C50** — descriptors **0=absent; 1=moderate OR uncertain; 2=obvious** (exact wording).
- **C57** — muzzle/whiskers lowest reliability, **ICC 0.55–0.67** (Evangelista Table 2).
- **C27** — landmark NME **9–26% by morphology** (Martvel 2024, fvets.2024.1442634, PMC11663861).
- **C28** — automated landmarks cost **~7 accuracy points** (same paper, Table 6: 0.73→0.66).

**Library/API (all load-bearing ones confirmed):**
- **C24 / C87** — RF-DETR trains natively on MPS as of **v1.6.0** (release notes fix grid_sample/bicubic CPU fallback).
- **C25 / C60 / C62 / C64** — coral-pytorch `corn_loss` / `corn_label_from_logits`, Linear(feat, 2) per head, 10 logits, decode→0–10.
- **C36 / C71** — `StratifiedGroupKFold`, `cohen_kappa_score(weights='quadratic')`, torchmetrics mAP fields.
- **C48 / C49** — Claude strict structured outputs **strip** min/max; use **enum [0,1,2]**; `messages.parse()` / `additionalProperties:false` (platform.claude.com structured-outputs docs).

**Dataset (measured directly from COCO v1 export + live API):**
- **C02** — boxes are centered head crops, **median area ~0.24–0.28** of frame.
- **C31** — preprocessing is **Stretch-to-640** on 4:3 sources (metadata `resize:{640,640,'Stretch to'}`), distorts AU geometry.
- **C32** — filename regexes match all 2040 names (594 CAT-prefixed, 1446 plain).
- **C40** — **valid = 53 pain / 356 no_pain images, unaugmented** (exact, COCO export).
- **C53** — **~2040 images** (metadata + manifest both exactly 2040).
- License = **CC BY 4.0** (project metadata).

**External datasets:** **C13** CatFLW CC BY-NC 4.0; **C72** 2079 faces / 48 CatFACS-relevant landmarks (Kaggle georgemartvel/catflw); **C17** horse-grimace 5 individuals, 3-of-5 AUs, CC BY 4.0 (HF oliveirabruno01/openfarm-horse-grimace-region); **no OPEN graded per-AU cat-FGS dataset exists** (HF zero hits; Evangelista data "on reasonable request").

---

## 3. Corrections needed (load-bearing first)

| Claim | Verdict | Evidence (source) | What the plan SHOULD say |
|---|---|---|---|
| **C73** — "Feighelstein 2023 (s41598-023-49031-2) achieved 95.5%" | **REFUTED** | DOI s41598-023-49031-2 is **Steagall et al. 2023** ("Fully automated deep learning models with smartphone applicability…"), Table 2 = 95.51%. Feighelstein 2023 is **s41598-023-35846-6** (77% landmark / 65% DL). | **Steagall et al. 2023** validated landmark(37)→geometric→XGBoost at **95.5%** (MSE 0.0096). Feighelstein 2023 is a *separate* 77%-vs-65% binary paper. Fix attribution everywhere (§5/§7). |
| **C74** — Mode for ear/orbital/muzzle/**head**, Min for whiskers | **PARTIALLY-TRUE** | **Steagall** 2023 p.4 (not Feighelstein): Mode best "except for whiskers change **AND head position** for which Minimum performed better"; Table 4 Head Min 0.1465 < Mode 0.1674. | Per **Steagall 2023**: Mode for ear/orbital/muzzle; **Min for BOTH whiskers AND head**. Move head to Min; reattribute. |
| **C12** — COSMIN review named calibration/CIs/reliability & "explicitly excluded automated scorers" | **REFUTED as written** | Lee & Steagall 2026 (JVIM 40(1) aalaf062): never uses "calibration"/"confidence intervals"; exclusions are chronic-pain / non-ordinal / non-English; **AI/automated never mentioned** (out of scope by topic, not stated exclusion). Names measurement-error/reliability/validation/interpretability/ROC-threshold as gaps. | Reframe: the review covers **only human-rater instruments** and flags measurement-error/reliability/validation as underreported; **calibration, CIs, and automated-scorer reliability fall outside its scope** — *that* is the white space we take. Drop "named calibration/CIs" and "explicitly excluded." |
| **C70 / C01** — leakage 201/337 clips | **REFUTED (number)** | Full 2040-image manifest, clip-grouped: **191 of 336 clips (56.8%)** in both train+valid. 201/337 is a 76%-sample extrapolation (C80). | Leakage is **191/336 clips (56.8%)**, computed on full data. Phenomenon is real and load-bearing; the number was wrong. Update Intro, §2, §4, §7. |
| **C39** — train split has 467 pain boxes | **REFUTED** | COCO v1 train export: **414 pain / 2880 no_pain boxes**; source (pre-aug) train pain ≈ 202. 467 unsupported. | Augmented train = **414 pain boxes** (~202 source + augmentation). |
| **C81** — train split = 1819 no_pain boxes | **REFUTED** | 1819 is **whole-project** metadata `classes.no_pain`; train source ≈1454, augmented ≈2880; valid ≈357. | State **1819 as dataset-wide**, not train-only. pos_weight basis (C41 sqrt(1819/264)=2.62) is fine but must be labeled metadata-based. |
| **C78** — bnb 4-bit QLoRA is CPU-only on macOS, the only hard CUDA blocker | **OUTDATED** | Official bitsandbytes support matrix (2026): macOS arm64 QLoRA(4-bit) **✅ CPU-supported**, Metal(MPS) **🐢 slow-supported**. Not strictly CPU-only, not the unique blocker. | Reword: "bnb 4-bit on Mac is **CPU/alpha-Metal and impractically slow**; use **mlx-vlm** for speed (not feasibility)." Colab-T4 half (bnb-4bit needs CUDA bnb) still stands. |
| **C06** — graded FGS automated "exactly once" | **PARTIALLY-TRUE** | Steagall 2023 is sole paper producing per-AU 0/1/2→0–10. But **"Feline SentiNet" 2023** (IEEE ICSES, CNN+RandomForest) does 5-category pain grading at 90% — automated *graded*, not FGS-structured. | Scope to: **"automated exactly once for the FGS per-AU 0/1/2→0–10 scoring structure (Steagall 2023)."** Cite Feline SentiNet as adjacent multi-class grading (ImageNet-CNN, so doesn't touch C10/C11). |
| **C47** — "Wu et al. 2025," 13% prevalence → 53% sensitivity | **PARTIALLY-TRUE** | arXiv 2506.07273 first author **Chavoshi** (not Wu); worked example is **10%** prevalence → ~53% sens at 95% spec. | Cite **Chavoshi et al. 2025 (arXiv 2506.07273)**; example is 10% prevalence; 13% is our dataset figure applied by extension. |
| **C65** — two-source rule = grimace + clinical reason | **CONFIRMED w/ precision** | Feighelstein 2023 p.4: requires **CMPS-feline ≥5 AND charted clinical reason**. | Pain label = **high behavioral pain-scale score (CMPS ≥5) + charted clinical reason** (not "grimace + reason"). |
| **C34 / C37 / C38 / C41** — prevalence 12.7%, 6.9:1, "264 unique pain faces" | **PARTIALLY-TRUE / source-ambiguous** | Metadata classes: 264 pain / 1819 no_pain → 12.7%, 6.89:1. Live annotation sum: **246 / 1811 → 12.0%, 7.36:1**. | Keep prevalence ~12–13% and imbalance ~6.9–7.4:1 as approximate; **pin every exact derived figure to the metadata basis (264/1819)** and stop calling 264 "unique pain **faces**" — it is a **box count** (distinct individuals are far fewer). |
| **C16** — single confounded "Flickr / CAT_01" source | **PARTIALLY-TRUE** | Manifest: **CAT_01 is the only CAT_NN id**, 84 clips (range 100–184). "Flickr" provenance not in API. | Single-individual confound confirmed (CAT_01, 84 clips); drop unverifiable "Flickr" attribution. |
| **C26** — DINOv2 "/14 patch grid" | **PARTIALLY-TRUE** | patch14 = 14-px patches; token grid = input/14 (37×37 @518). Operational rule (crops divisible by 14) is correct. | Say "patch size 14; crops must be **divisible by 14**." |
| **C61** — "CORN (Cao et al., 2111.08851)" | **PARTIALLY-TRUE** | Authors are **Shi, Cao & Raschka** (lead = Shi). Technical claim correct. | Cite **Shi, Cao & Raschka (2021/2023)**. |
| **C46** — cat is COCO class → "strong transfer" | **PARTIALLY-TRUE** | "cat" = COCO id 15 (confirmed). "Strong transfer" has no benchmark. | Present transfer as **expected**, not established. |
| **C14** — Steagall needs 8 "real-time" raters | **PARTIALLY-TRUE** | 8 raters (6F/2M) scored **still images**, not real-time. | Drop "real-time." |

---

## 4. Asserted-but-unverified — what we must actually MEASURE

These are stated as fact in the plan but were **never computed/sourced**:

1. **The 201/337 leakage number (C01/C70/C80) — the root cause.** C80 self-admits all clip-level stats (337 clips / 146 pain / 142 mixed / 201 leaked) were **extrapolated from a 76% sample (1562/2040)**. **MEASURE on full 2040 manifest** (already done by verifier): **336 clips, 138 pain-bearing, 135 mixed, 191 leaked.** Recompute C01, C33, C70 from full data and replace every propagated figure.
2. **Distinct-cat / "unique pain faces" counts (C22/C38).** C22 confirmed *directionally* (CAT_01 alone = 84 clips → upper bound ~253 individuals), but the **actual distinct-individual count is NOT computable without re-ID**. **MEASURE:** treat "264" strictly as a pain-box count; do not assert any "unique faces" number until re-ID is run. C20 warns CLIP/pHash are duplicate detectors, not re-ID — so they cannot substantiate it.
3. **C54 (VLM-vs-FGS, Sci Rep 2025 s41598-025-27404-z) — load-bearing, underpins choosing Claude.** Paper **not retrievable**; "VLMs underestimate FGS / only Claude acceptable bias" **unconfirmed**. **MUST fetch and confirm** before using as justification for Claude as weak-labeler.
4. **C86 (DINOv2 frozen > landmark-XGBoost in few-hundred-label regime) — load-bearing engine bet.** **Extrapolation, not a result** in arXiv 2304.07193 (no cat-FGS benchmark there). **MEASURE empirically in Gate 4/5**; present as hypothesis.
5. **C23 (768-dim CLIP vector free from Roboflow search API).** Verifier **could not retrieve** an embedding field. **MEASURE:** confirm the exact endpoint/field before relying on it for Gate 2 (else local DINOv2/CLIP extraction needed).
6. **C18 / C58 / C68 / C19 / C69 — kappa floors, sample-size floor, Monte-Carlo GO/NO-GO.** All **project pre-registration**, not published. Defensible (kappaSize-grounded, Evangelista-ICC-anchored) but label as **design decisions**; back C58 with an actual kappaSize/power calculation.
7. **C85 (≥0.90 Phase-A pain-recall gate).** Self-imposed target — **justify explicitly against C04's 90.7% reference sensitivity**, not as derived from prior automated work.

---

## 5. Novelty verdict

**SURVIVES IN SUBSTANCE.** The adversarial search (Semantic Scholar, web, arXiv, GitHub) found **zero counter-examples** to the three pillars:
- **C10** — no prior cat-pain work uses a **foundation model/VLM** (all are binary / ImageNet-CNN / landmark-geometry: Steagall, Feighelstein, Martvel, Feline SentiNet, Feline Feelings). **Confirmed.**
- **C11** — **VLM weak-labeling of the 5 FGS AUs has never been attempted.** Only adjacent work (Sci Rep 2025) *benchmarks* VLMs as direct raters, not as weak-labelers. **Confirmed.**
- **C07** — **no automated-FGS work has shipped calibration, CIs, open artifacts, or measured training-label reliability.** Corroborating zero-hit evidence. **Confirmed.**

**Genuinely unowned:** foundation-model/VLM engine + VLM AU weak-labeling + a calibration/CI/abstention validity wrapper on open artifacts.

**Must soften (two precise rescopings, or the framing is attackable):**
1. **C06** → "automated exactly once **for the FGS per-AU 0/1/2→0–10 structure** (Steagall 2023)," explicitly citing **Feline SentiNet 2023** (5-category, 90%, CNN+RF) as adjacent graded-but-not-FGS work.
2. **C12** → drop "named calibration/CIs" and "explicitly excluded automated scorers"; reframe as the review covering only human-rater instruments and leaving calibration/CI/automated-scorer reliability out of scope.

With those two edits the novelty stack is airtight.

---

## 6. What the plan SHOULD be — specific edits

**BUILD_PLAN.md:**
- **Intro / §2 / §4 / §7:** change leakage **201/337 → 191/336 (56.8%)**; change **337 clips → 336**, **146 → 138** pain-bearing, **142 → 135** mixed. Add a one-line note: "all clip-level figures recomputed on the full 2040-image manifest (the prior 76%-sample extrapolation in C80 is retired)."
- **§2:** train pain boxes **467 → 414** (augmented; ~202 source). Relabel **1819 no_pain as dataset-wide**, not train-split. Tag prevalence **12.7% / imbalance 6.9:1 / pos_weight 2.62** as **metadata-based (264/1819)**; note live annotation sum gives 12.0% / 7.4:1.
- **§5 / §7:** reassign **95.5%** and the **Mode/Min aggregation** guidance to **Steagall 2023 (s41598-023-49031-2)**; keep Feighelstein 2023 (s41598-023-35846-6) as the separate 77/65% binary paper; **move head position to the Min aggregation group.**
- **§0.5:** rescope **"graded FGS automated exactly once"** to the FGS scoring structure + cite Feline SentiNet; rewrite the **COSMIN (C12)** framing.
- **§1 / §7 (compute split):** reword the **bitsandbytes** line — bnb-4bit on Mac is CPU/alpha-Metal and slow (not impossible); mlx-vlm chosen for **speed**.
- **§3 / §3.4:** fix **C47** citation to **Chavoshi et al. 2025**, 10% example prevalence.
- **§3.3:** correct the **two-source rule (C65)** to CMPS ≥5 + clinical reason; fix **CORN attribution to Shi, Cao & Raschka (C61)**; phrase **DINOv2 patch (C26)** as "divisible by 14"; mark **C86** as a hypothesis to validate in Gate 4/5.
- **§3.1:** mark **C54** as **unverified** until the Sci Rep 2025 paper (s41598-025-27404-z) is fetched — flag it because it justifies Claude as labeler.
- **Stop asserting any "unique pain faces / distinct individuals" number** (C38) until re-ID is run; describe 264 as a pain-box count and CAT_01 (84 clips) as the lone known individual.
- **Gate 2:** add a verification TODO that the **768-dim CLIP field (C23)** actually returns from the search API before depending on it.

**Other docs:**
- **PAPER_DEBATE.md** — propagate the Steagall-vs-Feighelstein reattribution (95.5% is Steagall's).
- **DATA_DECISION.md** — note the metadata-vs-live annotation discrepancy (264/1819 vs 246/1811) and the 493 exact-duplicate filenames (2040 records / 1547 distinct) the plan currently omits.
- **MEMORY / dataset-facts.md** — replace the 76%-sample clip numbers with the full-manifest figures (336/138/135/191).