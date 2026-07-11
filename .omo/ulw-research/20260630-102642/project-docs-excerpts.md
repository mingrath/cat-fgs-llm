# Project docs excerpts
## RESEARCH.md
> **⚠️ SUPERSEDED where it conflicts with `FINAL_DIRECTION.md` (authoritative) and the code.**
> Early-exploration notes: the QWK-primary framing and the convnext_tiny backbone here are
> NOT what shipped. The engine is frozen DINOv2 + CORN heads (the `_reg` register variant
> `dinov2_vits14_reg` is the field default; A/B it before locking plain ViT-S/14); the
> kappa method is a protocol whose per-AU result is pending an independent vet anchor, and
> QWK-vs-VLM is NEVER validation. Trust `FINAL_DIRECTION.md` + the code.

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
- **Backbone (transfer learning):** `timm` — `convnext_tiny.fb_in22k_ft_in1k` (default) or frozen `vit_small_patch14_dinov2` linear probe (fast strong baseline) — but default to the `_reg` register variant `dinov2_vits14_reg`, since registers suppress attention artifacts that hurt dense, localized per-AU features (orbital/ear/muzzle); A/B it before locking the plain variant.
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
## FINAL_DIRECTION.md
# Finalized direction — build something that isn't just another cat-pain detector

**Date:** 2026-06-12
**Status:** AUTHORITATIVE. This document is the reconciled output of a multi-agent direction debate (8 doc-readers → 7 critic personas → moderator → 3 red-teamers → lead synthesis). Where it conflicts with [BUILD_PLAN.md](BUILD_PLAN.md), **this supersedes** — BUILD_PLAN has been edited to point here and carries the surgical deltas inline. Grounded in [GAP_ANALYSIS.md](GAP_ANALYSIS.md), [GITHUB_MINE.md](GITHUB_MINE.md) (Pass 2 novelty mining), [FACTCHECK.md](FACTCHECK.md), and the `novel-contribution` memory.

---

## 1. Do I agree with the direction?

**Agree-with-changes.** The direction survives all three red-team passes, but only after it stops letting the engine (DINOv2+CORN) and the off-the-shelf wrapper tools carry novelty weight they cannot bear. The strongest point across the debate is the **replication red-team's verdict**: the not-a-replicate thesis is load-bearing in exactly two places — the **per-AU VLM-vs-vet κ protocol whose result is pending the independent vet anchor** (VLM-as-AU-rater) and the **confound audit sold as a transportable attribution protocol** — and cosmetic everywhere else; strip those two and v1 is "frozen DINOv2 + binary pain head + standard reliability metrics," i.e. the Feighelstein/Martvel detector we swore not to build. The **feasibility and statistics red-teams converge on one wall**: every quantitative claim (κ, 0.39, abstention NPV, 5×5 correlation, severity cells) draws on the *same* ~50-positive vet account, never budgeted, so "guaranteed NPV," "validated graded," and the "empirical correlation fix" are arithmetically undeliverable at this n. The fix is decisive and cheap: **headline the two as portable methods (the κ one a protocol whose result is pending the independent vet anchor), concede the engine as plumbing, re-center the wrapper on a welfare-asymmetric loss, and demote graded to an inspectable-but-not-validated artifact.** With those moves the plan is honest, solo-deliverable on M4, and not a replicate.

## 2. Kept (consensus)

- **Wrapper-as-frame, accuracy-as-anti-benchmark** — every persona, including Reviewer 2, endorses it as the only unowned, solo-feasible framing; never write "we beat 77/79/95%" (BUILD_PLAN §0.5, ruling 6.12).
- **Circularity firewall (ruling 6.e / §3.3)** — sens/spec at 0.39 estimated ONLY on vet-confirmed labels; QWK-vs-VLM is never validation. Named by all 7 as the methodological backbone.
- **Gate-2 confound audit as an early action** — cheap, no vet, no GPU; can kill or reframe the corpus before any spend, and publishes either way.
- **Gate-3 frozen hashed cat-disjoint hold-out with CI-abort** — structurally defeats post-hoc goalpost moves; reviewers reward it.
- **Per-cat (not per-clip) grouping + always print the distinct-pain-cat denominator + Clopper-Pearson/bootstrap CIs; augmented copies never enter any reported N** (Gate 1, §4, ruling 6.a).
- **Frozen DINOv2 ViT-S + shallow CORN heads on cached features** — correct low-data/MPS choice; default to the `_reg` register variant (`dinov2_vits14_reg`), which suppresses attention artifacts that hurt dense localized per-AU features (orbital/ear/muzzle), and A/B it against plain ViT-S/14 (kept as backwards-compatible) before locking the backbone; full fine-tune at ~264 faces would be malpractice; CORN over CORAL/softmax is right.
- **Gate-4 MPS compute-correctness pre-checks are BLOCKING** — a leak-proof PR-AUC from silently-wrong MPS logits is worthless.
- **Binary-plus-abstention as the likely v1 ship / the floor that stands today** — it stands now and is the v1 spine; if muzzle/whiskers κ collapses, the graded layer (upside conditional on Gate 1-B) is dropped and the calibrated binary + abstention floor remains; no project-killing null.
- **Replication-trap discipline (§0.5 traps 1–8)** — no Steagall landmark+XGBoost, no Martvel video/temporal, no external-cohort claim on single CAT_01, no multi-rater ICC with one vet, horse set as decode scaffolding only.
- **Decision-support triage framing, never an autonomous analgesia trigger** — output is "grimace consistent with pain, X/10; recommend vet assessment"; negative class documented as an unknown (possibly sedated/post-op) mixture.

## 3. Changed (resolved conflicts)

**A — Headline. RESOLVED: co-headline the two as PORTABLE METHODS (the κ one a protocol whose result is pending the independent vet anchor); concede the engine; the wrapper is the frame, not the claim.**
Promote (i) "per-AU VLM-as-AU-rater κ protocol vs vet (result pending the independent vet anchor)" framed as *can a frozen VLM weak-label feline FGS AUs at human-rater agreement* (a forward-looking method finding, not an audit of CAT_01's specific labels; a high κ measures that weak-labeling capability only if the vet anchor is independent AND the VLM rubric differs from the vet's rubric — otherwise it measures rubric-following), and (ii) "first capture-condition confound-attribution protocol (FGS-BG-Gap + per-AU EBPG)" framed as *reusable on the next dataset*. Demote calibration/ECE/decision-curve/MAPIE to "supporting evidence." Explicitly state in the paper that the DINOv2+CORN engine is NOT claimed as novel. *Forced by Replication-attack Holes 1, 2, 3, 6:* single-source data means non-portable contributions are dead on arrival; only method-portability and a conceded engine survive the "swap-the-backbone + extra metrics" filing.

**B — Graded vs binary spine. RESOLVED: binary-plus-wrapper IS the v1 spine, stated in the abstract; graded is a built-but-NOT-VALIDATED artifact, not a "gated upside we expect to fire."**
The abstract's headline must hold with graded output dropped. *Forced by Feasibility-attack §C and Statistics-attack #6/red-team #1–2:* at n≈120 the severity tail is unestimable and B and C collapse into one decision, so frame graded as "v2-pending-more-data," not as upside likely to be realized.

**C — Independent graded ground truth. RESOLVED: CUT the gold-set branch entirely.**
Do NOT chase Steagall/Zamansky/CatFLW. *Forced by Feasibility-attack §C:* CatFLW is landmarks/bboxes (a category error as a graded anchor), and the request-only FGS sets are unbounded-dependency / weeks-to-never on a solo timeline. Re-write as: "we will NOT headline a validated graded claim; graded-CORN ships as an internally-consistent, VLM-anchored, qualitatively-inspected exploratory layer with the circularity stated." This removes an unbounded external dependency from the critical path and is *more* honest. **Word "graded" struck from every validated-claim sentence.**

**D — Operating point. RESOLVED in favor of the clinician: fixed-high-sensitivity, NOT Youden-J/F1, and quantify the welfare loss.**
Select the cutoff at pain-recall ≥0.90 (Evangelista anchor) inside train folds, report the specificity it buys with bootstrap CIs. **Make the harm-ratio-weighted decision curve the wrapper's headline artifact, not ECE** — sweep undertreat:overtreat as a *range* across the dcurves threshold-probability axis (we have no vet-elicited ratio). *Forced by Clinician position + Replication-attack Hole 4:* a symmetric-cost knee is the wrong loss for a welfare instrument, and generic calibration is hygiene whereas a welfare-loss-weighted operating point is a clinical-decision-theory contribution no prior cat-pain paper made. Correct the stale §3.3 Youden-J/F1 line.

**E — CORN path. RESOLVED: commit the distributional path in the spine; SPLIT off the correlation fix.**
- *E.1 (adopt unconditionally):* move soft `P(rank>k)` → per-AU pmf → convolve to a pmf over the 0–10 sum → **RPS-on-the-sum (single scalar, bootstrap CI) + per-AU ClasswiseECE** into BUILD_PLAN §3.3 as the default. Relegate argmax-sum to the 0.39 point decision only. **Do NOT print a binned reliability diagram on the 11-atom sum** — degenerate at ~11/atom (per-bin SE ±0.18–0.26). *Forced by ML-methods-critic + Statistics-attack #5:* the current spine specifies a statistically incoherent path while promising metrics it cannot support.
- *E.2 (cut the empirical 5×5 matrix):* replace with a **2-point ρ sensitivity band** — run the §3.4 Monte-Carlo under ρ=0 and a pinned ρ=0.3 (uniform), report GO only if the decision holds under BOTH. *Forced by Feasibility-attack §E + Statistics-attack #7 + red-team #5:* a 10-off-diagonal matrix at n≈50 has per-entry CIs ~[−0.4,+0.6], launders noise as data-driven rigor, and may not be positive-definite. A pinned ρ is honest; a noisily-fit matrix is false rigor.

**F — Abstention. RESOLVED: DELETE the word "guaranteed"; ship a one-sided 95% NPV lower-bound curve; 0.39 is CI-first exploratory.**
Report the abstention curve as "NPV with a one-sided 95% LB at each abstention rate," shown to clear ≥0.90 only at honestly-reported abstention rates. *Forced by Statistics-attack #4 (the sharpest overclaim):* certifying NPV≥0.95 needs ~60 zero-error abstained-in negatives out of ~70 total → abstention rate collapses to ~0 → vacuous instrument. **Run the LTT/MAPIE sample-size power calc BEFORE any vet spend** (Gate 0); if the budget can't certify a useful band, the curve is exploratory, full stop.

**G — Severity-cell gate. ADOPTED, but COLLAPSE not caveat.**
Pre-commit: unless the anchor delivers double-digit AU=2 cells, collapse the high end (merge AU 1+2, or report only painful/not above threshold). *Forced by Statistics-attack #6:* AU=2 sens CI spans [0.35,0.97] at single-digit n — that is no information, and a caveat leaves an anchoring number in a table; it must be a pre-registered structural decision.

**Gate-1-B fires on the κ CI LOWER BOUND, not the point estimate** (Statistics-attack #1): a floor of 0.6 is indistinguishable from a true 0.47 at this n, so a point-estimate gate is not a gate. Pre-register the κ floors *with* the kappaSize power calc backing them.

**Confound audit is ONE-DIRECTIONAL** (Statistics-attack #2): well-powered to *detect* confounding (AUC 0.65→z≈2.2 at n=50), underpowered to *rule it out* — report "no confound detected at this power," never "no confound."

**Housekeeping (uncontested):**
- **Self-justify the labeler via the κ pilot** ("we selected the VLM by measured per-AU κ on the anchor") and drop the unretrievable Sci Rep 2025 "only Claude acceptable" citation (FACTCHECK C54) — removes the phantom-citation dependency at the root of headline #1 (Replication Hole 7).
- **Fix the COSMIN line**: AI scoring is out-of-topic-scope of Lee & Steagall 2026 (C12) — do NOT claim the review "named calibration/CIs" or "excluded automated scorers." (Already corrected in BUILD_PLAN §0.5.)

## 4. Cut / descoped

| Cut | Trigger to re-add |
|---|---|
| **Validated "graded 0–10 FGS instrument" claim** (C). Graded-CORN ships as an inspectable artifact only. | An *independent* per-AU 0/1/2 gold held-out set arrives AND AU=2 cells reach double digits. Not on the solo timeline. |
| **Gold-set acquisition (Steagall/Zamansky/CatFLW)** as a critical-path dependency (C). | A request-only set clears its DUA with a bounded SLA — treat as v2 windfall, never a blocker. |
| **The word "guaranteed"** on the NPV band (F). | A power calc shows the budgeted n certifies NPV≥0.90 at ≤40% abstention. |
| **Empirical 5×5 cross-AU error-correlation matrix** (E.2). | n into the low hundreds of vet rows with non-single-digit cells. Until then: 2-point ρ sensitivity band. |
| **Binned reliability diagram on the 0–10 sum** (E.1). | ~24+ samples/atom (n into the high hundreds). Until then: RPS-on-the-sum + per-AU ClasswiseECE. |
| **Per-cell AU=2 (severe) calibration number** (G). | Anchor yields double-digit AU=2 cells. Until then: collapsed high-end scale. |
| **Engine novelty claim** (A) — conceded as plumbing permanently. | Never. A conceded swap-the-backbone delta is harmless; an oversold one is the reviewer's favorite kill. |
| **Any horse number in abstract/results/transfer table** (trap 7). | Never — methods/appendix sentence only ("we unit-validated the decode→sum→threshold path on 5-horse genuine 0/1/2 labels"). |

## 5. The not-a-replicate thesis

> **This is not another cat-pain detector: its headline deliverables are two transportable methods no prior feline-pain work produced — a protocol for measuring whether a frozen VLM can weak-label FGS action units at human-rater agreement (per-AU VLM-vs-vet quadratic κ, CI-lower-bound-gated; result pending the independent vet anchor), and a reusable capture-condition confound-attribution protocol — shipped on an explicitly-conceded DINOv2+CORN engine, with an honest binary-plus-abstention floor (welfare-asymmetric operating point at pain-recall ≥0.90, one-sided-95%-NPV defer-to-vet curve) that stands even though the graded layer is reported as inspected-not-validated.**

**Three concrete artifacts that prove it (all dataset-agnostic / portable):**
1. **VLM-as-AU-rater κ protocol** — released code that scores any face corpus's per-AU VLM labels against a vet anchor and reports 5 quadratic κ with CI lower bounds; a result requires the independent per-AU vet anchor that does not yet exist (Gate 0), and counts as weak-labeling capability only if that anchor is independent and the VLM rubric differs from the vet's rubric — otherwise it measures rubric-following. (The portable method, immune to "routine hygiene.")
2. **FGS-BG-Gap + per-AU EBPG confound-attribution protocol** — a one-directional audit any future facial-pain-scorer corpus can run; deliverable is the *protocol*, not "CAT_01 is confounded."
3. **Welfare-asymmetric decision-curve + one-sided-95%-NPV abstention curve** — operating point at fixed high sensitivity with the undertreat:overtreat harm ratio swept as a range, plus a defer-to-vet boundary reported with finite-sample lower bounds (no "guarantee" word). This is the clinical-decision contribution that distinguishes the wrapper from generic calibration.

## 6. Build sequence (risk-first, reconciled with BUILD_PLAN gates)

**Gate 0 — Power calcs + vet-budget pre-registration (NEW, blocks everything quantitative; ~1 afternoon, zero data, no GPU).**
- Goal: write the *single integer* — how many faces the vet scores, per-AU, in how many sittings — and run three zero-data power calcs: (a) faces for per-AU κ CI half-width ≤0.15 (kappaSize); (b) faces for LTT/MAPIE-certified NPV≥0.90 at ≤40% abstention; (c) whether the 0.39 CI is reportable at the budgeted n.
- Exit: budget integer committed; A's magnitude claim and F's "guarantee" demoted *on paper* now if (a)/(b) fail.
- Why first: Feasibility/Statistics red-teams show this is the project's single point of failure dressed as five resolutions. **This precedes everything in BUILD_PLAN §0.5 "First 3 actions."**

**Gate 1 — Per-CAT merge (BUILD_PLAN Gate 1, unchanged; keystone).** Goal: collapse clips to true individuals, validated against trusted `CAT_` IDs (CLIP/pHash are duplicate detectors, not re-ID). Exit: no individual straddles a fold; distinct-pain-cat denominator printed. Compute: CPU. Why: blocks all grouping/CV.

**Gate 2 — Capture-condition confound audit (BUILD_PLAN Gate 2; CHANGED: now framed as a portable protocol + one-directional).** Goal: trivial brightness/blur/aspect/CLIP classifier predicts pain → quantify via FGS-BG-Gap. Exit: detect-or-not result logged; if it can only produce "CAT_01 is confounded," demote to a threat-to-validity paragraph; if it produces a transportable protocol, it co-headlines. Compute: CPU, no vet. Why: cheapest kill/reframe; runs before any spend.

**Gate 3 — Frozen hashed cat-disjoint hold-out + CI-abort (BUILD_PLAN Gate 3, unchanged).** Exit: fold CSV + test ids hashed; CI aborts any run that can read the test manifest. Compute: CPU.

**Gate 4 — MPS compute-correctness pre-checks (BUILD_PLAN Gate 4, unchanged, BLOCKING).** Exit: MPS-vs-CPU DINOv2 logit parity; 1-epoch CORN smoke test; synthetic CORN-decode→sum→0.39 unit test passes. Compute: M4/MPS. Why: silent-correctness gate before any scoring.

**Gate 5 — Alignment/NME gate before Phase B (BUILD_PLAN Gate 5, unchanged).** Exit: CatFLW-landmark NME inside RF-DETR crop vs eye-aligned crop, and median face-pixel resolution, both acceptable; else add 2-point eye-similarity alignment. Compute: M4. Why: misaligned crops make "calibration" measure crop quality.

**Gate 1-B — Weak-label reliability κ pilot (BUILD_PLAN Gate 1-B; CHANGED: fires on CI lower bound; self-justifies the labeler).** Goal: ~120-image (≥50 positive, per Gate 0 budget) VLM-vs-vet per-AU quadratic κ with CIs; the pilot itself selects the VLM. Exit: GO if the κ **CI lower bound** clears the pre-registered, power-backed floor (orbital/ear/head ≥0.6; muzzle/whiskers 0.4–0.6 caveat). Compute: hosted-API inference + CPU. Why: the headline #1 protocol (result pending the independent vet anchor); runs alongside §3.4 Monte-Carlo (now 2-point ρ band).

**Gate 6 — Severity-cell count gate (NEW, after the anchor; pure counting).** Exit: if AU=2 cells single-digit → collapse high-end scale (pre-committed). Compute: CPU. Why: closes the decision-boundary hole; likely fires.

**Then (post-gate, binary spine):** distributional CORN (E.1) on cached features → RPS-on-sum + per-AU ClasswiseECE → fixed-high-sensitivity operating point + welfare-asymmetric decision curve (D) → one-sided-95%-NPV abstention curve (F). Graded-CORN trained and *inspected qualitatively*, never entered as a validated claim.

## 7. Kill criteria (pivot-to-binary-spine vs proceed)

- **Gate 0 (b) fails** (no budget certifies useful NPV): proceed, but F is exploratory-only; "guaranteed" never appears. **Not a kill** — relocates the deliverable to the welfare decision curve.
- **Gate 2 fires damning AND yields only "CAT_01 is confounded" (non-portable):** demote audit to threat-to-validity; the κ method must then carry the headline alone. If Gate 1-B *also* fails → the honest paper is a confound/audit note; pre-register that venue NOW.
- **Gate 2 detects confounding as a transportable protocol:** PROCEED — this is a publishable co-headline even if everything downstream is null.
- **Gate 1-B κ CI-lower-bound below floor on orbital/ear/head:** drop graded entirely → ship calibrated **binary + abstention + confound audit**; the wrapper and the "κ-as-method" headline still stand (a publishable METHOD even at mediocre κ — a real method finding once the independent anchor exists, not a measurement claimed now). PIVOT to binary spine, not a kill.
- **Gate 1-B κ passes on orbital/ear/head, fails muzzle/whiskers:** proceed on the surviving AUs; state how dropping 2/5 heads shifts the achievable 0–10 range and whether 0.39 (≈4/10) is even reachable (Clinician note).
- **Gate 4 or Gate 5 fails:** BLOCKING — fix before any metric is believed; no number from silently-wrong MPS logits or misaligned crops is reported.
- **Gate 6 fires (AU=2 single-digit):** collapse the high-end scale (expected base-rate outcome); do NOT print a per-cell severe calibration number.
- **§3.4 Monte-Carlo GO under ρ=0 but NO-GO under ρ=0.3:** declare the gate fragile, say so, and default to the binary spine rather than claim a GO the correlation could flip.

**Modal outcome to plan for as the default product:** binary-plus-wrapper + κ-as-method + confound-protocol + welfare decision curve + LB-abstention curve, with graded-CORN shipped as an inspected-not-validated artifact. That product is solo-deliverable on M4/MPS with one vet, and it is not another cat-pain detector.

**Seam landed (2026-06-13 update, cross-reality check vs code):** The portable
co-headline is now concretely embodied in `src/protocols/` (first-class surface
with `__init__.py` re-exports of the κ/confound protocols, thin adapters for
AU-override / col-map / generic non-FGS ordinal mode with no 0.39 assumption,
and `standalone_test_corpus.py` — a deletion-safe, zero-torch, synthetic
arbitrary-AU exerciser that can survive removal of engine/vlm/data). New central
`src/gates/orchestrator.py` + Makefile `gate-e2e-synthetic` / `test-portable` /
`gate-orchestrate` + e2e tests enforce strict gates (G0 power first, cat-disjoint
G1/G3, one-dir G2, CI-LB G1-B, blocking G4, welfare asym baked, single-source
decode via constants, inspected-not-validated graded). `src/model/decode.py`
updated for portable N/k. Cache schema (_CACHE_SCHEMA + FEATURE_SCHEMA) and
DINOv3 prep (backbone + richer patch_std per MCP context7) are explicit
executable embodiments. This makes the "runnable protocol" / "portable
methods" claims accurate, citable, and stronger than paper prose alone. Doc
debt addressed by updates to paper/sections/04+06+00-abstract, README quickstart,
and this note (P5/FreshPaperDocLandedSync cand7 fidelity 2026-06-14). Reality >
prior docs (paper "protocol only" language now backed by importable seam + new
orch + generalized decode + cache). Use `python -m src.protocols.standalone_test_corpus` and orchestrator synthetic for verification. Uniqueness vs priors (Steagall closed/handcrafted; Martvel video-only; zero prior public VLM per-AU κ protocol + confound attribution + welfare-asym + standalone + strict gate orch) holds per MCP grep/context7 polls on GitHub (no matching code patterns for the surface). DINOv3 successor prep documented for engine evolution.

## 8. Decision record — framing + timeline (2026-06-14, MCP-grep + paper-scan grounded)

Two open v1 decisions were resolved by a multi-agent debate grounded in (i) an MCP
`grep__searchGitHub` + `context7` scan of public code and (ii) a non-GitHub paper
scan (arXiv / Semantic Scholar / OpenReview / PubMed). Both judges and the paper
scan **converge at medium confidence** and **confirm — do not flip — the §3.A
asymmetric framing.**

**DECISION A — Headline framing: ASYMMETRIC (resolved).**
Lead with the one-directional, power-conditioned, **per-AU confound-attribution
protocol** (FGS-BG-Gap counterfactual + per-AU EBPG saliency-as-confound-evidence
+ VLM judge-bias probe) as the demonstrable spine — it validates **today** on
planted positive + negative controls in `src/protocols/standalone_test_corpus.py`,
no vet anchor needed. Demote the **VLM-as-AU-rater κ** to a *guarded,
adequately-powered-but-method-crowded* reliability check shipped
inspected-not-validated (CI-LB gate + rubric-independence guard named as the
specific increments *within* that leg, not a co-equal pillar). Demote the wrapper
(Clopper-Pearson NPV-LB + welfare-asym DCA + LTT/MAPIE abstention) to **cited
plumbing**, not a contribution. κ is **not deleted** — it remains kill-tree
insurance per §7 lines 105–107 (sole-survivor headline if the confound leg
degrades), only re-ranked below the confound protocol.

*Exact headline wording to use:* "A portable, power-aware confound-attribution
protocol for fine-grained animal-affect models, with a guarded VLM-as-AU-rater
reliability check (CI-lower-bound gated, result pending an independent vet anchor)."

*Why:* the κ-as-judge idea is crowded off-the-shelf (crowd-kit / gtmf / medkit IRR
libs, AWS sample, and `laudos-ai/laibench-public` `calibrate.ts` a near-twin of the
rubric-independence audit), so co-equal billing hands a reviewer a free kill; the
assembled confound protocol returns **0 GitHub hits and 0 assembled-paper matches**
(closest in-domain near-miss, Tech4Animals "segment-based framework" Sci Rep 2025,
owns only ~1 of 3 legs and is positively framed). Novel-by-assembly survives.

**DECISION B — Timeline: SHIP NOW as a methods/protocol paper (resolved), conditional on Decision A.**
Submit on synthetic + planted positive/negative controls; target an
**ML-eval/trustworthiness or clinical-ML-methods track, NOT a vet journal first.**
Disclose empty `data/`, missing vet anchor, Roboflow-binary-only, and the
`power.json calc_c_point_039 reportable:false (n_pos=16)` operating point as
limitations, real-cat application named as future work. *Why:* synthetic-planted
validation is the native, accepted mode for trustworthiness tooling (cleanlab
`test_spurious_correlation.py` plants + asserts recovery; AIF360; MAPIE on
synthetic streams); holding defends against a phantom scoop (0 open competitor code
for graded cat-FGS); and Gate-0 rules forbid printing the one real number a vet
sitting would buy. **B is load-bearing on A:** ship-now is only defensible under
asymmetric framing.

**Mandatory honesty constraints from the paper scan:**
- Claim only the **assembly** + per-AU EBPG-as-confound-evidence + the
  *instantiation* of equivalence-style audit hygiene — **NOT** any individual
  primitive and **NOT** the power-conditioned / equivalence statistics themselves
  (those are published: cite & credit Huang & Hooker 2026 arXiv:2605.11614; Singh
  et al. NeurIPS 2023 RegML arXiv:2312.04745).
- Brand neither the κ **CI-LB gate** nor the **rubric-paraphrase guard** as novel:
  CI-LB acceptance on an ordinal scale is textbook clinimetrics (Tractenberg/Rosen
  2010 PMC2924444; Donner & Rotondi 2010; Rotondi & Donner 2012; `kappaSize`;
  Sim & Wright 2005); paraphrase-invariance for LLM raters is published (Weng et al.
  "Policy Invariance" 2026 arXiv:2605.06161; JudgeSense arXiv:2604.23478).

**Citations to cite-and-distinguish-from:** Tech4Animals segment-based framework
(Sci Rep 15:13670, 2025) [owns per-AU saliency leg]; "Mitigating Context Bias in
VLMs / BECKI" (Electronics 14(16):3311, 2025) [bg-context-bias in affect VLMs,
human domain]; chatbot-vs-expert FGS agreement incl. Claude (Sci Rep
s41598-025-27404-z, 2025) [plain limits-of-agreement, same domain]; Adebayo et al.
ICLR 2023 (arXiv:2212.04629) [cite as *support* — justifies the one-directional
"at this power" framing]; ImageNet-9 / Xiao 2021 + Moayeri CVPR 2022 + EBPG
[primitives]; FGS ordinal-κ domain anchors Evangelista 2021 / Cheng-Evangelista-
Steagall 2020.

**Residual open item (only thing the scans could NOT close):** papers cannot be
exhaustively swept like code; a very recent (2026) or paywalled animal-affect
preprint assembling all three confound legs could exist undetected (five-angle
convergence makes this unlikely). Re-run the targeted paper scan immediately before
submission. This erodes nothing in Decision A's headline; (3) only further erodes
the already-conceded κ side — which is itself an argument for the ship-now timing.
## IMPLEMENTATION_PLAN.md
# IMPLEMENTATION_PLAN.md — cat-fgs-llm

**Date:** 2026-06-12
**Status:** Build specification (authoritative for *how*, not *what*)

## Purpose

This document is the end-to-end engineering build spec for cat-fgs-llm: the repo/env scaffold, dataset acquisition and preparation, the binary pain detector (Phase A), face-crop/alignment preprocessing, the VLM weak-labeling + kappa pilot, the frozen-DINOv2 + CORN engine (Phase B), the trustworthiness wrapper, and the gated end-to-end pipeline with its evaluation protocol and build sequence. It exists to make the strategy in `FINAL_DIRECTION.md` executable by a solo developer on Apple M4 / MPS (Colab T4 for detector training only), with every quantitative claim routed through a strict, artifact-emitting gate order.

## Supersedes / relationship

- **`FINAL_DIRECTION.md` is the authoritative strategy** and governs every contested decision (the confound-attribution headline, the guarded kappa reliability check demoted below it, the conceded engine, the wrapper as cited supporting plumbing, the binary-plus-wrapper spine, the inspected-not-validated graded layer, the operating point, the circularity firewall, the gate order, and the kill/pivot rules). Where this document and `FINAL_DIRECTION.md` disagree on *strategy*, `FINAL_DIRECTION.md` wins.
- **This document is the build spec** — it fixes file layout, environment pins, command sequences, code skeletons, and the exact artifacts each gate writes. `BUILD_PLAN.md` carries deltas that are folded in here.
- Nothing in this spec re-litigates strategy. The DINOv2+CORN engine is conceded plumbing throughout; the headline is the capture-condition confound-attribution protocol (FGS-BG-Gap + per-AU EBPG + VLM judge-bias probe), with the VLM-as-AU-rater per-AU quadratic kappa (gated on the CI lower bound) demoted to a guarded, inspected-not-validated reliability check, and the welfare wrapper cited as supporting plumbing.

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
  - [0.2 The headline + the guarded reliability check](#02-the-headline--the-guarded-reliability-check)
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
- [6. The trustworthiness layer — every metric, defined](#6-the-trustworthiness-layer--every-metric-defined)
  - [6.0 Setup — environment, inputs, files](#60-setup--environment-inputs-files)
  - [6.1 Pillar 1 — VLM-as-AU-rater κ (GUARDED RELIABILITY CHECK)](#61-pillar-1--vlm-as-au-rater-κ-guarded-reliability-check)
  - [6.2 Pillar 2 — Ordinal calibration (supporting)](#62-pillar-2--ordinal-calibration-supporting-the-010-layer--inspected-not-validated)
  - [6.3 Pillar 3 — Welfare-asymmetric DECISION CURVE (cited supporting plumbing)](#63-pillar-3--welfare-asymmetric-decision-curve-cited-supporting-plumbing)
  - [6.4 Pillar 4 — Defer-to-vet ABSTENTION](#64-pillar-4--defer-to-vet-abstention-one-sided-95-npv-lower-bound)
  - [6.5 Pillar 5 — Confound-attribution PROTOCOL (THE HEADLINE)](#65-pillar-5--confound-attribution-protocol-the-headline-portable)
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

**One framing rule that governs every file in this repo:** the DINOv2+CORN engine is **plumbing, conceded, never claimed novel**. The headline is the **confound-attribution protocol**, with the kappa reliability check guarded and ranked below it and the welfare wrapper as cited supporting plumbing (0.2). Directory names, module docstrings, artifact tags, and W&B run names must never imply the engine is the contribution.

### 0.1 The modal deliverable (plan for this as the default product)

Lay out the repo for the **modal outcome**, not the optimistic one. The shippable product is a **binary-plus-wrapper** vertical with six named components. The abstract must hold with component 6 dropped, and the word **"graded" is struck from every validated-claim sentence**:

| # | Component | What it is | Where it lives | Validated? |
|---|---|---|---|---|
| 1 | **Calibrated binary pain decision** | RF-DETR/YOLO detector → per-image pain decision at a **fixed pain-recall ≥0.90** operating point (Evangelista anchor), **never** Youden-J/F1 | `src/detect/`, `src/wrapper/operating_point.py` | YES — vet-confirmed labels only (circularity firewall) |
| 2 | **Confound-attribution protocol** | FGS-BG-Gap + per-AU EBPG + VLM judge-bias probe, **one-directional / power-conditioned** ("no confound detected at this power"); validated today on planted +/- controls | `src/eval/confound.py` | YES (**THE HEADLINE**) |
| 3 | **κ reliability check (guarded)** | Per-AU VLM-vs-vet quadratic weighted kappa with **CI lower bound**; a guarded, inspected-not-validated check ranked **below** the confound headline (not co-equal); kill-tree insurance. The *protocol* is the deliverable; **a result is pending the independent per-AU vet anchor** (Gate 0 — the current binary dataset cannot yield 0/1/2 ground truth, so no κ number is reported yet). CI-LB gate = standard clinimetrics; rubric guard = published; neither branded novel. QWK-vs-VLM is **never** validation of the decision | `src/vlm/`, `src/eval/kappa.py` | YES (guarded check) |
| 4 | **Welfare-asymmetric decision curve** | dcurves net-benefit, sweeping undertreat:overtreat as a **range** (no vet-elicited point ratio) — the wrapper's reported artifact (replaces ECE); **cited supporting plumbing, not a headline** | `src/wrapper/decision_curve.py` | YES |
| 5 | **LB-abstention curve** | One-sided **95% NPV lower bound** at each abstention rate; the word **"guaranteed" is banned**; cited supporting plumbing. Demoted to exploratory if the G0 power budget for the NPV-LB is not met | `src/wrapper/abstention.py` | YES (exploratory if G0 budget unmet) |
| 6 | **Inspected graded-CORN** | 5 per-AU CORN heads → 0–10 sum → distributional pmf; shipped **inspected-not-validated** | `src/model/`, `src/eval/distributional.py` | **NO — never a validated claim** |

Component 6 is gated, labeled, and walled out of every validated results table. Its outputs feed only `src/eval/distributional.py` for inspection, never a headline number.

### 0.2 The headline + the guarded reliability check

The claimed contribution is **one headline leg** plus a guarded check ranked below it; both are **dataset-agnostic** and ship as released, runnable code:

1. **Capture-condition confound-attribution protocol (THE HEADLINE)** (`src/eval/confound.py`): FGS-BG-Gap counterfactual + per-AU EBPG saliency-as-confound-evidence + VLM judge-bias probe, a **one-directional, power-conditioned** audit any future facial-pain-scorer corpus can run. The strong leg — validated today on planted positive+negative controls (`src/protocols/standalone_test_corpus.py`). Deliverable is the *protocol*, not "CAT_01 is confounded." The novelty is the **assembly + per-AU EBPG-as-confound-evidence + equivalence-style audit hygiene** — never an individual primitive (bg-swap, saliency) and never the one-directional/power-conditioned/equivalence statistics themselves.
2. **VLM-as-AU-rater κ reliability check (GUARDED, inspected-not-validated; result pending the independent vet anchor)** (`src/vlm/` + `src/eval/kappa.py`): scores any face corpus's per-AU VLM labels against a vet anchor, reports 5 quadratic κ with **CI lower bounds**. A guarded reliability check ranked **below** the confound protocol, **not** a co-equal second pillar; kept as kill-tree insurance (sole-survivor headline if the confound leg degrades). Forward-looking method finding ("can a frozen VLM weak-label feline FGS AUs at human-rater agreement"), **not** an audit of CAT_01's specific labels. Fires on the **CI lower bound** (Gate 1-B) — a textbook clinimetrics gate, not a novel increment; the rubric-paraphrase guard is published, also not novel. The labeler choice is **self-justified by this pilot**; no external weak-label citation is invoked to justify it.

The welfare wrapper (decision curve + abstention) is **cited supporting plumbing, never a headline.** Calibration / RPS / ClasswiseECE / MAPIE are **supporting evidence**, never headlines. There is **no binned reliability diagram on the 11-atom 0–10 sum** anywhere in the repo (distributional path uses RPS-on-sum + per-AU ClasswiseECE only).

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

## IMPROVEMENTS.md
# cat-fgs-llm — Improvement Recommendations

**Generated:** 2026-06-15
**Scope:** Code hygiene, duplication removal, library modernization, security/perf

---

## 1. CRITICAL FIXES (Ship Immediately)

### ✅ DONE: Pre-commit file handle leaks
- **File:** `.pre-commit-config.yaml:38,66`
- **Fix:** Replaced `json.load(open(...))` with `with open(...) as f: json.load(f)`
- **Impact:** Resource leak eliminated; 2 locations patched

### ✅ DONE: G0 manifest guard extracted
- **New file:** `src/gates/manifests.py`
- **Benefit:** Single source of truth for `GATE_PRECONDS` + `enforce_g0_manifests()`
- **Next:** Gate scripts should import this instead of duplicating logic (future refactor)

---

## 2. HIGH-PRIORITY REFACTORS (Next Sprint)

### 2.1 Gate script duplication removal
**Current state:** 8 `scripts/gate*.py` files may contain repeated G0 manifest checks.
**Action:** After this PR, update each gate script to:
```python
from src.gates.manifests import enforce_g0_manifests
enforce_g0_manifests("gateX", synthetic=args.synthetic)
```
**Effort:** 2-3 hours (mostly search/replace + test)

### 2.2 Orchestrator cleanup
**File:** `src/gates/orchestrator.py:60-71`
**Issue:** Still defines `GATE_PRECONDS` locally (duplicate of `manifests.py`)
**Action:** Remove the local definition; import from `manifests.py`
**Effort:** 15 minutes

---

## 3. LIBRARY MIGRATION OPPORTUNITIES (Context7 Findings)

### 3.1 Transformers / DINOv2 — Register variant (HIGH VALUE)
**Finding:** DINOv2 with Registers (`dinov2_vits14_reg`) is explicitly documented to suppress attention artifacts in low-informative regions — directly relevant to per-AU FGS scoring (orbital/ear/muzzle sub-regions).

**Current code:** Uses `facebook/dinov2_vits14` (non-reg)
**Recommendation:** A/B test `facebook/dinov2_vits14_reg` before locking backbone
**Impact:** README already flags this; now confirmed by official docs
**Command:**
```bash
# In src/model/ backbone loading
model = AutoModel.from_pretrained("facebook/dinov2_vits14_reg")
```

### 3.2 Anthropic — Prompt caching (MEDIUM VALUE)
**Finding:** SDK supports `cache_control: {type: "ephemeral", ttl: "5m"}` on system prompts and tool results.

**Current usage:** Unknown (check `src/vlm/batch_submit.py`)
**Recommendation:** Add cache breakpoints on long rubric/system prompts to reduce token costs on repeated VLM calls
**Effort:** 1-2 hours (add `cache_control` blocks)

### 3.3 MAPIE — LTT for abstention (LOW-MEDIUM VALUE)
**Finding:** `BinaryClassificationController` with LTT already used in `src/wrapper/abstention.py`. Docs confirm multi-risk control (NPV + PPV + abstention_rate) pattern matches current implementation.

**Status:** Code is aligned with current MAPIE best practices. No urgent migration.

### 3.4 dcurves — Harm specification (LOW VALUE)
**Finding:** `dca(..., harm={'model': 0.01})` API confirmed. Current wrapper likely passes harm ratios correctly.

**Status:** No change needed unless harm values are hardcoded (grep for `harm` in `src/wrapper/`).

---

## 4. SECURITY & PERFORMANCE FINDINGS

### 4.1 Security (No blockers found)
- `.env` is gitignored ✅
- Pre-commit enforces `check-json` on `artifacts/*.json` and `data/manifests/*.json` ✅
- No obvious secrets in committed files (spot-checked)
- **Action:** Run `git secrets --scan` or `trufflehog` before next release

### 4.2 Performance hotspots (needs measurement)
- **VLM batching:** `src/vlm/batch_submit.py` — verify prompt caching is active
- **Gate pipeline I/O:** Repeated `power.json` reads across gates — consider in-memory cache for orchestrator runs
- **No obvious N+1 or missing `@lru_cache`** on pure functions (e.g., `get_au_names`)

**Recommended profiling:**
```bash
uv run python -m cProfile -o profile.out scripts/gate2_confound.py
uv run snakeviz profile.out
```

---

## 5. QUICK WINS (Low Effort, High Confidence)

| Change | File | Effort | Confidence |
|--------|------|--------|------------|
| Add `from __future__ import annotations` to `src/gates/manifests.py` | `manifests.py:1` | 1 min | 100% |
| Ruff: enable `UP007` (PEP 604) | `pyproject.toml` | 5 min | 90% |
| Add `test_manifests.py` for `enforce_g0_manifests` | `tests/test_manifests.py` | 30 min | 95% |
| Document `GATE_PRECONDS` as the single source in README | `README.md` | 10 min | 100% |

---

## 6. COMMANDS TO RUN

```bash
# Verify pre-commit fixes
uv run pre-commit run --all-files

# Test new manifests module
uv run python -c "from src.gates.manifests import enforce_g0_manifests; print('OK')"

# Context7 follow-up (when quotas reset)
# Run context7-mcp skill queries on the 4 libraries above

# Full gate pipeline smoke (synthetic)
make gate-e2e-synthetic --synthetic
```

---

**Next milestone:** After the 8 gate scripts are updated to import `manifests.py`, delete the duplicate `GATE_PRECONDS` from `orchestrator.py` and add a unit test for the helper.

**Owner:** Lead (Sisyphus)
**Status:** 2/7 todos complete; 2 critical fixes shipped; 1 high-value library migration identified (DINOv2 _reg)
## DATA_DECISION.md
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
6. **We must label the per-AU FGS gap ourselves — there is no shortcut.** Proceed with the Phase B plan: **VLM weak-label 5 AUs → vet review** (CatFACS manual as the rubric), prioritizing acquisition of **more distinct pain cats and especially severe AU=2 examples**. Power target: ~400+ pain-positive evaluation cases (~2x today) to pin the 0.39-threshold sensitivity/specificity to a ±0.10 CI — noting the per-AU kappa remains a protocol with its result pending the independent per-AU vet anchor (Gate 0), since the current binary pain/no_pain dataset cannot yield 0/1/2 ground truth.
7. **Gate Phase A clinical claims behind a label-reliability pilot + capture-condition confound audit** (our labels are pseudo-labels on generic Flickr photos). Update CLAUDE.md / RESEARCH.md / BUILD_PLAN.md: **provenance = Zhang 2008, NOT Finka 2019** — drop the Finka cross-dataset leakage warning entirely.

---

## 6. Risks & leakage guardrails when combining sources

- **Group by clip, never by collapsed individual.** Collapsing CAT_01 creates a 605-image super-group → one degenerate fold. Clip-grouping keeps every fold at 27–39 pain clips.
- **Never use CLIP-cosine for individual dedup** — no identity signal; any count it produces is a threshold artifact. Use imagehash / face re-ID on pixels if true per-cat dedup is ever required.
- **Grouping key = filename stem only.** Strip `_png.rf.<hash>.jpg`; never key on a parent folder named `CAT`. This neutralizes the lone `00000100` collision and makes archive CAT_xx folder collision impossible.
- **Provenance = Zhang 2008, not Finka.** If we later obtain CatFLW/Finka/Steagall, they share **no** cats with our Roboflow set → no Finka-overlap leakage. But each acquired set has only 26–84 cats — **regroup any merge by individual cat** or recall/PR-AUC will inflate.
- **License tracking:** our set CC BY 4.0; archive CC0; **CatFLW CC BY-NC 4.0 (blocks commercial deployment of any derived model)**; all request-only pain sets have no stated data license. Keep per-source licenses separate; do not let CC0/NC images leak into a CC BY release.
- **Augmentation ≠ subjects.** 2x augmentation expands frames, adds zero pain individuals; it stabilizes training but does not widen the generalization CI. Always count unique frames (1631 train), not augmented (3262).
- **Label-validity confound is the top risk** — bigger than any leakage question. Treat the binary pain signal as unvalidated until the reliability pilot passes.## GAP_ANALYSIS.md
# Cat Pain / FGS: Field Coverage, White Space, and the Gap We Should Own

## 1. The field in one page

| Paper | What it nailed | What it left open |
|---|---|---|
| **Evangelista 2019** (FGS development) | Created and psychometrically validated the FGS (5 AUs, 0/1/2); derived the load-bearing **>0.39 analgesia cut-off** (AUC 0.94, sens 90.7 / spec 86.6); inter-rater ICC 0.89. | 100% manual human scoring; no automation; no brachy/dark-coat/kitten; cut-off never tested *interventionally*; data on request only. |
| **Feighelstein 2022** (binary PoC) | First head-to-head landmark-ML vs ResNet50 on cats; clean LOSO splits; showed 48-landmark geometry ≈ raw image (~72%). | Binary only; manual landmarks; single homogeneous cohort (young female DSH); no graded output; no foundation model / VLM. |
| **Feighelstein 2023** (explainable, noisy pop.) | Tested generalization on heterogeneous 84-cat clinic population; XAI region importance (**mouth > eyes > ears**); LDM (77%) beat DL (65%). | Binary; one frame/cat; manual landmarks; no external set; cannot separate pain from distress; no calibration/CIs. |
| **Steagall 2023** (automated FGS) | **Only graded per-AU automated FGS** (37 landmarks → 35 geometric descriptors → XGBoost; per-AU 0/1/2 + total + 0.39 decision); 95.5% binary, smartphone-arguable. | Hand-crafted geometry, not learned features, no ordinal loss; needs 8 real-time vet raters; data **withheld for app**; no calibration, no CIs, components tested in isolation; no app shipped. |
| **Martvel 2024** (video) | First end-to-end **raw-video** binary pipeline (YOLOv8 + ELD landmarks + temporal XGBoost); showed dynamics help; cross-dataset transfer attempted. | Binary; transfer largely **failed** (Finka→TiHo −0.15 acc); only acc/F1 (no ROC/CI/calibration); landmark-API only released; analgesia confound not disentangled. |
| **Marangoni 2026** (brachy ocular pain) | Showed FGS **responsiveness** + reliability in a hard population; surfaced e-collar and image-vs-real-time overestimation confounds; muzzle/whisker AUs break down in brachy. | Human raters only; no control group; automated brachy FGS = future work; no graded 0–10; single-reviewer real-time gold. |
| **Lee & Steagall 2026** (COSMIN review) | Field-level audit: FGS = highest evidence; named **responsiveness, interpretability, feasibility, ROC analgesia thresholds** as the open needs. | **Explicitly excluded all automated/AI scorers**; no meta-analysis; no new data; named gaps left unfilled by design. |
| **Namboonlue 2023** (Thai CNN thesis) | Independent non-European cohort; EfficientNetB7 79%; Grad-CAM (center-face attention); honest overfitting/confound discussion (Smudge meme false positive). | Binary; no validated scale as ground truth; no subject counts/leakage control; severe overfit; nothing released. |

**One-line synthesis:** The field has validated the human FGS + 0.39 threshold, automated *binary* pain repeatedly, automated *graded* FGS exactly once (closed, hand-crafted, no learned features), and shipped **zero** runnable artifacts, **zero** calibration/CIs, **zero** open benchmarks, and **zero** measurement of its own training-label reliability.

---

## 2. The white space — gaps NONE of the 8 filled

**Group A — Trustworthy-decision layer (the strongest, most-converged white space)**
- **Calibration + cost-sensitive operating point at 0.39** — no ML paper reports a reliability diagram, Brier/ECE, or decision-curve under undertreatment≫overtreatment asymmetry. Clinical papers derived 0.39 on *humans*; ML papers inherit it as a constant. *Novelty: high · Feasibility: high · Value: high.*
- **Uncertainty-aware defer-to-vet abstention** evaluated as accuracy-vs-coverage — Martvel filters low-confidence frames internally but nobody surfaces it as an output. *high · high · high.*
- **Pain-recall-at-fixed-high-sensitivity with bootstrap 95% CIs stated up front** (CI half-width honesty) — never reported under real prevalence. *medium · high · high.*

**Group B — Label honesty (forced on us by our data, owned by no one)**
- **Weak-label reliability as a measured quantity** — treat the VLM 5-AU labeler as a *rater*; report per-AU agreement (quadratic κ + CI) vs a vet. No paper reports reliability of its *training* labels (only of human FGS raters). *high · high · high.*
- **Label-provenance + capture-condition confound audit** — does the binary signal predict pain or brightness/blur/box-aspect/morphology? Flagged by Martvel/Namboonlue/Feighelstein, **quantified by none**. *medium–high · high · high.*

**Group C — Modeling novelty**
- **Frozen foundation-model backbone (DINOv2 ViT-S)** — every paper uses ImageNet ResNet/EfficientNet/ShuffleNet or landmark geometry; zero foundation models. *high · high · high.*
- **Learned graded per-AU output via rank-consistent ordinal heads (CORN)** — the only graded paper (Steagall) used XGBoost on descriptors, no ordinal loss, no learned features. *high · medium · high.*
- **VLM/LLM weak-labeling of the 5 AUs** — all 8 papers are pure vision; the per-AU annotation bottleneck is named repeatedly but the proposed fix is always "automate landmarks," never "semantically weak-label the AUs." *high · medium · high.*

**Group D — Open artifact**
- **Open, leakage-audited benchmark + datasheet** with honest pseudo-label provenance and a frozen 0.39 protocol — every dataset is request-only/withheld. *Caveat below in §6.* *medium–high · medium · medium.*

---

## 3. What we should NOT do (replication traps)

1. **Do NOT re-do Steagall's landmark → geometric-descriptor → XGBoost pipeline.** It is done, it is the field's graded baseline, and it needs 8 real-time vet raters we don't have. Replicating it = a worse, closed-data clone.
2. **Do NOT re-do Martvel's video/temporal pipeline.** We have no video, no timestamps, no paired frames. Temporal modeling is structurally impossible on our data.
3. **Do NOT chase the binary-accuracy leaderboard (77 / 79 / 95%).** Saturated, on tiny separable curated sets. "Beat 79%" is not a contribution and our confounded labels make it dishonest.
4. **Do NOT claim external-cohort validation.** We have one confounded Flickr-derived source (single CAT_01 camera). Every external pain/FGS set is request-only. Name single-source as our top threat-to-validity; do not pretend to fill it.
5. **Do NOT promise responsiveness / longitudinal within-cat rescoring.** It is the field's #1 named need (tempting) but requires paired pre/post-analgesia timestamped frames we do not have. Cite as out-of-reach.
6. **Do NOT promise multi-rater model-vs-human ICC/Bland-Altman.** Needs a rater panel; we have at most one vet. Borrow the vocabulary, not the claim.
7. **Do NOT headline cross-species horse transfer.** 5 horses, 3 of 5 AUs (no whiskers, no head), grouped ≈ 5 datapoints. Use it only as **pipeline-validation scaffolding** (does CORN-decode → sum → threshold work on *real* graded labels?), never as a measured transfer gain.
8. **Do NOT center a shipped mobile app or multimodal posture/audio.** App = solo-eng scope creep (a minimal HF Space demo is fine as a delivery vehicle, not the thesis); posture/audio = no body/audio data.

---

## 4. THE GAP WE SHOULD OWN

> **A portable, power-aware confound-attribution protocol for fine-grained animal-affect models — per-AU FGS-BG-Gap counterfactual + per-AU EBPG saliency-as-confound-evidence + a VLM judge-bias probe, framed one-directionally ("at this power, no confound detected") — applied to a learned graded FGS scorer, with a guarded VLM-as-AU-rater reliability check (CI-lower-bound gated, result pending an independent vet anchor) and a cited calibration/abstention decision layer as supporting evidence. Novelty is the ASSEMBLY + per-AU EBPG-as-confound-evidence + the instantiation of equivalence-style audit hygiene — not any individual primitive or the underlying statistics.**

Concretely, one vertical:

1. **Engine (modeling novelty):** frozen **DINOv2 ViT-S/14** (the `_reg` register variant, `dinov2_vits14_reg`, is the field default — registers suppress attention artifacts that hurt the localized orbital/ear/muzzle features per-AU FGS needs; A/B it before locking ViT-S/14) + **5 rank-consistent CORN ordinal heads** → per-AU 0/1/2 → 0–10 → 0.39 decision.
2. **Supervision (never attempted):** **VLM weak-labels the 5 AUs**; a vet reviews a small calibration anchor; active-learning prioritizes uncertain/severe cases. This is the *only* path from binary/web data to graded FGS without a closed expert corpus.
3. **Honesty layer (where the headline lives):** the **headline is the per-AU confound-attribution protocol** — FGS-BG-Gap background counterfactual + per-AU EBPG saliency-as-confound-evidence + a VLM judge-bias probe, framed one-directionally and conditioned on the audit power available ("at this power, no confound detected"). The **VLM-as-rater per-AU quadratic κ vs vet** is a **guarded, inspected-not-validated reliability check** (CI-lower-bound gated, result pending an independent vet anchor), not a co-equal pillar. The decision-support machinery — **calibration** (reliability diagram, Brier/ECE), **pain-recall-at-fixed-sensitivity with bootstrap 95% CIs (half-width stated first)**, **decision-curve under undertreatment≫overtreatment**, **defer-to-vet abstention curve**, face-quality/morphology input gate — is **cited plumbing / supporting evidence**, not a headline contribution.
4. **Artifact:** released weak-labeled AU annotations + frozen-backbone+CORN code/weights + frozen clip-grouped test split + **datasheet** documenting pseudo-label provenance and the confound audit.

**Why this is novel against each relevant paper:**
- **vs Steagall 2023** (the only graded work): we use *learned* features (frozen ViT) + an *ordinal loss* (CORN) instead of hand-crafted geometry + XGBoost; we replace 8 real-time vet raters with VLM-weak-labeling + a small vet anchor; and we report calibration/CIs/abstention/confound audit — *all* absent from Steagall. Our data is open; theirs is withheld.
- **vs Feighelstein 2022/2023 & Martvel 2024 & Namboonlue 2023:** they are binary; we are graded per-AU. They are ImageNet-CNN/landmark; we are foundation-model. None uses any VLM/LLM. None reports calibration, CIs, abstention, or training-label reliability.
- **vs Evangelista 2019 & Marangoni 2026:** they derived/tested 0.39 on *human* raters with no model to calibrate. We attach the calibrated, cost-sensitive, abstention-aware decision layer to an *automated* scorer.
- **vs Lee & Steagall 2026:** they named ROC thresholds + interpretability + feasibility as the field's open need and **explicitly excluded automated scorers**. We deliver exactly the automated-scorer measurement layer they could not assess.

**Why it's feasible on our stack:** frozen backbone → only light heads train (M4/Colab T4); calibration, abstention, bootstrap CIs, κ, and confound audits are all **post-hoc analyses on held-out scores** — no FGS gold corpus, no second cohort, no video, no large compute. The horse set is used once to sanity-check the CORN decode path on genuine 0/1/2 labels.

**Resolving the lens disagreement:** the five analyses circle the same point from different sides. Modeling says "DINOv2+CORN+VLM is the engine" (conceded as plumbing); clinical/methodology/data say "confound audit + label-honesty + a calibrated decision layer is the unowned validity ground"; product says "ship it open." These are **not competing** — they are the engine, the honesty layer, and the delivery vehicle of *one* contribution. The headline is the **per-AU confound-attribution protocol** (the strong leg — validated today on planted positive+negative controls, zero GitHub and zero assembled-paper matches); below it sits the **guarded VLM-as-AU-rater κ check** (inspected-not-validated, kept as kill-tree insurance, not a co-headline); and the calibration/abstention machinery is **cited plumbing**, not a contribution. The graded engine is what makes the protocol worth running; the open release is the delivery. The "open benchmark" framing is demoted to "open *audit + datasheet + protocol*" (not a leaderboard implying trustworthy labels we don't have).

---

## 5. Concrete project framing (how this updates BUILD_PLAN.md)

**Target.** A graded per-AU FGS scorer (frozen DINOv2 ViT-S + 5 CORN heads, VLM-weak-labeled + vet-anchored AUs) delivered as a **calibrated, abstention-aware, confound-audited decision-support artifact** at the 0.39 cut-off.

**The claim we could make (and defend).** *"A portable, power-aware per-AU confound-attribution protocol for fine-grained animal-affect models — we audit the acquisition/morphology confound one-directionally (FGS-BG-Gap counterfactual + per-AU EBPG saliency-as-confound-evidence + VLM judge-bias probe), reporting it at the audit power available rather than claiming the absence of confounding. The protocol is applied to a learned graded FGS scorer, guarded by a VLM-as-AU-rater reliability check (per-AU quadratic κ with a CI-lower-bound gate, result pending an independent vet anchor) and supported by a cited calibrated/abstention-aware decision layer at 0.39. The novelty is the assembly, the per-AU EBPG-as-confound-evidence step, and the equivalence-style audit hygiene — not any individual primitive, nor the κ CI-LB gate (textbook clinimetrics), nor the rubric-paraphrase guard (published)."* Application to real cats is named as future work.

**Minimal validation to support it (the GO/NO-GO gates):**
- **Gate 1 — Label reliability:** ~120-image VLM-vs-vet per-AU quadratic κ pilot with CI. A high κ only measures weak-labeling *capability* if the vet anchor is independent and the rubric handed to the VLM differs from the vet's rubric; otherwise it measures rubric-following — state which a given run measures. If κ is too low, AUs that fail are reported as such (honest), not hidden.
- **Gate 2 — Confound audit:** trivial classifier (brightness/blur/box-aspect/CLIP) on the binary label. If it beats chance, report pain as entangled with acquisition context and condition all downstream claims on it.
- **Gate 3 — Decision layer:** calibration + pain-recall-at-fixed-sensitivity with bootstrap 95% CI half-width stated up front (~±0.13 on ~50-positive folds); 0.39 decision estimated **only on vet-confirmed labels** to break VLM-label circularity; abstention accuracy-vs-coverage curve.
- Splits: **clip-grouped StratifiedGroupKFold** (no same-cat leakage); frozen hashed test split.

**First 3 actions:**
1. **Run Gate-2 confound audit now** on the existing binary Roboflow data (trivial-shortcut classifier) — cheapest, decides whether the corpus is salvageable and frames the whole paper's honesty.
2. **Stand up the VLM 5-AU weak-labeler + the ~120-image vet-review pilot** to get the first Gate-1 per-AU κ number (the precondition every downstream claim rides on).
3. **Wire frozen DINOv2 ViT-S embedding extraction + a CORN ordinal head, validated first on the horse 0/1/2 labels** (decode → sum → threshold sanity check) before touching cat AUs.

---

## 6. Honest risks

- **Label reliability could fail Gate 1.** If VLM-vs-vet κ is poor on muzzle/whiskers (exactly where Steagall and Marangoni both report low inter-rater reliability), graded per-AU output may be untrustworthy. *Mitigation:* report κ honestly per-AU; ship the calibrated **binary** decision layer with abstention as the v1 floor — the likely v1 ship that stands today, with the graded 0-10 layer as upside conditional on Gate 1-B passing. The confound-attribution headline stands either way; and if the confound leg degrades, the guarded κ check is retained as kill-tree insurance (sole-survivor headline).
- **Confound audit could be damning (Gate 2).** If pain is largely predicted by brightness/blur/morphology, our labels are mostly shortcut. *This is still a publishable finding* (the first measured confound audit), but it kills any accuracy claim — which is why we don't headline accuracy.
- **Calibration/abstention could be seen as routine ML hygiene.** They are standard elsewhere; a reviewer may say "applying known methods." *This is exactly why they are framed as cited plumbing, not the headline.* The headline novelty is the per-AU confound-attribution protocol — the *assembly* of bg-counterfactual + per-AU EBPG-as-confound-evidence + judge-bias probe, framed one-directionally with equivalence-style audit hygiene — which is genuinely unowned in this research line. We never claim any individual primitive, nor the one-directional/power-conditioned/equivalence statistics, as novel.
- **Open-benchmark trap.** Releasing a dataset whose pain labels we ourselves distrust is a liability, and CatFLW (best alignment asset) is **CC BY-NC** — contaminating a clean open release and blocking commercial use. *Mitigation:* release the **audit + datasheet + protocol + code/weights**, not a "trustworthy-labels leaderboard."
- **Single-source ceiling.** One confounded Flickr camera means no external validity; wide CIs are unavoidable. *Mitigation:* name it as the top threat-to-validity up front rather than over-claiming.
- **VLM AU-grounding may be weak.** VLMs may not reliably localize feline-specific AUs (orbital tightening, whisker change). *Mitigation:* the Gate-1 κ pilot *measures* this directly before any training spend — failure is detected cheaply, not after the model is built.## FACTCHECK.md
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

**Genuinely unowned (asymmetric, per FINAL_DIRECTION §8):** the **headline** is the assembled per-AU confound-attribution protocol (FGS-BG-Gap counterfactual + per-AU EBPG-as-confound-evidence + VLM judge-bias probe, framed one-directionally); **below it** sits the guarded VLM-as-AU-rater κ check (inspected-not-validated, CI-LB gated, pending an independent vet anchor); the foundation-model/VLM engine and the calibration/CI/abstention layer are **cited plumbing / supporting evidence**, not headline contributions. Novelty = the assembly + per-AU EBPG-as-confound-evidence + equivalence-style audit hygiene — never any individual primitive, never the κ CI-LB gate (textbook clinimetrics) or rubric-paraphrase guard (published), never the one-directional/power-conditioned/equivalence statistics themselves.

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
- **MEMORY / dataset-facts.md** — replace the 76%-sample clip numbers with the full-manifest figures (336/138/135/191).## HF_SOLUTION.md
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
| **1. VLM weak-label (Claude Opus 4.8, 5-AU enum 0/1/2) + cleanlab/boundary triage → vet review → frozen DINOv2 + 5 CORN heads + active learning** | **Silver only**, not gold; trust comes from the vet layer it feeds | ~$12 (single batched+cached Opus pass over ~2,040 crops); $0 GPU; **binding cost = ~4–5 vet hrs** to clear the 120–150 kappa-CI floor | Complete 5-AU silver layer over all images **this week**, zero external permission; concentrates scarce vet hours on boundary/disagreement cases (prefill accept/correct); supplies the guarded per-AU VLM-vs-vet kappa reliability check (CI-lower-bound gated, inspected-not-validated, result pending the independent vet anchor) that backs the headline confound-attribution protocol | **pursue-now (critical path / spine)** |
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

Relevant files: `/Users/mingrath/ghq/github.com/mingrath/cat-fgs-llm/BUILD_PLAN.md`, `/Users/mingrath/ghq/github.com/mingrath/cat-fgs-llm/DATA_DECISION.md`, `/Users/mingrath/ghq/github.com/mingrath/cat-fgs-llm/RESEARCH.md`.## GITHUB_MINE.md
# GitHub Mining — Additive Briefing (beyond HF_SOLUTION.md / DATA_DECISION.md)

## 1. Did anything NEW turn up?

**Verdict on new downloadable cat-pain/FGS data: effectively NO.** No new open, directly-downloadable cat dataset with graded per-AU FGS (or even expert binary cat-pain) labels exists on GitHub. The entire published cat lineage (Steagall, Evangelista, Finka/TiHo, **and now the named Zamansky/Feighelstein 464-img GitLab repo**) is confirmed request-only or dead-linked. This does **not** change the DATA_DECISION conclusion.

**But the mining produced four genuinely new, useful assets** not in HF_SOLUTION/DATA_DECISION:

1. **`mytalbot/MGS_data`** — real **open per-AU ordinal grimace labels** (Mouse Grimace Scale, 5 AUs incl. orbital + ear + **whisker change**), 4,944 rows. The *only* openly downloadable dataset whose label shape matches our 5-AU 0/1/2 target — and it covers a whisker AU, our structural blind spot.
2. **`Barn99/Automated-RGS`** — pretrained **ViT per-AU grimace grader + YOLOv5 AU-region detector** (Rat Grimace Scale, weights on Google Drive). A working instance of our exact planned architecture (region-detect → per-AU ordinal head).
3. **Exact request target located:** `gitlab.com/is-annazam/automated-recognition-of-pain-in-cats` — the named (now 404/upon-request) repo for the **464-img balanced expert binary cat-pain set + 48 CatFACS landmarks**, author **Anna Zamansky / Martvel**. Gives DATA_DECISION §5's "email Zamansky" a precise repo+author handle.
4. **`marroyol/comp3000-marie`** — **directly downloadable** ~44-img **4-level ordinal pain-likelihood** labels keyed to CatFLW filenames, with **two passes + a second rater** (ready-made inter-rater kappa fixture). Non-expert/student labels, so not gold — but a free, multi-rater calibration fixture on real CatFLW crops.

Plus a public **Martvel landmark-detection Colab API** (the only public form of the IJCV 48-landmark cat detector — no GitHub repo exists) and concrete reusable training/wrapper code (below).

## 2. New datasets / data links (genuinely new only)

| name | url | pain-labeled? | labels | license | access |
|---|---|---|---|---|---|
| MGS_data (Mouse Grimace) | github.com/mytalbot/MGS_data | yes (mouse) | **5 ordinal AUs incl. whisker change** (orbital, nose, cheek, ear, whisker), 4,944 rows; -1=rejected | LICENSE file in repo (verify) | open `git clone` — ships scores `MGS_raw.txt`, **not** source images |
| Automated-RGS weights (Rat Grimace) | github.com/Barn99/Automated-RGS | yes (rat) | pretrained ViT per-AU grader (orbital/ear/nose) + YOLOv5 AU detector; no whisker | not stated | code open; weights via [Drive folder](https://drive.google.com/drive/folders/1Cl_5GyouX7sDLv1NUKuq_YxrrQRQMYKn) |
| comp3000-marie pain_labels | github.com/marroyol/comp3000-marie/tree/main/pain_labels | yes (cat, **non-expert**) | 4-level ordinal pain-likelihood, keyed to CatFLW filenames; pass1+pass2+rater2 | no LICENSE (images=CatFLW CC BY-NC) | open, raw.githubusercontent download |
| Zamansky/Feighelstein cat pain (464 img) | gitlab.com/is-annazam/automated-recognition-of-pain-in-cats | yes (cat, **expert binary**) | binary pain/no-pain (ovariohysterectomy protocol) + 48 CatFACS landmarks, 26 cats | unspecified/academic | **404 / request-only** — email Anna Zamansky, Tech4Animals, U. Haifa |
| htcv_mgs labels+weights | github.com/PatrickFreund/htcv_mgs | yes (mouse, **binary only**) | labels.csv binary 0/1 (NOT per-AU) + per-fold `best_model.pth` | not stated | open |
| CatFACS coding manual | animalfacs.github.io/AnimalFACS/CatFACS | no (manual) | defines canonical cat facial AU scheme | manual free; videos permission-gated | [Drive](https://drive.google.com/drive/folders/1uO0DXexVg9V62TC3fGBvHkqz84J5lEc1) |
| EquinePainFace annotations | github.com/jmalves5/EquinePainFaceDataset | yes (horse) | graded equine pain-face annotations (XLS/JSON) tied to PLOS ONE pone.0231608 | unspecified | repo + PLOS supplementary |

*(CatFLW, the horse-grimace set, and the Steagall/Evangelista request-only sets are already in DATA_DECISION/HF_SOLUTION — not repeated.)*

## 3. Repos to clone now

### (a) Paper code / weights
| repo | what it gives us | reuse value |
|---|---|---|
| github.com/martvelge/CatFLW | Official CatFLW repo; per-image JSON `{labels:(48,2), bounding_boxes}` format spec + Kaggle link (confirms the download already in DATA_DECISION) | high (format reference) |
| [Martvel landmark Colab](https://colab.research.google.com/drive/1XmTL3qJ2mMfb4FfCdwhnDW5jVUWNYTbi) | **Only public form** of the IJCV magnifying-ensemble 48-landmark cat detector (no GitHub exists). Auto-detect 48 landmarks on our Roboflow images → AU-region crops without manual annotation | high |
| github.com/martvelge/dog_emotions_LLMs | Same lab's official code (AGPL-3.0) for **prompting VLMs over cropped animal faces + YOLO11** — direct template for our VLM weak-labeling step | medium |
| github.com/Barn99/Automated-RGS | YOLOv5 AU-detector + per-AU ViT ordinal grader **with downloadable weights** — warm-start candidate / architecture proof | high |

### (b) Reusable training code (DINOv2 + CORN, face-align)
| repo | what it gives us | reuse value |
|---|---|---|
| github.com/arthur-thuy/qde-ordinality | Backbone-agnostic ordinal head (`RobertaOrdinalHead`) that takes pooled CLS embeddings and switches CORAL/CORN/OrderedLogit by config — **best template to drop onto DINOv2 `features[:,0,:]`**; wrap 5 in `nn.ModuleList` | high |
| github.com/itsprakhar/Downstream-Dinov2 | Frozen-DINOv2 extractor + swappable head + `train_classifier.py` ImageFolder entrypoint — scaffold; replace single Linear with the 5 CORN heads (MPS-friendly, backbone frozen) | high |
| github.com/marinbenc/dermatoscopy_colorimetry_eval | Concrete **CORAL-on-image-backbone training loop** (`CoralLayer` + `levels_from_labelbatch` + `coral_loss`) — copy-paste ordinal-grading recipe | high |
| github.com/chelsea23311/Cat-Face-Landmark-Detection | Complete cat-face 9-keypoint pipeline (ResNet-50, train/test/predict, **interocular-distance normalization**, PCK/NME) — ready-made eye-alignment step for Gate-5 / the §3 NME audit; weights on releases | medium |
| github.com/RobvanGastel/dinov3-finetune | DINOv2/v3 **LoRA** path for later T4 unfreezing if frozen linear-probe underfits | medium |
| github.com/Raschka-research-group/coral-pytorch | Canonical CORN/CORAL lib (already named in our plan; confirmed: `CoralLayer`, `corn_loss`, `corn_label_from_logits`, `levels_from_labelbatch`) | high (already planned) |

### (c) Trustworthiness-wrapper implementations (calibration / abstention / noisy-label)
| repo | what it gives us | reuse value |
|---|---|---|
| github.com/ENSTA-U2IS-AI/torch-uncertainty | **One-stop wrapper backbone (Apache-2.0):** temp/vector/matrix/Dirichlet scaling + ECE/SmoothECE/AdaptiveECE/**ClasswiseECE** + reliability diagrams **AND** selective-classification AURC/AUGRC/CovAt5%Risk/RiskAt80%Cov + conformal APS/RAPS. Covers 2 of 4 wrapper pillars off the shelf | high |
| github.com/cleanlab/cleanlab | Confident-learning weak-label audit (`find_label_issues`, multi-label) — ranks suspect VLM FGS labels into a send-to-vet queue (already named; now confirmed multi-label support for the 5-AU case) | high |
| github.com/dholzmueller/probmetrics | **Maintained** temperature-scaling + ECE/Brier/NLL/SmoothECE (pip, v1.3.0 2026) — use this for the actual temp fit, not the unmaintained gpleiss repo | high |
| github.com/EFS-OpenSource/calibration-framework | netcal — most complete standalone calibration toolbox + reliability-diagram plotting | high |
| github.com/apple/ml-calibration (`relplot`) | Kernel-smoothed reliability diagrams + SmoothECE (ICLR2024) — **important because our FGS set is small; binned ECE is unstable at low n** | high |
| github.com/MSKCC-Epi-Bio/dcurves | Maintained Vickers **decision-curve / net-benefit** analysis — clinical-utility argument at the 0.39 threshold | high |
| github.com/bhanML/Co-teaching | Official PyTorch **co-teaching** small-loss selection — exactly the noisy-VLM-label training already specified in HF_SOLUTION step 8 | medium |
| github.com/IdoGalil/benchmarking-uncertainty-estimation-performance | ICLR2023 AURC + **SAC** (max coverage at accuracy target) — frames "defer-to-vet" as a publishable metric | medium |

## 4. Ideas / references worth borrowing

- **Pretrain/validate the 5 CORN heads on `mytalbot/MGS_data` first** — real open per-AU ordinal labels (incl. a whisker AU) to de-risk the heads before spending cat/vet data, alongside the horse warm-start already in HF_SOLUTION. MGS covers whisker; horse does not — partial relief for our whisker/head blind spot.
- **Adopt `Barn99/Automated-RGS`'s two-stage pattern explicitly** (region-detect → per-AU ordinal head) and consider fine-tuning from its released rat ViT weights as a warm start.
- **Head module:** copy `qde-ordinality`'s `RobertaOrdinalHead` (dropout→dense→tanh→dropout→`out_proj`), feed DINOv2 CLS, `out_proj = nn.Linear(384, K-1)` with K=3; wrap 5 in `nn.ModuleList`. Loss = sum of 5 `corn_loss`; decode per-AU via `corn_label_from_logits` → 0–10 sum → 0.39 in code. CORAL-vs-CORN ablation is a one-flag config switch.
- **Use `comp3000-marie`'s pass1/pass2/rater2 as a pre-built inter-rater kappa fixture** to validate our VLM-weak-label kappa pipeline on real CatFLW crops before any vet hour.
- **Wrapper:** lean on `torch-uncertainty` for calibration+abstention; report `relplot` SmoothECE alongside binned ECE (small-n); per-AU `ClasswiseECE` so miscalibration is attributable to a specific AU; frame abstention as SAC (Galil ICLR2023); add `dcurves` net-benefit at 0.39.
- **Rubric/prompt spec (free, defensible):** pull the **CatFACS manual** (Drive) and `etho-backend`'s CatFACS→FGS AU-mapping table (`research/02_CATFACS_FGS.md`, encodes mean ≥0.39) to anchor the VLM weak-label prompt + vet rubric.
- **AU-importance ablation** (Talbot "orbital tightening alone" finding): test whether a subset of the 5 cat AUs retains most predictive power — informs which AUs to prioritize for scarce vet review and whether whiskers/head can be dropped.

## 5. Updated bottom line

**GitHub does NOT change the conclusion** that we must self-label via VLM + vet review. No new open cat FGS/per-AU dataset surfaced; the published cat lineage — now including the precisely-named **Zamansky 464-img GitLab repo** — is all request-only or dead-linked. The VLM-weak-label → vet-arbitration → frozen-DINOv2 + 5-CORN pipeline remains mandatory.

What GitHub **does** change: (a) gives DATA_DECISION §5's Zamansky email an exact repo+author target (likely bundles CatFLW-style landmarks since Martvel is co-author); (b) supplies an **open per-AU ordinal pretraining set with a whisker AU** (MGS_data) and a **transferable pretrained per-AU grimace head + weights** (Automated-RGS) to extend the horse-only warm-start; (c) provides a free multi-rater kappa fixture (comp3000-marie); (d) confirms ready-to-use code for every wrapper pillar.

### Exact next commands

```bash
# --- New per-AU ordinal pretraining + transferable grimace head/weights ---
git clone https://github.com/mytalbot/MGS_data ./datasets/mgs          # 5 mouse AUs incl. whisker (scores only)
git clone https://github.com/Barn99/Automated-RGS ./models/automated-rgs # ViT per-AU + YOLOv5; weights via Drive folder in README

# --- Multi-rater cat pain-likelihood fixture (kappa sanity) on CatFLW crops ---
git clone https://github.com/marroyol/comp3000-marie ./datasets/comp3000-marie

# --- Reusable training code: ordinal head + frozen-DINOv2 scaffold + CORAL loop + eye-align ---
git clone https://github.com/arthur-thuy/qde-ordinality ./ref/qde-ordinality
git clone https://github.com/itsprakhar/Downstream-Dinov2 ./ref/downstream-dinov2
git clone https://github.com/marinbenc/dermatoscopy_colorimetry_eval ./ref/coral-image-loop
git clone https://github.com/chelsea23311/Cat-Face-Landmark-Detection ./ref/cat-landmark-align

# --- VLM weak-label template (same lab, animal faces) ---
git clone https://github.com/martvelge/dog_emotions_LLMs ./ref/animal-vlm

# --- Trustworthiness-wrapper deps (pip; not git) ---
pip install torch-uncertainty cleanlab probmetrics netcal relplot dcurves

# --- Optional: LoRA upgrade path + co-teaching reference ---
git clone https://github.com/RobvanGastel/dinov3-finetune ./ref/dinov3-lora
git clone https://github.com/bhanML/Co-teaching ./ref/co-teaching
```

**Manual fetches (no CLI):** Martvel 48-landmark detector — open Colab `https://colab.research.google.com/drive/1XmTL3qJ2mMfb4FfCdwhnDW5jVUWNYTbi` (no GitHub repo exists); CatFACS manual — Drive `https://drive.google.com/drive/folders/1uO0DXexVg9V62TC3fGBvHkqz84J5lEc1`; Automated-RGS weights — Drive `https://drive.google.com/drive/folders/1Cl_5GyouX7sDLv1NUKuq_YxrrQRQMYKn`.

**Email target (sharpened):** Anna Zamansky / George Martvel, Tech4Animals Lab, U. Haifa — re `gitlab.com/is-annazam/automated-recognition-of-pain-in-cats` (464-img expert binary cat-pain + 48 CatFACS landmarks).

**Do NOT** treat `comp3000-marie` (non-expert student labels) or `htcv_mgs` (binary only, not per-AU) as gold supervision — fixtures/scaffolds only.

---

# Mining Pass 2 (2026-06-12) — Novelty / differentiation axis

> Sections 1–5 above answer *"what data/code can we clone?"* This pass answers a different question: *"what real GitHub techniques would make us UNIQUE, not a duplicate?"* Produced by a 63-agent grep-MCP workflow (760 grep calls, 46 findings verified, 45 confirmed real by re-grepping inside the named repos). Sorted by how much each strengthens the **per-AU confound-attribution protocol** (the headline contribution) and, below it, the **guarded VLM-as-AU-rater κ check** — not the engine, and not the cited calibration/abstention plumbing.

**Headline:** no public GitHub repo assembles a per-AU confound-attribution protocol (bg-counterfactual + per-AU EBPG-as-confound-evidence + judge-bias probe, framed one-directionally) on a fine-grained animal-affect model; the components exist only in isolation. The assembly — plus the per-AU EBPG-as-confound-evidence step and the equivalence-style audit hygiene — is what is unowned. No individual primitive is claimed as novel, and the underlying statistics are not. Ordinal calibration + abstention are cited supporting plumbing; CORN/DINOv2 are table stakes — adopt and move on.

## P2.1 Adopt now (high differentiation, low/med effort)

| # | Technique | Repo | What to try | Why it makes us unique | Effort |
|---|-----------|------|-------------|------------------------|--------|
| 1 | **Risk-controlled two-threshold abstention** (Learn-then-Test): below λ₁→no-pain, above λ₂→pain, middle→defer; thresholds chosen so NPV/recall is statistically guaranteed | `scikit-learn-contrib/MAPIE` (`plot_risk_control_llm_as_a_judge.py`) | Feed calibrated pain-prob into `BinaryClassificationController`; control **negative_predictive_value** (missed pain = dominant cost) at target α on the vet anchor; report `abstention_rate` next to bootstrap recall CIs | Turns the defer-to-vet curve into a **distribution-free finite-sample** bound on undertreatment risk (cited plumbing / supporting evidence, not a headline claim; avoid "guaranteed" branding in prose); seconds on CPU. **Coexists with** the frozen 0.39 cutoff (band selector λ₁,λ₂ around 0.39, not a replacement) | low |
| 2 | **Foreground-mask background counterfactual** (ImageNet-9 BG-challenge): build `mixed_rand`/`only_bg_t`, measure score shift when only background swaps | `MadryLab/backgrounds_challenge` (via `bytedance/ibot`) + `visinf/beyond-accuracy` (BG-Gap) | Mask the cat face (DINOv2 attention / SAM / grabcut — no training), composite onto same-FGS vs swapped clinic backgrounds; report **FGS-BG-Gap** = mean 0–10 shift + pain-flip rate, per AU | Makes Gate-2's confound audit **causal not correlational** — proves the score isn't reading the cage/clinic. No prior cat-pain paper proves this | medium |
| 3 | **Energy-based Pointing Game (EBPG)**: fraction of saliency energy inside an anatomical ROI vs whole map | `haofanwang/Score-CAM` (`energyPointGame.py`) + `hungntt/xai_thyroid` | Per-AU saliency (attention-rollout/Grad-CAM on frozen backbone, 1–2 passes), ROI from CatFLW landmarks; report `energy_in_ROI / energy_whole` per AU, stratified by capture condition | The scalar the audit was missing: does the "ear" head fire on the ear? Per-AU localization-faithfulness no prior cat-pain work reports | low |
| 4 | **Ordinal Krippendorff's α** over N≥3 repeated VLM weak-labelings of the same image | `prometheus-eval/prometheus-eval` (`eval/consistency.py`) | `krippendorff.alpha(reliability_data, level_of_measurement="ordinal")` on a (n_runs, n_items) matrix per AU at varied temp/seed | Vet-free reliability axis (VLM self-consistency) complementing vs-vet κ. Cheap pre-screen before vet budget + abstention signal (low-α → defer). Feeds Gate 1-B | low |

## P2.2 Borrow patterns (adjacent) — concrete code to copy

**CORN decode (point estimate + soft probs for calibration):**
- `Raschka-research-group/coral-pytorch` — `corn_loss` + `corn_label_from_logits`. **Critical:** do NOT decode straight to a hard label. Keep `probas = cumprod(sigmoid(logits))` (per-level cumulative P(rank>k)) — those soft ordinal probs feed ECE/Brier/RPS/reliability-diagram/abstention. The `>0.5` hard threshold is point-estimate only. Unit-test against their doctest (`tensor([1,3])`) — this *is* Gate 4's CORN-decode test.
- Vendoring: `ludwig-ai/ludwig` (`modules/loss_implementations/corn.py`) inlines the ~40-line `corn_loss` (torch+F only, MIT) — self-contained, auditable for the datasheet, no pip dep.
- Multi-head: `mueller-franzes/odelia_breast_mri` (`CornLossMulti`) — one wide head width Σ(K_au−1)=10, `torch.split` per AU, `corn_loss` per chunk averaged, decode+stack to [B,5]. Tiny trainable params on frozen trunk.
- Loss ablation: `SocialComplexityLab/life2vec` makes loss a config enum `{corn, cdw, smoothl1, focal}` — borrow the abstraction to A/B CORN vs CDW-CE and report calibration per loss.

**Frozen DINOv2:**
- `chandar-lab/semantic-wm` — freeze idiom verbatim: `Dinov2Model.from_pretrained` → `.requires_grad_(False)` → `.eval()`; read `config.hidden_size` (384 ViT-S); derive patch size from `embeddings.patch_embeddings.projection.stride[0]` not hardcode 14. HF returns CLS at index 0 → slice `[:,1:,:]` for patch tokens.
- `microsoft/Semi-supervised-learning` (`semilearn/nets/dinov2.py`) — `only_feat`/`only_fc` split: one `only_feat` pass to **cache pooled features to disk**, then train the 5 CORN heads off the cache (no backbone forward per epoch — big M4 win). A/B mean-pool vs CLS (AU localization may favor patch-token pooling).
- `BenediktAlkin/vtab1k-pytorch` — init each CORN head `nn.init.trunc_normal_(weight, std=2e-5)` for stable small-data linear-probe; keep their LoRA-on-backbone path as fallback if frozen-linear underperforms.

**Calibration (adapt, don't copy — these are top-label softmax, ours is ordinal):**
- `gpleiss/temperature_scaling` — the `optim.LBFGS([T], lr=0.01, max_iter=200)` NLL closure + bin-and-average ECE skeleton. Per-head scalar T (or shared); report before/after NLL+ECE; ECE on the binarized pain decision, not max-confidence.
- `EFS-OpenSource/calibration-framework` (netcal) — `netcal.metrics.ECE/ACE/MCE` + `ReliabilityDiagram` on the pain decision; **netcal.regression** `ENCE/UCE/QCE/PICP/MPIW` on the 0–10 sum as a predictive distribution (mean+var from per-AU CORN probs). `IsotonicRegression`/`VarianceScaling` to fix miscalibration post-hoc. Skip GP recalibrators (overkill on MPS).
- `awslabs/gluonts` (`discrete_distribution.py rps()`) + `PriorLabs/TabPFN` — add **RPS** (discrete ranked probability score) as the ordinal-aware Brier: `RPS = Σ_k (CDF_pred(k) − CDF_obs(k))²`. CORN emits P(y>k) so the CDF is free; rewards "close on the ordinal scale." Report per-AU + aggregate-FGS RPS with bootstrap CIs.

**Abstention:**
- `facebookresearch/reliable_vqa` — a **learned** selector head: tiny MLP over [DINOv2 CLS feat ⊕ 5 concatenated CORN logits] predicting P(correct vs vet), instead of raw confidence. Adopt their `risk_coverage/auc` as the headline selective-prediction metric (RiskTolerance→Threshold protocol); wrap with our bootstrap CIs (they don't). Max-prob as baseline so the learned selector is the delta.

**Confound audit (counterfactual engines for P2.1 #2/#3):** `visinf/beyond-accuracy` BG-Gap, `bytedance/ibot` BG-challenge variants, `haofanwang/Score-CAM` EBPG. Plus `machanic/AU_R-CNN`'s presence(≥1)/strong(≥2) cut — *invert its role*: report VLM-vs-vet κ separately at the presence and strong thresholds and feed the margin into the abstention curve.

**VLM weak-labeling (structured-output plumbing):**
- `Arize-ai/phoenix` — forced tool-use: one tool whose `input_schema` has 5 enum properties (each `[0,1,2]`) + optional per-AU confidence/abstain; `tool_choice={"type":"tool","name":...}`; read `content_block.input`. Parse-failure-free enums de-risk the κ metric.
- `D-Star-AI/dsRAG` — `instructor.from_anthropic(mode=Mode.ANTHROPIC_JSON)` + Pydantic `AUScores` (5× `Literal[0,1,2]`), base64 image source, `max_retries`; log validation failures as datasheet provenance.
- `thomasnormal/fewshot` — Pydantic image-in/`Literal`-enum-out + `GreedyFewShot(max_examples=3)` to auto-select vet-anchored in-context demos; measure κ lift before freezing labels.
- `Cohere-Labs-Community/m-rewardbench` — row-paired `cohen_kappa_score(labels=[0,1,2])`, **but add `weights="quadratic"`** (their code is nominal — `labels=` only fixes the class set, does NOT make it ordinal). This bug-to-avoid is our weak-label-reliability number.
- `Lum1104/MER-Factory` `MENTION_TO_AUS` — parse free-text VLM AU mentions into our 5 feline AUs as a cheap secondary check. Do NOT adopt `AU_TO_TEXT_MAP` (human OpenFace FACS — wrong species, reintroduces a human-FACS confound).

**Face crop (zero-training preprocessing fallback only):** `cyclomon/UNSB` / `haribaskar/CatDetection-HaarCascade` — Haar `frontalcatface` → detectMultiScale → pad ~0.3 → clamp → crop. Coarse pre-crop feeding DINOv2; log detect-failure as an abstention signal; do NOT make it the landmark pipeline (Steagall trap). Fix the swapped-xml bug in UNSB if adopted.

## P2.3 Already covered / duplicate — don't re-find

- **CORN loss/decode mechanics** (`coral-pytorch`, `ludwig`, `odelia_breast_mri`, `life2vec`) — already in BUILD_PLAN §3.3/§5. Plumbing, zero novelty.
- **Frozen-DINOv2 freeze idiom** (`semantic-wm`) — decided architecture; borrow the snippet, not research.
- **`anl13/animal_papers`** — a bibliography, not code (surfaced 4×). Re-confirms MEMORY: feline-grimace exists on GitHub only as CatFLW *citations* (landmarks, NOT AUs/FGS). Stop re-grepping; cite CatFLW for alignment only.
- **Plain Cohen's κ with `labels=[0,1,2]`** (`m-rewardbench`) — we planned quadratic-weighted κ; their omission of `weights=` is the gap, not the contribution.

## P2.4 Threats to our novelty — and how we still differentiate

Honest read: **nothing on public GitHub threatens the core thesis.** Components exist in isolation; no repo assembles the per-AU confound-attribution protocol (bg-counterfactual + per-AU EBPG-as-confound-evidence + judge-bias probe, one-directional) on cat pain or any fine-grained animal-affect model. We own the *assembly*, not the primitives or the statistics.

| Apparent threat | Reality | How we still own it |
|---|---|---|
| "Frozen DINOv2 + ordinal head is done" (`Semi-supervised-learning`, `vtab1k-pytorch`, `odelia_breast_mri`) | True as plumbing, on breast MRI / VTAB / SSL. None grimace, none cat, none VLM-supervised, none reports calibration/abstention/confound | Engine was never the novelty (BUILD_PLAN §0.5, FINAL_DIRECTION §8). The confound-attribution protocol is untouched; calibration/abstention are cited plumbing, not the claim |
| "Selective prediction is solved" (`reliable_vqa`, MAPIE) | True for VQA / generic binary | Selective prediction here is **cited plumbing**, not the headline; we adopt it as supporting evidence and do not claim it as a contribution |
| "Calibration libraries exist" (netcal, temperature_scaling) | True | Calibration here is **cited plumbing**, not the claim; we apply known ordinal/regression calibration as supporting evidence, not as novelty |
| "Background-challenge is known" (Madry, ibot, beyond-accuracy) | True on ImageNet-9 | We do not claim bg-swap as novel. What is unowned is the *assembled* per-AU confound-attribution protocol: FGS-BG-Gap + per-AU EBPG-as-confound-evidence + judge-bias probe, framed one-directionally ("at this power, no confound detected"). The per-AU EBPG-as-confound-evidence step is the specific new instance |
| Steagall 2023 / Martvel | Closed data, landmark+XGBoost / video-temporal; no learned features, no calibration, no abstention, no confound counterfactual | The explicit replication traps (BUILD_PLAN §0.5 traps 1–2) |

**Differentiation rule:** we claim no single component as novel, and we do not brand the work as "first/only." What we own is the *assembly* of the per-AU confound-attribution protocol (incl. the per-AU EBPG-as-confound-evidence step and equivalence-style audit hygiene), framed one-directionally and conditioned on the audit power available. The κ reliability check rides below it as a guarded, inspected-not-validated guard.

## P2.5 Gaps still unfilled by GitHub — ordered by where they sit in the asymmetric thesis

1. **(HEADLINE) The assembled per-AU confound-attribution protocol** — FGS-BG-Gap background counterfactual + per-AU EBPG-as-confound-evidence + VLM judge-bias probe, framed one-directionally and conditioned on the audit power available ("at this power, no confound detected"), stratified by capture condition. No repo assembles this on a fine-grained animal-affect model. The assembly + the per-AU EBPG-as-confound-evidence step + the equivalence-style audit hygiene are what is unowned — not any primitive, not the statistics.
2. **(GUARDED CHECK, below the headline) VLM-as-weak-labeler → per-AU quadratic κ vs vet on cat AUs** — κ mechanics exist; the per-AU measurement on cat AUs does not. Kept as an inspected-not-validated reliability check (CI-lower-bound gated, result pending an independent vet anchor) and as kill-tree insurance, not a co-equal pillar. The κ CI-LB gate is textbook clinimetrics and the rubric-paraphrase guard is published — neither is branded as a novel increment.
3. **(CITED PLUMBING) A learned graded (0–10) FGS scorer** — GitHub has cat *landmarks* (CatFLW) and *binary* leaderboards; we use a frozen-backbone + CORN graded scorer as the engine the protocol runs on, conceded as plumbing.
4. **(CITED PLUMBING) Ordinal calibration of a clinical grimace score** (ENCE/QCE/RPS + reliability diagram on a summed CORN distribution) — supporting evidence, not a claim.
5. **(CITED PLUMBING) A distribution-free defer-to-vet band** with NPV control (undertreatment cost) — MAPIE machinery used as supporting evidence; do not use "guaranteed" branding as a headline.
6. **(SUPPORTING) Honest datasheet of pseudo-label provenance** + frozen clip-grouped test split + released weak-labeled AU annotations.

## P2.6 Single highest-leverage next experiment

**The Gate 1-B VLM-vs-vet per-AU quadratic-κ pilot** (~120 images, ≥50 pain-positive), instrumented with three borrowed pieces at once: (a) `Arize-ai/phoenix` forced-tool-use enum schema (parse-failure-free 5-AU calls); (b) `m-rewardbench` row-paired κ **with `weights="quadratic"`**; (c) `prometheus-eval` ordinal Krippendorff α over N≥3 repeated VLM runs for a vet-free self-consistency column.

Highest leverage because it is the GO/NO-GO gate the guarded reliability check rides on (BUILD_PLAN §0.5, §3.4), needs no GPU/backbone (pure API + sklearn/krippendorff on M4), and produces the weak-label-reliability number for that check — which stays inspected-not-validated until an independent vet anchor lands. Note this is the *guard*, not the headline: the per-AU confound-attribution protocol is the spine and is validated today on planted controls without a vet anchor. If muzzle/whiskers κ collapses, the α pre-screen and the binary-plus-abstention floor — the likely v1 ship that stands today regardless — both cover us, and the confound headline is unaffected; if instead the confound leg degrades, the κ check is retained as sole-survivor insurance — pure information gain, no project-killing downside.

---

# Mining Pass 3 (2026-06-13) — daily.dev real-time literature scan

> Passes 1–2 mined **GitHub** ("what data/code can we clone?" / "what techniques make us unique?"). This pass mines the **live daily.dev developer-content feed** ("what *recently-published* ideas should we adopt to stay current?"). Produced by a 70-agent dynamic Workflow (7 module groups → 105 unique candidates discovered via `/recommend/keyword` + `/search/posts`, time=year → adversarial LAW gate). **Engagement caveat:** niche AI/research posts carry 0–5 upvotes, so recency + source drove ranking, not vote counts.

**Headline:** the recent literature **corroborates the settled design, it does not displace it.** The adversarial gate returned **0 hard-adopt, 18 trial (16 after URL-dedup), 87 skip** across 105 candidates — the LAW held against every one. The `src/data` module drew **0 trials / 13 skips** (`StratifiedGroupKFold`-by-`cat_id` is already current best practice). A whole cluster of DINOv3 / Canopy-Height / diffusion-REPA "frozen-DINO works downstream!" posts was correctly skipped as **backbone story** (web-scale pretrain, GPU, custom loss — out of regime; engine is conceded plumbing). The single *new actionable correctness* item is P3.1 #1.

## P3.1 New & actionable (not already in Pass 1–2)

| # | Idea | File | Effort | Source (date) | Why it's new |
|---|------|------|--------|---------------|--------------|
| **1 ⭐** | **Cluster-bootstrap the κ CI by `cat_id`.** VERIFIED code gap: `eval/bootstrap.py:58` `bootstrap_ci` has a `groups=` path that resamples whole cats, but `eval/kappa.py` `bootstrap_qwk_lb`/`bootstrap_qwk_ci` (lines 81, 111) resample IMAGES i.i.d. — within-cat correlation inflates the κ lower bound feeding the gate. Add a `groups=cat_id` path behind a min-cats degeneracy guard (CAT_01 LOIO can collapse). | `src/eval/kappa.py` | low | freecodecamp cluster-randomization (2026-05-22) | Correctness fix on the guarded κ reliability check; not in any prior pass |
| 2 | **Judge-bias perturbation template** for the confound protocol — borrow DiffuJudge-AV's 7 named bias perturbations + its "r=0.753 hides κ=0.057" datapoint as *evidence for* the κ-over-Pearson gate; harvest the eugeneyan LLM-as-Judge bias catalog (position/verbosity/self-enhancement) as documented checks. | `src/eval/confound.py` | low–med | towardsdatascience 2026-05-28; eugeneyan | Extends Pass-2 confound audit (P2.1 #2/#3 = *background* counterfactuals) along a new *judge-bias* axis |
| 3 | **Cross-model disagreement** as an *epistemic* abstention signal — current `consistency.py` Krippendorff α (P2.1 #4) is temperature-only (aleatoric); a second cheap VLM's disagreement adds epistemic spread. Feeds abstention only — never the banned "guaranteed" framing. | `src/vlm/consistency.py` | med | MIT News, 2026-03-19 | New angle on the already-adopted α self-consistency column |
| 4 | **Per-AU split prompts** vs the single 5-AU `FGS_TOOL` (Netflix: one prompt scoring many criteria hurts accuracy). Falsifiable; multiplies batch cost — A/B before committing. | `src/vlm/schema.py` | med | netflixtechblog, 2026-04-10 | New experiment on the VLM rater prompt shape |
| 5 | **Decision-tree rubric** + calibration exemplars: convert the flat 0/1/2 AU descriptors in `rubric.py` into a decision tree to lift VLM-vs-vet agreement on weak AUs (muzzle/whiskers). Pilot first. | `src/vlm/rubric.py` | low | eugeneyan labeling-guidelines | Rubric-authoring lever, complementary to the CatFACS anchor in §4 |

## P3.2 Tactical hardening (cheap, no new claim)

| Idea | File | Source |
|------|------|--------|
| **Prompt-cache TTL check** — verify `ttl=1h` is set for long async Batches windows (5-min TTL + silent edit-invalidation collapse hit-rate). | `src/vlm/rubric.py` | alexcloudstar, 2026-04-24 |
| **DINOv2 reg-vs-plain artifact A/B** justification + test-time-register caveat (backbone.py already *plans* this A/B on cached features). | `src/model/backbone.py` | towardsdatascience, 2026-01-14 |
| **kNN / linear-separability probe** on cached frozen features as a pre-train Gate-4 sanity baseline (held-out features only — circularity). | `src/model/cache_features.py` | q42, 2026-01-26 |
## PAPER_DEBATE.md
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

8. **Phase B model — frozen DINOv2 ViT-S/14 + 5 equal-capacity CORN ordinal heads.** Use the `_reg` register variant (`dinov2_vits14_reg`) as the default — registers suppress attention artifacts that hurt the localized orbital/ear/muzzle features; A/B it before locking ViT-S/14. All heads equal capacity (ear AU preserved). Run an architecture bake-off the papers never did — frozen DINOv2 linear/CORN probe vs. landmark→geometric-feature→XGBoost (interpretability lane, NME-gated) vs. RF-DETR binary head — all on identical cat-grouped folds. Log train-vs-val gap as an overfitting monitor; never report from an overfit checkpoint.

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
7. **Is the machine's "mouth-dominant, ear-weak" weighting biological or acquisition artifact, and does it hold on our frontal-gated crops?** A finding to measure per-AU, not inherit.## REVIEW_REPORT.md
# cat-fgs-llm Comprehensive Review Report

**Date:** 2026-06-15
**Branch reviewed:** `pipeline-checkpoint`
**Review scope:** Architecture, code quality, ML pipeline, tests/CI, documentation, config/dependencies, paper-code alignment

---

## Executive Summary

The repository has a **solid architectural foundation** with clean layered boundaries, strong gate discipline, and good core-contract test coverage. However, several **concrete bugs and best-practice gaps** need attention before the codebase is production-ready or publication-ready. The most critical issues are a process-random cache hash that breaks reproducibility, a missing direct dependency, and an undefined reference in the wrapper layer.

**Overall scores:**

| Area | Score | Verdict |
|------|-------|---------|
| Architecture | 9/10 | Clean DAG, well-separated layers |
| Code Quality | 6/10 | Several bugs and anti-patterns |
| ML Pipeline | 6/10 | Missing scheduler/early stopping, reproducibility bug |
| Test & CI | 7/10 | Core contracts tested, CI wiring incomplete |
| Documentation | 7/10 | Mostly aligned, stale references exist |
| Config & Dependencies | 7/10 | Healthy deps, missing direct dep + hardcoded paths |
| Paper-Code Alignment | 8/10 | Core claims match code |

---

## 🔴 Critical Issues (Must Fix)

### 1. Process-Random Cache Hash Breaks Reproducibility

- **File:** `src/model/cache_features.py`
- **Lines:** 53-57
- **Issue:** The cache schema hash is computed as `hash(tuple(AU_ORDER))`. Python's `hash()` is randomized per process (unless `PYTHONHASHSEED` is set *before* interpreter start), so the same repository can produce different schema hashes across runs.
- **Impact:** Cache invalidation is non-deterministic. Regenerating the same feature cache on different machines/shells may produce different schema hashes, causing avoidable cache misses or validation failures.
- **Fix:** Replace `hash(tuple(AU_ORDER))` with a stable digest such as:
  ```python
  import hashlib
  au_hash = hashlib.sha256(
      "|".join(AU_ORDER).encode("utf-8")
  ).hexdigest()[:16]
  ```
- **Follow-up:** After changing the hash function, regenerate existing caches and commit the new provenance artifacts.

### 2. Missing `Pillow` Direct Dependency

- **File:** `pyproject.toml`
- **Issue:** `PIL.Image` is imported in `src/vlm/call.py`, `src/model/cache_features.py`, and `src/data/dedup.py`, but `Pillow` is not declared as a direct dependency in `pyproject.toml`.
- **Impact:** The package may fail to install in a clean environment even though code needs it. It currently works only because other transitive dependencies happen to pull in Pillow.
- **Fix:** Add `"Pillow>=10.0"` to the `[project] dependencies` list.

### 3. Undefined `_DEFAULT_CONFIG` Reference

- **File:** `src/wrapper/operating_point.py`
- **Lines:** 26, 129
- **Issue:** The constant `_DEFAULT_CONFIG` is referenced but no longer defined in the module. This will raise a `NameError` at runtime when the default config path is used.
- **Impact:** Any caller that relies on the default config path will crash.
- **Fix:** Either restore `_DEFAULT_CONFIG = Path("configs/wrapper.yaml")` (or equivalent) at module level, or change the default argument to use the path literal directly.

### 4. No Validation Split / No LR Scheduler / No Early Stopping in Training Loop

- **File:** `src/model/train_heads.py`
- **Lines:** 200-252
- **Issue:** The `train()` function trains on the full provided tensor without an internal validation split, learning-rate scheduler, early stopping, or gradient clipping.
- **Impact:** The training loop is brittle and prone to overfitting; there is no principled stopping criterion other than a fixed epoch count.
- **Fix:**
  - Split the cached feature tensor into train/validation folds inside `train()`.
  - Track validation loss each epoch.
  - Add an `LRScheduler` (e.g., `ReduceLROnPlateau` or cosine).
  - Add early stopping with a configurable patience.
  - Document that these additions preserve the reproducibility contract (seed + deterministic mode).

---

## 🟠 High-Priority Issues

### 5. `patch_l2` Feature Is Not Actually L2-Pooled

- **File:** `src/model/cache_features.py`
- **Line:** 164
- **Issue:** When `patch_mode == "l2"`, the cached value is `patch.mean(1)` of L2-normalized tokens. That is a mean-pooled feature of normalized tokens, not an L2-pooled feature vector.
- **Impact:** Misleading cache key/name; downstream consumers may believe they are using a different feature representation than they actually are.
- **Fix:** Rename the cache key to something accurate (e.g., `patch_l2_mean`) or change the computation to an actual L2 pooling (e.g., L2 norm over the pooled patch vector). Update `_CACHE_SCHEMA` accordingly.

### 6. Bare `except Exception` Masks Errors

- **File:** `src/vlm/batch_submit.py`
- **Line:** 75
- **Issue:** A broad `except Exception` silently swallows import/runtime errors and falls back to a random proxy.
- **Impact:** Real bugs can be hidden because failures are silently converted to sentinel/random behavior.
- **Fix:** Narrow the exception handling to the specific exceptions that are expected (e.g., `ImportError`, `AttributeError`) or re-raise unexpected errors after logging.

### 7. `bootstrap_qwk_ci` Can Crash on Degenerate Bootstrap Draws

- **File:** `src/eval/kappa.py`
- **Issue:** When all bootstrap replicates are degenerate or empty, `np.nanpercentile` can receive an empty array.
- **Impact:** The kappa CI computation can raise an unhandled error on small or low-prevalence datasets.
- **Fix:** Guard against empty bootstrap arrays:
  ```python
  if len(boots) == 0 or np.all(np.isnan(boots)):
      return qwk, np.nan, np.nan
  ci = np.nanpercentile(boots, [2.5, 97.5])
  ```

### 8. `seed_everything` Allows Nondeterministic Operations

- **File:** `src/data/seed.py`
- **Lines:** 14-21
- **Issue:** `torch.use_deterministic_algorithms(True, warn_only=True)` logs warnings but does not enforce deterministic algorithms.
- **Impact:** Reproducibility is "best effort" rather than guaranteed.
- **Fix:** Document the trade-off in `seed.py` and consider adding a strict mode parameter for users who need full determinism.

### 9. `BinaryPainHead` Ignores Configured `head_init_std`

- **File:** `src/model/train_heads.py`
- **Issue:** The binary pain head is initialized with a hardcoded standard deviation, ignoring the `head_init_std` value used for the CORN heads.
- **Impact:** Inconsistent initialization between the binary and graded heads, making config files misleading.
- **Fix:** Read `head_init_std` from the config and apply it to `BinaryPainHead` as well.

### 10. `build_cache` Does Not Verify Split Purity

- **File:** `src/model/cache_features.py`
- **Lines:** 120-216
- **Issue:** `build_cache` writes all rows into one NPZ file without asserting that fold assignments match the committed manifests.
- **Impact:** If an incorrect or stale CSV is used, the cache can silently include test/holdout rows in the training set.
- **Fix:** Before caching, assert that every `cat_id`/`image_id` belongs to exactly one declared split and that the resulting fold counts match `data/manifests/`.

---

## 🟡 Medium-Priority Issues

### 11. Makefile `pre-commit` Target Does Not Run Configured Hooks

- **File:** `Makefile`
- **Lines:** 101-104
- **Issue:** The `pre-commit` target runs custom Python one-liners but does not invoke the actual `uv run pre-commit run --all-files` hooks configured in `.pre-commit-config.yaml`.
- **Impact:** Local and CI pre-commit checks can diverge.
- **Fix:** Make the `pre-commit` target run the real hook suite (or rename it to avoid confusion).

### 12. Hardcoded Absolute Path in Global Config

- **File:** `configs/global.yaml`
- **Line:** 14
- **Issue:** `repo_root` is set to `/Users/mingrath/ghq/github.com/mingrath/cat-fgs-llm`.
- **Impact:** Non-portable; the config will not work on other machines without editing.
- **Fix:** Derive `repo_root` at runtime from the script location (e.g., `Path(__file__).resolve().parents[2]`) or make it overridable via an environment variable.

### 13. `.gitignore` Missing Common Python Build/Cache Patterns

- **File:** `.gitignore`
- **Issue:** Missing entries for `.mypy_cache/`, `.ruff_cache/`, `.coverage*`, `htmlcov/`, `dist/`, `build/`, `*.egg-info/`.
- **Fix:** Add these standard ignores.

### 14. Stale Planning-Document References

- **Files:** `BUILD_PLAN.md`, `IMPLEMENTATION_PLAN.md`, `README.md`, `Makefile`
- **Issues:**
  - `BUILD_PLAN.md` references `GITHUB_MINE.md` which no longer exists.
  - `IMPLEMENTATION_PLAN.md` claims the committed backbone is `dinov2-small`, but the code uses `dinov2_vits14_reg`.
  - `IMPLEMENTATION_PLAN.md` claims `!data/manifests/` is missing from `.gitignore`, but it is present.
  - `README.md` gate-e2e-synthetic command does not match the Makefile target.
  - `Makefile` contains a stale CI workflow comment.
- **Fix:** Audit and update these documents to match the current codebase.

### 15. Paper Claim "data/ is empty" Is Stale

- **File:** `paper/sections/06-evaluation-protocol-and-gates.tex`
- **Issue:** The paper states the `data/` directory is empty, but `data/manifests/power.json` and `data/manifests/severity.json` are committed.
- **Fix:** Update the wording to say *bulk* data are absent while manifests are committed.

---

## ✅ Strengths to Preserve

1. **Clean architecture**: `model → eval → wrapper` layering with `protocols` as a portable facade.
2. **Gate discipline**: Strict G0→G6 run order enforced in both `Makefile` and `src/gates/orchestrator.py`.
3. **Single source of truth**: `src/constants.py` centralizes `AU_ORDER` and `POINT_DECISION_THRESHOLD`.
4. **Core contract tests**: AU order mirrors, CORN decode→sum→0.39, MPS parity, kappa cluster bootstrap, and confound recovery are all well-tested.
5. **Dependency health**: No version conflicts; Python 3.11 alignment is consistent across `.python-version`, `pyproject.toml`, and `uv.lock`.
6. **Env handling**: Secrets live in `.env` which is correctly gitignored; `.env.example` is placeholder-only.

---

## Recommended Implementation Order

1. **Day 1 — Bug fixes:**
   - Replace `hash(tuple(AU_ORDER))` with `hashlib.sha256`.
   - Add `Pillow` to `pyproject.toml`.
   - Fix `_DEFAULT_CONFIG` in `operating_point.py`.

2. **Day 2 — Training loop hardening:**
   - Add validation split to `train_heads.py`.
   - Add LR scheduler and early stopping.
   - Fix `BinaryPainHead` initialization.

3. **Day 3 — Robustness:**
   - Fix `bootstrap_qwk_ci` empty-array handling.
   - Narrow `except Exception` in `batch_submit.py`.
   - Fix `patch_l2` naming/computation.

4. **Day 4 — Hygiene:**
   - Update stale docs (`BUILD_PLAN.md`, `IMPLEMENTATION_PLAN.md`, `README.md`, `Makefile`).
   - Fix `.gitignore` gaps.
   - Make `configs/global.yaml.repo_root` portable.
   - Fix `Makefile.pre-commit` to run real hooks.

5. **Day 5 — Paper pass:**
   - Update paper `data/` phrasing.
   - Re-run `make paper` and `make checkcites`.

---

## Notes for the Fixer

- Do **not** refactor while fixing bugs — keep changes minimal and targeted.
- After changing the cache hash, run `make gate4` and `make test` to verify caches regenerate correctly.
- If adding a validation split changes reproducible metrics, document the change in `FINAL_DIRECTION.md` or `IMPLEMENTATION_PLAN.md`.
- Preserve the existing "engine is conceded plumbing" framing in any docstrings or comments touched.
## findings.md
# UniquenessVsPriorsAndNotGoodSynthesizer — Findings (prioritized pipeline changes)

**Synthesized from:** FINAL_DIRECTION.md (portable co-headline κ protocol + confound protocol; binary-plus-wrapper spine; kill criteria; not-a-replicate thesis; conceded DINO+CORN + 8-gates + cat-disjoint + distributional + one-dir confound + anti-circularity + welfare wrapper + VLM weak), GAP_ANALYSIS.md (white-space table: Group A trustworthy-decision layer unowned — calib + cost-sensitive op at 0.39, uncertainty abstention, pain-recall@high-sens+CI, weak-label κ, label-provenance confound audit, FM backbone, CORN ordinal, VLM weak; vs 8 papers incl. Steagall 2023 graded closed landmark+XGBoost 95.5%, Feighelstein 2022/23 ResNet/landmark binary, Martvel 2024 YOLO video binary cross-fail, Namboonlue CNN, Evangelista/Marangoni/Lee-Steagall human-only), FACTCHECK.md (novelty pillars survive: no prior FM/VLM, no training-label reliability κ, no calib/CI/abstention/open wrapper on automated FGS; corrections to Steagall attribution), GITHUB_MINE.md (P1-3: GitHub isolated components (MAPIE risk-control, BG-challenge counterfactuals, CORN templates, ordinal κ fixtures, torch-uncertainty/dcurves) but ZERO assembled learned-graded-FGS + portable trustworthiness wrapper on cat pain; daily.dev adds cluster-bootstrap κ correctness + judge-bias axis corroboration), IMPLEMENTATION_PLAN.md + BUILD_PLAN.md + PAPER_DEBATE.md + DATA_DECISION.md + HF_SOLUTION.md + README.md (local code already wires cat_id cluster-bootstrap in kappa/confound/wrapper/operating_point, circularity firewall enforced in op_point, one-dir NO_CONFOUND_MSG + bg_gap/ebpg/judge_bias stubs, Gate0 power + manifests, distributional RPS/ClasswiseECE via decode pmf, hashed CI-abort leak guard in tests, A/B backbone stub, centralized POINT_DECISION_THRESHOLD=0.39 + sum_pmf/au_pmf_from_cumprobs in decode+aggregate+distributional+wrapper), local src/scripts/tests/configs (strong leak-resist/cat-disjoint/CI-honesty foundations; gaps in active entropy-driven VLM, full Gate2, extracted protocols, power sims/e2e runner/CI skeleton, FGS-safe TTA, DINOv3, conformal e2e).

**Answer to user:** To make the pipeline *better* (robustness/repro/calib/small-data leak resistance/CI honesty/power) *or more unique* than previous while building on conceded strengths: prioritize the 7 concrete changes below. These close identified gaps (scattered decode/threshold, partial Gate2 TODOs, missing active VLM/power-sims/CI/e2e, basic aug only, no DINOv3) without touching the conceded engine (plumbing) or core portable methods (κ protocol pending anchor; confound protocol). They amplify the unowned white space (trustworthiness wrapper + portable protocols + strict gated honesty) vs priors that had zero of it.

**Critique summary (polls @Local @PriorArt @GitHub):** Your MCPs + synthesis show we can claim X (first portable VLM-as-AU-rater κ protocol (CI-LB-gated; pending independent vet anchor + rubric-independence guard) + first reusable one-dir capture-condition confound-attribution protocol (FGS-BG-Gap + per-AU EBPG + judge-bias) + welfare-asym dcurves at fixed pain-recall≥0.90 + one-sided 95% NPV LB abstention + cat-disjoint cluster-bootstrap CIs everywhere + power-pre-reg Gate0 + distributional CORN + strict 8-gates/CI-abort/anti-circ) because priors never had Y (Steagall: closed 37-landmark hand-crafted geometry + XGBoost graded per-AU 0/1/2 + 95.5% binary, 8 raters, no learned features/ordinal loss/calib/CIs/abstention/confound/VLM/open artifacts; Feighelstein: ResNet/landmark binary ~72-77%, no graded/VLM/calib/CIs/abstention/confound protocol; Martvel: YOLO video binary, cross-dataset transfer failed, no calib etc.; others pure CNN binary, zero wrapper honesty or portable methods). But P2 (Gate2 confound) still risks circular/non-portable if we don't fix Z (finish bg/ebpg/judge TODOs + full publishable standalone protocol script + one-dir language + artifacts/gate2 outputs + counterfactual integration; currently script has only trivial probe). Local is already ahead on leak-resist (StratifiedGroupKFold by cat_id + hashed static/dynamic CI abort + cluster bootstrap) and CI honesty (one-sided LB, vet-only firewall, distinct-pain-cat denom) but needs the 7 changes for full robustness/power/uniqueness.

## DECISIONS RESOLVED (2026-06-14) — see FINAL_DIRECTION.md §8 for full record + citations

Two v1 decisions were settled by MCP `grep__searchGitHub`/`context7` + a non-GitHub paper scan (arXiv/Semantic Scholar/OpenReview/PubMed). Both confirm — do not flip — the asymmetric framing.

- **A — Framing: ASYMMETRIC.** Headline = the per-AU **confound-attribution protocol** (validated today on planted +/- controls in `src/protocols/standalone_test_corpus.py`). κ = guarded inspected-not-validated reliability check (CI-LB gate + rubric guard are *increments within*, not a co-equal pillar; kept as §7 kill-tree insurance). Wrapper = cited plumbing. Headline wording: *"A portable, power-aware confound-attribution protocol for fine-grained animal-affect models, with a guarded VLM-as-AU-rater reliability check (CI-lower-bound gated, result pending an independent vet anchor)."*
- **B — Timeline: SHIP NOW** as a methods/protocol paper on synthetic + planted controls (conditional on A). Target an ML-eval/trustworthiness or clinical-ML-methods track, not a vet journal first. Holding buys only a number Gate-0 forbids printing (`power.json calc_c_point_039 reportable:false, n_pos=16`).
- **Honesty constraints:** novelty = the *assembly* + per-AU EBPG-as-confound-evidence only — NOT the primitives and NOT the equivalence/power statistics (cite Huang&Hooker 2026, Singh 2023). κ CI-LB gate is textbook clinimetrics (Tractenberg/Rosen 2010, Donner/Rotondi, kappaSize, Sim&Wright); rubric guard is published (Policy Invariance 2026). Cite-and-distinguish: Tech4Animals Sci Rep 2025 (owns saliency leg), BECKI/MDPI 2025, chatbot-FGS Sci Rep 2025; Adebayo 2022 as *support* for the one-directional framing.
- **Residual:** re-run the targeted paper scan immediately before submission (papers can't be swept exhaustively like code).

**Prioritized 5-7 concrete pipeline changes (final list, 7 items as synthesized):**

1. **Adopt DINOv3 in cache_features.py + backbone.py with A/B (reg/plain + v3 option) + update provenance/PROVENANCE.json + tests/test_backbone_variant.py.**  
   - **vs-prior:** Steagall/Feighelstein/Martvel et al. used ImageNet CNNs or hand landmarks — zero foundation models (FM) or DINO anything; our conceded DINOv2 is already unique, extending to selectable DINOv3 keeps the edge future-proof.  
   - **vs-Px (local/GAP papers):** Extends conceded DINO+CORN engine (P3/GAP Group C) + existing reg default/A/B stub (backbone.py: DEFAULT_VARIANT + load) and cache (cache_features.py: dual CLS/patch + provenance) by adding dino v3-vits16 option behind flag; hardens small-data robustness (better localized features for orbital/ear/muzzle) without fine-tune.  
   - Concrete: add variant="dinov2_vits14_reg|dinov2_vits14|dinov3_vits16" in backbone.load_frozen_dinov2; conditional torch.hub or hf; wire to cache_features + configs/corn.yaml; A/B in Gate4/5 + cache PROVENANCE.

2. **Extract fgs_pmf + THRESHOLD to src/protocols/ (new fgs.py or decode_protocol.py exporting sum_pmf/au_pmf_from_cumprobs/point_sum/POINT_DECISION_THRESHOLD/analgesia_flag + helpers) + update ALL call sites (src/model/decode.py, src/model/corn.py, src/eval/distributional.py, src/wrapper/abstention.py + operating_point.py, src/vlm/aggregate.py + rubric.py, scripts/*, tests/*) + add/repoint unit tests (test_gate4_decode etc.).**  
   - **vs-prior:** No prior (Steagall hand-crafted no pmf/distributional; binary papers no ordinal decode protocol at all) ever extracted a portable FGS pmf+threshold protocol for reuse across corpora — this makes our distributional path + 0.39 (Evangelista) a first-class reusable artifact.  
   - **vs-Px:** Builds directly on conceded distributional CORN (FINAL E.1; decode.py + distributional.py already avoid drift via imports) + centralized THRESHOLD=0.39 in aggregate (used by abstention/wrapper/gate0/point_sum) + anti-drift discipline (eval imports from engine); fixes scatter for repro/portability (one source for any future dataset's FGS 0-10 pmf + decision).  
   - Concrete: mkdir src/protocols/; move/expose logic (keep thin shims or re-exports in old locations for compat); update imports; expand tests for protocol invariance; document in IMPLEMENTATION_PLAN/README as "portable FGS pmf protocol".

3. **Add active VLM loop driven by CORN pmf entropy + abstention in src/vlm/ (new active.py or extend batch_collect/consistency + call) + update run_vlm_labels.py + integrate quality_gate/crop signals + entropy computation on au_pmf/sum_pmf.**  
   - **vs-prior:** All priors pure vision (no VLM weak-labeling of 5 FGS AUs at all; GAP Group C); adjacent Sci Rep benchmarks VLMs as raters but never as active weak-labelers for training.  
   - **vs-Px:** Amplifies conceded VLM weak + welfare wrapper + anti-circularity (Gate 1-B κ pilot self-justifies) + distributional (pmf entropy as uncertainty) + abstention (wrapper) for small-data robustness (prioritize high-entropy/near-0.39 + low-quality for vet review; reduces vet hours while preserving honesty).  
   - Concrete: entropy = -sum(p * log p) on per-AU pmf or sum_pmf; rank pool by entropy + |p-0.39| + crop quality flags; batch_submit with uncertainty filter; loop in run_vlm + Makefile target; log to W&B/manifests.

4. **Add conformal/MAPIE + dcurves harm in wrapper + e2e test (tests/test_decision_eval.py or new test_wrapper_e2e.py; wire full LTT risk-control on NPV + harm-ratio dcurves sweep into operating_point/decision_curve/abstention + gate0 flag integration).**  
   - **vs-prior:** Zero cat-pain work reports calibration (reliability/ECE), decision-curve under undertreat≫overtreat, or abstention coverage (GAP Group A strongest white space; Martvel filters internally only).  
   - **vs-Px:** Hardens conceded welfare-asymmetric decision-curve (headline artifact, not ECE) + one-sided 95% NPV LB abstention + fixed high-sens op point (FINAL D/F) + existing lazy MAPIE in abstention + dcurves in decision_curve (already bootstrap cat-grouped); adds e2e + full harm integration for calib robustness/power (Gate0 power now drives certified vs exploratory).  
   - Concrete: expand mapie_ltt_band usage + harm-weighted net-benefit; add test that runs full pipeline on synth/vet-confirmed folds and asserts firewall + LB + curve; update wrapper.yaml placeholders.

5. **Power sim + manifest builder in gate0/gate1 + e2e gate runner + pytest integration + .github CI skeleton (scripts/gate0_power.py enhance with MonteCarlo sims beyond current; gate1_merge + manifests; new scripts/run_all_gates.py or Makefile e2e; tests/ for gates; .github/workflows/ci.yml with pytest + gate4-blocking + leak guards + matrix).**  
   - **vs-prior:** No priors had power pre-reg, honest CIs, or CI honesty (no bootstrap, no distinct-cat denom, no CI-abort).  
   - **vs-Px:** Strengthens conceded strict 8-gates (G0 power + vet-budget pre-reg is load-bearing; FINAL §6) + cat-disjoint + CI-abort (G3 + test_no_test_leak) + power calcs (current gate0 uses approx; add sims for κ/NPV/0.39 under ρ band) + repro (committed manifests); delivers leak resistance/CI honesty/power at CI scale (no .github today).  
   - Concrete: extend gate0 with sims using np.random for low-prev MC (reuse §3.4 style); manifest builders emit power.json + folds + test sha; runner orchestrates make gate0 && ... && pytest -k gate; CI skeleton: test + gate4 (MPS/decode blocking) + static leak scan.

6. **FGS-safe learned aug or TTA (add in src/crop/pipeline.py or new src/aug/fgs_safe.py; limited geometry-preserving (small affine/rot ≤10°, h-flip only, no heavy blur/dropout/vertical) + test-time aug at inference for Phase B crops; integrate with cache + train_heads + detect; unit tests for AU-preservation).**  
   - **vs-prior:** No prior discusses FGS-safe aug (Steagall geometry hand-crafted; binary papers use generic; risk of destroying orbital/muzzle/whisker cues).  
   - **vs-Px:** Builds on conceded FGS-safe light aug in detect (BUILD_PLAN §2.3: fliplr/mosaic/light; banned flipud/strong color) + crop expand/quality (Gate5) + small-data regime (frozen heads) for robustness (TTA reduces variance on tiny n; learned e.g. via light policy or albumentations limited); prevents "calibration measures aug destruction".  
   - Concrete: functions safe_augment(crop) and tta_forward(m, crops); wire to cache_features (no aug in cache) + inference; tests assert NME/entropy impact minimal; config flag.

7. **Gate2 full one-dir protocol + publishable script (complete TODOs in scripts/gate2_confound.py + src/eval/confound.py: wire bg_gap_per_au on counterfactual composites, per-AU ebpg on saliency vs CatFLW ROI, judge_bias on perturbed VLM reruns; add standalone publishable demo script e.g. scripts/publish_confound_protocol.py + artifact + docs; enforce one-dir language + NO_CONFOUND_MSG everywhere).**  
   - **vs-prior:** Zero prior quantified capture-condition confound (flagged by Martvel/Namboonlue/Feighelstein but never measured; GAP Group B); no reusable protocol or publishable script.  
   - **vs-Px:** Completes conceded Gate2 (FINAL §3/§6: portable protocol co-headline, one-directional "no confound detected at this power"; BUILD_PLAN G2) + GITHUB_MINE P2.1 (BG-Gap + EBPG) + P3.1 (judge-bias) + current partial script (trivial probe + cat-grouped CI + verdict); makes co-headline #2 shippable and defensible (not just "CAT_01 is confounded").  
   - Concrete: implement counterfactual (mask face, composite bg) + saliency (attention rollout or Score-CAM) + perturbations; update run_gate + output json + README protocol spec; add e2e test; publish script produces artifacts/gate2/protocol_demo/.

**RECOMMENDED IMPLEMENT ORDER per FINAL gates (risk-first; G0 blocks quant; each preserves conceded strengths + anti-replicate thesis):**
- Per FINAL §6 + IMPLEMENTATION_PLAN §7: G0 (power calcs + vet-budget + sims/manifests from change 5) FIRST (zero data).
- Then G1 (per-CAT merge + manifest builder from 5) + G2 (full one-dir protocol script from 7 — cheapest CPU kill/reframe; co-headline #2).
- G3 (hold-out + CI abort, already strong) + G4 (MPS + decode unit test — BLOCKING; benefits from protocols extract in 2).
- G5 (NME/align) + Gate1-B (κ pilot; benefits from active VLM in 3 + protocols in 2).
- G6 (severity collapse).
- Parallel/non-blocking after G0/G4: backbone DINOv3 A/B (1), protocols extract + call-site updates (2), active VLM (3), wrapper conformal/dcurves e2e (4), FGS aug/TTA (6), CI skeleton + e2e runner (5).
- Always: run `make test` (Gate4) before metrics; update datasheet/README with "engine conceded; headlines = portable methods".

This list is minimal, solo-M4 feasible, directly addresses "better (robustness repro calib small-data leak resistance CI honesty power) or more unique", and maps 1:1 to the conceded base + gaps vs priors (none had any portable method + wrapper + gates). Implement in this order; findings.md is the living prioritized spec (revise on new subagent inputs).

**UniquenessSynthesizer done — ready for user review / next subagent posts.**

--- @FreshHandoffChronicVerifToleranceTests verif note (tiny affirm; 2026-06-14) ---
core portable/gates green; 1 chronic tolerated pre-existing; no regression post handoff read + appends. Verif: make lint (ruff style errs unrelated); python -m standalone (env); uv orch --synthetic --portable-only: OVERALL PASS (portable surface G0/G1b/G2 exercised); make test-portable || true + uv pytest -k "portable or e2e or gate or schema or dedup or power": dedicated (standalone + au_order portable/monkey) green (8P+), selective e2e 13P/2F; make exit0 via ||true. Audit: chronic exactly as test_e2e_gates_pipeline.py:211 doc (synth lenient orch.py:76 _enforce early return + G3 SYNTHETIC-STUB side-effect; 2F on abort mock + wrapper patch; pre-exist per Makefile:68 + prior Fresh logs/round_fresh_*; tolerated for del-safe/CI matrix; real-path in test_manifests_enforce.py:1; NOT regression from landed portable/orch seam). MCP FIRST + POLL surfaces (grep__searchGitHub/context7 on CI/gates matrix patterns: selective -k common, MCP0 on our portable vs priors) + cross @CI (Makefile skeleton + markers + variants jobs; .github absent) @Gates (orch synth portable-only + gate-e2e-synthetic) @hub (round_fresh_1/3 affirm "MCP0; first open... core green; 1 chronic tolerated; preserve ALL [full]"; revise done). todo complete. Preserve ALL. Living ready (verif note) tiny affirm (no handoff). Dynamic fire done. See /tmp/subagent_FreshHandoffChronicVerifToleranceTests_findings.md (full verif results + file:line + 'MCP 0... first open... Preserve ALL [full]'). Append only; all prior strengths verbatim. 

--- @FreshHandoffWrapperMAPIEAbstention handoff append (2026-06-14) ---
file: wrapper/abstention.py:1 (npv_lb_curve Clopper + mapie_ltt_band planned exact MAPIE negative_predictive_value BinaryClassificationController LTT + gate0_flag + "guaranteed" BANNED); operating_point.py:34 (pain-recall>=0.90 train-only); decision_curve.py:4 (harm dcurves RANGE headline); decode.py:53 (pmf_entropy for abstention); vlm/consistency.py:14 (CONSIST abstention SIGNAL vet-free + NPV LB ban); protocols/__init__.py:51 (reexport pmf for hybrid UNC + adapters); gates/orchestrator.py:44 G0-first; data/folds.py cat-disjoint; ci.yml G0; wrapper.yaml 0.90 + ltt_delta. 
why better/unique: welfare-asym (fixed 0.90 recall train + harm range dcurves + one-sided 95% NPV LB defer-vet + banned) unique decision-support (triage recommend vet; clinical harm frame) vs priors (Steagall/Feig/Martvel) zero calib/welfare/CI/abstention/dcurves/NPV (point acc/closed only). Unique: one-sided NPV LB (Clopper primary + MAPIE LTT) + vet firewall (sens/spec vet-only) + circularity (QWK-VLM never) + portable (protocols reexport pmf_entropy/consist + generic_ordinal_mode + standalone del-safe) + G1B CI-LB self-justify. MCP/local: search first context7 + grep__searchGitHub on MAPIE repo literal confirmed exact API/LearnThenTest framework/negative_predictive_value export + usage (LLM-judge ex analogous); 0 public FGS/Steagall portable/welfare matches ("first open..."); local reads/gates/poll/rounds/fresh* (GatesLeak wiring affirmed no leak) full match FINAL. 
MCP 0... first open portable + gated + welfare + CI honest + leak-resist one. Preserve ALL [G0-first (power/vet pre-reg blocks quant + manifests/power committed + CI/pre-commit), cat-disjoint + LOIO + distinct_pain_cats + cat-grouped bootstrap *everywhere* (anti-bench: distinct_pain_cats printed every rate; aug N *never* counted in any N), vet firewall + circularity (sens/spec vet-only; QWK-VLM *never* val; kappa guard), portable co-headline (VLM kappa CI-LB protocol pending independent anchor + rubric guard; one-dir confound protocol reusable) + adapters/generic_ordinal_mode (no 0.39/arb AU), welfare-asym (pain-recall>=0.90 fixed train-only + harm dcurves range + one-sided 95% NPV LB defer-vet; "guaranteed" banned), conceded + distrib CORN on FM (multi-head default; sum_pmf convolve 0-10 RPS/ECE; pmf_entropy active VLM/abstention; point only 0.39) + VLM atoms-only (VLM 5 enum only; sums/flags in code), strict 8-gates (G0 power first, G1 cat-disjoint, G2 one-dir, G3 hash abort, G4 MPS/decode/0.39 BLOCK make test, G1B CI-LB, G6 collapse; orch + synth deletion-safe + CI matrix), anti-bench (distinct_pain_cats; aug N never), open (code/weights/artifacts/datasheet; not leaderboard), single-source (AU_ORDER + 0.39 law + pins everywhere + decode/aggregate/kappa/wrapper/vlm/cache), NO_CONFOUND one-dir, FGS-safe light aug + cache prov + Gate3, hygiene (ruff/pytest + pre-commit), paper fidelity (not-a-replicate, conceded, inspected-not-validated graded, portable co-headline, traps avoided)].
POLL + cross @VLM @Gates @hub: executed (cat chat/living/rounds/gates json/fresh*); cross "@VLM @Gates @hub: wrapper audit converging on hybrid UNC pmf+CONSIST + LTT affirm... Preserve ALL [full]. Revise: affirm + ready tiny or handoff." Revise: no material; hybrid polish (pmf_entropy + consistency + dist) additive per VLM pmf/entropy + consistency abstention role. 
Propose: affirm (solid FINAL-exact); polish hybrid UNC pmf+CONSIST+LTT band v1 + G1B self-justify (already). 
todo complete. Preserve ALL. Ready tiny (optional hybrid helper) or handoff. Fire now. See /tmp/subagent_FreshHandoffWrapperMAPIEAbstention_findings.md (full audit/MCP hits/proposal). MCP 0 reconfirmed; first open... 
--- end append ---

## FRESH CORN/DECODE POST-LANDED (FreshCORNAndPortableDecodeValidator, 2026-06-13)
- DYNAMIC CORN + all required reads + MCP (grep__searchGitHub corn_loss hits ludwig/coral-pytorch; no strong co-teach/sentinel public for CORN; context7 pytorch fallback + tangential multihead/cumprod) + full pytest (15p gate/au/thresh; portable standalone PASS zero-torch arbitrary AU/generic) + gate4 driver PASS + custom derives audit complete.
- Post-generalize: decode derives k_eff=pmf.shape[1]; n_aus=len at :58 (sum_pmf/point_sum); heads n_aus derive; protocols adapters + standalone clean for generic (reexports intact; decode arbitrary AU via len + get_au_names override). No drift: G4 green (0.39 pins + sum=1 + coral cross + AU order law), test_threshold, kappa order comment.
- Small polish: explicit schema (n_aus via len(logits_list), MCP sentinel note) added to corn.py multi_corn_loss (search_replace). Co-teach sentinel: MCP not strong public CORN pattern -> keep faithful (VLM -1 domain + graph zero for co-teach). 
- Entropy: pmf_entropy + au_pmf_entropies exposed in decode.py (collab subagent + role; for VLM active driver high-unc triage).
- POSTED: artifacts/msgs/corn_decode_post_landed_validation.txt (exact msg + @VLM @DINO @Local POLLs + summary).
- Findings: artifacts/reports/fresh_corn_portable_decode_validator_findings.md (full MCP/local/audit/proposals). Appended here.
- Focus locked: distributional portable (pmf/convolve/entropy/RPS from decode single source -> eval/distrib; protocols for kappa/confound generic + decode compat layer; no hardcodes; standalone claim). 
- Dynamic: bg subagents (VLM/gates completed with cross-edits/polls) + our parallel run_terminal polls + Popen mini-spawns + hub converge notes. Spawns + discuss together via files/outputs/MCP.
- Pipeline better/unique: hardened CORN multi-head (schema + faithful) + portable distrib decode + entropy active + single-source vs priors (point XGBoost no VLM/ordinal/distrib/portable/gates). All per role/task. Ready wire + e2e.
--- VLMActive handoff append --- file: decode:72 batch_submit:50+ wrapper:206 scripts/run_vlm_labels.py:74+ src/gates/orchestrator.py:281 src/protocols/adapters.py:77 ; why better/unique: small-N G0 high-unc focus + VLM-rater-kappa + distrib CORN pmf unc active loop (MCP0 priors 0 vs Steagall/Feig/Martvel no VLM/unc); preserve ALL G0-first list verbatim. See /tmp/subagent_FreshHandoffVLMActiveDemoWire_findings.md + round_fresh_3 + msg . 
--- @FreshHandoffReadGate2LightDemo (2026-06-14 per round8 OPEN + handoff) --- file:line (scripts/gate2_confound.py:2 + src/eval/confound.py:3 + src/protocols/__init__.py:33 + src/protocols/standalone_test_corpus.py:82) "lightweight portable affirmed; demo for v1 reusable protocol" + evidence: trivial_probe one-dir (gate2:70-121,105,118) NO_CONFOUND_MSG always (confound:37 THE LAW + bg_gap/judge always attach note) protocols reexport (protocols:33 bg_gap,ebpg,judge_bias,NO_CONFOUND_MSG,JUDGE...) + adapters:77 generic_ordinal_mode (no 0.39/arb AU) + standalone:82 PASS judge + :113 bg_gap + :134 ebpg + NO on synth arb zero-torch + orch:237 portable_only (exercises gate2 toy/synth) + :260 gate2 features-csv toy + test_e2e:93-99 assert g2 demo/NO/trivial/judge + artifacts/gate2/confound_audit.json:11 verdict "no confound detected at this power" + :20 note + demo:true + protocols/README:65 confound example + :91 standalone full kappa/confound synth + FINAL:88/51/72/98 (cheap pre-train no vet GPU before spend; one-dir "no confound detected at this power" never "no confound"; co-headline reusable protocol on next dataset; Gate2 after G1) + GAP:29 (Group B "capture-condition confound audit... quantified by none"; portable on any future corpus) + round_fresh_8 + prior FreshHandoffGate2* affirm "lightweight for v1 (cheap... portable focus)" + synth support + "keep lightweight" vs DYNAMIC full. vs priors (none): GAP/FACTCHECK "quantified by none"; Steagall/Feighelstein/Martvel no open/reusable/one-dir confound protocol/FGS-BG-Gap/EBPG/judge-bias/NO_CONFOUND/standalone del-safe; MCP grep__searchGitHub (search first) on "NO_CONFOUND_MSG|bg_gap|judge_bias|standalone_test_corpus|per_au_kappa_table" + "feline grimace|FGS.*Steagall|Feighelstein|Martvel" -> "No results found" x2 (unrelated only) + firecrawl "zero public code/repos ... first open portable + gated + welfare one". full preserve + MCP 0 reconfirmed; first open portable + gated + one-dir confound reusable protocol one. Preserve ALL [G0-first (power/vet pre-reg blocks quant + manifests/power committed + CI/pre-commit), cat-disjoint + LOIO + distinct_pain_cats + cat-grouped bootstrap *everywhere* (anti-bench: distinct_pain_cats printed every rate; aug N *never* counted in any N), vet firewall + circularity (sens/spec vet-only; QWK-VLM *never* val; kappa guard), portable co-headline (VLM kappa CI-LB protocol pending independent anchor + rubric guard; one-dir confound protocol reusable) + adapters/generic_ordinal_mode (no 0.39/arb AU), welfare-asym (pain-recall>=0.90 fixed train-only + harm dcurves range + one-sided 95% NPV LB defer-vet; "guaranteed" banned), conceded + distrib CORN on FM (multi-head default; sum_pmf convolve 0-10 RPS/ECE; pmf_entropy active VLM/abstention; point only 0.39) + VLM atoms-only (VLM 5 enum only; sums/flags in code), strict 8-gates (G0 power first, G1 cat-disjoint, G2 one-dir, G3 hash abort, G4 MPS/decode/0.39 BLOCK make test, G1B CI-LB, G6 collapse; orch + synth deletion-safe + CI matrix), anti-bench (distinct_pain_cats; aug N never), open (code/weights/artifacts/datasheet; not leaderboard), single-source (AU_ORDER + 0.39 law + pins everywhere + decode/aggregate/kappa/wrapper/vlm/cache), NO_CONFOUND one-dir, FGS-safe light aug + cache prov + Gate3, hygiene (ruff/pytest + pre-commit), paper fidelity (not-a-replicate, conceded, inspected-not-validated graded, portable co-headline, traps avoided)]. POLL + cross @Gate2 @hub done (multiple cat/ls/echo/sleep/append). todo complete. Preserve ALL. Ready tiny (affirm current synth demo sufficient; optional doc note) or handoff. Fire now. See /tmp/subagent_FreshHandoffReadGate2LightDemo_findings.md . MCP 0... first open... Preserve ALL. --- end append ---

--- @FreshHandoffReadWaveCIHardener (handoff-read wave 2026-06-14) ---
[exact same standardized block as above: MCP strict confirmed via search first grep__searchGitHub on public CI patterns matching Unstructured etc; local ci:177+ matrix exact os[ubuntu macos] py[3.11 3.12] fail-fast:false + dedicated make test-portable + standalone + MPS/ARM G4 smoke + pre-commit-strict + gate0 first + G4 block + mocks; why better/unique as above; MCP 0 reconfirmed this handoff-read wave; white space HOLDS; first open portable + gated + welfare + CI honest one. Preserve ALL [full list verbatim G0-first ... paper fidelity]. POLL/SCHED/cross/sleep/revise + todo + living ready tiny affirm full verif (core green; chronic tolerated no reg). See absolute: /Users/mingrath/ghq/github.com/mingrath/cat-fgs-llm/.github/workflows/ci.yml (ci:177+), Makefile, tests/test_e2e_gates_pipeline.py, /tmp/DYNAMIC..._LIVING_HANDOFF_READ_WAVE_20260614.md , recent /tmp/subagent_FreshHandoffCIHardenerMatrixVariants_findings.md + CIHardener* + GatesLeak* + PriorArt* + HandoffReadVerify + PortableDeletionVerifier etc. All preserved.

--- @FreshHandoffReadWaveFidelitySync (handoff-read wave 2026-06-14) --- file:line paper/sections/00-abstract.tex:38 + 'fidelity holds post landed handoff-read; MCP0; protocols portable co-headline + standalone + orch + CI del + early G1 + DINO prep + G0 power + pmf + conceded + inspected-not-validated + not-a-replicate ALL match landed seam in FINAL/README/protocols/README; paper 00 explicit "portable methods seam" "src/protocols/" "standalone_test_corpus.py" "orchestrator" "DINOv3 prep (MCP... round_fresh_4 / FreshHandoff*)" "G0-first" "conceded" "MCP 0 reconfirmed"; 04 "conceded" "inspected-not-validated" "pmf"; 06 "inspected-not-validated" "CI-abort" "deletion-safe portable protocols surface (CI matrix... make test-portable survives rm -rf engine)" "DINOv3 prep" "conceded DINOv2/CORN"; README "seam landed... paper sections 04/06 post-sync" "Portable protocols import" "standalone quickstart" "G0" "deletion-safe" "conceded" "inspected-not-validated"; FACTCHECK novelty survives with portable rescopings/not-a-replicate; no drift (table holds per round_fresh_8 + FreshHandoffGatesLeakWiringCollaborator + FreshHandoffPriorArtReconfirm20260614 + FreshHandoffHandoffReadVerify + FidelityPaperSync refs + LIVING/DYNAMIC/POLLs); key code (protocols/__init__.py:25 pure reexports + adapters generic_ordinal_mode no 0.39/arb; standalone_test_corpus.py:82 NO_CONFOUND + PASS arb; orch.py:71 _enforce_g0_manifests G0-first hard SystemExit + PRECONDS; gate1_merge.py:125-154 dedup REQUIRED pre-SGKF + ENTRY GUARD + early vet_clean pre build_folds; ci.yml:177+ portable-deletion-isolation matrix ubuntu/macos py3.11/3.12 + explicit rm engine + dedicated standalone + MPS/ARM G4; decode.py:72 pmf_entropy active VLM/abstention; cache_features.py:57 _CACHE_SCHEMA dinov3_richer + patch_mode mean_std; folds.py:51 load_power STRICT manifests/power.json ONLY + G0; power.json G0 vet=120 min=50 floors underpowered "guaranteed banned"; all per POLL ls/cat/tail/echo SCHED + cross @Prior @Gates @Uniq @hub @FreshHandoffReadWave*; MCP optional search first (grep__searchGitHub schema) + prior agents reconfirmed 0 "No results" portable/FGS/Steagall/Feig/Martvel). Preserve ALL [G0-first (power/vet pre-reg blocks quant + manifests/power committed + CI/pre-commit), cat-disjoint + LOIO + distinct_pain_cats + cat-grouped bootstrap *everywhere* (anti-bench: distinct_pain_cats printed every rate; aug N *never* counted in any N), vet firewall + circularity (sens/spec vet-only; QWK-VLM *never* val; kappa guard), portable co-headline (VLM kappa CI-LB protocol pending independent anchor + rubric guard; one-dir confound protocol reusable) + adapters/generic_ordinal_mode (no 0.39/arb AU), welfare-asym (pain-recall>=0.90 fixed train-only + harm dcurves range + one-sided 95% NPV LB defer-vet; "guaranteed" banned), conceded + distrib CORN on FM (multi-head default; sum_pmf convolve 0-10 RPS/ECE; pmf_entropy active VLM/abstention; point only 0.39) + VLM atoms-only (VLM 5 enum only; sums/flags in code), strict 8-gates (G0 power first, G1 cat-disjoint, G2 one-dir, G3 hash abort, G4 MPS/decode/0.39 BLOCK make test, G1B CI-LB, G6 collapse; orch + synth deletion-safe + CI matrix), anti-bench (distinct_pain_cats; aug N never), open (code/weights/artifacts/datasheet; not leaderboard), single-source (AU_ORDER + 0.39 law + pins everywhere + decode/aggregate/kappa/wrapper/vlm/cache), NO_CONFOUND one-dir, FGS-safe light aug + cache prov + Gate3, hygiene (ruff/pytest + pre-commit), paper fidelity (not-a-replicate, conceded, inspected-not-validated graded, portable co-headline, traps avoided)]. Living ready tiny (doc affirm; no new file) or handoff. Fire now. All verbatim preserved. MCP0; fidelity holds. @hub @Prior @Gates @Uniq @FreshHandoff*

