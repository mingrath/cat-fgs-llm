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
