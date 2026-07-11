# Local line references
## README.md
     1	# cat-fgs-llm
     2	
     3	**This is not another cat-pain detector.** Its headline deliverable is a portable,
     4	power-aware **confound-attribution protocol** for fine-grained animal-affect models —
     5	a one-directional, per-AU audit (FGS-BG-Gap counterfactual + per-AU EBPG
     6	saliency-as-confound-evidence + VLM judge-bias probe) that any future facial-pain-scorer
     7	corpus can run. It ships with a **guarded VLM-as-AU-rater reliability check**
     8	(per-AU VLM-vs-vet quadratic kappa, CI-lower-bound-gated; a result pending an
     9	independent vet anchor), on an explicitly-conceded DINOv2 + CORN engine, with an honest
    10	binary-plus-abstention floor (welfare-asymmetric operating point at pain-recall >= 0.90,
    11	one-sided 95% NPV defer-to-vet curve — cited supporting plumbing, not a headline) that
    12	stands even though the 0-10 layer is reported as inspected-not-validated.
    13	
    14	The novelty is the **assembly** plus the per-AU EBPG-as-confound-evidence step plus the
    15	instantiation of equivalence-style audit hygiene — never any individual primitive
    16	(bg-swap, saliency, the one-directional / power-conditioned statistics are all prior
    17	work) and never the kappa CI-lower-bound gate (textbook clinimetrics) or the
    18	rubric-paraphrase guard (published).
    19	
    20	**Status of the legs:** the confound protocol is the strong leg — it validates today on
    21	planted positive and negative controls (`src/protocols/standalone_test_corpus.py`). The
    22	kappa reliability check is the guarded second leg: a *protocol with a result pending* —
    23	it cannot report a number until the per-AU vet anchor exists (see Gate 0). The current
    24	dataset is binary pain/no_pain and **cannot** yield 0/1/2 AU ground truth, so until the
    25	anchor is built, the kappa check ships as a runnable protocol, not a finding. Kappa is
    26	not deleted: it is kill-tree insurance — the sole-survivor headline if the confound leg
    27	degrades. The binary-plus-abstention floor is what stands today as the engineering spine;
    28	treat it as the likely v1 ship, not the fallback.
    29	
    30	## The engine is conceded plumbing, not claimed novel
    31	
    32	The frozen DINOv2 ViT-S/14 + 5 per-AU CORN heads -> 0-10 sum -> 0.39 decision engine
    33	(`src/model/`) is **plumbing, explicitly conceded as not-novel**. Directory names,
    34	module docstrings, artifact tags, and run names never imply the engine is the
    35	contribution. Its outputs feed only `src/wrapper/` (the welfare frame, cited supporting
    36	plumbing) and `src/eval/` (the confound-attribution headline + the guarded kappa check).
    37	The v1 spine is **binary pain/no_pain + wrapper**; the
    38	0-10 layer is built, decoded, and **inspected-not-validated** — it never emits a
    39	validated-claim number, and QWK-vs-VLM is never validation.
    40	
    41	**Backbone note:** the `_reg` (register) DINOv2 variants are now the field default
    42	(`dinov2_vits14_reg`) because registers suppress attention artifacts that hurt dense,
    43	localized features — and per-AU FGS scoring is exactly localized (orbital, ear, muzzle
    44	sub-regions). A/B the reg variant before locking ViT-S/14. Expect orbital/ear/head to
    45	be where the Gate 1-B kill-switch fires: a small frozen backbone with no fine-tuning on
    46	a tiny corpus is most likely to miss those AUs, so budget for binary-plus-abstention
    47	being the v1 ship rather than the fallback.
    48	
    49	## The portable artifacts (all dataset-agnostic)
    50	
    51	1. **FGS-BG-Gap + per-AU EBPG confound-attribution protocol (THE HEADLINE)**
    52	   (`src/eval/confound.py`) — a one-directional, power-conditioned audit ("no confound
    53	   detected at this power") any future facial-pain-scorer corpus can run. Combines a
    54	   background-swap counterfactual, per-AU EBPG saliency-as-confound-evidence, and a VLM
    55	   judge-bias probe. Validated today on planted positive and negative controls
    56	   (`src/protocols/standalone_test_corpus.py`). The contribution is the assembly + the
    57	   per-AU EBPG-as-confound-evidence step + the equivalence-style audit hygiene; the
    58	   underlying primitives and the one-directional / power-conditioned statistics are prior
    59	   work and are cited, not claimed.
    60	2. **VLM-as-AU-rater kappa reliability check (GUARDED, inspected-not-validated)**
    61	   (`src/vlm/`, `src/eval/kappa.py`) — scores any face corpus's per-AU VLM labels against
    62	   a vet anchor; reports 5 quadratic kappa with CI lower bounds. Fires on the CI lower
    63	   bound (Gate 1-B). This is a guarded reliability check ranked below the confound
    64	   protocol, NOT a co-equal contribution; the CI-lower-bound gate is standard
    65	   clinimetrics and the rubric-independence guard is published, so neither is branded as
    66	   novel. **A result requires an independent vet anchor that does not yet exist**; the
    67	   artifact shipped today is the protocol. **Interpretation guard:** a high kappa only
    68	   measures capability if the anchor is independent and the rubric handed to the VLM is
    69	   not the same rubric the vet scored from — otherwise it measures rubric-following, not
    70	   weak-labeling skill. State which one a given run measures. Kept as kill-tree insurance
    71	   (sole-survivor headline if the confound leg degrades).
    72	3. **Welfare-asymmetric decision curve + one-sided-95%-NPV abstention curve (CITED
    73	   SUPPORTING PLUMBING, not a headline)** (`src/wrapper/`) — operating point at fixed
    74	   pain-recall >= 0.90 with the undertreat:overtreat harm ratio swept as a range, plus a
    75	   defer-to-vet boundary with finite-sample lower bounds (the word "guaranteed" is
    76	   banned).
    77	
    78	## Gate run-order (strict; each gate blocks downstream)
    79	
    80	Each gate writes an immutable artifact; no downstream number is believed until its
    81	prerequisite gate's artifact exists and passes.
    82	
    83	```
    84	G0  gate0_power      power calcs + vet-budget integer (no data, no GPU) — blocks all quantitative work
    85	G1  gate1_merge      per-CAT merge vs CAT_ ids (not raw CLIP/pHash); cat-disjoint folds
    86	G2  gate2_confound   one-directional capture-condition audit
    87	G3  gate3_holdout    frozen hashed cat-disjoint hold-out + CI abort
    88	G4  gate4_mps_check  MPS<->CPU logit parity + CORN decode->sum->0.39 (BLOCKING; `make test`)
    89	G5  gate5_nme        alignment / NME / face-pixel-resolution audit
    90	G1-B kappa pilot     run_vlm_labels + eval/kappa (fires on CI lower bound; self-justifies the labeler)
    91	G6  gate6_severity   AU=2 severity-cell collapse decision
    92	```
    93	
    94	Run via the Makefile: `make gate0`, `make gate1`, ..., plus `make test` (Gate 4 must
    95	be green before any quantitative target runs). The central orchestrator
    96	(`make gate-orchestrate` / `gate-pipeline` / `gate-e2e-synthetic --synthetic`)
    97	enforces strict order + artifacts + abort (G0 first). Use `make test-portable`
    98	for the deletion-safe protocols surface (zero-torch import isolation + synthetic
    99	arbitrary-AU exerciser).
   100	
   101	**Portable protocols import (the citable surface — confound headline + guarded kappa check):**
   102	```python
   103	from src.protocols import (
   104	    per_au_kappa_table, bootstrap_qwk_lb, bg_gap, judge_bias,
   105	    get_au_names, map_df_columns, generic_ordinal_mode, FGS_AU_NAMES,
   106	)
   107	from src.protocols.adapters import apply_au_override
   108	# Standalone / deletion-safe test: python -m src.protocols.standalone_test_corpus
   109	```
   110	**src.protocols standalone quickstart** (see also `src/protocols/README.md:17` for full examples + `python -m src.protocols.standalone_test_corpus` deletion-safe zero-torch proof after `rm -rf` engine; adapters/generic for non-FGS reuse):
   111	```python
   112	from src.protocols import (
   113	    per_au_kappa_table, bootstrap_qwk_lb, bg_gap, judge_bias,
   114	    get_au_names, map_df_columns, generic_ordinal_mode, FGS_AU_NAMES,
   115	)
   116	from src.protocols.adapters import apply_au_override
   117	```
   118	See `src/protocols/` (adapters for generic/non-FGS reuse; standalone corpus
   119	exercises with no engine leak). The confound-attribution headline and the guarded kappa
   120	check are both first-class importable + gate-enforced (see FINAL_DIRECTION.md, paper
   121	sections 04/06 post-sync, Makefile test-portable / gate-e2e-synthetic, and
   122	src/gates/orchestrator.py). DINOv3 prep is staged for an engine successor.
   123	
   124	**Kill/pivot:** if orbital/ear/head kappa lower bound falls below the Gate 1-B floor,
   125	drop the 0-10 layer entirely and ship calibrated binary + abstention + the confound
   126	audit; the confound protocol carries the headline alone and the guarded kappa check is
   127	recorded as not-met. The confound protocol is the leg the paper rests on; kappa is the
   128	insurance.
   129	
   130	## Environment
   131	
   132	uv-managed, Python 3.11 (system 3.9.6 is not used). Local = Apple M4 / MPS, **no
   133	local CUDA**; CUDA is confined to `notebooks/colab_train_rfdetr.ipynb` for detector
   134	training only. All code runs through `uv run`. See `pyproject.toml` and
   135	`configs/*.yaml` (all hyperparameters and seeds live in configs, never hardcoded).
## FINAL_DIRECTION.md
     1	# Finalized direction — build something that isn't just another cat-pain detector
     2	
     3	**Date:** 2026-06-12
     4	**Status:** AUTHORITATIVE. This document is the reconciled output of a multi-agent direction debate (8 doc-readers → 7 critic personas → moderator → 3 red-teamers → lead synthesis). Where it conflicts with [BUILD_PLAN.md](BUILD_PLAN.md), **this supersedes** — BUILD_PLAN has been edited to point here and carries the surgical deltas inline. Grounded in [GAP_ANALYSIS.md](GAP_ANALYSIS.md), [GITHUB_MINE.md](GITHUB_MINE.md) (Pass 2 novelty mining), [FACTCHECK.md](FACTCHECK.md), and the `novel-contribution` memory.
     5	
     6	---
     7	
     8	## 1. Do I agree with the direction?
     9	
    10	**Agree-with-changes.** The direction survives all three red-team passes, but only after it stops letting the engine (DINOv2+CORN) and the off-the-shelf wrapper tools carry novelty weight they cannot bear. The strongest point across the debate is the **replication red-team's verdict**: the not-a-replicate thesis is load-bearing in exactly two places — the **per-AU VLM-vs-vet κ protocol whose result is pending the independent vet anchor** (VLM-as-AU-rater) and the **confound audit sold as a transportable attribution protocol** — and cosmetic everywhere else; strip those two and v1 is "frozen DINOv2 + binary pain head + standard reliability metrics," i.e. the Feighelstein/Martvel detector we swore not to build. The **feasibility and statistics red-teams converge on one wall**: every quantitative claim (κ, 0.39, abstention NPV, 5×5 correlation, severity cells) draws on the *same* ~50-positive vet account, never budgeted, so "guaranteed NPV," "validated graded," and the "empirical correlation fix" are arithmetically undeliverable at this n. The fix is decisive and cheap: **headline the two as portable methods (the κ one a protocol whose result is pending the independent vet anchor), concede the engine as plumbing, re-center the wrapper on a welfare-asymmetric loss, and demote graded to an inspectable-but-not-validated artifact.** With those moves the plan is honest, solo-deliverable on M4, and not a replicate.
    11	
    12	## 2. Kept (consensus)
    13	
    14	- **Wrapper-as-frame, accuracy-as-anti-benchmark** — every persona, including Reviewer 2, endorses it as the only unowned, solo-feasible framing; never write "we beat 77/79/95%" (BUILD_PLAN §0.5, ruling 6.12).
    15	- **Circularity firewall (ruling 6.e / §3.3)** — sens/spec at 0.39 estimated ONLY on vet-confirmed labels; QWK-vs-VLM is never validation. Named by all 7 as the methodological backbone.
    16	- **Gate-2 confound audit as an early action** — cheap, no vet, no GPU; can kill or reframe the corpus before any spend, and publishes either way.
    17	- **Gate-3 frozen hashed cat-disjoint hold-out with CI-abort** — structurally defeats post-hoc goalpost moves; reviewers reward it.
    18	- **Per-cat (not per-clip) grouping + always print the distinct-pain-cat denominator + Clopper-Pearson/bootstrap CIs; augmented copies never enter any reported N** (Gate 1, §4, ruling 6.a).
    19	- **Frozen DINOv2 ViT-S + shallow CORN heads on cached features** — correct low-data/MPS choice; default to the `_reg` register variant (`dinov2_vits14_reg`), which suppresses attention artifacts that hurt dense localized per-AU features (orbital/ear/muzzle), and A/B it against plain ViT-S/14 (kept as backwards-compatible) before locking the backbone; full fine-tune at ~264 faces would be malpractice; CORN over CORAL/softmax is right.
    20	- **Gate-4 MPS compute-correctness pre-checks are BLOCKING** — a leak-proof PR-AUC from silently-wrong MPS logits is worthless.
    21	- **Binary-plus-abstention as the likely v1 ship / the floor that stands today** — it stands now and is the v1 spine; if muzzle/whiskers κ collapses, the graded layer (upside conditional on Gate 1-B) is dropped and the calibrated binary + abstention floor remains; no project-killing null.
    22	- **Replication-trap discipline (§0.5 traps 1–8)** — no Steagall landmark+XGBoost, no Martvel video/temporal, no external-cohort claim on single CAT_01, no multi-rater ICC with one vet, horse set as decode scaffolding only.
    23	- **Decision-support triage framing, never an autonomous analgesia trigger** — output is "grimace consistent with pain, X/10; recommend vet assessment"; negative class documented as an unknown (possibly sedated/post-op) mixture.
    24	
    25	## 3. Changed (resolved conflicts)
    26	
    27	**A — Headline. RESOLVED: co-headline the two as PORTABLE METHODS (the κ one a protocol whose result is pending the independent vet anchor); concede the engine; the wrapper is the frame, not the claim.**
    28	Promote (i) "per-AU VLM-as-AU-rater κ protocol vs vet (result pending the independent vet anchor)" framed as *can a frozen VLM weak-label feline FGS AUs at human-rater agreement* (a forward-looking method finding, not an audit of CAT_01's specific labels; a high κ measures that weak-labeling capability only if the vet anchor is independent AND the VLM rubric differs from the vet's rubric — otherwise it measures rubric-following), and (ii) "first capture-condition confound-attribution protocol (FGS-BG-Gap + per-AU EBPG)" framed as *reusable on the next dataset*. Demote calibration/ECE/decision-curve/MAPIE to "supporting evidence." Explicitly state in the paper that the DINOv2+CORN engine is NOT claimed as novel. *Forced by Replication-attack Holes 1, 2, 3, 6:* single-source data means non-portable contributions are dead on arrival; only method-portability and a conceded engine survive the "swap-the-backbone + extra metrics" filing.
    29	
    30	**B — Graded vs binary spine. RESOLVED: binary-plus-wrapper IS the v1 spine, stated in the abstract; graded is a built-but-NOT-VALIDATED artifact, not a "gated upside we expect to fire."**
    31	The abstract's headline must hold with graded output dropped. *Forced by Feasibility-attack §C and Statistics-attack #6/red-team #1–2:* at n≈120 the severity tail is unestimable and B and C collapse into one decision, so frame graded as "v2-pending-more-data," not as upside likely to be realized.
    32	
    33	**C — Independent graded ground truth. RESOLVED: CUT the gold-set branch entirely.**
    34	Do NOT chase Steagall/Zamansky/CatFLW. *Forced by Feasibility-attack §C:* CatFLW is landmarks/bboxes (a category error as a graded anchor), and the request-only FGS sets are unbounded-dependency / weeks-to-never on a solo timeline. Re-write as: "we will NOT headline a validated graded claim; graded-CORN ships as an internally-consistent, VLM-anchored, qualitatively-inspected exploratory layer with the circularity stated." This removes an unbounded external dependency from the critical path and is *more* honest. **Word "graded" struck from every validated-claim sentence.**
    35	
    36	**D — Operating point. RESOLVED in favor of the clinician: fixed-high-sensitivity, NOT Youden-J/F1, and quantify the welfare loss.**
    37	Select the cutoff at pain-recall ≥0.90 (Evangelista anchor) inside train folds, report the specificity it buys with bootstrap CIs. **Make the harm-ratio-weighted decision curve the wrapper's headline artifact, not ECE** — sweep undertreat:overtreat as a *range* across the dcurves threshold-probability axis (we have no vet-elicited ratio). *Forced by Clinician position + Replication-attack Hole 4:* a symmetric-cost knee is the wrong loss for a welfare instrument, and generic calibration is hygiene whereas a welfare-loss-weighted operating point is a clinical-decision-theory contribution no prior cat-pain paper made. Correct the stale §3.3 Youden-J/F1 line.
    38	
    39	**E — CORN path. RESOLVED: commit the distributional path in the spine; SPLIT off the correlation fix.**
    40	- *E.1 (adopt unconditionally):* move soft `P(rank>k)` → per-AU pmf → convolve to a pmf over the 0–10 sum → **RPS-on-the-sum (single scalar, bootstrap CI) + per-AU ClasswiseECE** into BUILD_PLAN §3.3 as the default. Relegate argmax-sum to the 0.39 point decision only. **Do NOT print a binned reliability diagram on the 11-atom sum** — degenerate at ~11/atom (per-bin SE ±0.18–0.26). *Forced by ML-methods-critic + Statistics-attack #5:* the current spine specifies a statistically incoherent path while promising metrics it cannot support.
    41	- *E.2 (cut the empirical 5×5 matrix):* replace with a **2-point ρ sensitivity band** — run the §3.4 Monte-Carlo under ρ=0 and a pinned ρ=0.3 (uniform), report GO only if the decision holds under BOTH. *Forced by Feasibility-attack §E + Statistics-attack #7 + red-team #5:* a 10-off-diagonal matrix at n≈50 has per-entry CIs ~[−0.4,+0.6], launders noise as data-driven rigor, and may not be positive-definite. A pinned ρ is honest; a noisily-fit matrix is false rigor.
    42	
    43	**F — Abstention. RESOLVED: DELETE the word "guaranteed"; ship a one-sided 95% NPV lower-bound curve; 0.39 is CI-first exploratory.**
    44	Report the abstention curve as "NPV with a one-sided 95% LB at each abstention rate," shown to clear ≥0.90 only at honestly-reported abstention rates. *Forced by Statistics-attack #4 (the sharpest overclaim):* certifying NPV≥0.95 needs ~60 zero-error abstained-in negatives out of ~70 total → abstention rate collapses to ~0 → vacuous instrument. **Run the LTT/MAPIE sample-size power calc BEFORE any vet spend** (Gate 0); if the budget can't certify a useful band, the curve is exploratory, full stop.
    45	
    46	**G — Severity-cell gate. ADOPTED, but COLLAPSE not caveat.**
    47	Pre-commit: unless the anchor delivers double-digit AU=2 cells, collapse the high end (merge AU 1+2, or report only painful/not above threshold). *Forced by Statistics-attack #6:* AU=2 sens CI spans [0.35,0.97] at single-digit n — that is no information, and a caveat leaves an anchoring number in a table; it must be a pre-registered structural decision.
    48	
    49	**Gate-1-B fires on the κ CI LOWER BOUND, not the point estimate** (Statistics-attack #1): a floor of 0.6 is indistinguishable from a true 0.47 at this n, so a point-estimate gate is not a gate. Pre-register the κ floors *with* the kappaSize power calc backing them.
    50	
    51	**Confound audit is ONE-DIRECTIONAL** (Statistics-attack #2): well-powered to *detect* confounding (AUC 0.65→z≈2.2 at n=50), underpowered to *rule it out* — report "no confound detected at this power," never "no confound."
    52	
    53	**Housekeeping (uncontested):**
    54	- **Self-justify the labeler via the κ pilot** ("we selected the VLM by measured per-AU κ on the anchor") and drop the unretrievable Sci Rep 2025 "only Claude acceptable" citation (FACTCHECK C54) — removes the phantom-citation dependency at the root of headline #1 (Replication Hole 7).
    55	- **Fix the COSMIN line**: AI scoring is out-of-topic-scope of Lee & Steagall 2026 (C12) — do NOT claim the review "named calibration/CIs" or "excluded automated scorers." (Already corrected in BUILD_PLAN §0.5.)
    56	
    57	## 4. Cut / descoped
    58	
    59	| Cut | Trigger to re-add |
    60	|---|---|
    61	| **Validated "graded 0–10 FGS instrument" claim** (C). Graded-CORN ships as an inspectable artifact only. | An *independent* per-AU 0/1/2 gold held-out set arrives AND AU=2 cells reach double digits. Not on the solo timeline. |
    62	| **Gold-set acquisition (Steagall/Zamansky/CatFLW)** as a critical-path dependency (C). | A request-only set clears its DUA with a bounded SLA — treat as v2 windfall, never a blocker. |
    63	| **The word "guaranteed"** on the NPV band (F). | A power calc shows the budgeted n certifies NPV≥0.90 at ≤40% abstention. |
    64	| **Empirical 5×5 cross-AU error-correlation matrix** (E.2). | n into the low hundreds of vet rows with non-single-digit cells. Until then: 2-point ρ sensitivity band. |
    65	| **Binned reliability diagram on the 0–10 sum** (E.1). | ~24+ samples/atom (n into the high hundreds). Until then: RPS-on-the-sum + per-AU ClasswiseECE. |
    66	| **Per-cell AU=2 (severe) calibration number** (G). | Anchor yields double-digit AU=2 cells. Until then: collapsed high-end scale. |
    67	| **Engine novelty claim** (A) — conceded as plumbing permanently. | Never. A conceded swap-the-backbone delta is harmless; an oversold one is the reviewer's favorite kill. |
    68	| **Any horse number in abstract/results/transfer table** (trap 7). | Never — methods/appendix sentence only ("we unit-validated the decode→sum→threshold path on 5-horse genuine 0/1/2 labels"). |
    69	
    70	## 5. The not-a-replicate thesis
    71	
    72	> **This is not another cat-pain detector: its headline deliverables are two transportable methods no prior feline-pain work produced — a protocol for measuring whether a frozen VLM can weak-label FGS action units at human-rater agreement (per-AU VLM-vs-vet quadratic κ, CI-lower-bound-gated; result pending the independent vet anchor), and a reusable capture-condition confound-attribution protocol — shipped on an explicitly-conceded DINOv2+CORN engine, with an honest binary-plus-abstention floor (welfare-asymmetric operating point at pain-recall ≥0.90, one-sided-95%-NPV defer-to-vet curve) that stands even though the graded layer is reported as inspected-not-validated.**
    73	
    74	**Three concrete artifacts that prove it (all dataset-agnostic / portable):**
    75	1. **VLM-as-AU-rater κ protocol** — released code that scores any face corpus's per-AU VLM labels against a vet anchor and reports 5 quadratic κ with CI lower bounds; a result requires the independent per-AU vet anchor that does not yet exist (Gate 0), and counts as weak-labeling capability only if that anchor is independent and the VLM rubric differs from the vet's rubric — otherwise it measures rubric-following. (The portable method, immune to "routine hygiene.")
    76	2. **FGS-BG-Gap + per-AU EBPG confound-attribution protocol** — a one-directional audit any future facial-pain-scorer corpus can run; deliverable is the *protocol*, not "CAT_01 is confounded."
    77	3. **Welfare-asymmetric decision-curve + one-sided-95%-NPV abstention curve** — operating point at fixed high sensitivity with the undertreat:overtreat harm ratio swept as a range, plus a defer-to-vet boundary reported with finite-sample lower bounds (no "guarantee" word). This is the clinical-decision contribution that distinguishes the wrapper from generic calibration.
    78	
    79	## 6. Build sequence (risk-first, reconciled with BUILD_PLAN gates)
    80	
    81	**Gate 0 — Power calcs + vet-budget pre-registration (NEW, blocks everything quantitative; ~1 afternoon, zero data, no GPU).**
    82	- Goal: write the *single integer* — how many faces the vet scores, per-AU, in how many sittings — and run three zero-data power calcs: (a) faces for per-AU κ CI half-width ≤0.15 (kappaSize); (b) faces for LTT/MAPIE-certified NPV≥0.90 at ≤40% abstention; (c) whether the 0.39 CI is reportable at the budgeted n.
    83	- Exit: budget integer committed; A's magnitude claim and F's "guarantee" demoted *on paper* now if (a)/(b) fail.
    84	- Why first: Feasibility/Statistics red-teams show this is the project's single point of failure dressed as five resolutions. **This precedes everything in BUILD_PLAN §0.5 "First 3 actions."**
    85	
    86	**Gate 1 — Per-CAT merge (BUILD_PLAN Gate 1, unchanged; keystone).** Goal: collapse clips to true individuals, validated against trusted `CAT_` IDs (CLIP/pHash are duplicate detectors, not re-ID). Exit: no individual straddles a fold; distinct-pain-cat denominator printed. Compute: CPU. Why: blocks all grouping/CV.
    87	
    88	**Gate 2 — Capture-condition confound audit (BUILD_PLAN Gate 2; CHANGED: now framed as a portable protocol + one-directional).** Goal: trivial brightness/blur/aspect/CLIP classifier predicts pain → quantify via FGS-BG-Gap. Exit: detect-or-not result logged; if it can only produce "CAT_01 is confounded," demote to a threat-to-validity paragraph; if it produces a transportable protocol, it co-headlines. Compute: CPU, no vet. Why: cheapest kill/reframe; runs before any spend.
    89	
    90	**Gate 3 — Frozen hashed cat-disjoint hold-out + CI-abort (BUILD_PLAN Gate 3, unchanged).** Exit: fold CSV + test ids hashed; CI aborts any run that can read the test manifest. Compute: CPU.
    91	
    92	**Gate 4 — MPS compute-correctness pre-checks (BUILD_PLAN Gate 4, unchanged, BLOCKING).** Exit: MPS-vs-CPU DINOv2 logit parity; 1-epoch CORN smoke test; synthetic CORN-decode→sum→0.39 unit test passes. Compute: M4/MPS. Why: silent-correctness gate before any scoring.
    93	
    94	**Gate 5 — Alignment/NME gate before Phase B (BUILD_PLAN Gate 5, unchanged).** Exit: CatFLW-landmark NME inside RF-DETR crop vs eye-aligned crop, and median face-pixel resolution, both acceptable; else add 2-point eye-similarity alignment. Compute: M4. Why: misaligned crops make "calibration" measure crop quality.
    95	
    96	**Gate 1-B — Weak-label reliability κ pilot (BUILD_PLAN Gate 1-B; CHANGED: fires on CI lower bound; self-justifies the labeler).** Goal: ~120-image (≥50 positive, per Gate 0 budget) VLM-vs-vet per-AU quadratic κ with CIs; the pilot itself selects the VLM. Exit: GO if the κ **CI lower bound** clears the pre-registered, power-backed floor (orbital/ear/head ≥0.6; muzzle/whiskers 0.4–0.6 caveat). Compute: hosted-API inference + CPU. Why: the headline #1 protocol (result pending the independent vet anchor); runs alongside §3.4 Monte-Carlo (now 2-point ρ band).
    97	
    98	**Gate 6 — Severity-cell count gate (NEW, after the anchor; pure counting).** Exit: if AU=2 cells single-digit → collapse high-end scale (pre-committed). Compute: CPU. Why: closes the decision-boundary hole; likely fires.
    99	
   100	**Then (post-gate, binary spine):** distributional CORN (E.1) on cached features → RPS-on-sum + per-AU ClasswiseECE → fixed-high-sensitivity operating point + welfare-asymmetric decision curve (D) → one-sided-95%-NPV abstention curve (F). Graded-CORN trained and *inspected qualitatively*, never entered as a validated claim.
   101	
   102	## 7. Kill criteria (pivot-to-binary-spine vs proceed)
   103	
   104	- **Gate 0 (b) fails** (no budget certifies useful NPV): proceed, but F is exploratory-only; "guaranteed" never appears. **Not a kill** — relocates the deliverable to the welfare decision curve.
   105	- **Gate 2 fires damning AND yields only "CAT_01 is confounded" (non-portable):** demote audit to threat-to-validity; the κ method must then carry the headline alone. If Gate 1-B *also* fails → the honest paper is a confound/audit note; pre-register that venue NOW.
   106	- **Gate 2 detects confounding as a transportable protocol:** PROCEED — this is a publishable co-headline even if everything downstream is null.
   107	- **Gate 1-B κ CI-lower-bound below floor on orbital/ear/head:** drop graded entirely → ship calibrated **binary + abstention + confound audit**; the wrapper and the "κ-as-method" headline still stand (a publishable METHOD even at mediocre κ — a real method finding once the independent anchor exists, not a measurement claimed now). PIVOT to binary spine, not a kill.
   108	- **Gate 1-B κ passes on orbital/ear/head, fails muzzle/whiskers:** proceed on the surviving AUs; state how dropping 2/5 heads shifts the achievable 0–10 range and whether 0.39 (≈4/10) is even reachable (Clinician note).
   109	- **Gate 4 or Gate 5 fails:** BLOCKING — fix before any metric is believed; no number from silently-wrong MPS logits or misaligned crops is reported.
   110	- **Gate 6 fires (AU=2 single-digit):** collapse the high-end scale (expected base-rate outcome); do NOT print a per-cell severe calibration number.
   111	- **§3.4 Monte-Carlo GO under ρ=0 but NO-GO under ρ=0.3:** declare the gate fragile, say so, and default to the binary spine rather than claim a GO the correlation could flip.
   112	
   113	**Modal outcome to plan for as the default product:** binary-plus-wrapper + κ-as-method + confound-protocol + welfare decision curve + LB-abstention curve, with graded-CORN shipped as an inspected-not-validated artifact. That product is solo-deliverable on M4/MPS with one vet, and it is not another cat-pain detector.
   114	
   115	**Seam landed (2026-06-13 update, cross-reality check vs code):** The portable
   116	co-headline is now concretely embodied in `src/protocols/` (first-class surface
   117	with `__init__.py` re-exports of the κ/confound protocols, thin adapters for
   118	AU-override / col-map / generic non-FGS ordinal mode with no 0.39 assumption,
   119	and `standalone_test_corpus.py` — a deletion-safe, zero-torch, synthetic
   120	arbitrary-AU exerciser that can survive removal of engine/vlm/data). New central
   121	`src/gates/orchestrator.py` + Makefile `gate-e2e-synthetic` / `test-portable` /
   122	`gate-orchestrate` + e2e tests enforce strict gates (G0 power first, cat-disjoint
   123	G1/G3, one-dir G2, CI-LB G1-B, blocking G4, welfare asym baked, single-source
   124	decode via constants, inspected-not-validated graded). `src/model/decode.py`
   125	updated for portable N/k. Cache schema (_CACHE_SCHEMA + FEATURE_SCHEMA) and
   126	DINOv3 prep (backbone + richer patch_std per MCP context7) are explicit
   127	executable embodiments. This makes the "runnable protocol" / "portable
   128	methods" claims accurate, citable, and stronger than paper prose alone. Doc
   129	debt addressed by updates to paper/sections/04+06+00-abstract, README quickstart,
   130	and this note (P5/FreshPaperDocLandedSync cand7 fidelity 2026-06-14). Reality >
   131	prior docs (paper "protocol only" language now backed by importable seam + new
   132	orch + generalized decode + cache). Use `python -m src.protocols.standalone_test_corpus` and orchestrator synthetic for verification. Uniqueness vs priors (Steagall closed/handcrafted; Martvel video-only; zero prior public VLM per-AU κ protocol + confound attribution + welfare-asym + standalone + strict gate orch) holds per MCP grep/context7 polls on GitHub (no matching code patterns for the surface). DINOv3 successor prep documented for engine evolution.
   133	
   134	## 8. Decision record — framing + timeline (2026-06-14, MCP-grep + paper-scan grounded)
   135	
   136	Two open v1 decisions were resolved by a multi-agent debate grounded in (i) an MCP
   137	`grep__searchGitHub` + `context7` scan of public code and (ii) a non-GitHub paper
   138	scan (arXiv / Semantic Scholar / OpenReview / PubMed). Both judges and the paper
   139	scan **converge at medium confidence** and **confirm — do not flip — the §3.A
   140	asymmetric framing.**
   141	
   142	**DECISION A — Headline framing: ASYMMETRIC (resolved).**
   143	Lead with the one-directional, power-conditioned, **per-AU confound-attribution
   144	protocol** (FGS-BG-Gap counterfactual + per-AU EBPG saliency-as-confound-evidence
   145	+ VLM judge-bias probe) as the demonstrable spine — it validates **today** on
   146	planted positive + negative controls in `src/protocols/standalone_test_corpus.py`,
   147	no vet anchor needed. Demote the **VLM-as-AU-rater κ** to a *guarded,
   148	adequately-powered-but-method-crowded* reliability check shipped
   149	inspected-not-validated (CI-LB gate + rubric-independence guard named as the
   150	specific increments *within* that leg, not a co-equal pillar). Demote the wrapper
   151	(Clopper-Pearson NPV-LB + welfare-asym DCA + LTT/MAPIE abstention) to **cited
   152	plumbing**, not a contribution. κ is **not deleted** — it remains kill-tree
   153	insurance per §7 lines 105–107 (sole-survivor headline if the confound leg
   154	degrades), only re-ranked below the confound protocol.
   155	
   156	*Exact headline wording to use:* "A portable, power-aware confound-attribution
   157	protocol for fine-grained animal-affect models, with a guarded VLM-as-AU-rater
   158	reliability check (CI-lower-bound gated, result pending an independent vet anchor)."
   159	
   160	*Why:* the κ-as-judge idea is crowded off-the-shelf (crowd-kit / gtmf / medkit IRR
   161	libs, AWS sample, and `laudos-ai/laibench-public` `calibrate.ts` a near-twin of the
   162	rubric-independence audit), so co-equal billing hands a reviewer a free kill; the
   163	assembled confound protocol returns **0 GitHub hits and 0 assembled-paper matches**
   164	(closest in-domain near-miss, Tech4Animals "segment-based framework" Sci Rep 2025,
   165	owns only ~1 of 3 legs and is positively framed). Novel-by-assembly survives.
   166	
   167	**DECISION B — Timeline: SHIP NOW as a methods/protocol paper (resolved), conditional on Decision A.**
   168	Submit on synthetic + planted positive/negative controls; target an
   169	**ML-eval/trustworthiness or clinical-ML-methods track, NOT a vet journal first.**
   170	Disclose empty `data/`, missing vet anchor, Roboflow-binary-only, and the
   171	`power.json calc_c_point_039 reportable:false (n_pos=16)` operating point as
   172	limitations, real-cat application named as future work. *Why:* synthetic-planted
   173	validation is the native, accepted mode for trustworthiness tooling (cleanlab
   174	`test_spurious_correlation.py` plants + asserts recovery; AIF360; MAPIE on
   175	synthetic streams); holding defends against a phantom scoop (0 open competitor code
   176	for graded cat-FGS); and Gate-0 rules forbid printing the one real number a vet
   177	sitting would buy. **B is load-bearing on A:** ship-now is only defensible under
   178	asymmetric framing.
   179	
   180	**Mandatory honesty constraints from the paper scan:**
## DATA_DECISION.md
     1	> **⚠️ SUPERSEDED where it conflicts with `FINAL_DIRECTION.md` (authoritative) and the code.**
     2	> In particular this doc's "group by CLIP, never collapse the individual" guidance is
     3	> REVERSED in the shipped pipeline: CV is grouped by `cat_id` (see `src/data/folds.py`,
     4	> `scripts/gate2_confound.py`). Trust the code + `FINAL_DIRECTION.md` over this briefing.
     5	
     6	# DECISION BRIEFING: Is Our Data Enough, and Should We Use the Local Cat Archive?
     7	
     8	## 1. Is our current dataset enough?
     9	
    10	**Phase A (binary pain detector): YES to TRAIN, NOT YET to make a clinical recall claim — but the picture is materially better than feared.**
    11	
    12	The ground-truth count is **260 unique pain frames** (not 264 — that is a box-instance count; one image is class-mixed) out of **2040 unique source images** (train 1631 / valid 409; the shipped "3262 train" is 2x augmentation, not real data). 260 pain-positives clears the hard floors for transfer learning: the ~100-positive minimum and the ~10-events-per-effective-parameter EPV rule. With a frozen/transfer backbone, focal/class-weighted loss, and oversampling of the 260 pain faces, this is a textbook data-starved-but-trainable regime.
    13	
    14	**The distinct-pain-cat reality (the real sample size) is far healthier than the early "low-tens (15–40)" guess.** Filename parsing (BUILD_PLAN regex, 260/260 matched) yields **173 distinct pain source-clips** and a defensible **~132–173 distinct pain individuals** — 4–10x the placeholder. Only **one** CAT_xx camera (CAT_01) exists in the entire dataset. The much-discussed "CLIP cosine > 0.6 connected-components" merge is **unusable** (CLIP carries no individual-identity signal here; it collapses 98% of pain pairs into one blob) — do not use it; if true per-individual dedup is ever needed, do it on downloaded pixels with imagehash or a cat-face re-ID model.
    15	
    16	What this buys us: `StratifiedGroupKFold(5)` **grouped by CLIP** gives pain-clips per fold of [39, 36, 27, 33, 38] — **no fold at 0 or 1**, so subject-exclusive CV is safe and validatable. (Grouping by collapsed individual is a trap: it fuses CAT_01 into one 605-image super-group and produces a degenerate fold. **Group by clip, not by collapsed cat.**)
    17	
    18	The honest limit: the *shipped* 409-image valid split holds only ~50 pain images and is leaky. A pain-recall point estimate off ~50 positives carries a 95% CI half-width of ~±0.13 — not clinically credible. **Fix: replace the single shipped split with repeated cat/clip-grouped k-fold CV, pool out-of-fold predictions, and report PR-AUC + pain recall with bootstrap 95% CIs and the CI width stated up front.**
    19	
    20	**Phase B (5-AU ordinal 0–10 FGS): NO — currently impossible, and underpowered even after labeling.** The dataset has **zero** per-AU / 0–10 / FGS labels (binary boxes only), so current Phase B statistical power is literally **0**. Even if all 260 pain images were vet-labeled across 5 AUs, the **severe (level-2) cells** of each AU would fall to single-digits–low-tens — far below any per-class floor — making per-AU QWK/MAE unstable and the summed-score calibration untrustworthy exactly at the 0.39 (4/10) decision boundary. Phase B is a **data-acquisition and labeling problem, not a tuning problem.**
    21	
    22	> ⚠️ **Bigger-than-leakage validity threat surfaced during investigation:** our pain/no_pain labels are almost certainly **unvalidated subjective pseudo-labels on generic Flickr pet photos** (see §2), not clinically grounded surgical pain. The binary signal may be capturing flat-faced/grumpy breed morphology and acquisition context rather than analgesia-relevant pain. A **label-reliability pilot and capture-condition confound audit must be GO/NO-GO gates** before any clinical claim.
    23	
    24	---
    25	
    26	## 2. What is the local archive, really?
    27	
    28	The local archive is the **Zhang/Sun/Tang ECCV 2008 "Cat Head Detection" dataset** (redistributed as Kaggle `crawford/cat-dataset`), confidence VERY HIGH. On disk it shows 19,994 `.jpg` files, but the `cats/` subfolder is a **byte-identical duplicate** — the real count is **9,997 unique** full-scene Flickr cat photos (whole animals in natural scenes, ~500px long edge, NOT pre-cropped faces), each paired with a `.jpg.cat` sidecar holding **9 coarse facial landmarks** (1 left eye, 1 right eye, 1 mouth, 2×3 ear points). It has **no bounding boxes, no pain/no_pain class, no per-AU FGS labels, no 0–10 scores** — zero pain signal. License is CC0 on the Kaggle redistribution (unverified locally; underlying Flickr per-image rights may apply). It is purely a **face-detection / landmark** asset. **Notably, this same image family is the upstream parent of our Roboflow set** (identical 8-digit Flickr IDs + CAT_xx fingerprint), confirming our "pain" labels are pseudo-labels layered on this generic corpus.
    29	
    30	---
    31	
    32	## 3. Should we use the archive — and exactly how?
    33	
    34	**Decision: Largely SKIP it. Pursue at most ONE narrow, conditional use; it is dominated by CatFLW on every dimension that matters.**
    35	
    36	Of five candidate uses, four are dead on their merits:
    37	
    38	- **Cat-face detector — DROP.** BUILD_PLAN already verified RF-DETR boxes are tight frontal head crops (~27–30% of frame): "no separate face detector is needed." The archive has no boxes anyway.
    39	- **SSL / DINOv2 domain-adaptation pretraining — DROP.** 10k uncropped Flickr cats cannot meaningfully shift DINOv2's pretraining; the backbone is deliberately **frozen**. Real risk of degrading features for ~zero gain.
    40	- **Hard-negative mining — DROP.** Out-of-domain Flickr cats with no boxes are distribution shift, not useful hard negatives; the in-domain no_pain set (1819) + Smudge negative-control suite already cover this.
    41	- **The one viable use — the test-time EYE-ALIGNMENT helper (BUILD_PLAN items 204/253).** The flagged NME risk needs only a **2-point eye-similarity transform**, and the archive's 2 eye landmarks on ~10k cats are plentiful for that. **BUT** use this **only as a fallback if CatFLW is blocked** — CatFLW's 48 CatFACS-aligned landmarks dominate the archive's 9 coarse, non-FACS points (1 mouth point, ear tips) for any FGS geometry. Realistically: **CatFLW wins, the archive contributes nothing, and that is an acceptable, honest outcome.** Do not pre-build the aligner before the 30–50-image NME audit says it is needed.
    42	
    43	**CAT_xx-overlap leakage — explicitly a FALSE ALARM.** The archive's `CAT_00..CAT_06` are *parent folders* bucketing unrelated Flickr cats; our `CAT_01` is an *in-filename camera id*. They never collide **as long as the grouping key is derived from filename stems only** (strip `_png.rf.<hash>.jpg`), never from a parent path. Our export tree is just `train|valid/{images,labels}` with **no CAT_xx directories**, so cross-archive phantom-merge is structurally impossible. The only real internal collision is clip `00000100`, which appears in both `CAT_01_00000100_*` and bare `00000100_*` forms; the BUILD_PLAN regex correctly maps these to **distinct** keys (`CAT01_00000100` vs `P_00000100`). Add a unit assertion that `00000100` yields two groups. **Keep the archive namespaced and stored outside the `datasets/` export path so it can never enter pain loaders.**
    44	
    45	---
    46	
    47	## 4. External datasets to add
    48	
    49	**No open dataset closes the pain or per-AU FGS gap.** Every pain/FGS-labeled corpus is **request-only** (email the authors). Everything openly downloadable is **landmark-only with zero pain signal**. Verified directly via the authenticated Roboflow REST API (bypasses the Cloudflare 403) and PMC full-text + supplements.
    50	
    51	### Tier 1 — PAIN / FGS-labeled (high value, all request-only)
    52	
    53	| Name | URL | Pain/FGS-labeled? | Size | License | How to get it | Why |
    54	|---|---|---|---|---|---|---|
    55	| **Steagall / DeepMGS FGS corpus** | pmc.ncbi.nlm.nih.gov/articles/PMC10703818/ | **YES — 1188 imgs with true 5-AU 0/1/2 FGS** | 3447 imgs (1188 FGS-scored), 37 landmarks | Request-only; withheld pending commercial app | Email **P.V. Steagall** (Montreal), academic non-commercial framing | The single richest match to our exact Phase B target. Hardest to get. |
    56	| **Evangelista 2019 FGS validation** | pmc.ncbi.nlm.nih.gov/articles/PMC6911058/ | **YES — 110 imgs scored 0–2 on all 5 AUs** | 110 imgs / 55 cats | Request-only | Email P.V. Steagall (same group) | Canonical gold-standard graded FGS; tiny but exact. |
    57	| **Feighelstein / Finka + TiHo pain sets** | nature.com/articles/s41598-024-78406-2 | YES but **composite (MCPS/CMPS), not per-AU** | Finka ~26–29 cats; TiHo 72 videos | Request-only | Email **A. Zamansky / G. Martvel** (Haifa/TiHo) | Extra binary pain-positives + video path; NOT per-AU supervision. |
    58	
    59	### Tier 2 — Landmark-only (open, for preprocessing only — ZERO pain signal)
    60	
    61	| Name | URL | Pain/FGS-labeled? | Size | License | How to get it | Why |
    62	|---|---|---|---|---|---|---|
    63	| **CatFLW** (Cat Facial Landmarks in the Wild) | kaggle.com/datasets/georgemartvel/catflw | No | ~2079 imgs, **48 CatFACS-aligned landmarks** + bbox | **CC BY-NC 4.0** | Open Kaggle/GitHub download, no request | **Best open asset** for DETECT+LANDMARK+EYE-ALIGN+CROP + NME validation. Strict upgrade over the archive. |
    64	| **Zhang 2008 / crawford cat-dataset** (our archive) | kaggle.com/datasets/crawford/cat-dataset | No | 9997 unique, 9 landmarks | CC0 (Kaggle) | Already in hand | Fallback eye-aligner only; inferior to CatFLW. Dedupe (it is 2x-duplicated; one CAT folder has corrupt `.cat`). |
    65	
    66	**Not worth pursuing:** Roboflow Universe emotion/expression/"sick" proxies (Cat Moods Scanner, Cat Emotions, Cat Expression Detection) — coarse, smaller than our 260, no AU structure; `icu-egjok/pain-paitents` is **human** ICU. CatFACS is a free coding *manual*, not a dataset (use it as the AU rubric for vet review).
    67	
    68	---
    69	
    70	## 5. THE DECISION (numbered, tied to BUILD_PLAN.md)
    71	
    72	1. **Phase A pain training — use ONLY our Roboflow set (260 pain / 1819 no_pain).** Train the binary RF-DETR/YOLO detector with focal/class-weighted loss + oversample/copy-paste of the 260 pain faces (BUILD_PLAN lines 11, 51, 251). Do **not** add external cats to pain training.
    73	2. **Re-split FIRST: `StratifiedGroupKFold(5)` grouped by CLIP** (group_id from filename stem per BUILD_PLAN line 28). Pool out-of-fold predictions; report **PR-AUC + pain recall with bootstrap 95% CIs** and the CI width. Report distinct-pain-individual count (**~132–173**) as the true sample size, not "264/2040." Add a unit assertion that clip `00000100` → two distinct groups.
    74	3. **Face-alignment / NME validation — adopt CatFLW (48 landmarks, CC BY-NC).** Download now; use it to train/validate the DETECT+LANDMARK+EYE-ALIGN+CROP stage (BUILD_PLAN lines 78, 110). Run the **30–50-image NME audit** (PAPER_DEBATE 204/253) comparing RF-DETR crops vs eye-aligned crops *before* building any aligner.
    75	4. **Local archive — fallback only.** Build the 2-point eye-aligner from it **only if** CatFLW access/license stalls **and** the NME audit shows alignment is needed. Otherwise do nothing with it. Keep it namespaced and outside `datasets/`.
    76	5. **Acquire next (parallel, do not gate the timeline):** email **Steagall** (1188-img + 110-img FGS sets) and **Zamansky/Martvel** (Finka/TiHo). Budget for non-response, especially Steagall (withheld for a commercial app).
    77	6. **We must label the per-AU FGS gap ourselves — there is no shortcut.** Proceed with the Phase B plan: **VLM weak-label 5 AUs → vet review** (CatFACS manual as the rubric), prioritizing acquisition of **more distinct pain cats and especially severe AU=2 examples**. Power target: ~400+ pain-positive evaluation cases (~2x today) to pin the 0.39-threshold sensitivity/specificity to a ±0.10 CI — noting the per-AU kappa remains a protocol with its result pending the independent per-AU vet anchor (Gate 0), since the current binary pain/no_pain dataset cannot yield 0/1/2 ground truth.
    78	7. **Gate Phase A clinical claims behind a label-reliability pilot + capture-condition confound audit** (our labels are pseudo-labels on generic Flickr photos). Update CLAUDE.md / RESEARCH.md / BUILD_PLAN.md: **provenance = Zhang 2008, NOT Finka 2019** — drop the Finka cross-dataset leakage warning entirely.
    79	
    80	---
    81	
    82	## 6. Risks & leakage guardrails when combining sources
    83	
    84	- **Group by clip, never by collapsed individual.** Collapsing CAT_01 creates a 605-image super-group → one degenerate fold. Clip-grouping keeps every fold at 27–39 pain clips.
    85	- **Never use CLIP-cosine for individual dedup** — no identity signal; any count it produces is a threshold artifact. Use imagehash / face re-ID on pixels if true per-cat dedup is ever required.
    86	- **Grouping key = filename stem only.** Strip `_png.rf.<hash>.jpg`; never key on a parent folder named `CAT`. This neutralizes the lone `00000100` collision and makes archive CAT_xx folder collision impossible.
    87	- **Provenance = Zhang 2008, not Finka.** If we later obtain CatFLW/Finka/Steagall, they share **no** cats with our Roboflow set → no Finka-overlap leakage. But each acquired set has only 26–84 cats — **regroup any merge by individual cat** or recall/PR-AUC will inflate.
    88	- **License tracking:** our set CC BY 4.0; archive CC0; **CatFLW CC BY-NC 4.0 (blocks commercial deployment of any derived model)**; all request-only pain sets have no stated data license. Keep per-source licenses separate; do not let CC0/NC images leak into a CC BY release.
    89	- **Augmentation ≠ subjects.** 2x augmentation expands frames, adds zero pain individuals; it stabilizes training but does not widen the generalization CI. Always count unique frames (1631 train), not augmented (3262).
    90	- **Label-validity confound is the top risk** — bigger than any leakage question. Treat the binary pain signal as unvalidated until the reliability pilot passes.## FACTCHECK.md
     1	# FACT-CHECK REPORT — Cat FGS Pain Detector BUILD_PLAN.md
     2	
     3	The plan's **novelty thesis survives**, but six load-bearing factual claims are wrong (mostly mis-attributed citations and sample-extrapolated dataset numbers), one compute claim is outdated, and three load-bearing claims that underpin the engine/labeler choice were never verified against their source.
     4	
     5	---
     6	
     7	## 1. Scorecard
     8	
     9	| Verdict | Total | Load-bearing |
    10	|---|---|---|
    11	| **Confirmed** | 38 | 24 |
    12	| **Partially-true** | 18 | 10 |
    13	| **Outdated** | 1 | 1 |
    14	| **Refuted** | 6 | 5 |
    15	| **Unverifiable / asserted-not-verified** | 12 | 7 |
    16	| **Totals (87 claims)** | 87 | ~47 load-bearing |
    17	
    18	Headline: **5 refuted load-bearing** (C73, C12, C70/C01, C39, C81 — plus root-cause C80), **1 outdated load-bearing** (C78), **and the novelty spine holds** with two required rescopings (C06, C12).
    19	
    20	---
    21	
    22	## 2. Confirmed solid (rely on as-is)
    23	
    24	**Clinical spine (Evangelista 2019, s41598-019-55693-8; Steagall 2023, s41598-023-49031-2):**
    25	- **C04** — FGS operating point **AUC 0.94, sens 90.7%, spec 86.6%** (Evangelista 2019 Fig 7).
    26	- **C05** — analgesia threshold **>0.39/1.0 (~4/10)** (Evangelista abstract; COSMIN 2026 Table 5).
    27	- **C08 / C09** — 5 AUs (ear, orbital, muzzle, whiskers, head), each 0/1/2, sum 0–10.
    28	- **C50** — descriptors **0=absent; 1=moderate OR uncertain; 2=obvious** (exact wording).
    29	- **C57** — muzzle/whiskers lowest reliability, **ICC 0.55–0.67** (Evangelista Table 2).
    30	- **C27** — landmark NME **9–26% by morphology** (Martvel 2024, fvets.2024.1442634, PMC11663861).
    31	- **C28** — automated landmarks cost **~7 accuracy points** (same paper, Table 6: 0.73→0.66).
    32	
    33	**Library/API (all load-bearing ones confirmed):**
    34	- **C24 / C87** — RF-DETR trains natively on MPS as of **v1.6.0** (release notes fix grid_sample/bicubic CPU fallback).
    35	- **C25 / C60 / C62 / C64** — coral-pytorch `corn_loss` / `corn_label_from_logits`, Linear(feat, 2) per head, 10 logits, decode→0–10.
    36	- **C36 / C71** — `StratifiedGroupKFold`, `cohen_kappa_score(weights='quadratic')`, torchmetrics mAP fields.
    37	- **C48 / C49** — Claude strict structured outputs **strip** min/max; use **enum [0,1,2]**; `messages.parse()` / `additionalProperties:false` (platform.claude.com structured-outputs docs).
    38	
    39	**Dataset (measured directly from COCO v1 export + live API):**
    40	- **C02** — boxes are centered head crops, **median area ~0.24–0.28** of frame.
    41	- **C31** — preprocessing is **Stretch-to-640** on 4:3 sources (metadata `resize:{640,640,'Stretch to'}`), distorts AU geometry.
    42	- **C32** — filename regexes match all 2040 names (594 CAT-prefixed, 1446 plain).
    43	- **C40** — **valid = 53 pain / 356 no_pain images, unaugmented** (exact, COCO export).
    44	- **C53** — **~2040 images** (metadata + manifest both exactly 2040).
    45	- License = **CC BY 4.0** (project metadata).
    46	
    47	**External datasets:** **C13** CatFLW CC BY-NC 4.0; **C72** 2079 faces / 48 CatFACS-relevant landmarks (Kaggle georgemartvel/catflw); **C17** horse-grimace 5 individuals, 3-of-5 AUs, CC BY 4.0 (HF oliveirabruno01/openfarm-horse-grimace-region); **no OPEN graded per-AU cat-FGS dataset exists** (HF zero hits; Evangelista data "on reasonable request").
    48	
    49	---
    50	
    51	## 3. Corrections needed (load-bearing first)
    52	
    53	| Claim | Verdict | Evidence (source) | What the plan SHOULD say |
    54	|---|---|---|---|
    55	| **C73** — "Feighelstein 2023 (s41598-023-49031-2) achieved 95.5%" | **REFUTED** | DOI s41598-023-49031-2 is **Steagall et al. 2023** ("Fully automated deep learning models with smartphone applicability…"), Table 2 = 95.51%. Feighelstein 2023 is **s41598-023-35846-6** (77% landmark / 65% DL). | **Steagall et al. 2023** validated landmark(37)→geometric→XGBoost at **95.5%** (MSE 0.0096). Feighelstein 2023 is a *separate* 77%-vs-65% binary paper. Fix attribution everywhere (§5/§7). |
    56	| **C74** — Mode for ear/orbital/muzzle/**head**, Min for whiskers | **PARTIALLY-TRUE** | **Steagall** 2023 p.4 (not Feighelstein): Mode best "except for whiskers change **AND head position** for which Minimum performed better"; Table 4 Head Min 0.1465 < Mode 0.1674. | Per **Steagall 2023**: Mode for ear/orbital/muzzle; **Min for BOTH whiskers AND head**. Move head to Min; reattribute. |
    57	| **C12** — COSMIN review named calibration/CIs/reliability & "explicitly excluded automated scorers" | **REFUTED as written** | Lee & Steagall 2026 (JVIM 40(1) aalaf062): never uses "calibration"/"confidence intervals"; exclusions are chronic-pain / non-ordinal / non-English; **AI/automated never mentioned** (out of scope by topic, not stated exclusion). Names measurement-error/reliability/validation/interpretability/ROC-threshold as gaps. | Reframe: the review covers **only human-rater instruments** and flags measurement-error/reliability/validation as underreported; **calibration, CIs, and automated-scorer reliability fall outside its scope** — *that* is the white space we take. Drop "named calibration/CIs" and "explicitly excluded." |
    58	| **C70 / C01** — leakage 201/337 clips | **REFUTED (number)** | Full 2040-image manifest, clip-grouped: **191 of 336 clips (56.8%)** in both train+valid. 201/337 is a 76%-sample extrapolation (C80). | Leakage is **191/336 clips (56.8%)**, computed on full data. Phenomenon is real and load-bearing; the number was wrong. Update Intro, §2, §4, §7. |
    59	| **C39** — train split has 467 pain boxes | **REFUTED** | COCO v1 train export: **414 pain / 2880 no_pain boxes**; source (pre-aug) train pain ≈ 202. 467 unsupported. | Augmented train = **414 pain boxes** (~202 source + augmentation). |
    60	| **C81** — train split = 1819 no_pain boxes | **REFUTED** | 1819 is **whole-project** metadata `classes.no_pain`; train source ≈1454, augmented ≈2880; valid ≈357. | State **1819 as dataset-wide**, not train-only. pos_weight basis (C41 sqrt(1819/264)=2.62) is fine but must be labeled metadata-based. |
    61	| **C78** — bnb 4-bit QLoRA is CPU-only on macOS, the only hard CUDA blocker | **OUTDATED** | Official bitsandbytes support matrix (2026): macOS arm64 QLoRA(4-bit) **✅ CPU-supported**, Metal(MPS) **🐢 slow-supported**. Not strictly CPU-only, not the unique blocker. | Reword: "bnb 4-bit on Mac is **CPU/alpha-Metal and impractically slow**; use **mlx-vlm** for speed (not feasibility)." Colab-T4 half (bnb-4bit needs CUDA bnb) still stands. |
    62	| **C06** — graded FGS automated "exactly once" | **PARTIALLY-TRUE** | Steagall 2023 is sole paper producing per-AU 0/1/2→0–10. But **"Feline SentiNet" 2023** (IEEE ICSES, CNN+RandomForest) does 5-category pain grading at 90% — automated *graded*, not FGS-structured. | Scope to: **"automated exactly once for the FGS per-AU 0/1/2→0–10 scoring structure (Steagall 2023)."** Cite Feline SentiNet as adjacent multi-class grading (ImageNet-CNN, so doesn't touch C10/C11). |
    63	| **C47** — "Wu et al. 2025," 13% prevalence → 53% sensitivity | **PARTIALLY-TRUE** | arXiv 2506.07273 first author **Chavoshi** (not Wu); worked example is **10%** prevalence → ~53% sens at 95% spec. | Cite **Chavoshi et al. 2025 (arXiv 2506.07273)**; example is 10% prevalence; 13% is our dataset figure applied by extension. |
    64	| **C65** — two-source rule = grimace + clinical reason | **CONFIRMED w/ precision** | Feighelstein 2023 p.4: requires **CMPS-feline ≥5 AND charted clinical reason**. | Pain label = **high behavioral pain-scale score (CMPS ≥5) + charted clinical reason** (not "grimace + reason"). |
    65	| **C34 / C37 / C38 / C41** — prevalence 12.7%, 6.9:1, "264 unique pain faces" | **PARTIALLY-TRUE / source-ambiguous** | Metadata classes: 264 pain / 1819 no_pain → 12.7%, 6.89:1. Live annotation sum: **246 / 1811 → 12.0%, 7.36:1**. | Keep prevalence ~12–13% and imbalance ~6.9–7.4:1 as approximate; **pin every exact derived figure to the metadata basis (264/1819)** and stop calling 264 "unique pain **faces**" — it is a **box count** (distinct individuals are far fewer). |
    66	| **C16** — single confounded "Flickr / CAT_01" source | **PARTIALLY-TRUE** | Manifest: **CAT_01 is the only CAT_NN id**, 84 clips (range 100–184). "Flickr" provenance not in API. | Single-individual confound confirmed (CAT_01, 84 clips); drop unverifiable "Flickr" attribution. |
    67	| **C26** — DINOv2 "/14 patch grid" | **PARTIALLY-TRUE** | patch14 = 14-px patches; token grid = input/14 (37×37 @518). Operational rule (crops divisible by 14) is correct. | Say "patch size 14; crops must be **divisible by 14**." |
    68	| **C61** — "CORN (Cao et al., 2111.08851)" | **PARTIALLY-TRUE** | Authors are **Shi, Cao & Raschka** (lead = Shi). Technical claim correct. | Cite **Shi, Cao & Raschka (2021/2023)**. |
    69	| **C46** — cat is COCO class → "strong transfer" | **PARTIALLY-TRUE** | "cat" = COCO id 15 (confirmed). "Strong transfer" has no benchmark. | Present transfer as **expected**, not established. |
    70	| **C14** — Steagall needs 8 "real-time" raters | **PARTIALLY-TRUE** | 8 raters (6F/2M) scored **still images**, not real-time. | Drop "real-time." |
    71	
    72	---
    73	
    74	## 4. Asserted-but-unverified — what we must actually MEASURE
    75	
    76	These are stated as fact in the plan but were **never computed/sourced**:
    77	
    78	1. **The 201/337 leakage number (C01/C70/C80) — the root cause.** C80 self-admits all clip-level stats (337 clips / 146 pain / 142 mixed / 201 leaked) were **extrapolated from a 76% sample (1562/2040)**. **MEASURE on full 2040 manifest** (already done by verifier): **336 clips, 138 pain-bearing, 135 mixed, 191 leaked.** Recompute C01, C33, C70 from full data and replace every propagated figure.
    79	2. **Distinct-cat / "unique pain faces" counts (C22/C38).** C22 confirmed *directionally* (CAT_01 alone = 84 clips → upper bound ~253 individuals), but the **actual distinct-individual count is NOT computable without re-ID**. **MEASURE:** treat "264" strictly as a pain-box count; do not assert any "unique faces" number until re-ID is run. C20 warns CLIP/pHash are duplicate detectors, not re-ID — so they cannot substantiate it.
    80	3. **C54 (VLM-vs-FGS, Sci Rep 2025 s41598-025-27404-z) — load-bearing, underpins choosing Claude.** Paper **not retrievable**; "VLMs underestimate FGS / only Claude acceptable bias" **unconfirmed**. **MUST fetch and confirm** before using as justification for Claude as weak-labeler.
    81	4. **C86 (DINOv2 frozen > landmark-XGBoost in few-hundred-label regime) — load-bearing engine bet.** **Extrapolation, not a result** in arXiv 2304.07193 (no cat-FGS benchmark there). **MEASURE empirically in Gate 4/5**; present as hypothesis.
    82	5. **C23 (768-dim CLIP vector free from Roboflow search API).** Verifier **could not retrieve** an embedding field. **MEASURE:** confirm the exact endpoint/field before relying on it for Gate 2 (else local DINOv2/CLIP extraction needed).
    83	6. **C18 / C58 / C68 / C19 / C69 — kappa floors, sample-size floor, Monte-Carlo GO/NO-GO.** All **project pre-registration**, not published. Defensible (kappaSize-grounded, Evangelista-ICC-anchored) but label as **design decisions**; back C58 with an actual kappaSize/power calculation.
    84	7. **C85 (≥0.90 Phase-A pain-recall gate).** Self-imposed target — **justify explicitly against C04's 90.7% reference sensitivity**, not as derived from prior automated work.
    85	
    86	---
    87	
    88	## 5. Novelty verdict
    89	
    90	**SURVIVES IN SUBSTANCE.** The adversarial search (Semantic Scholar, web, arXiv, GitHub) found **zero counter-examples** to the three pillars:
    91	- **C10** — no prior cat-pain work uses a **foundation model/VLM** (all are binary / ImageNet-CNN / landmark-geometry: Steagall, Feighelstein, Martvel, Feline SentiNet, Feline Feelings). **Confirmed.**
    92	- **C11** — **VLM weak-labeling of the 5 FGS AUs has never been attempted.** Only adjacent work (Sci Rep 2025) *benchmarks* VLMs as direct raters, not as weak-labelers. **Confirmed.**
    93	- **C07** — **no automated-FGS work has shipped calibration, CIs, open artifacts, or measured training-label reliability.** Corroborating zero-hit evidence. **Confirmed.**
    94	
    95	**Genuinely unowned (asymmetric, per FINAL_DIRECTION §8):** the **headline** is the assembled per-AU confound-attribution protocol (FGS-BG-Gap counterfactual + per-AU EBPG-as-confound-evidence + VLM judge-bias probe, framed one-directionally); **below it** sits the guarded VLM-as-AU-rater κ check (inspected-not-validated, CI-LB gated, pending an independent vet anchor); the foundation-model/VLM engine and the calibration/CI/abstention layer are **cited plumbing / supporting evidence**, not headline contributions. Novelty = the assembly + per-AU EBPG-as-confound-evidence + equivalence-style audit hygiene — never any individual primitive, never the κ CI-LB gate (textbook clinimetrics) or rubric-paraphrase guard (published), never the one-directional/power-conditioned/equivalence statistics themselves.
    96	
    97	**Must soften (two precise rescopings, or the framing is attackable):**
    98	1. **C06** → "automated exactly once **for the FGS per-AU 0/1/2→0–10 structure** (Steagall 2023)," explicitly citing **Feline SentiNet 2023** (5-category, 90%, CNN+RF) as adjacent graded-but-not-FGS work.
    99	2. **C12** → drop "named calibration/CIs" and "explicitly excluded automated scorers"; reframe as the review covering only human-rater instruments and leaving calibration/CI/automated-scorer reliability out of scope.
   100	
   101	With those two edits the novelty stack is airtight.
   102	
   103	---
   104	
   105	## 6. What the plan SHOULD be — specific edits
   106	
   107	**BUILD_PLAN.md:**
   108	- **Intro / §2 / §4 / §7:** change leakage **201/337 → 191/336 (56.8%)**; change **337 clips → 336**, **146 → 138** pain-bearing, **142 → 135** mixed. Add a one-line note: "all clip-level figures recomputed on the full 2040-image manifest (the prior 76%-sample extrapolation in C80 is retired)."
   109	- **§2:** train pain boxes **467 → 414** (augmented; ~202 source). Relabel **1819 no_pain as dataset-wide**, not train-split. Tag prevalence **12.7% / imbalance 6.9:1 / pos_weight 2.62** as **metadata-based (264/1819)**; note live annotation sum gives 12.0% / 7.4:1.
   110	- **§5 / §7:** reassign **95.5%** and the **Mode/Min aggregation** guidance to **Steagall 2023 (s41598-023-49031-2)**; keep Feighelstein 2023 (s41598-023-35846-6) as the separate 77/65% binary paper; **move head position to the Min aggregation group.**
   111	- **§0.5:** rescope **"graded FGS automated exactly once"** to the FGS scoring structure + cite Feline SentiNet; rewrite the **COSMIN (C12)** framing.
   112	- **§1 / §7 (compute split):** reword the **bitsandbytes** line — bnb-4bit on Mac is CPU/alpha-Metal and slow (not impossible); mlx-vlm chosen for **speed**.
   113	- **§3 / §3.4:** fix **C47** citation to **Chavoshi et al. 2025**, 10% example prevalence.
   114	- **§3.3:** correct the **two-source rule (C65)** to CMPS ≥5 + clinical reason; fix **CORN attribution to Shi, Cao & Raschka (C61)**; phrase **DINOv2 patch (C26)** as "divisible by 14"; mark **C86** as a hypothesis to validate in Gate 4/5.
   115	- **§3.1:** mark **C54** as **unverified** until the Sci Rep 2025 paper (s41598-025-27404-z) is fetched — flag it because it justifies Claude as labeler.
   116	- **Stop asserting any "unique pain faces / distinct individuals" number** (C38) until re-ID is run; describe 264 as a pain-box count and CAT_01 (84 clips) as the lone known individual.
   117	- **Gate 2:** add a verification TODO that the **768-dim CLIP field (C23)** actually returns from the search API before depending on it.
   118	
   119	**Other docs:**
   120	- **PAPER_DEBATE.md** — propagate the Steagall-vs-Feighelstein reattribution (95.5% is Steagall's).
   121	- **DATA_DECISION.md** — note the metadata-vs-live annotation discrepancy (264/1819 vs 246/1811) and the 493 exact-duplicate filenames (2040 records / 1547 distinct) the plan currently omits.
   122	- **MEMORY / dataset-facts.md** — replace the 76%-sample clip numbers with the full-manifest figures (336/138/135/191).## src/model/backbone.py
     1	"""Frozen DINOv2 ViT-S/14 backbone (IMPLEMENTATION_PLAN §5.1).
     2	
     3	CONCEDED PLUMBING (FINAL_DIRECTION §A); never claimed as novel. ViT-S (hidden 384,
     4	patch 14), forward-only, frozen because the supervised set is ~120-300 vet/VLM
     5	labels; only the ~5 light heads train.
     6	
     7	BACKBONE VARIANT (README §"engine is conceded plumbing"): the register variant
     8	``dinov2_vits14_reg`` is the FIELD DEFAULT — registers suppress attention artifacts
     9	that hurt dense, localized features, and per-AU FGS scoring (orbital/ear/muzzle
    10	sub-regions) is exactly localized. The plain ``dinov2_vits14`` is kept selectable so
    11	the reg-vs-plain choice can be A/B'd before locking. The choice is recorded in the
    12	cache PROVENANCE.json (an input to every reported number), so a run can never hide
    13	which backbone produced its features. Both variants are ViT-S (embed_dim 384).
    14	
    15	Patch/hidden are derived from the model (m.embed_dim), not hardcoded in
    16	load-bearing paths; the 14/384/518 constants in comments are for the reader only.
    17	"""
    18	
    19	import torch
    20	import torch.nn.functional as F
    21	from torchvision import transforms
    22	
    23	from src.model.device import DEVICE
    24	
    25	IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    26	
    27	# Selectable ViT-S/14 variants. Reg (registers) is the field default; plain is the
    28	# A/B comparator. Both are embed_dim 384, patch 14 — preprocess is identical.
    29	DEFAULT_VARIANT = "dinov2_vits14_reg"
    30	SUPPORTED_VARIANTS = ("dinov2_vits14_reg", "dinov2_vits14", "dinov3_vits16")
    31	
    32	# 518 = 37*14 -> 37x37 patch grid; divisible-by-14 mandatory for DINOv2.
    33	# For dinov3_vits16 (patch 16) use 224/512 multiple of 16; richer per MCP context7 (dense oob frozen for small-N medical facial ordinal AU).
    34	VARIANT_PATCH_SIZES = {"dinov2_vits14_reg": 14, "dinov2_vits14": 14, "dinov3_vits16": 16}
    35	VARIANT_INPUT_SIZES = {"dinov2_vits14_reg": 518, "dinov2_vits14": 518, "dinov3_vits16": 224}  # 224 base for vits16; caller may override for dense
    36	
    37	preprocess = transforms.Compose([
    38	    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    39	    transforms.CenterCrop(518),
    40	    transforms.ToTensor(),
    41	    transforms.Normalize(IMNET_MEAN, IMNET_STD),
    42	])
    43	
    44	
    45	def load_frozen_dinov2(device: str = DEVICE, variant: str = DEFAULT_VARIANT):
    46	    """Load a frozen ViT-S/14 DINOv2 (or dinov3_vits16 successor) backbone.
    47	
    48	    variant: ``dinov2_vits14_reg`` (field default, registers suppress attention
    49	    artifacts on localized per-AU features) or the plain ``dinov2_vits14`` A/B
    50	    comparator, or ``dinov3_vits16`` (MCP /facebookresearch/dinov3 primary A/B:
    51	    high-quality dense features out-of-the-box / without fine-tuning per MODEL_CARD;
    52	    same forward_features contract: x_norm_clstoken + x_norm_patchtokens;
    53	    supports get_intermediate_layers(n=list) + L2 patch for richer localized AU).
    54	    Returns (model, hidden_dim).
    55	    """
    56	    if variant not in SUPPORTED_VARIANTS:
    57	        raise ValueError(
    58	            f"unsupported backbone variant {variant!r}; expected one of {SUPPORTED_VARIANTS} "
    59	            "(reg is the field default; plain is the A/B comparator; dinov3_vits16 MCP dense oob successor for small-N imbal ordinal facial AU)"
    60	        )
    61	    if variant.startswith("dinov3"):
    62	        # MCP context7 confirmed: torch.hub "facebookresearch/dinov3" dinov3_vits16 or dinov3.hub.backbones import; frozen oob dense ideal
    63	        m = torch.hub.load("facebookresearch/dinov3", variant, source="github")
    64	    else:
    65	        m = torch.hub.load("facebookresearch/dinov2", variant)
    66	    m.requires_grad_(False)      # freeze ALL params
    67	    m.eval()                     # disable dropout/BN-update; deterministic features
    68	    m.to(device)
    69	    hidden = m.embed_dim         # 384 for ViT-S; 384 for vits16 too
    70	    # assert relaxed for dinov3 vits16 (still ~384d small); caller derives
    71	    if "vits" in variant:
    72	        assert hidden == 384, f"expected 384 for small ViT-S, got {hidden}"
    73	    return m, hidden
    74	
    75	
    76	@torch.inference_mode()
    77	def extract(m, x, layer: str = "last", patch_l2: bool = False):
    78	    """Extract (cls, patch) tokens from frozen backbone (conceded engine, no-FT default).
    79	
    80	    layer: "last" (forward_features default) | list[int]/"intermed" for get_intermediate_layers
    81	           (MCP-confirmed dinov3 vision_transformer contract: n=..., return_class_token=True, norm=True
    82	            for dense localized features).
    83	    patch_l2: apply F.normalize(..., p=2, dim=-1) gated by caller config (e.g. corn.yaml patch_mode=l2).
    84	    Supports dinov3_vits16 richer oob frozen dense per-AU (orbital/ear/muzzle) for small-N imbal ordinal AU.
    85	    Updates callers + provenance/schema (the 'hash' of config for repro/G3).
    86	    """
    87	    if layer == "last" or layer is None:
    88	        out = m.forward_features(x)  # dict
    89	        cls = out["x_norm_clstoken"]        # [B,384]   (CLS, == index 0 of the token seq)
    90	        patch = out["x_norm_patchtokens"]   # [B,N,384] (patch tokens; HF-equiv of [:,1:,:])
    91	    else:
    92	        # MCP dinov3: get_intermediate_layers for multi-layer dense/localized (eval encoder, classifiers)
    93	        # e.g. n=[5,11,17,23] or range; norm=True, return_class_token=True per vision_transformer + notebooks
    94	        # PRACTICAL A/B richer: for n=list do mean+std concat across layers for denser localized cues (orbital/ear/muzzle) + L2 full
    95	        if isinstance(layer, str) and layer.lower() == "intermed":
    96	            layer = [-4, -3, -2, -1]  # richer last-4 intermed per MCP dinov3 contract for dense localized
    97	        n = layer if isinstance(layer, (list, tuple, range)) else [-1]
    98	        inter = m.get_intermediate_layers(x, n=n, reshape=False, return_class_token=True, norm=True)
    99	        if inter:
   100	            # richer: concat mean+std of patch feats from selected layers (or single); supports full intermed mean_std
   101	            patches = [i[0] for i in inter]  # patch tokens per layer
   102	            if len(patches) > 1:
   103	                pcat = torch.cat(patches, dim=-1)  # concat layers first for richer dim
   104	                pmean = pcat.mean(dim=1)  # [B, D*nl]
   105	                pstd = pcat.std(dim=1)
   106	                patch = torch.cat([pmean, pstd], dim=-1)  # mean+std concat richer
   107	                # cls from last
   108	                cls = inter[-1][1]
   109	            else:
   110	                patch, cls = inter[-1][0], inter[-1][1]  # patch + cls from selected (last) intermed layer
   111	                # for single also support mean_std style if needed
   112	                if patch.shape[1] > 0:  # ensure
   113	                    pmean = patch.mean(dim=1)
   114	                    pstd = patch.std(dim=1)
   115	                    # keep patch as-is for compat; richer via caller pool or intermed mode
   116	        else:
   117	            out = m.forward_features(x)
   118	            cls = out["x_norm_clstoken"]
   119	            patch = out["x_norm_patchtokens"]
   120	    if patch_l2:
   121	        patch = F.normalize(patch, p=2, dim=-1)
   122	    return cls, patch
## src/model/cache_features.py
     1	"""Feature caching to disk (IMPLEMENTATION_PLAN §5.2) -- the big M4 win.
     2	
     3	CONCEDED PLUMBING. The frozen DINOv2 backbone is NEVER claimed as novel
     4	(FINAL_DIRECTION §A); it only carries the binary-plus-wrapper spine.
     5	
     6	Principle (GITHUB_MINE P2.2): the backbone is frozen, so its output for a given
     7	crop never changes. Run ONE pass over every crop, write pooled features to disk,
     8	then train the heads off the cache with ZERO backbone forwards per epoch. On M4
     9	this turns each head epoch from minutes into milliseconds and makes the §5.6
    10	pooling x loss x seed sweep cheap.
    11	
    12	The cache holds BOTH poolings (CLS + mean-pooled patch tokens) so the trainer can
    13	flip `pool` with no re-extraction. Labels carry a -1 sentinel where the VLM/vet
    14	did not score that AU; multi_corn_loss skips sentinel AUs row-wise.
    15	
    16	Datasheet line: PROVENANCE.json records the DINOv2 commit, preprocess params,
    17	device, and torch version -- the cache is an input to EVERY reported number.
    18	"""
    19	
    20	import csv
    21	import json
    22	import platform
    23	import subprocess
    24	from datetime import datetime, timezone
    25	from pathlib import Path
    26	
    27	import numpy as np
    28	import torch
    29	from PIL import Image
    30	from torchvision import transforms
    31	
    32	from src.constants import AU_ORDER
    33	from src.model.backbone import (
    34	    DEFAULT_VARIANT,
    35	    IMNET_MEAN,
    36	    IMNET_STD,
    37	    VARIANT_INPUT_SIZES,
    38	    extract,
    39	    load_frozen_dinov2,
    40	)
    41	from src.model.device import DEVICE
    42	
    43	# manifest_csv columns (IMPLEMENTATION_PLAN §5.2):
    44	#   img_path, cat_id, fold,
    45	#   y_ear, y_orbital, y_muzzle, y_whiskers, y_head,  (-1 sentinel where unscored)
    46	#   y_pain, is_vet_clean
    47	AUS = AU_ORDER
    48	
    49	# _CACHE_SCHEMA (hard enforce per FreshDINOv3Context7RicherEnforcer audit + separability/train tests):
    50	# Full keys + values for versioned cache (no legacy allow_pickle drift; supports richer patch_std / layer intermed / L2 / dinov3_vits16 A/B per MCP /facebookresearch/dinov3 "dense features without fine-tuning" + exact forward_features x_norm_* + get_intermediate_layers contract).
    51	# extract() now extended (layer='last'|'intermed' + patch_l2 gated by config; see backbone.py) + hash via layer/patch_mode in schema/provenance.
    52	# Preserves ALL: cat-disjoint manifests, vet firewall early, provenance sidecar, single-source AU_ORDER, G3 abort, conceded FM, portable protocols compatibility.
    53	_CACHE_SCHEMA = {
    54	    "schema_version": "v1_dinov3_richer_20260614",
    55	    "n_aus": len(AU_ORDER),
    56	    "au_hash": hash(tuple(AU_ORDER)),  # simple for enforce (use stable in prod)
    57	    "k": 3,  # ordinal levels 0/1/2
    58	    "feature_dim": 384,  # ViT-S small (v2 reg + dinov3_vits16)
    59	    "variant": DEFAULT_VARIANT,  # will be overridden per build
    60	    "layer": "last",  # "last" | list[int] for get_intermediate_layers n= per MCP
    61	    "patch_mode": "mean_std",  # mean_std | l2 | intermed  (richer for localized facial AU orbital/ear/muzzle vs mean only)
    62	}
    63	# Uniform assert no legacy (enforce per @FreshHandoffDINOv3RicherEnforcer MCP audit + full embed): 
    64	# all cache writes/loads must carry schema_version + layer/patch_mode + variant + au_hash; 
    65	# A/B hash for dinov3_vits16 primary vs dinov2_reg; richer intermed n=list mean+std + L2 per public dinov github contract (vision_transformer forward_features x_norm_* + get_intermediate_layers).
    66	assert _CACHE_SCHEMA["schema_version"].startswith("v1_dinov3_richer"), "legacy schema drift forbidden"
    67	SCHEMA_VERSION = _CACHE_SCHEMA["schema_version"]
    68	
    69	
    70	def _dinov2_commit() -> str:
    71	    """Best-effort torch.hub DINOv2 checkout commit for the datasheet.
    72	
    73	    Returns 'unknown' rather than raising so caching never fails on provenance."""
    74	    hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    75	    try:
    76	        out = subprocess.run(
    77	            ["git", "-C", str(hub_dir), "rev-parse", "HEAD"],
    78	            capture_output=True, text=True, check=True,
    79	        )
    80	        return out.stdout.strip()
    81	    except Exception:
    82	        return "unknown"
    83	
    84	
    85	def _write_provenance(out_npz, device: str, n_rows: int, variant: str, layer: str = "last", patch_mode: str = "mean_std") -> Path:
    86	    """Write <cache-stem>.PROVENANCE.json next to the cache (datasheet line).
    87	
    88	    Named after the npz so the cat and horse caches each keep their own
    89	    provenance instead of overwriting a shared file. Records the backbone
    90	    variant + layer/patch_mode (dinov3 richer MCP) so a reported number can never hide which engine
    91	    produced its features. Cross-checked in separability + train + _CACHE_SCHEMA enforce."""
    92	    out_npz = Path(out_npz)
    93	    prov_path = out_npz.parent / f"{out_npz.stem}.PROVENANCE.json"
    94	    prov_path.parent.mkdir(parents=True, exist_ok=True)
    95	    prov = {
    96	        "cache_npz": str(out_npz),
    97	        "n_rows": n_rows,
    98	        "backbone_variant": variant,  # dinov2_vits14_reg (field default) or dinov2_vits14 or dinov3_vits16 (MCP primary A/B dense oob)
    99	        "layer": layer,
   100	        "patch_mode": patch_mode,  # mean_std (default richer) | l2 | intermed per MCP get_intermediate_layers n=list + L2 patch
   101	        "dinov_commit": _dinov2_commit() if not variant.startswith("dinov3") else "dinov3_hub_mcp",
   102	        "preprocess": {
   103	            "resize": VARIANT_INPUT_SIZES.get(variant, 518),
   104	            "center_crop": VARIANT_INPUT_SIZES.get(variant, 518),
   105	            "interpolation": "bicubic",
   106	            "normalize_mean": [0.485, 0.456, 0.406],
   107	            "normalize_std": [0.229, 0.224, 0.225],
   108	        },
   109	        "device": device,
   110	        "torch_version": torch.__version__,
   111	        "python_version": platform.python_version(),
   112	        "platform": platform.platform(),
   113	        "built_at_utc": datetime.now(timezone.utc).isoformat(),
   114	        "_CACHE_SCHEMA": _CACHE_SCHEMA,  # hard embed for cross enforce
   115	    }
   116	    prov_path.write_text(json.dumps(prov, indent=2))
   117	    return prov_path
   118	
   119	
   120	def build_cache(manifest_csv, out_npz, device: str = DEVICE, variant: str = DEFAULT_VARIANT, layer: str = "last", patch_mode: str = "mean_std"):
   121	    """One frozen-DINO (v2 or dinov3_vits16) forward over every crop in manifest_csv -> out_npz.
   122	
   123	    variant: dinov2_vits14_reg (default) | dinov2_vits14 | dinov3_vits16 (MCP context7 primary A/B
   124	      for dense oob frozen high-quality localized features without fine-tuning; same
   125	      forward_features x_norm_clstoken/x_norm_patchtokens contract + get_intermediate_layers).
   126	    layer: "last" | n=list[int] (e.g. [-4,-3,-2,-1] or [5,11,17,23]) passed to get_intermediate_layers for multi-layer dense per MCP.
   127	    patch_mode: "mean_std" (richer: mean+std of patches for localized AU cues) | "l2" (full L2 norm on patch feats) | "intermed" (concat/mean intermed layers).
   128	      Per corn.yaml + separability _POOLS incl patch_std; hard _CACHE_SCHEMA enforce.
   129	
   130	    Saves cls + richer patch_* + 5-AU labels (-1 sentinel) + y_pain + is_vet_clean + schema keys,
   131	    writes PROVENANCE.json (extended with layer/patch_mode). Hard schema on write (no drift).
   132	    """
   133	    out_npz = Path(out_npz)
   134	    out_npz.parent.mkdir(parents=True, exist_ok=True)
   135	
   136	    m, hidden = load_frozen_dinov2(device, variant=variant)
   137	    # relaxed assert for dinov3 vits16 (still 384d small model)
   138	    if variant.startswith("dinov3"):
   139	        assert 380 <= hidden <= 390, f"expected ~384 for small ViT-S dinov3, got {hidden}"
   140	    else:
   141	        assert hidden == 384, f"expected 384-d ViT-S features, got {hidden}"
   142	
   143	    # per-variant size (MCP dinov3 vits16 16px patch; caller ensures divisible)
   144	    input_size = VARIANT_INPUT_SIZES.get(variant, 518)
   145	    # Note: preprocess is global 518; for dinov3 A/B in prod use variant-specific compose or resize in loop (here stub for contract; real uses corn.yaml crop_edge)
   146	
   147	    rows, cls_feats, patch_mean_feats, patch_std_feats, patch_l2_feats = [], [], [], [], []
   148	    intermed_cache = []  # for intermed mode
   149	    with open(manifest_csv) as f:
   150	        for r in csv.DictReader(f):
   151	            # simple resize per variant for contract (full prod wires VARIANT_INPUT_SIZES + crop)
   152	            img_pil = Image.open(r["img_path"]).convert("RGB")
   153	            img = transforms.Compose([
   154	                transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
   155	                transforms.CenterCrop(input_size),
   156	                transforms.ToTensor(),
   157	                transforms.Normalize(IMNET_MEAN, IMNET_STD),
   158	            ])(img_pil)[None].to(device)
   159	            cls, patch = extract(m, img, layer=layer, patch_l2=(patch_mode == "l2"))
   160	            cls_feats.append(cls.squeeze(0).cpu().numpy())
   161	            pmean = patch.mean(1).squeeze(0).cpu().numpy()
   162	            pstd = patch.std(1).squeeze(0).cpu().numpy() if patch_mode in ("mean_std", "std") else np.zeros_like(pmean)
   163	            # l2 now handled inside extract when patch_l2=True (per DINOv3RicherSub extension + MCP L2 F.normalize p=2)
   164	            pl2 = patch.mean(1).squeeze(0).cpu().numpy() if patch_mode == "l2" else np.zeros_like(pmean)
   165	            patch_mean_feats.append(pmean)
   166	            patch_std_feats.append(pstd)
   167	            patch_l2_feats.append(pl2)
   168	            # intermed: via layer=list in extract (MCP get_intermediate_layers n=... norm/return_class); richer mean+std concat for dense localized (align backbone per dinov3 oob contract); no legacy compat
   169	            if patch_mode == "intermed" and hasattr(m, "get_intermediate_layers"):
   170	                n_layers = layer if isinstance(layer, (list, tuple, range)) else [-1]
   171	                inter = m.get_intermediate_layers(img, n=n_layers, reshape=False, return_class_token=True, norm=True)
   172	                if inter:
   173	                    patches = [i[0].cpu().numpy() for i in inter]
   174	                    if len(patches) > 1:
   175	                        pcat = np.concatenate(patches, axis=-1)
   176	                        pmean = pcat.mean(axis=1)
   177	                        pstd = pcat.std(axis=1)
   178	                        inter_patch = np.concatenate([pmean, pstd], axis=-1)
   179	                    else:
   180	                        p = patches[0]
## src/vlm/aggregate.py
     1	"""In-code aggregation: the 0-10 sum and the 0.39 flag (IMPLEMENTATION_PLAN §4.3).
     2	
     3	THIS is the single threshold definition reused by both the VLM path and the engine
     4	(CORN) decode path -- the Gate-4 unit test pins ``decode -> per-AU 0-2 -> sum -> 0.39``
     5	against this exact contract so there is ONE threshold definition in the repo.
     6	
     7	The VLM NEVER computes any of these; it emits only the 5 atoms in {0,1,2}. The
     8	0.39 flag is triage decision-support only -- never an autonomous analgesia trigger
     9	-- and the sum it rides on is an INSPECTED-NOT-VALIDATED quantity.
    10	"""
    11	
    12	# Fixed Evangelista 5-AU order — canonical declaration in src.constants.AU_ORDER
    13	# (a leaf module, so this import adds no schema/anthropic-stack dependency).
    14	from src.constants import AU_ORDER
    15	
    16	AU_NAMES = list(AU_ORDER)
    17	
    18	# THE one 0.39 definition (Evangelista sum/10 cut, ~4/10). Engine decode
    19	# (src.model.decode.point_sum) and the abstention band import it from here;
    20	# never re-declare the literal elsewhere.
    21	POINT_DECISION_THRESHOLD = 0.39
    22	
    23	
    24	def fgs_sum(result_dict):
    25	    """Sum the 5 per-AU scores into 0..10. ``result_dict[au]['score']`` in {0,1,2}."""
    26	    return sum(result_dict[au]["score"] for au in AU_NAMES)
    27	
    28	
    29	def analgesia_flag(s):
    30	    """Triage flag: ratio = sum/10; clinical cut at >= 0.39 (~4/10).
    31	
    32	    Works on scalars, numpy arrays, and torch tensors (s/10.0 promotes to float)."""
    33	    return (s / 10.0) >= POINT_DECISION_THRESHOLD
    34	
    35	
    36	def any_abstain(result_dict):
    37	    """For routing/triage only, not a clinical output."""
    38	    return any(result_dict[au]["abstain"] for au in AU_NAMES) or (
    39	        result_dict["image_quality"] != "frontal_clear"
    40	    )
## src/model/decode.py
     1	"""Distributional decode (IMPLEMENTATION_PLAN §5.5).
     2	
     3	PORTABLE PLUMBING — supporting infrastructure for the headline confound-attribution
     4	protocol and its guarded kappa check, not a contribution in its own right.
     5	
     6	DEFAULT path (FINAL_DIRECTION §E.1): keep soft cumulative P(rank>k); build each
     7	AU's pmf over {0,1,2}; convolve the 5 pmfs into one pmf over the 0-10 sum. From
     8	that distribution come RPS-on-the-sum (one scalar, bootstrap CI) + per-AU
     9	ClasswiseECE (computed downstream in src/eval/). NO binned reliability diagram on
    10	the 11-atom sum (degenerate at ~11 atoms).
    11	
    12	argmax / hard-decode (point_sum) is reserved for the 0.39 POINT decision ONLY.
    13	The 0-10 sum is INSPECTED-NOT-VALIDATED; no validated-claim number is emitted here.
    14	"""
    15	
    16	import numpy as np
    17	import torch
    18	
    19	from src.constants import AU_ORDER
    20	from src.model.corn import corn_label_from_logits
    21	
    22	
    23	N_DEFAULT = len(AU_ORDER)
    24	
    25	
    26	def au_pmf_from_cumprobs(cum):          # cum: [B,2] = [P(y>0), P(y>0 & y>1)]
    27	    p_gt0, p_gt1 = cum[:, 0], cum[:, 1]
    28	    p0 = 1 - p_gt0
    29	    p1 = p_gt0 - p_gt1                   # = P(y>0) - P(y>1)
    30	    p2 = p_gt1
    31	    pmf = torch.stack([p0, p1, p2], dim=1)          # [B,3]
    32	    return torch.clamp(pmf, min=0)      # guard tiny negatives from float error
    33	
    34	
    35	def sum_pmf(au_pmfs):                    # au_pmfs: list of 5 x [B,3] (numpy)
    36	    # convolve per-AU pmfs -> pmf over 0..(n_aus * max_au_score)
    37	    B = au_pmfs[0].shape[0]
    38	    n_aus = len(au_pmfs)
    39	    max_sum = sum(int(p.shape[1]) - 1 for p in au_pmfs)
    40	    out = np.zeros((B, max_sum + 1))
    41	    for b in range(B):
    42	        acc = np.array([1.0])
    43	        for a in range(n_aus):
    44	            acc = np.convolve(acc, au_pmfs[a][b])
    45	        out[b] = acc / acc.sum()         # renormalize
    46	    return out                           # [B,max_sum+1], sums to 1 over the sum support
    47	
    48	
    49	def point_sum(logits_list):              # hard decode for the decision ONLY
    50	    # threshold imported from src.vlm.aggregate — the single 0.39 definition in
    51	    # the repo, shared with the VLM path so the two flags can never drift.
    52	    from src.vlm.aggregate import analgesia_flag
    53	
    54	    labels = [corn_label_from_logits(lg) for lg in logits_list]   # 5 x [B]
    55	    s = torch.stack(labels, dim=1).sum(1)        # [B] in 0..10
    56	    return s, analgesia_flag(s.float())          # painful flag
    57	
    58	
    59	def pmf_entropy(pmf):
    60	    """Entropy of pmf (scalar or per-row).
    61	
    62	    Used for active VLM/abstention: high entropy = high uncertainty (CORN pmf driver).
    63	    Per-AU: call on au_pmf_from_cumprobs output; image-level: on sum_pmf output.
    64	    Preserves atoms-only contract: unc computed in code, never from VLM.
    65	    Pure-np portable derive (N/k from input shape; use protocols + adapters.generic_ordinal_mode for non-FGS).
    66	    """
    67	    p = np.asarray(pmf, dtype=float)
    68	    if p.ndim == 1:
    69	        p = p / (p.sum() + 1e-12)
    70	        p = np.clip(p, 1e-12, 1.0)
    71	        return float(-np.sum(p * np.log(p)))
    72	    else:
    73	        # batch: per-row entropy
    74	        p = p / (p.sum(axis=-1, keepdims=True) + 1e-12)
    75	        p = np.clip(p, 1e-12, 1.0)
    76	        return -np.sum(p * np.log(p), axis=-1)
    77	
    78	
    79	def au_pmf_entropies(au_pmfs_list):
    80	    """List of per-AU mean entropy over batch (for triage/score)."""
    81	    return [float(np.mean(pmf_entropy(au))) for au in au_pmfs_list]
## configs/corn.yaml
     1	# ENGINE config (conceded plumbing, never claimed novel).
     2	# Frozen DINOv2 ViT-S/14 (register variant default) + 5 per-AU CORN heads. Distributional decode is DEFAULT.
     3	num_aus: 5                  # 5 per-AU CORN heads (one per FGS action unit)
     4	K: 3                        # ordinal levels per AU -> {0,1,2}, so K-1 = 2 CORN logits per head
     5	
     6	backbone:
     7	  # torch.hub id consumed by src.model.backbone (reg = field default; plain = A/B comparator).
     8	  # _reg registers suppress attention artifacts that hurt dense, localized per-AU FGS
     9	  # features (orbital/ear/muzzle); A/B vs plain dinov2_vits14 before locking.
    10	  # DINOv3: dinov3_vits16 (MCP: same forward_features x_norm_clstoken/patchtokens + get_intermediate_layers
    11	  # for dense localized AU cues; frozen no-FT small data per context7).
    12	  name: dinov2_vits14_reg
    13	  frozen: true
    14	  hidden_dim: 384
    15	  patch_size: 14
    16	  crop_edge: 518            # crop input edge (518 = 37 patches * 14)
    17	  # A/B knob for richer patch per MCP dinov3 (mean_std default; l2 or intermed for orbital/ear localization; dinov3_vits16 primary successor per FINAL A/B + context7 'dense w/o FT' 'oob' for small/medical/facial/ordinal; exact forward x_norm + get_intermed n=list/L2/mean(patch))
    18	  patch_mode: mean_std      # mean_std | l2 | intermed  (updates cache build + load pool support; patch_std richer for localized AU)
    19	  layer: last               # or n=[-4,-3,-2,-1] (list/seq/range) via get_intermediate_layers (MCP n=... for multi-layer dense); dinov3_vits16 A/B primary (compat default v2_reg)
    20	  # dinov3_vits16 promote: backbone load + corn A/B + e2e portable matrix probe (MCP 374 snips + landed contract)
    21	  # Enforce richer (per DINOv3RicherEnforcer): intermed n=list -> mean+std concat in backbone extract + cache; L2 option; uniform no-legacy asserts + A/B hash in cache/train; dinov3_vits16 primary for dense oob frozen small-N imbal ordinal AU (orbital/ear/muzzle) vs priors no FM. Full schema embed + prov cross. 
    22	
    23	decode:
    24	  mode: distributional      # DEFAULT: P(rank>k)=cumprod(sigmoid(logits)) -> per-AU pmf -> convolve to 0-10 sum pmf
    25	  # Hard argmax decode is used ONLY for the 0.39 point decision; NO binned reliability
    26	  # diagram on the 11-atom 0-10 sum (RPS-on-sum + per-AU ClasswiseECE only).
    27	  # The 0.39 itself is NOT configurable here: the single definition lives in
    28	  # src/vlm/aggregate.py POINT_DECISION_THRESHOLD (Gate-4-pinned decode contract).
    29	
    30	co_teach:
    31	  num_gradual: 10           # epochs to ramp keep-rate 1.0 -> (1 - tau); canonical Han et al. 2018 R(T)
    32	  # tau = estimated VLM noise rate. RUNTIME-OWNED by Gate-1-B: train_corn.py reads
    33	  # est_noise_rate.overall from artifacts/gate1b/kappa_report.json (--tau overrides).
    34	  # Deliberately NOT a config key — a placeholder here silently drops good rows.
    35	  # vet-clean rows (is_vet_clean==1) are NEVER dropped from either selection.
    36	
    37	# Head-training knobs (frozen-backbone linear probe, ~120-300 labels, M4/MPS).
    38	# Literal defaults mirror IMPLEMENTATION_PLAN §5.6 table.
    39	train:
    40	  optimizer: adamw
    41	  lr: 1.0e-3                # tiny linear probe; AdamW stable
    42	  weight_decay: 1.0e-2
    43	  epochs: 80               # 60-100; default 80. converges fast on cached feats
    44	  batch_size: 32           # whole train set fits one batch on [N,384] in RAM
    45	  dropout: 0.1             # light reg
    46	  head_init_std: 2.0e-5    # trunc_normal_ std; small-data linear-probe stability
    47	  num_workers: 0           # small data / MPS; deterministic loading
    48	  pool: cls                # cls (default) vs patch_mean; cache holds both
    49	  loss: corn               # corn (default) vs coral ablation
    50	  train_folds: [0, 1, 2]   # cat-disjoint G3 folds used for training
    51	  seeds: [0, 1, 2, 3, 4]   # 5 seeds; report mean +/- bootstrap CI (n is small)
    52	  au_weights: null         # optional [1,1,1.5,1.5,1] to upweight muzzle/whiskers imbalance
    53	
    54	# Sibling BINARY pain head (the v1 SPINE). Linear(384,1) + BCEWithLogitsLoss(pos_weight).
    55	# Trained on y_pain, independent of CORN; this is the validated-claim path. The 0-10 CORN
    56	# sum is INSPECTED-NOT-VALIDATED and never emits a validated-claim number.
    57	binary_pain:
    58	  pos_weight: null         # RUNTIME: set to n_neg/n_pos from the train split if null
## configs/power.yaml
     1	# GATE 0 — power calcs + vet-budget pre-registration (blocks all quantitative work).
     2	#
     3	# PLACEHOLDERS owned by Gate 0 / the clinician (replace before any quantitative claim):
     4	#   - vet_budget_integer : the single committed integer — how many faces the vet scores.
     5	#     ~120 is an UNCONFIRMED placeholder; Gate 0 writes the frozen value. TODO(Gate 0).
     6	#   - min_pain_pos       : minimum pain-positive faces inside the budget (>= 50). TODO(Gate 0).
     7	# Until Gate 0 commits these, no downstream kappa / NPV-LB / 0.39 number is believed.
     8	
     9	kappa:
    10	  ci_half_width_max: 0.15  # per-AU quadratic kappa CI half-width target (kappaSize); G0 gate
    11	
    12	npv_lb:
    13	  target: 0.90             # NPV lower-bound target (MAPIE / LTT) for the abstention curve
    14	  max_abstention: 0.40     # certified at <= 40% abstention
    15	
    16	# The 0.39 point-decision threshold is deliberately NOT a key here: gate0_power.py
    17	# imports the single definition (src.vlm.aggregate.POINT_DECISION_THRESHOLD), so the
    18	# pre-registered power calc can never drift from the operating threshold.
    19	
    20	vet_budget_integer: 120    # PLACEHOLDER ~120 — TODO(Gate 0): commit the frozen integer
    21	min_pain_pos: 50           # PLACEHOLDER >= 50 pain-positive faces — TODO(Gate 0)
## artifacts/gate1b/kappa_report.json
     1	{
     2	  "gate": "1B",
     3	  "floors_source": "power.json",
     4	  "n_boot": 5000,
     5	  "alpha": 0.05,
     6	  "per_au": {
     7	    "ear": {
     8	      "kappa": 0.8717948717948718,
     9	      "ci_lb": 0.7407407407407407,
    10	      "floor": 0.6,
    11	      "pass": true,
    12	      "degenerate": false,
    13	      "n": 30,
    14	      "distinct_pain_cats": 6
    15	    },
    16	    "orbital": {
    17	      "kappa": 0.5263157894736843,
    18	      "ci_lb": 0.25,
    19	      "floor": 0.6,
    20	      "pass": false,
    21	      "degenerate": false,
    22	      "n": 30,
    23	      "distinct_pain_cats": 6
    24	    },
    25	    "muzzle": {
    26	      "kappa": 0.6739130434782609,
    27	      "ci_lb": 0.4382022471910113,
    28	      "floor": 0.4,
    29	      "pass": true,
    30	      "degenerate": false,
    31	      "n": 30,
    32	      "distinct_pain_cats": 6
    33	    },
    34	    "whiskers": {
    35	      "kappa": 0.8,
    36	      "ci_lb": 0.6622516556291391,
    37	      "floor": 0.4,
    38	      "pass": true,
    39	      "degenerate": false,
    40	      "n": 30,
    41	      "distinct_pain_cats": 6
    42	    },
    43	    "head": {
    44	      "kappa": 0.5377503852080123,
    45	      "ci_lb": 0.2941176470588235,
    46	      "floor": 0.6,
    47	      "pass": false,
    48	      "degenerate": false,
    49	      "n": 30,
    50	      "distinct_pain_cats": 6
    51	    }
    52	  },
    53	  "est_noise_rate": {
    54	    "per_au": {
    55	      "ear": 0.13333333333333333,
    56	      "orbital": 0.3,
    57	      "muzzle": 0.16666666666666666,
    58	      "whiskers": 0.2,
    59	      "head": 0.26666666666666666
    60	    },
    61	    "overall": 0.21333333333333332
    62	  },
    63	  "decision": {
    64	    "graded_go": false,
    65	    "muzzle_whiskers": {
    66	      "muzzle": "ok",
    67	      "whiskers": "ok"
    68	    },
    69	    "on_core_fail": "PIVOT to binary spine (drop the 0-10 layer); confound-attribution headline is unaffected, and this kappa check remains kill-tree insurance"
    70	  },
    71	  "interpretation_guard": {
    72	    "note": "A high kappa only measures capability if BOTH: (i) the vet anchor is independent, AND (ii) the VLM rubric differs from the vet rubric. Otherwise it measures rubric-following. See README section 1 Interpretation Guard.",
    73	    "anchor_independent": null,
    74	    "rubric_match": null
    75	  }
    76	}## artifacts/gates_run_20260630_032044.json
     1	{
     2	  "ts": "20260630_032044",
     3	  "synthetic": true,
     4	  "portable_only": true,
     5	  "order": [
     6	    "gate0",
     7	    "gate1",
     8	    "gate2",
     9	    "gate3",
    10	    "gate4",
    11	    "gate5",
    12	    "gate1b",
    13	    "gate6"
    14	  ],
    15	  "results": {
    16	    "gate0": {
    17	      "exit": 0,
    18	      "output_tail": "DRY-RUN"
    19	    },
    20	    "gate1": {
    21	      "skipped": "portable_only",
    22	      "note": "see src.protocols for full isolation"
    23	    },
    24	    "gate2": {
    25	      "exit": 0,
    26	      "output_tail": "DRY-RUN"
    27	    },
    28	    "gate3": {
    29	      "skipped": "portable_only",
    30	      "note": "see src.protocols for full isolation"
    31	    },
    32	    "gate4": {
    33	      "skipped": "portable_only",
    34	      "note": "see src.protocols for full isolation"
    35	    },
    36	    "gate5": {
    37	      "skipped": "portable_only",
    38	      "note": "see src.protocols for full isolation"
    39	    },
    40	    "gate1b": {
    41	      "exit": 0,
    42	      "output_tail": "DRY-RUN"
    43	    },
    44	    "gate6": {
    45	      "skipped": "portable_only",
    46	      "note": "see src.protocols for full isolation"
    47	    }
    48	  },
    49	  "artifacts_verified": {
    50	    "gate0": [
    51	      [
    52	        "data/manifests/power.json",
    53	        true
    54	      ]
    55	    ],
    56	    "gate2": [
    57	      [
    58	        "artifacts/gate2/confound_audit.json",
    59	        true
    60	      ]
    61	    ],
    62	    "gate1b": [
    63	      [
    64	        "artifacts/gate1b/kappa_report.json",
    65	        true
    66	      ]
    67	    ]
    68	  },
    69	  "aborted_at": null,
    70	  "power_first": true,
    71	  "honesty_notes": [
    72	    "G0 power pre-registration first (blocks quantitative).",
    73	    "Single-source 0.39 via src.vlm.aggregate (decode + VLM paths).",
    74	    "One-directional G2; CI-LB G1b; hash G3; BLOCKING G4.",
    75	    "All via committed manifests/artifacts (uniqueness enforcement).",
    76	    "Portable protocols (src.protocols) surface: deletion-safe / import-isolated kappa+confound+adapters reusable on the next corpus."
    77	  ],
    78	  "toy_dir": "/var/folders/xx/68h1kgjd235cjzz7w6px59580000gn/T/gate_e2e_toy_wxjs0hcb",
    79	  "overall": "PASS",
    80	  "note": "Gates enforce order + artifact uniqueness + power-first (G0). --synthetic for safe e2e/repro."
    81	}