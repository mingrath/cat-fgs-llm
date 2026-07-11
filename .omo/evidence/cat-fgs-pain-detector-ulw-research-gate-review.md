# Gate Review — cat FGS pain detector ULW research re-review

recommendation: APPROVE

## blockers
None.

## originalIntent
User asked for `omo:ulw-research` on this project: if building a new cat FGS pain detector, what recent technique makes the pipeline worth improving?

## desiredOutcome
A research deliverable, not code implementation: `SYNTHESIS.md` and `REPORT.html` should answer the technique-choice question with cited, repo-grounded, recent evidence; avoid clinical/VLM overclaiming; include claim ledger, expansion/convergence evidence, and verification artifacts.

## userOutcomeReview
The refreshed deliverable now satisfies the requested outcome. It gives a concrete recommendation: vet-anchor-first foundation-feature A/B (`dinov2_vits14_reg` vs `dinov3_vits16` plus a strong baseline such as ConvNeXt/SigLIP2), with calibration/abstention and independent veterinary AU anchoring before graded FGS claims. It explicitly rejects unsafe direct VLM clinical scoring and owner-app/diagnosis claims. The recommendation is grounded in local repo contracts and recent external literature, and the remaining gaps are accurately framed as future validation needs rather than solved facts.

## checked artifact paths
- `.omo/ulw-research/20260630-102642/SYNTHESIS.md`
- `.omo/ulw-research/20260630-102642/REPORT.html`
- `.omo/ulw-research/20260630-102642/claim-ledger.md`
- `.omo/ulw-research/20260630-102642/expansion-log.md`
- `.omo/ulw-research/20260630-102642/closure-malformed-expand.md`
- `.omo/ulw-research/20260630-102642/wave-3-deployment-product-constraints.md`
- `.omo/ulw-research/20260630-102642/closure-optional-leads.md`
- `.omo/ulw-research/20260630-102642/recency-search-log.md`
- `.omo/ulw-research/20260630-102642/worktree-scope.md`
- `.omo/ulw-research/20260630-102642/verify-local-contracts.md`
- `.omo/ulw-research/20260630-102642/verify-targeted-tests.md`
- `.omo/ulw-research/20260630-102642/verify-portable-suite.md`
- Fresh commands: `uv run python .omo/ulw-research/20260630-102642/verify-local-contracts.py`; `uv run pytest tests/test_backbone_variant.py tests/test_threshold_single_source.py tests/test_gate4_decode.py -q`; partial `make test-portable` until user requested immediate verdict.

## previous rejection items
1. Malformed/missing `## EXPAND` tails: fixed by `closure-malformed-expand.md`, mapping extracted leads to wave artifacts or non-blocking caveats.
2. Deployment/product axis: fixed by `wave-3-deployment-product-constraints.md`, covering inference/product posture, direct VLM risk, mobile/clinic constraints, licensing, crop/occlusion/defer behavior, subgroup/breed risk.
3. Optional material leads: fixed by `closure-optional-leads.md`, covering RF-DETR vs YOLO, validation design, model-card wording, and methods matrix.
4. Recency wording/ledger: fixed. `SYNTHESIS.md` and `claim-ledger.md` use search-bounded wording: no verified 2026 primary cat-pain/FGS method paper surfaced, not universal nonexistence. Fresh web spot-check found known 2023 automated FGS, 2024 video-landmark, 2025 chatbot/VLM, and 2026 observer-gender FGS paper, but no 2026 automated primary method counterexample.
5. HTML polish: fixed. `REPORT.html` has rendered `<h1>/<h2>/<ul>/<strong>/<code>` markup; direct raw Markdown check found `**`: 0, backtick: 0, `## `: 0.
6. Dirty diff/code-review scope gap: fixed by `worktree-scope.md`. The dirty code files are explicitly outside this research deliverable; only `.omo/ulw-research/20260630-102642/` artifacts are in reviewed scope for this re-review.

## remove-ai-slops / programming pass
Direct anti-slop pass found no unresolved blocker in the deliverables. Tests are not used as tautological proof of the research claim; they verify local repo contracts only. Production code dirty diff is excluded from scope by `worktree-scope.md`, so no unresolved programming/slop blocker remains for this research artifact review.

## verification evidence
- Fresh local contract command: PASS; output confirms DINOv3 support path, DINOv2-reg configured default, 0.39 point threshold, and 384-d feature cache contract.
- Fresh targeted tests: PASS; `12 passed in 1.69s`.
- Portable suite artifact: `verify-portable-suite.md` records `make test-portable` pass, including standalone portable tests and synthetic orchestrator PASS lines. Fresh rerun was interrupted only because user requested immediate verdict; completed portions showed standalone corpus PASS and first pytest segment `6 passed, 10 warnings` before interruption.
- Fresh web spot-check: no 2026 automated cat-pain/FGS method paper surfaced; a 2026 Scientific Reports observer-gender FGS paper is not an automated detector method.

## exact evidence gaps
None required for approval. Remaining scientific/product gaps are correctly disclosed inside `SYNTHESIS.md` as future work: independent vet AU labels, public graded dataset absence, empirical A/B not yet run, subgroup robustness not validated.
