# Expansion Log

## Phase 0
- Opened session and axes.
- Initial wave planned across repo, web, OSS, datasets, deployment, skeptic.

## Wave 1 return — codebase-docs-decisions
- Journaled worker 019f1691-d329-7252-aa2c-f233f45402e1.
- New leads: independent vet anchor; public graded dataset gap; κ cluster bootstrap; DINOv3/prompt/rubric improvements.
- Status: leads opened for expansion wave 2 unless already covered by active workers.

## Wave 1 return — codebase-executable-pipeline
- Journaled worker 019f1691-c82a-77c2-ac77-88835917e9e6.
- Leads opened: 0.39 coupling; crop/detect fragility; VLM bottleneck; cache/schema; detector branch; test coverage.
- Some are local implementation hardening, not central to technique recommendation; keep as repo-fit constraints.

## Wave 1 return — oss-architecture
- Journaled worker 019f1691-f14b-7de3-ad04-879dacfd9c13.
- Leads opened: recent cat datasets/training recipes; RF-DETR vs YOLO transfer cost; FGS OSS repos.

## Wave 1 return — skeptic-risk
- Journaled worker 019f1691-fb4f-7851-9f9c-a8af9f7eb741.
- Lead opened: claim-by-claim model-card checklist.

## Wave 1 return — datasets-oss
- Journaled worker 019f1691-e74b-7cd1-b33f-e08e5c4da182.
- Lead opened: citation/supplement mining around 2019 and 2024 papers.

## Wave 2 return — repo-dinov3-fit
- Journaled worker 019f1695-bec2-7102-84d0-b913aeed07d1.
- New actionable leads: none beyond DINOv3 empirical A/B and separability harness validation.

## Wave 1 return — literature-methods
- Journaled worker 019f1691-dd61-7ad1-8dbf-a6f04b150c9d.
- Leads: PRISMA/bibliography/matrix; not necessary for concise final answer, covered by synthesis.

## Wave 2 convergence status
- Opened DINOv3 repo-fit expansion: returned, no new core leads.
- Opened FGS/VLM/vet-anchor expansion: still pending at time of ledger draft; external Exa/web evidence independently captured.
- Literature methods worker returned; requested optional bibliography/matrix only, no core unchecked scientific lead.
- Remaining leads are report-format or implementation hardening extras: model-card sentence list, PRISMA/BibTeX, deeper RF-DETR vs YOLO benchmark. They are not necessary to answer the user's technique-choice question.
- Convergence reason: two waves completed for core axes; no unchecked lead changes the recommendation.

## Wave 2 return — fgs-vlm-vet-anchor
- Journaled worker 019f1695-cae9-7e11-9cd5-a8deed96da4a.
- New leads are detailed validation design/report tables; core recommendation unchanged.

## Final convergence
- Zero unchecked core leads remain for the user's question.
- Optional leads (BibTeX, PRISMA, model-card wording, prospective validation design) are report extensions, not blockers.
- Convergence reached after two waves.

## Rejection fix — malformed EXPAND closure
- Added `closure-malformed-expand.md`.
- Closed malformed tails by extracting leads from worker bodies and mapping each to journaled wave or caveat.

## Rejection fix — deployment/product constraints
- Added `wave-3-deployment-product-constraints.md`.
- Product/deployment axis closed.

## Rejection fix — optional lead closure
- Added `closure-optional-leads.md` covering RF-DETR vs YOLO, validation design, model-card wording, methods matrix.

## Rejection fix — recency claim
- Added `recency-search-log.md`.
- Downgraded synthesis wording to search-bounded claim.

## Rejection fix — HTML/worktree
- Re-rendered `REPORT.html` with h1/h2/ul structure and checklist pass.
- Added `worktree-scope.md` explicitly excluding unrelated pre-existing dirty code diff from research deliverable.

## Rejection fix — synthesis/report refresh
- Updated `SYNTHESIS.md` with deployment constraints, optional lead closures, recency log, portable verification, and dirty diff exclusion.
- Re-rendered `REPORT.html`; checklist passed: no raw headings, no raw bold markdown, no escape characters.
