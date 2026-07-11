# Wave 1 — Codebase docs / decision record

Worker: 019f1691-d329-7252-aa2c-f233f45402e1
Axis: existing research/docs decisions and claimed future improvements.

## Key findings
- Repo already reframed away from “another cat-pain detector”: headline is the portable confound-attribution protocol; VLM-as-AU-rater κ is a guarded secondary protocol pending independent vet anchor; engine is conceded plumbing.
- V1 spine is binary pain/no_pain + welfare wrapper + abstention. Graded 0–10 is inspect-only until independent vet labels and adequate AU=2 cells exist.
- Highest unimplemented blocker: independent vet anchor. Biggest data gap: no open graded cat-FGS ground truth; current corpus is binary/pseudo-labeled and confounded.
- Technical follow-up: κ bootstrap clustering by `cat_id` noted as correctness gap in docs/history.

## Sources / local refs from worker
- README.md:3, README.md:20, README.md:51, README.md:60, README.md:72, README.md:83, README.md:124
- FINAL_DIRECTION.md:10, FINAL_DIRECTION.md:27, FINAL_DIRECTION.md:33, FINAL_DIRECTION.md:36, FINAL_DIRECTION.md:49, FINAL_DIRECTION.md:57, FINAL_DIRECTION.md:72
- IMPLEMENTATION_PLAN.md:137, IMPLEMENTATION_PLAN.md:459, IMPLEMENTATION_PLAN.md:1088, IMPLEMENTATION_PLAN.md:1416, IMPLEMENTATION_PLAN.md:1988, IMPLEMENTATION_PLAN.md:2110
- GAP_ANALYSIS.md:56, GAP_ANALYSIS.md:79, GAP_ANALYSIS.md:84, GAP_ANALYSIS.md:181
- DATA_DECISION.md:12, DATA_DECISION.md:22, DATA_DECISION.md:49
- FACTCHECK.md:3, FACTCHECK.md:51, FACTCHECK.md:88
- HF_SOLUTION.md:5, HF_SOLUTION.md:31, HF_SOLUTION.md:63
- GITHUB_MINE.md:5, GITHUB_MINE.md:190, GITHUB_MINE.md:204, GITHUB_MINE.md:230
- paper/sections/00-abstract.tex:23, paper/sections/01-introduction.tex:166, paper/sections/08-ethics-welfare-limitations.tex:89

## EXPAND markers verbatim
Worker returned malformed tail only: `## EXPAND tail`. Treat as no concrete lead; follow-up not required because body included actionable leads already known: independent vet anchor, graded dataset gap, κ cluster bootstrap, DINOv3/prompt/rubric future work.
