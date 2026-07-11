# ULW Research Notepad

Tier: HEAVY — user explicitly requested `omo:ulw-research`; project + external literature + pipeline recommendation require exhaustive research.

Skills selected:
- `omo:ulw-research` — explicit user request; controls research swarm, journal, synthesis.
- `agent-reach` — task is research/search; use citation-backed external retrieval.

Success criteria:
1. Map current project pipeline/code/data assumptions from repo evidence.
2. Identify most recent and strongest techniques for cat FGS pain detection, including model families, labels, datasets, and deployment constraints.
3. Compare candidate improvements against this repo’s current pipeline and recommend the highest-value upgrade path.
4. Cite every claim with sources or local verification artifacts; record unresolved gaps.
5. Produce `SYNTHESIS.md` and final concise answer with evidence refs.


## Final status
- Synthesis complete: `SYNTHESIS.md`.
- HTML report complete: `REPORT.html`.
- Claim ledger complete: `claim-ledger.md`.
- Expansion convergence logged: `expansion-log.md`.
- Verification:
  - `verify-local-contracts.md` PASS.
  - `verify-targeted-tests.md` PASS (`12 passed`).
  - `verify-portable-suite.md` PASS.
- Gate reviewer: APPROVED (`.omo/evidence/cat-fgs-pain-detector-ulw-research-gate-review.md`).

Final answer: recommend DINOv3/DINOv2-reg measured A/B + expert-vet AU anchor + threshold safety/calibration/abstention, not direct VLM clinical scoring or a pure landmark replacement.

## Stop-hook continuation verification
- Artifact integrity: PASS.
- Fresh targeted tests: `uv run pytest tests/test_backbone_variant.py tests/test_threshold_single_source.py tests/test_gate4_decode.py -q` => `12 passed in 1.20s`.
- Reviewer approval still present in `.omo/evidence/cat-fgs-pain-detector-ulw-research-gate-review.md`.
- `omx cancel` executed: `Cancelled: ultrawork`.
