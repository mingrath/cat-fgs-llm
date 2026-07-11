# Closure — malformed EXPAND tails

## wave-1-codebase-docs-decisions.md
Worker returned `## EXPAND tail` instead of the required bullet list. Body still surfaced concrete leads:
- independent vet anchor
- public graded dataset gap
- κ cluster bootstrap
- DINOv3/prompt/rubric future work

Closure:
- Independent vet anchor investigated in `wave-2-fgs-vlm-vet-anchor.md`.
- Public graded dataset gap investigated in `wave-1-datasets-oss.md`.
- DINOv3 investigated in `wave-2-repo-dinov3-fit.md`.
- κ cluster bootstrap is repo-hardening, not technique-selection blocker; local tests include `tests/test_kappa_cluster_bootstrap.py` existence from codebase lane, but not run in targeted verification.

## wave-2-repo-dinov3-fit.md
Worker returned `## EXPAND tail` without bullets. Body concluded no new actionable leads beyond DINOv3 empirical A/B and separability harness validation.

Closure:
- DINOv3 A/B is the central recommendation in `SYNTHESIS.md`.
- Separability harness validation is an implementation-hardening prerequisite before trusting A/B diagnostics; noted as caveat.
- No further EXPAND lead required for the research answer.
