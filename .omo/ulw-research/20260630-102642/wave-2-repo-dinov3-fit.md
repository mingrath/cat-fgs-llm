# Wave 2 — Repo DINOv3 / cache fit

Worker: 019f1695-bec2-7102-84d0-b913aeed07d1
Axis: repo-fit of DINOv3 and cache changes.

## Key findings
- DINOv3 is already partially supported as an A/B successor path, not empirically validated default.
- `src/model/backbone.py` supports `dinov2_vits14_reg`, `dinov2_vits14`, `dinov3_vits16`; default remains `dinov2_vits14_reg`.
- Cache schema persists `variant`, `layer`, `patch_mode`; current feature dimension contract is 384.
- Breakpoints: patch-size/crop contract (`dinov3_vits16` vs 14-patch path), schema rigidity around `feature_dim=384`, evaluation harness bug risk in separability probe, A/B missing.
- Recommendation: keep `_reg` default; run DINOv3 as explicit alternate behind current cache/provenance; do not widen schema until empirical gain.

## Source anchors from worker
- `src/model/backbone.py:29-35`, `src/model/backbone.py:45-75`, `src/model/backbone.py:78-108`
- `src/model/cache_features.py:53-66`, `src/model/cache_features.py:120-213`
- `src/model/train_heads.py:43-52`
- `src/eval/separability.py:29-31`, `src/eval/separability.py:77-110`
- `configs/corn.yaml:10-21`
- `tests/test_backbone_variant.py:17-40`
- `tests/test_separability_probe.py:43-56`
- `scripts/cache_features.py:83-116`
- `README.md:35-42`, `README.md:122`
- `docs/architecture_audit_report.md:13-18`

## EXPAND markers verbatim
Worker returned `## EXPAND tail` without concrete new leads. Treat as no new actionable lead; DINOv3 A/B already covered.
