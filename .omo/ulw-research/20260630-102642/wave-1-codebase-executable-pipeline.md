# Wave 1 — Codebase executable pipeline

Worker: 019f1691-c82a-77c2-ac77-88835917e9e6
Axis: current executable pipeline and model architecture.

## Key findings
- Executable architecture: ingest/merge/split → detector/crop/quality gate → frozen backbone feature cache → CORN heads + sibling binary pain head → VLM weak-labeling + kappa protocol → wrapper/decision curve → confound/separability/e2e tests.
- Central wiring: `src/gates/orchestrator.py`, `Makefile`, `configs/*.yaml`, gate scripts.
- Backbone loader already supports `dinov2_vits14`, `dinov2_vits14_reg`, and `dinov3_vits16`; current docs/config choose `dinov2_vits14_reg` as default.
- Highest practical upgrade seams: crop/detect quality, VLM weak-label/vet anchor, detector branch choice, cache schema, test gaps.

## Sources / local refs from worker
- `src/data/parse.py`, `src/data/folds.py`
- `src/crop/pipeline.py`, `src/crop/quality_gate.py`, `src/crop/align_eyes.py`, `configs/crop.yaml`
- `src/detect/train_rfdetr.py`, `src/detect/train_yolo.py`, `configs/detect_rfdetr.yaml`, `configs/detect_yolo.yaml`
- `src/model/backbone.py`, `src/model/cache_features.py`, `src/model/heads.py`, `src/model/corn.py`, `src/model/decode.py`, `src/model/train_heads.py`, `configs/corn.yaml`
- `src/vlm/schema.py`, `src/vlm/rubric.py`, `src/vlm/call.py`, `src/vlm/batch_submit.py`, `src/vlm/batch_collect.py`, `src/vlm/aggregate.py`, `configs/vlm_fgs.yaml`
- `src/wrapper/operating_point.py`, `src/wrapper/decision_curve.py`, `src/eval/*`
- `tests/test_*`, `docs/architecture_audit_report.md`

## EXPAND markers verbatim
- LEAD: shared 0.39 threshold coupling — WHY: highest cross-layer risk and most central invariant — ANGLE: trace all imports/uses of `POINT_DECISION_THRESHOLD`, `analgesia_flag`, and `point_sum`
- LEAD: crop/detect fragility — WHY: upstream quality dominates all downstream numbers — ANGLE: inspect `scripts/gate5_nme.py`, `src/crop/quality_gate.py`, `src/crop/align_eyes.py`, detector training configs
- LEAD: VLM weak-label bottleneck — WHY: kappa pilot is the weakest unclosed empirical dependency — ANGLE: inspect `scripts/run_vlm_labels.py`, `scripts/gate1b_kappa_pilot.py`, `src/eval/kappa.py`, `configs/vlm_fgs.yaml`
- LEAD: cache/schema rigidity — WHY: any representation change will break training and evaluation unless coordinated — ANGLE: trace `_CACHE_SCHEMA`, provenance, and all feature-cache consumers
- LEAD: detector branch choice — WHY: RF-DETR vs YOLO is the biggest active model fork — ANGLE: inspect `configs/detect_rfdetr.yaml`, `configs/detect_yolo.yaml`, downstream data interface assumptions
- LEAD: test coverage gaps — WHY: docs already flag missing direct tests on crop/dedup/train surfaces — ANGLE: inspect `tests/test_no_test_leak.py`, `docs/architecture_audit_report.md`, and add coverage map from source to tests
