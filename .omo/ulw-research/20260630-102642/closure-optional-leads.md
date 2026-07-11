# Closure — material optional leads

## RF-DETR vs YOLO
- Evidence from OSS lane: YOLO is easiest/mature practical detector; RF-DETR is modern, transformer-based, already in repo, heavier but plausible.
- Repo fit: project already has both `src/detect/train_rfdetr.py` and `src/detect/train_yolo.py`; detector choice is upstream crop quality, not current strongest bottleneck.
- Closure: do not make detector swap the main recommendation unless Gate 5/NME/crop quality fails. Keep RF-DETR/YOLO as an A/B front-end, lower priority than vet anchor and backbone A/B.

## Validation design
- Evidence from FGS/VLM lane: new detector must validate expert AU agreement, 0.39 threshold safety, Bland–Altman bias/LoA, analgesia responsiveness, breed/subgroup robustness, image/video quality robustness.
- Closure: folded into `SYNTHESIS.md` recommendation and product constraints.

## Model-card wording
Allowed:
> Prototype decision-support research system for feline facial pain assessment; binary triage spine and portable confound-audit protocol; graded 0–10 FGS inspected-not-validated pending independent veterinary AU labels and subgroup validation.
Forbidden:
> validated cat pain detector; diagnoses pain; owner app ready; VLM labels are reliable; no confounding; guaranteed NPV; breed-invariant; beats prior work.
- Closure: included in final answer risk language.

## Methods matrix
| Method | Evidence strength | Repo fit | Recommendation |
|---|---|---|---|
| DINOv2-reg | local default + mature DINOv2 | high | keep baseline |
| DINOv3 | newest foundation features + local branch | medium-high | A/B candidate, not default |
| SigLIP2/ConvNeXt | OSS/research support | medium | optional baseline |
| Landmark-XGBoost | strongest direct FGS prior but request-only/closed data | medium | reference/interpretability, not replacement |
| VLM direct scoring | 2025 negative/caution | low | do not use clinically |
| YOLO/RF-DETR | mature detector options | high | A/B only if crop quality blocks |
| SAM2/video smoothing | temporal/crop value | conditional | use for video/occlusion, not first still-image ROI |

