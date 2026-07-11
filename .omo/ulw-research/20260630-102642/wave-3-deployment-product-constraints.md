# Wave 3 — Deployment / product constraints

Axis: deployment/product constraints for a new cat FGS pain detector.

## Findings
- Product posture must be decision-support triage, not autonomous diagnosis or analgesia trigger. Local repo states this in README/paper; skeptic lane confirmed external evidence makes stronger claims unsafe.
- Direct VLM/chatbot scoring is not product-safe because 2025 FGS chatbot LoA crosses the 0.39 analgesia threshold. Use VLM only as candidate weak-labeler behind vet-anchor gates.
- Owner-facing deployment requires: explicit consent/image policy, defer-to-vet behavior for poor crop/occlusion/e-collar, breed/morphology subgroup reporting, and no “no pain” reassurance from low-confidence negatives.
- M4/local deployment: frozen-feature cache + shallow heads are practical; full training of large foundation models is not. DINOv3 can be feature-extractor A/B, not an end-user claim.
- Smartphone/mobile: the strongest direct FGS prior is the 2023 smartphone-app paper, but its dataset is not public and it is not evidence this repo’s model generalizes.
- Commercial constraints: CatFLW is CC BY-NC 4.0 per dataset lane, so do not build commercial aligner/product dependencies on CatFLW-derived weights without license review.

## Sources
- Local: README.md:20-28, README.md:60-76, paper/sections/08-ethics-welfare-limitations.tex from skeptic lane.
- VLM/chatbot risk: https://www.nature.com/articles/s41598-025-27404-z
- Smartphone prior: https://www.nature.com/articles/s41598-023-49031-2
- CatFLW dataset/license: https://www.kaggle.com/datasets/georgemartvel/catflw and https://github.com/martvelge/CatFLW

## Closure
Deployment/product axis is covered. It reinforces the final recommendation: improve the pipeline with measured backbone/vet-anchor/calibration gates, not a standalone clinical app.
