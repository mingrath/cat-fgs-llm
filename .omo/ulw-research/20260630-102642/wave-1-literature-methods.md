# Wave 1 — Literature methods 2024–2026

Worker: 019f1691-dd61-7ad1-8dbf-a6f04b150c9d
Axis: recent automated FGS / cat pain techniques.

## Key findings
- No verified 2026 primary cat-pain/FGS method paper surfaced by 2026-06-30.
- Strongest FGS-specific automation remains 2023 smartphone FGS pipeline: landmark CNN + geometric descriptors + FGS prediction.
- Strongest fully automated temporal pain detector is 2024 raw-video landmark pipeline: automated face+48-landmark detection + temporal modeling; 70%/66% accuracy in two datasets.
- 2025 segment-based explainability framework reports DINO-pretrained ViT classifier accuracy 0.86 on cat pain dataset, but is an XAI framework, not a dedicated detector; evidence supports DINO-family features as plausible upgrade candidate.
- 2025 chatbot/VLM FGS paper is negative/cautionary: poor agreement / clinical risk; VLMs cannot replace vet validation.
- Landmark methods repeatedly beat plain CNNs on small/mid cat pain datasets; temporal signal matters; generalization remains modest.

## Source anchors from worker
- 2024 video landmarks: https://www.nature.com/articles/s41598-024-78406-2 ; PubMed https://pubmed.ncbi.nlm.nih.gov/39543343/ ; DOI 10.1038/s41598-024-78406-2
- 2023 smartphone FGS: https://www.nature.com/articles/s41598-023-49031-2 ; PubMed https://pubmed.ncbi.nlm.nih.gov/38062194/ ; DOI 10.1038/s41598-023-49031-2
- 2024 automated landmark-based analysis: https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2024.1442634/full ; DOI 10.3389/fvets.2024.1442634
- 2023 explainable automated pain: https://www.nature.com/articles/s41598-023-35846-6 ; PubMed https://pubmed.ncbi.nlm.nih.gov/37268666/ ; DOI 10.1038/s41598-023-35846-6
- 2025 segment XAI: https://www.nature.com/articles/s41598-025-96634-y ; PMC https://pmc.ncbi.nlm.nih.gov/articles/PMC12012102/ ; DOI 10.1038/s41598-025-96634-y
- 2025 chatbot FGS: https://www.nature.com/articles/s41598-025-27404-z ; DOI 10.1038/s41598-025-27404-z
- 2025 IEEE keypoint+FGS lead: https://ieeexplore.ieee.org/document/10961966/ ; DOI 10.1109/ECTIDAMTNCON64748.2025.10961966
- 2024 CatFLW landmark detector: https://link.springer.com/article/10.1007/s11263-024-02006-w ; arXiv https://arxiv.org/abs/2310.09793 ; DOI 10.1007/s11263-024-02006-w
- 2022 automated recognition baseline: https://www.nature.com/articles/s41598-022-13348-1 ; PubMed https://pubmed.ncbi.nlm.nih.gov/35688852/ ; DOI 10.1038/s41598-022-13348-1

## EXPAND markers verbatim
- I can turn this into:
  - CSV / BibTeX
  - a PRISMA-style search log
  - an annotated bibliography
  - a ranked “methods vs datasets vs metrics” matrix

## CLAIMS markers verbatim
- I verified primary evidence for the main cat pain / FGS automation papers from **2022, 2023, 2024, and 2025**.
- The latest verified primary method lead in the search set is **December 2025**; I found **no verified 2026 primary method paper**.
- The field’s main technical arc is: **manual landmarks → automated landmarks → smartphone FGS → temporal video pipelines → explainability frameworks**.
- The current ceiling is still modest for raw-video pain detection, while FGS scoring automation is strongest when the model is still-image + landmark/geometry based.
