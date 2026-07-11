# Wave 1 — Datasets and public assets

Worker: 019f1691-e74b-7cd1-b33f-e08e5c4da182
Axis: public datasets, labels, Kaggle/Hugging Face/GitHub assets.

## Key findings
- No public downloadable graded cat-FGS pain dataset surfaced across web, GitHub, Hugging Face, Kaggle, Nature/Springer, Zenodo, figshare, Dataverse.
- Pain/FGS papers’ datasets are request-only or not public.
- Usable public assets are geometry/pretraining assets, not pain ground truth: CatFLW, Cat Dataset, cats_vs_dogs, Bingsu/Cat_and_Dog.
- Useful OSS references: myayash/cat-pain-recog, koshian2/cats-face-landmarks, WhiskerWatch.

## Source anchors from worker
- CatFLW Kaggle: https://www.kaggle.com/datasets/georgemartvel/catflw ; README mirror https://github.com/martvelge/CatFLW/blob/11f15176bdd7a449e378a6157a85b2ebec876159/README.md
- Cat Dataset Kaggle: https://www.kaggle.com/datasets/crawford/cat-dataset
- HF cats_vs_dogs: https://huggingface.co/datasets/microsoft/cats_vs_dogs
- HF Bingsu/Cat_and_Dog: https://huggingface.co/datasets/Bingsu/Cat_and_Dog
- FGS paper: https://www.nature.com/articles/s41598-019-55693-8
- Automated recognition: https://www.nature.com/articles/s41598-022-13348-1
- Smartphone/deep learning FGS: https://www.nature.com/articles/s41598-023-49031-2
- Video landmarks: https://www.nature.com/articles/s41598-024-78406-2
- myayash/cat-pain-recog: https://github.com/myayash/cat-pain-recog
- koshian2/cats-face-landmarks: https://github.com/koshian2/cats-face-landmarks
- WhiskerWatch: https://github.com/waleedalfar/whisker-watch

## EXPAND markers verbatim
- Next best expansion path: mine the supplementary material / citation graph around the 2024 landmark paper and the 2019 FGS paper for any hidden download links, author-hosted mirrors, or annotation tooling.

## CLAIMS markers verbatim
- CatFLW = 2079 images, 48 landmarks, bbox, Kaggle access, CC BY-NC 4.0, per repo README and Kaggle page.
- FGS label schema = 5 AUs, 0–2 each, 110 scored images, per 2019 Scientific Reports paper.
- Public pain datasets were not found; the main pain papers say “request to corresponding author” / not publicly available.
- The GitHub repos above are real, public, and directly usable as code/asset references for cat-face detection and FGS-style pipelines.
