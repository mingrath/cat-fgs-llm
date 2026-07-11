# Wave 1 — OSS architecture / practical CV stack

Worker: 019f1691-f14b-7de3-ad04-879dacfd9c13
Axis: latest practical CV building blocks for a cat FGS pipeline.

## Key findings
- Best ROI stack for small veterinary data: YOLO/RF-DETR for face localization; SAM2 only for masks/video; DINOv2 or SigLIP2/ConvNeXt for transfer; shallow head; CORN when labels are ordinal; temperature scaling/calibration; temporal smoothing for video; Grad-CAM for audit only.
- RF-DETR is practical and already in repo, but heavier than YOLO and best if transformer detector/keypoints/segmentation path matters.
- SAM2 is mature for image/video promptable segmentation and multi-object tracking; valuable only if crop stability/masks/videos are key.
- DINOv2 remains mature for frozen feature + linear/simple heads; SigLIP2 offers semantic/dense improvements but adds complexity; ConvNeXt is a strong supervised baseline.
- Landmark tools (MMPose/DeepLabCut) need cat-specific labeling; not the best immediate v1 path.

## Source anchors from worker
- RF-DETR: https://github.com/roboflow/rf-detr/blob/7a4bdbe92c446c57c2bba7c4f085ca6b7a4ba72d/README.md
- SAM2: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/README.md
- DINOv2: https://github.com/facebookresearch/dinov2/blob/7764ea0f912e53c92e82eb78a2a1631e92725fc8/README.md
- SigLIP2: https://github.com/google-research/big_vision/blob/0127fb6b337ee2a27bf4e54dea79cff176527356/big_vision/configs/proj/image_text/README_siglip2.md
- CLIP: https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/README.md
- ConvNeXt: https://github.com/facebookresearch/ConvNeXt/blob/048efcea897d999aed302f2639b6270aedf8d4c8/models/convnext.py
- Ultralytics: https://github.com/ultralytics/ultralytics/blob/1934d8a2c52e5d4644b4fcec81c970161879a432/docs/en/models/yolov8.md
- MMPose: https://mmpose.readthedocs.io/en/latest/overview.html
- DeepLabCut: https://deeplabcut.github.io/DeepLabCut/docs/maDLC_UserGuide.html
- Grad-CAM: https://github.com/jacobgil/pytorch-grad-cam
- CORN/CORAL: https://github.com/Raschka-research-group/coral-pytorch and https://github.com/Raschka-research-group/corn-ordinal-neuralnet
- Temperature scaling: https://arxiv.org/abs/1706.04599

## EXPAND markers verbatim
- Benchmark recent cat-face / feline veterinary datasets and training recipes.
- Pull exact paper SHAs / arXiv versions for CORN, calibration, and recent ordinal/uncertainty work.
- Compare RF-DETR vs YOLO on small custom-data transfer cost.
- Find any cat-specific landmark / feline pain / FGS OSS repos, if they exist.

## CLAIMS markers verbatim
- RF-DETR, SAM2, DINOv2, SigLIP2, CLIP, ConvNeXt, YOLO, MMPose, DeepLabCut, Grad-CAM, CORN, and temperature scaling are all evidenced above with source-linked snippets.
- For small veterinary data, the safest default is pretrained backbone + small head + calibration + smoothing, not training a large model from scratch.
- Cat-face landmarking is the weakest-maturity area here; general animal-pose tooling exists, but cat-specific out-of-the-box support is sparse.
