# Quality gate with retake guidance

Status: accepted

The LINE OA prototype will include an image-quality gate before showing any model-estimated FGS-style score. The pipeline should tolerate ordinary real-world Thai caregiver photos, including imperfect lighting and camera quality, but must return a cannot-score response when the face, crop, pose, occlusion, or image quality is below a defined minimum threshold.

## Considered Options

- **Quality gate + retake guidance**: chosen because it protects trust while still supporting real-world LINE usage.
- **Always show a score**: rejected because poor-image scores would look precise but be unreliable.
- **Require ideal photos only**: rejected because it would fail too often in the target Thai LINE OA environment.

## Consequences

The prototype must include retake instructions when quality is too low, such as asking for a front-facing photo, better lighting, visible eyes/ears/muzzle/whiskers, and no heavy occlusion. Quality-gate thresholds become part of model evaluation, not just UI behavior.
