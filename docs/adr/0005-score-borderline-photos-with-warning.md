# Score borderline photos with a warning

Status: accepted

The LINE OA prototype will show a model-estimated FGS-style score for borderline-quality photos when the image still passes the minimum quality gate. The score must be paired with a low-confidence warning that explains the photo quality may reduce reliability.

## Considered Options

- **Score borderline photos with warning**: chosen because Thai LINE users may often submit imperfect photos, and returning no result too often would hurt the prototype experience.
- **Require retake before scoring**: rejected because it may create too much friction for the target LINE OA workflow.
- **Always score every photo**: rejected because images below the minimum quality gate remain unscorable.

## Consequences

The system needs at least three quality states: scorable, borderline-scorable, and cannot-score. Evaluation must track not only model accuracy but also how often Thai real-world photos fall into each quality state.
