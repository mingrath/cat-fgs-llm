# Image-first flow with conditional follow-up

Status: accepted

The LINE OA prototype will use an image-first flow: the caregiver primarily sends a cat photo and receives a model-estimated FGS-style score, component estimates, and concern guidance. Follow-up questions are allowed only when needed for safety, uncertainty, image quality, or additional guidance.

## Considered Options

- **Image-first flow with conditional follow-up**: chosen because it matches LINE user behavior and keeps the prototype low-friction while preserving safety options.
- **Questionnaire-first flow**: rejected because it would make the prototype feel heavier and reduce the value of photo-first pain screening.
- **Photo-only with no questions ever**: rejected because some cases need context to avoid unsafe reassurance or unclear guidance.

## Consequences

The prototype must define explicit trigger rules for when a follow-up question appears. Emergency safety wording can appear in every result without forcing a questionnaire.
