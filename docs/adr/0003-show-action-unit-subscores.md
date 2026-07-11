# Show action-unit subscores as score details

Status: accepted

The LINE OA prototype will show the overall 0-10 model-estimated FGS-style score first, then show five 0-2 action-unit subscores as supporting details. This preserves the caregiver-facing clarity of a single score while making the result more scientifically interpretable and tied to the Feline Grimace Scale structure.

## Considered Options

- **Total score + action-unit details**: chosen because it gives a clear result and lets interested caregivers see which facial components influenced the estimate.
- **Total score only**: rejected because it hides the FGS structure and makes the result feel like an unexplained black box.
- **Full action-unit breakdown upfront**: rejected because it may overemphasize numerical precision and confuse caregivers.

## Consequences

The UI must phrase action-unit subscores as model-estimated component details, not definitive facial or muscle findings. The overall recommendation remains driven by the total score, image quality, uncertainty, and cannot-score rules.
