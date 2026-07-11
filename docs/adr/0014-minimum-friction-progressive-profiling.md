# Minimum-friction progressive profiling

Status: accepted

The LINE OA prototype will prioritize a low-friction image-first experience and avoid asking users too many questions before returning value. Dataset fields should be collected through progressive profiling: ask only when the information naturally improves the current result, safety guidance, repeated-use experience, or research consent flow.

## Considered Options

- **Minimum-friction progressive profiling**: chosen because it supports dataset growth without making the prototype feel like a research form.
- **Up-front cat profile and context form**: rejected because it adds friction before the caregiver receives a result.
- **No contextual questions ever**: rejected because symptom and identity context can improve safety, grouping, and future model evaluation.

## Consequences

The default flow remains send photo → receive score. Cat identity and context should be collected only when blended into existing value moments, such as symptom follow-up, repeated submissions, or optional research contribution prompts.
