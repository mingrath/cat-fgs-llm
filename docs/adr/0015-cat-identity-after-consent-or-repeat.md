# Cat identity after consent or repeat submission

Status: accepted

The LINE OA prototype will ask cat identity only after research storage consent or on repeat submission. It will not ask cat identity during the automatic symptom follow-up, because that follow-up is safety-critical and should stay focused.

## Considered Options

- **Ask cat identity after consent or repeat submission**: chosen because it supports cat-level grouping while preserving the low-friction image-first flow.
- **Ask cat identity during symptom follow-up for scores 4-10**: rejected because it mixes research profiling into a safety moment.
- **Never ask cat identity**: rejected because repeated-cat grouping is valuable for leakage control and longitudinal dataset quality.

## Consequences

The first submission can remain photo → score. Cat identity collection should be optional and framed as helping improve future tracking, not as a requirement for receiving the prototype result.
