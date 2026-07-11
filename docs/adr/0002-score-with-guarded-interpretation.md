# Score with guarded interpretation

Status: accepted

The LINE OA prototype will show a 0-10 model-estimated FGS-style score because the caregiver experience benefits from a concrete result and the Feline Grimace Scale provides a scientifically grounded scoring structure. The score will be presented as a prototype model estimate rather than an official clinical FGS result or diagnosis. Every score must be paired with a cautious interpretation, veterinary-assessment guidance, and a cannot-score path for insufficient photos.

## Considered Options

- **0-10 model-estimated FGS-style score + guarded interpretation**: chosen because it uses the FGS scoring structure and satisfies the desired user experience while reducing false clinical certainty.
- **0-100 concern score**: rejected because it is harder to explain scientifically and less directly tied to published FGS scoring.
- **Triage category only**: safer but rejected because it does not satisfy the desired prototype experience.
- **Raw official clinical FGS score only**: rejected because it would imply unsupported diagnostic validity.

## Consequences

The UI may say the score is based on Feline Grimace Scale action-unit structure, but must not imply veterinarian-confirmed diagnosis or treatment need. Clinic and specialist recommendations are explicitly later-version scope.
