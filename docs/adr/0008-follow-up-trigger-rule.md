# Follow-up trigger rule

Status: accepted

The LINE OA prototype will ask an automatic one-question follow-up when the model-estimated FGS-style score is 4-10. For scores 0-3, the prototype will not force a question but may offer an optional more-guidance action.

## Considered Options

- **Automatic one-question follow-up for 4-10, optional for 0-3**: chosen because symptom context matters once visible facial concern starts, while low-score cases should remain low-friction.
- **Automatic follow-up for every score**: rejected because it makes the image-first flow feel like a questionnaire.
- **Follow-up only after user request**: rejected because moderate/high scores need safer context before final guidance.

## Consequences

The follow-up question must be short, structured, and answerable quickly in LINE. It should influence concern guidance without turning the prototype into a diagnostic interview.
