# Multi-select symptom check

Status: accepted

The LINE OA prototype will use a structured multi-select symptom check as the one automatic follow-up question for scores 4-10. The question asks whether the cat recently showed eating less, hiding or moving less, limping or guarding a body part, unusual vocalizing, emergency signs, none of these, or not sure.

## Considered Options

- **Multi-select symptom check**: chosen because it is fast in LINE, easy to localize, and gives enough context to adjust concern guidance.
- **Free-text symptom question**: rejected because it is harder to parse safely and may make the prototype behave like a diagnostic chatbot.
- **No symptom question**: rejected because score-only guidance can miss emergency or behavior context.

## Consequences

Emergency signs must override the image score and produce urgent veterinary guidance. Non-emergency symptoms may strengthen the recommendation but must not become a diagnosis.
