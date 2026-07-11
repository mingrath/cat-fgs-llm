# Pseudonymous LINE identity for research records

Status: accepted

Consented research records will store a pseudonymous LINE user or session identifier, not raw personal identity. This allows grouping repeated submissions, detecting leakage between train/test splits, and analyzing real-world usage while reducing privacy risk.

## Considered Options

- **Store pseudonymous LINE identity**: chosen because research evaluation needs grouping and repeated-submission tracking.
- **Store no identity link**: rejected because duplicate or repeated cat photos could contaminate evaluation splits.
- **Store raw LINE identity**: rejected because it creates unnecessary privacy risk for the research dataset.

## Consequences

The ingestion pipeline should generate durable `submission_id` values and store pseudonymous identity separately from raw LINE platform identifiers. The dataset schema should include consent version, submitted photo URI, crop/mask URIs, quality state, score estimates, symptom answers, model version, and creation time.
