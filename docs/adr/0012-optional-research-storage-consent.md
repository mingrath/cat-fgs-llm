# Optional research storage consent

Status: accepted

Research storage consent will be optional but strongly encouraged. A caregiver can receive a LINE OA prototype result without agreeing to long-term storage, but the prototype will clearly invite them to contribute the submitted photo and derived annotations to improve the research model.

## Considered Options

- **Optional but strongly encouraged consent**: chosen because it supports dataset growth while preserving trust and cleaner consent.
- **Required consent before using the prototype**: rejected because it increases friction and weakens the voluntariness of consent.
- **No consent prompt**: rejected because it creates privacy, PDPA, and trust risk.

## Consequences

The system must support separate processing paths: transient result generation for non-consenting submissions and durable dataset ingestion for consenting submissions. Consent state must be stored with every durable research record.
