# Roboflow as synced dataset workspace

Status: accepted

Roboflow is a good fit for the LINE OA prototype's consented data pipeline as an annotation, dataset management, dataset-versioning, and selected training workspace. The LINE backend should remain the system of record for consent, pseudonymous identity, submission IDs, and provenance, then sync de-identified or pseudonymous images and derived annotations into Roboflow.

## Considered Options

- **Backend source of truth + Roboflow synced workspace**: chosen because it combines Roboflow convenience with defensible consent/provenance control.
- **Roboflow as the only source of truth**: not chosen because consent state, withdrawal/deletion, pseudonymous identity, LINE event lineage, and research audit logs are safer in our own backend.
- **No Roboflow**: not chosen because Roboflow can reduce annotation, dataset versioning, and model-training overhead.

## Consequences

The ingestion pipeline should upload consented images, crops, masks, and supported annotation formats to Roboflow, using batches, tags, and metadata such as `submission_id`, `consent_version`, `quality_state`, `score_band`, and `model_version`. Raw LINE identifiers should not be sent. Final FGS scoring, calibration, abstention, vet-anchor tracking, and research metrics may remain outside Roboflow if they require custom ordinal or statistical logic.
