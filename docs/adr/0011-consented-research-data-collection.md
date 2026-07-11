# Consented research data collection

Status: accepted

The LINE OA prototype may collect and store submitted cat photos for research and model improvement, including derived classifications, segmentation masks, crops, quality states, action-unit estimates, and metadata. This collection must be consented research storage, not silent collection, because LINE photo intake can create privacy, trust, and PDPA compliance risk.

## Considered Options

- **Consented research storage with derived annotations**: chosen because it supports dataset growth while keeping the prototype defensible.
- **No long-term storage**: rejected because it blocks dataset improvement and future model training.
- **Silent/default storage without clear consent**: rejected because it creates unnecessary legal and trust risk.

## Consequences

The system must separate normal image processing for a result from long-term research storage. The consent copy should explain storage purpose, data types, retention, deletion/withdrawal route, and research/model-improvement use. The stored dataset should track provenance and derived annotations so it can support later classification, segmentation, and model evaluation.

## Notes

As of 2026-07-01, Thailand PDPA is active. Current design should treat consent, purpose limitation, retention notice, and data-subject rights as engineering requirements, not optional paperwork.
