"""Gates package: central orchestration (scripts-only delegation).

Enforces run-order, artifacts, abort-on-fail, --synthetic (deletion-safe toy e2e).
Gates as strict honesty/uniqueness layer (power G0-first, single-source contracts, committed manifests).
"""

from .orchestrator import run_pipeline  # noqa: F401
