# cat-fgs-llm — thin task runner. One target per gate (strict run-order; each gate
# blocks downstream). Gate 4 (test) must be green before any quantitative target is
# believed. All code runs through `uv run`.
#
# Run order: gate0 -> gate1 -> gate2 -> gate3 -> gate4 (test) -> gate5 -> gate1b -> gate6

.PHONY: gate0 gate1 gate2 gate3 gate4 gate5 gate6 gate1b test features labels lint

# --- G0: power calcs + vet-budget integer (no data, no GPU; blocks all quantitative work)
gate0:
	uv run python scripts/gate0_power.py

# --- G1: per-CAT merge vs CAT_ ids (not raw CLIP/pHash); cat-disjoint folds
gate1:
	uv run python scripts/gate1_merge.py

# --- G2: one-directional capture-condition confound audit
gate2:
	uv run python scripts/gate2_confound.py

# --- G3: frozen hashed cat-disjoint hold-out + CI abort
gate3:
	uv run python scripts/gate3_holdout.py

# --- G4: MPS<->CPU logit parity + CORN decode->sum->0.39 unit driver (BLOCKING)
gate4:
	uv run python scripts/gate4_mps_check.py

# --- G5: alignment / NME / face-pixel-resolution audit
gate5:
	uv run python scripts/gate5_nme.py

# --- G6: severity-cell collapse (if AU=2 cells single-digit, collapse high end)
gate6:
	uv run python scripts/gate6_severity.py

# --- G1-B: VLM weak-label kappa pilot (fires on CI lower bound; self-justifies labeler)
gate1b:
	uv run python scripts/run_vlm_labels.py
	uv run python scripts/gate1b_kappa_pilot.py

# --- Gate-4 BLOCKING tests: CORN decode->sum->0.39 + MPS<->CPU parity + leakage guards
test:
	uv run pytest tests/

# --- frozen DINOv2 forward -> data/features/
features:
	uv run python scripts/cache_features.py

# --- Phase B VLM weak-label batch run (Gate 1-B pilot input)
labels:
	uv run python scripts/run_vlm_labels.py

# --- lint
lint:
	uv run ruff check .
