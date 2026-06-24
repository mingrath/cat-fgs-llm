# cat-fgs-llm — Improvement Recommendations

**Generated:** 2026-06-15
**Scope:** Code hygiene, duplication removal, library modernization, security/perf

---

## 1. CRITICAL FIXES (Ship Immediately)

### ✅ DONE: Pre-commit file handle leaks
- **File:** `.pre-commit-config.yaml:38,66`
- **Fix:** Replaced `json.load(open(...))` with `with open(...) as f: json.load(f)`
- **Impact:** Resource leak eliminated; 2 locations patched

### ✅ DONE: G0 manifest guard extracted
- **New file:** `src/gates/manifests.py`
- **Benefit:** Single source of truth for `GATE_PRECONDS` + `enforce_g0_manifests()`
- **Next:** Gate scripts should import this instead of duplicating logic (future refactor)

---

## 2. HIGH-PRIORITY REFACTORS (Next Sprint)

### 2.1 Gate script duplication removal
**Current state:** 8 `scripts/gate*.py` files may contain repeated G0 manifest checks.
**Action:** After this PR, update each gate script to:
```python
from src.gates.manifests import enforce_g0_manifests
enforce_g0_manifests("gateX", synthetic=args.synthetic)
```
**Effort:** 2-3 hours (mostly search/replace + test)

### 2.2 Orchestrator cleanup
**File:** `src/gates/orchestrator.py:60-71`
**Issue:** Still defines `GATE_PRECONDS` locally (duplicate of `manifests.py`)
**Action:** Remove the local definition; import from `manifests.py`
**Effort:** 15 minutes

---

## 3. LIBRARY MIGRATION OPPORTUNITIES (Context7 Findings)

### 3.1 Transformers / DINOv2 — Register variant (HIGH VALUE)
**Finding:** DINOv2 with Registers (`dinov2_vits14_reg`) is explicitly documented to suppress attention artifacts in low-informative regions — directly relevant to per-AU FGS scoring (orbital/ear/muzzle sub-regions).

**Current code:** Uses `facebook/dinov2_vits14` (non-reg)
**Recommendation:** A/B test `facebook/dinov2_vits14_reg` before locking backbone
**Impact:** README already flags this; now confirmed by official docs
**Command:**
```bash
# In src/model/ backbone loading
model = AutoModel.from_pretrained("facebook/dinov2_vits14_reg")
```

### 3.2 Anthropic — Prompt caching (MEDIUM VALUE)
**Finding:** SDK supports `cache_control: {type: "ephemeral", ttl: "5m"}` on system prompts and tool results.

**Current usage:** Unknown (check `src/vlm/batch_submit.py`)
**Recommendation:** Add cache breakpoints on long rubric/system prompts to reduce token costs on repeated VLM calls
**Effort:** 1-2 hours (add `cache_control` blocks)

### 3.3 MAPIE — LTT for abstention (LOW-MEDIUM VALUE)
**Finding:** `BinaryClassificationController` with LTT already used in `src/wrapper/abstention.py`. Docs confirm multi-risk control (NPV + PPV + abstention_rate) pattern matches current implementation.

**Status:** Code is aligned with current MAPIE best practices. No urgent migration.

### 3.4 dcurves — Harm specification (LOW VALUE)
**Finding:** `dca(..., harm={'model': 0.01})` API confirmed. Current wrapper likely passes harm ratios correctly.

**Status:** No change needed unless harm values are hardcoded (grep for `harm` in `src/wrapper/`).

---

## 4. SECURITY & PERFORMANCE FINDINGS

### 4.1 Security (No blockers found)
- `.env` is gitignored ✅
- Pre-commit enforces `check-json` on `artifacts/*.json` and `data/manifests/*.json` ✅
- No obvious secrets in committed files (spot-checked)
- **Action:** Run `git secrets --scan` or `trufflehog` before next release

### 4.2 Performance hotspots (needs measurement)
- **VLM batching:** `src/vlm/batch_submit.py` — verify prompt caching is active
- **Gate pipeline I/O:** Repeated `power.json` reads across gates — consider in-memory cache for orchestrator runs
- **No obvious N+1 or missing `@lru_cache`** on pure functions (e.g., `get_au_names`)

**Recommended profiling:**
```bash
uv run python -m cProfile -o profile.out scripts/gate2_confound.py
uv run snakeviz profile.out
```

---

## 5. QUICK WINS (Low Effort, High Confidence)

| Change | File | Effort | Confidence |
|--------|------|--------|------------|
| Add `from __future__ import annotations` to `src/gates/manifests.py` | `manifests.py:1` | 1 min | 100% |
| Ruff: enable `UP007` (PEP 604) | `pyproject.toml` | 5 min | 90% |
| Add `test_manifests.py` for `enforce_g0_manifests` | `tests/test_manifests.py` | 30 min | 95% |
| Document `GATE_PRECONDS` as the single source in README | `README.md` | 10 min | 100% |

---

## 6. COMMANDS TO RUN

```bash
# Verify pre-commit fixes
uv run pre-commit run --all-files

# Test new manifests module
uv run python -c "from src.gates.manifests import enforce_g0_manifests; print('OK')"

# Context7 follow-up (when quotas reset)
# Run context7-mcp skill queries on the 4 libraries above

# Full gate pipeline smoke (synthetic)
make gate-e2e-synthetic --synthetic
```

---

**Next milestone:** After the 8 gate scripts are updated to import `manifests.py`, delete the duplicate `GATE_PRECONDS` from `orchestrator.py` and add a unit test for the helper.

**Owner:** Lead (Sisyphus)
**Status:** 2/7 todos complete; 2 critical fixes shipped; 1 high-value library migration identified (DINOv2 _reg)
