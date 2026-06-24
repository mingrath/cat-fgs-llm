# cat-fgs-llm Comprehensive Review Report

**Date:** 2026-06-15
**Branch reviewed:** `pipeline-checkpoint`
**Review scope:** Architecture, code quality, ML pipeline, tests/CI, documentation, config/dependencies, paper-code alignment

---

## Executive Summary

The repository has a **solid architectural foundation** with clean layered boundaries, strong gate discipline, and good core-contract test coverage. However, several **concrete bugs and best-practice gaps** need attention before the codebase is production-ready or publication-ready. The most critical issues are a process-random cache hash that breaks reproducibility, a missing direct dependency, and an undefined reference in the wrapper layer.

**Overall scores:**

| Area | Score | Verdict |
|------|-------|---------|
| Architecture | 9/10 | Clean DAG, well-separated layers |
| Code Quality | 6/10 | Several bugs and anti-patterns |
| ML Pipeline | 6/10 | Missing scheduler/early stopping, reproducibility bug |
| Test & CI | 7/10 | Core contracts tested, CI wiring incomplete |
| Documentation | 7/10 | Mostly aligned, stale references exist |
| Config & Dependencies | 7/10 | Healthy deps, missing direct dep + hardcoded paths |
| Paper-Code Alignment | 8/10 | Core claims match code |

---

## 🔴 Critical Issues (Must Fix)

### 1. Process-Random Cache Hash Breaks Reproducibility

- **File:** `src/model/cache_features.py`
- **Lines:** 53-57
- **Issue:** The cache schema hash is computed as `hash(tuple(AU_ORDER))`. Python's `hash()` is randomized per process (unless `PYTHONHASHSEED` is set *before* interpreter start), so the same repository can produce different schema hashes across runs.
- **Impact:** Cache invalidation is non-deterministic. Regenerating the same feature cache on different machines/shells may produce different schema hashes, causing avoidable cache misses or validation failures.
- **Fix:** Replace `hash(tuple(AU_ORDER))` with a stable digest such as:
  ```python
  import hashlib
  au_hash = hashlib.sha256(
      "|".join(AU_ORDER).encode("utf-8")
  ).hexdigest()[:16]
  ```
- **Follow-up:** After changing the hash function, regenerate existing caches and commit the new provenance artifacts.

### 2. Missing `Pillow` Direct Dependency

- **File:** `pyproject.toml`
- **Issue:** `PIL.Image` is imported in `src/vlm/call.py`, `src/model/cache_features.py`, and `src/data/dedup.py`, but `Pillow` is not declared as a direct dependency in `pyproject.toml`.
- **Impact:** The package may fail to install in a clean environment even though code needs it. It currently works only because other transitive dependencies happen to pull in Pillow.
- **Fix:** Add `"Pillow>=10.0"` to the `[project] dependencies` list.

### 3. Undefined `_DEFAULT_CONFIG` Reference

- **File:** `src/wrapper/operating_point.py`
- **Lines:** 26, 129
- **Issue:** The constant `_DEFAULT_CONFIG` is referenced but no longer defined in the module. This will raise a `NameError` at runtime when the default config path is used.
- **Impact:** Any caller that relies on the default config path will crash.
- **Fix:** Either restore `_DEFAULT_CONFIG = Path("configs/wrapper.yaml")` (or equivalent) at module level, or change the default argument to use the path literal directly.

### 4. No Validation Split / No LR Scheduler / No Early Stopping in Training Loop

- **File:** `src/model/train_heads.py`
- **Lines:** 200-252
- **Issue:** The `train()` function trains on the full provided tensor without an internal validation split, learning-rate scheduler, early stopping, or gradient clipping.
- **Impact:** The training loop is brittle and prone to overfitting; there is no principled stopping criterion other than a fixed epoch count.
- **Fix:**
  - Split the cached feature tensor into train/validation folds inside `train()`.
  - Track validation loss each epoch.
  - Add an `LRScheduler` (e.g., `ReduceLROnPlateau` or cosine).
  - Add early stopping with a configurable patience.
  - Document that these additions preserve the reproducibility contract (seed + deterministic mode).

---

## 🟠 High-Priority Issues

### 5. `patch_l2` Feature Is Not Actually L2-Pooled

- **File:** `src/model/cache_features.py`
- **Line:** 164
- **Issue:** When `patch_mode == "l2"`, the cached value is `patch.mean(1)` of L2-normalized tokens. That is a mean-pooled feature of normalized tokens, not an L2-pooled feature vector.
- **Impact:** Misleading cache key/name; downstream consumers may believe they are using a different feature representation than they actually are.
- **Fix:** Rename the cache key to something accurate (e.g., `patch_l2_mean`) or change the computation to an actual L2 pooling (e.g., L2 norm over the pooled patch vector). Update `_CACHE_SCHEMA` accordingly.

### 6. Bare `except Exception` Masks Errors

- **File:** `src/vlm/batch_submit.py`
- **Line:** 75
- **Issue:** A broad `except Exception` silently swallows import/runtime errors and falls back to a random proxy.
- **Impact:** Real bugs can be hidden because failures are silently converted to sentinel/random behavior.
- **Fix:** Narrow the exception handling to the specific exceptions that are expected (e.g., `ImportError`, `AttributeError`) or re-raise unexpected errors after logging.

### 7. `bootstrap_qwk_ci` Can Crash on Degenerate Bootstrap Draws

- **File:** `src/eval/kappa.py`
- **Issue:** When all bootstrap replicates are degenerate or empty, `np.nanpercentile` can receive an empty array.
- **Impact:** The kappa CI computation can raise an unhandled error on small or low-prevalence datasets.
- **Fix:** Guard against empty bootstrap arrays:
  ```python
  if len(boots) == 0 or np.all(np.isnan(boots)):
      return qwk, np.nan, np.nan
  ci = np.nanpercentile(boots, [2.5, 97.5])
  ```

### 8. `seed_everything` Allows Nondeterministic Operations

- **File:** `src/data/seed.py`
- **Lines:** 14-21
- **Issue:** `torch.use_deterministic_algorithms(True, warn_only=True)` logs warnings but does not enforce deterministic algorithms.
- **Impact:** Reproducibility is "best effort" rather than guaranteed.
- **Fix:** Document the trade-off in `seed.py` and consider adding a strict mode parameter for users who need full determinism.

### 9. `BinaryPainHead` Ignores Configured `head_init_std`

- **File:** `src/model/train_heads.py`
- **Issue:** The binary pain head is initialized with a hardcoded standard deviation, ignoring the `head_init_std` value used for the CORN heads.
- **Impact:** Inconsistent initialization between the binary and graded heads, making config files misleading.
- **Fix:** Read `head_init_std` from the config and apply it to `BinaryPainHead` as well.

### 10. `build_cache` Does Not Verify Split Purity

- **File:** `src/model/cache_features.py`
- **Lines:** 120-216
- **Issue:** `build_cache` writes all rows into one NPZ file without asserting that fold assignments match the committed manifests.
- **Impact:** If an incorrect or stale CSV is used, the cache can silently include test/holdout rows in the training set.
- **Fix:** Before caching, assert that every `cat_id`/`image_id` belongs to exactly one declared split and that the resulting fold counts match `data/manifests/`.

---

## 🟡 Medium-Priority Issues

### 11. Makefile `pre-commit` Target Does Not Run Configured Hooks

- **File:** `Makefile`
- **Lines:** 101-104
- **Issue:** The `pre-commit` target runs custom Python one-liners but does not invoke the actual `uv run pre-commit run --all-files` hooks configured in `.pre-commit-config.yaml`.
- **Impact:** Local and CI pre-commit checks can diverge.
- **Fix:** Make the `pre-commit` target run the real hook suite (or rename it to avoid confusion).

### 12. Hardcoded Absolute Path in Global Config

- **File:** `configs/global.yaml`
- **Line:** 14
- **Issue:** `repo_root` is set to `/Users/mingrath/ghq/github.com/mingrath/cat-fgs-llm`.
- **Impact:** Non-portable; the config will not work on other machines without editing.
- **Fix:** Derive `repo_root` at runtime from the script location (e.g., `Path(__file__).resolve().parents[2]`) or make it overridable via an environment variable.

### 13. `.gitignore` Missing Common Python Build/Cache Patterns

- **File:** `.gitignore`
- **Issue:** Missing entries for `.mypy_cache/`, `.ruff_cache/`, `.coverage*`, `htmlcov/`, `dist/`, `build/`, `*.egg-info/`.
- **Fix:** Add these standard ignores.

### 14. Stale Planning-Document References

- **Files:** `BUILD_PLAN.md`, `IMPLEMENTATION_PLAN.md`, `README.md`, `Makefile`
- **Issues:**
  - `BUILD_PLAN.md` references `GITHUB_MINE.md` which no longer exists.
  - `IMPLEMENTATION_PLAN.md` claims the committed backbone is `dinov2-small`, but the code uses `dinov2_vits14_reg`.
  - `IMPLEMENTATION_PLAN.md` claims `!data/manifests/` is missing from `.gitignore`, but it is present.
  - `README.md` gate-e2e-synthetic command does not match the Makefile target.
  - `Makefile` contains a stale CI workflow comment.
- **Fix:** Audit and update these documents to match the current codebase.

### 15. Paper Claim "data/ is empty" Is Stale

- **File:** `paper/sections/06-evaluation-protocol-and-gates.tex`
- **Issue:** The paper states the `data/` directory is empty, but `data/manifests/power.json` and `data/manifests/severity.json` are committed.
- **Fix:** Update the wording to say *bulk* data are absent while manifests are committed.

---

## ✅ Strengths to Preserve

1. **Clean architecture**: `model → eval → wrapper` layering with `protocols` as a portable facade.
2. **Gate discipline**: Strict G0→G6 run order enforced in both `Makefile` and `src/gates/orchestrator.py`.
3. **Single source of truth**: `src/constants.py` centralizes `AU_ORDER` and `POINT_DECISION_THRESHOLD`.
4. **Core contract tests**: AU order mirrors, CORN decode→sum→0.39, MPS parity, kappa cluster bootstrap, and confound recovery are all well-tested.
5. **Dependency health**: No version conflicts; Python 3.11 alignment is consistent across `.python-version`, `pyproject.toml`, and `uv.lock`.
6. **Env handling**: Secrets live in `.env` which is correctly gitignored; `.env.example` is placeholder-only.

---

## Recommended Implementation Order

1. **Day 1 — Bug fixes:**
   - Replace `hash(tuple(AU_ORDER))` with `hashlib.sha256`.
   - Add `Pillow` to `pyproject.toml`.
   - Fix `_DEFAULT_CONFIG` in `operating_point.py`.

2. **Day 2 — Training loop hardening:**
   - Add validation split to `train_heads.py`.
   - Add LR scheduler and early stopping.
   - Fix `BinaryPainHead` initialization.

3. **Day 3 — Robustness:**
   - Fix `bootstrap_qwk_ci` empty-array handling.
   - Narrow `except Exception` in `batch_submit.py`.
   - Fix `patch_l2` naming/computation.

4. **Day 4 — Hygiene:**
   - Update stale docs (`BUILD_PLAN.md`, `IMPLEMENTATION_PLAN.md`, `README.md`, `Makefile`).
   - Fix `.gitignore` gaps.
   - Make `configs/global.yaml.repo_root` portable.
   - Fix `Makefile.pre-commit` to run real hooks.

5. **Day 5 — Paper pass:**
   - Update paper `data/` phrasing.
   - Re-run `make paper` and `make checkcites`.

---

## Notes for the Fixer

- Do **not** refactor while fixing bugs — keep changes minimal and targeted.
- After changing the cache hash, run `make gate4` and `make test` to verify caches regenerate correctly.
- If adding a validation split changes reproducible metrics, document the change in `FINAL_DIRECTION.md` or `IMPLEMENTATION_PLAN.md`.
- Preserve the existing "engine is conceded plumbing" framing in any docstrings or comments touched.
