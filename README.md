# cat-fgs-llm

A portable, power-aware **confound-attribution protocol** for fine-grained
animal-affect models — a per-AU audit that any future facial-pain-scorer corpus can
run — built on a deliberately conceded DINOv2 + CORN scoring engine.

The research question is not "can a model score cat pain?" It is **"when a model
appears to score pain, how do you show it isn't scoring the background, the camera, or
its own rubric?"**

> **Where the code is.** This is the `main` branch and it holds documentation only.
> All implementation, the ADRs, the evaluation gates and the paper live on the
> [`pipeline-checkpoint`](https://github.com/mingrath/cat-fgs-llm/tree/pipeline-checkpoint)
> branch. Every link below points there. `pipeline-checkpoint` is a working research
> branch, not a released artifact — it carries session scratch alongside the code.

## The three artifacts

1. **FGS-BG-Gap + per-AU EBPG confound-attribution protocol — the headline.**
   [`src/eval/confound.py`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/src/eval/confound.py)
   A one-directional, power-conditioned audit that reports "no confound detected at this
   power" rather than proving absence. Combines a background-swap counterfactual, per-AU
   EBPG saliency-as-confound-evidence, and a VLM judge-bias probe. It validates today
   against planted positive and negative controls in
   [`src/protocols/standalone_test_corpus.py`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/src/protocols/standalone_test_corpus.py).
   The claimed contribution is the *assembly* plus the per-AU-EBPG-as-confound-evidence
   step; the individual primitives (bg-swap, saliency, the one-directional statistics)
   are prior work and are cited, not claimed.

2. **VLM-as-AU-rater kappa reliability check — guarded, result pending.**
   [`src/eval/kappa.py`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/src/eval/kappa.py),
   [`src/vlm/`](https://github.com/mingrath/cat-fgs-llm/tree/pipeline-checkpoint/src/vlm)
   Scores per-AU VLM labels against a vet anchor with CI-lower-bound gating. **It ships
   as a runnable protocol, not a finding** — the current dataset is binary pain/no-pain
   and cannot yield the 0/1/2 AU ground truth the check needs. Kept as insurance: the
   sole-survivor headline if the confound leg degrades.

3. **Welfare-asymmetric decision curve + abstention boundary — supporting plumbing.**
   [`src/wrapper/`](https://github.com/mingrath/cat-fgs-llm/tree/pipeline-checkpoint/src/wrapper)
   Operating point at fixed pain-recall ≥ 0.90 with the undertreat:overtreat harm ratio
   swept as a range, plus a one-sided 95% NPV defer-to-vet curve. Cited supporting work,
   not a headline.

**The engine is conceded, not claimed.** The frozen DINOv2 ViT-S/14 + five per-AU CORN
heads → 0–10 sum → 0.39 decision engine
([`src/model/`](https://github.com/mingrath/cat-fgs-llm/tree/pipeline-checkpoint/src/model))
is explicitly labelled not-novel throughout the codebase. The 0–10 severity layer is
built, decoded and **inspected-not-validated** — it never emits a validated-claim
number. The v1 spine is binary pain/no-pain plus the abstention wrapper.

## Evaluation gates

Eight gates run in strict order; each writes an immutable artifact and no downstream
number is believed until its prerequisite passes. Gate 0 is a power calculation that
blocks all quantitative work; Gate 4 is a blocking MPS↔CPU logit-parity check.

| Gate | Script | Blocks |
| --- | --- | --- |
| G0 | [`gate0_power.py`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/scripts/gate0_power.py) | all quantitative work |
| G1 | [`gate1_merge.py`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/scripts/gate1_merge.py) | cat-disjoint folds |
| G1-B | [`gate1b_kappa_pilot.py`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/scripts/gate1b_kappa_pilot.py) | the labeler |
| G2 | [`gate2_confound.py`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/scripts/gate2_confound.py) | the headline claim |
| G3 | [`gate3_holdout.py`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/scripts/gate3_holdout.py) | hold-out reporting |
| G4 | [`gate4_mps_check.py`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/scripts/gate4_mps_check.py) | every quantitative target |
| G5 | [`gate5_nme.py`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/scripts/gate5_nme.py) | alignment claims |
| G6 | [`gate6_severity.py`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/scripts/gate6_severity.py) | severity-cell collapse |

Order is enforced centrally by
[`src/gates/orchestrator.py`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/src/gates/orchestrator.py),
not by convention. Fifteen test modules cover leakage, fold disjointness, decode parity and
confound recovery —
[`tests/`](https://github.com/mingrath/cat-fgs-llm/tree/pipeline-checkpoint/tests).

## Reading it without running it

- **[The paper](https://github.com/mingrath/cat-fgs-llm/tree/pipeline-checkpoint/paper)** — LaTeX, eleven sections. Start with
  [`04-methodology-engine.tex`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/paper/sections/04-methodology-engine.tex)
  and [`06-evaluation-protocol-and-gates.tex`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/paper/sections/06-evaluation-protocol-and-gates.tex).
- **[The 16 ADRs](https://github.com/mingrath/cat-fgs-llm/tree/pipeline-checkpoint/docs/adr)** — every product and research-ethics decision, including
  [consented research data collection](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/docs/adr/0011-consented-research-data-collection.md)
  and [pseudonymous identity for research records](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/docs/adr/0013-pseudonymous-line-identity-for-research-records.md).
- **[`src/protocols/README.md`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/src/protocols/README.md)** — the portable surface, importable with the engine deleted.

## Running it

uv-managed, Python 3.11. Local target is Apple M4 / MPS with **no local CUDA**; CUDA is
confined to
[`notebooks/colab_train_rfdetr.ipynb`](https://github.com/mingrath/cat-fgs-llm/blob/pipeline-checkpoint/notebooks/colab_train_rfdetr.ipynb)
for detector training. Every hyperparameter and seed lives in
[`configs/*.yaml`](https://github.com/mingrath/cat-fgs-llm/tree/pipeline-checkpoint/configs),
never hardcoded.

```sh
git clone -b pipeline-checkpoint https://github.com/mingrath/cat-fgs-llm.git
cd cat-fgs-llm
uv sync
make test            # Gate 4 parity — must be green before any quantitative target
make test-portable   # zero-torch protocols surface, engine-deletion safe
```

The protocols surface runs standalone with no dataset and no engine:

```sh
python -m src.protocols.standalone_test_corpus
```

## Status and limitations

- The confound protocol is the **strong leg** — validated on planted controls today.
- The kappa check is a **protocol with a result pending**; it needs an independent per-AU
  vet anchor that does not yet exist.
- The 0–10 severity layer is **inspected, not validated**.
- No dataset or imagery is committed to this repository.

Research code, released for review. Not a clinical or diagnostic tool.
