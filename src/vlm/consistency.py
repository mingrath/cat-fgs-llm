"""N>=3 repeated runs -> ordinal Krippendorff alpha per AU (IMPLEMENTATION_PLAN §4.5).

A VET-FREE reliability axis with two complementary signals:

  - alpha_per_au() — ALEATORIC self-consistency. Each image is scored N>=3 times at
    temperature=1.0 (varied seed tag); ordinal Krippendorff alpha per AU over the
    (n_runs x n_items) matrix. Captures within-model sampling noise.
  - cross_model_disagreement_per_au() — EPISTEMIC between-model disagreement. Two (or
    more) independently-prompted VLMs score the same images; the per-AU mean pairwise
    ordinal gap is the signal temperature-only self-consistency cannot see. Where two
    models disagree, that AU/image is a candidate to ROUTE TO VET.

Both are (a) a cheap pre-screen before spending vet budget, (b) an abstention SIGNAL
(low alpha / high disagreement AUs route to vet), and (c) a complement to -- never a
replacement for -- the vs-vet quadratic kappa. They are self-consistency signals ONLY:
NEVER offered as validation of the 0.39 flag (§4.7); QWK-vs-VLM is never validation
either. Any downstream abstention claim stays a one-sided 95% NPV LOWER bound -- the
word "guaranteed" is BANNED.

Low alpha (or high disagreement) on muzzle/whiskers is EXPECTED (the weakest AUs even
for human experts, ICC 0.55-0.67).

Custom_id conventions: ``"{image_id}__run{k}"`` for single-model self-consistency
(matches batch_submit); ``"{image_id}__{model}__run{k}"`` (or a separate ``model``
column) for the cross-model pass. ``AU_NAMES`` is imported from src.vlm.schema so the
pivots are keyed on the fixed 5-AU order.
"""

import krippendorff
import pandas as pd

from src.vlm.schema import AU_NAMES


def alpha_per_au(parquet_path):
    """Return {au: ordinal_alpha} over N>=3 runs read from a collected parquet."""
    df = pd.read_parquet(parquet_path)  # custom_id = "{image_id}__run{k}"
    df[["image_id", "run"]] = df["custom_id"].str.split("__run", expand=True)
    out = {}
    for au in AU_NAMES:
        # reliability_data: rows = runs (coders), cols = items (images); values in {0,1,2}
        wide = df.pivot(index="run", columns="image_id", values=f"{au}_score")
        out[au] = krippendorff.alpha(
            reliability_data=wide.values.astype(float),
            level_of_measurement="ordinal",
        )
    return out


def cross_model_disagreement_per_au(parquet_path):
    """Return {au: mean pairwise |ordinal gap|} between >=2 distinct models per AU.

    EPISTEMIC (between-model) disagreement, complementary to alpha_per_au's ALEATORIC
    (within-model, temperature) self-consistency. For each AU: collapse each model's
    repeated runs to its per-image mean score, then average -- over images -- the mean
    pairwise absolute score gap between models. HIGHER = more epistemic disagreement =
    stronger candidate to ROUTE TO VET. nan for an AU when fewer than two models are
    present (no between-model signal).

    Reads a collected parquet whose custom_id is ``"{image_id}__{model}__run{k}"`` OR
    that carries an explicit ``model`` column. This is a vet-FREE abstention SIGNAL only
    -- NEVER validation of the 0.39 flag, and any downstream NPV claim remains a one-sided
    95% LOWER bound ("guaranteed" is BANNED).
    """
    df = pd.read_parquet(parquet_path)
    if "model" not in df.columns or "image_id" not in df.columns:
        # "{image_id}__{model}__run{k}" -> strip the run tag, then split off the model.
        # rsplit keeps image_ids that themselves contain "__" intact (model has no "__").
        base = df["custom_id"].str.rsplit("__run", n=1).str[0]
        df = df.assign(
            image_id=base.str.rsplit("__", n=1).str[0],
            model=base.str.rsplit("__", n=1).str[1],
        )
    out = {}
    for au in AU_NAMES:
        # rows = image, cols = model; cell = that model's mean score over its runs
        consensus = df.groupby(["model", "image_id"])[f"{au}_score"].mean().unstack("model")
        models = list(consensus.columns)
        if len(models) < 2:
            out[au] = float("nan")
            continue
        pair_gaps = [
            (consensus[models[i]] - consensus[models[j]]).abs()
            for i in range(len(models))
            for j in range(i + 1, len(models))
        ]
        # mean over model-pairs (per image), then over images (images missing a model drop)
        out[au] = float(pd.concat(pair_gaps, axis=1).mean(axis=1).mean())
    return out
