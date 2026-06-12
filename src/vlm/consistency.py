"""N>=3 repeated runs -> ordinal Krippendorff alpha per AU (IMPLEMENTATION_PLAN §4.5).

A VET-FREE reliability axis: each image is scored N>=3 times at temperature=1.0
(varied seed tag), and ordinal Krippendorff alpha is computed per AU over the
(n_runs x n_items) matrix. This is (a) a cheap pre-screen before spending vet
budget, (b) an abstention signal (low-alpha AUs route to vet), and (c) a complement
to -- never a replacement for -- the vs-vet quadratic kappa.

Low alpha on muzzle/whiskers is EXPECTED (the weakest AUs even for human experts,
ICC 0.55-0.67). Alpha is a self-consistency signal ONLY: it is NEVER offered as
validation of the 0.39 flag (§4.7). QWK-vs-VLM is never validation either.

The custom_id convention is ``"{image_id}__run{k}"`` (matches batch_submit).
``AU_NAMES`` is imported from src.vlm.schema so the pivot is keyed on the fixed
5-AU order.
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
