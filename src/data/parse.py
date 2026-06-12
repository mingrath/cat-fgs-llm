"""Filename -> group_id parser (IMPLEMENTATION_PLAN §1.1 step 4).

Recover the original PNG stem from the Roboflow export suffix, then map the
stem to a CV grouping key. The group key is the unit one cat's clips collapse
toward at Gate 1; it is consumed by ``src.data.folds`` as the
``StratifiedGroupKFold`` group. KEY ON THE FILENAME STEM ONLY -- never on a
parent folder named ``CAT`` (that neutralizes the phantom Zhang-archive
``CAT_00..06`` folder collision).

This module is data plumbing for the binary spine; it makes no validated claim.
"""

import re

# Roboflow suffix: "<stem>_png.rf.<32hex>.jpg"  -> recover "<stem>.png" first.
SUFFIX = re.compile(r"_png\.rf\.[0-9a-f]+\.jpg$", re.IGNORECASE)
CAM = re.compile(r"^CAT_(\d{2})_(\d{8})_(\d{3})\.png$")   # CAT_01_00000100_007.png
PLAIN = re.compile(r"^(\d{8})_(\d{3})\.png$")             # 00000100_007.png


def group_id(fname: str) -> str:
    """Map a Roboflow export filename to its CV group key.

    ``CAT_NN_<clip>_<frame>`` -> ``"CATnn_<clip>"`` (namespaced camera clip);
    ``<clip>_<frame>``        -> ``"P_<clip>"``     (plain clip).

    Raises ``ValueError`` on any filename that matches neither pattern -- an
    unparseable name must fail loud before any split is built.
    """
    stem = SUFFIX.sub(".png", fname)            # -> "<stem>.png"
    m = CAM.match(stem)
    if m:
        cam, clip, _frame = m.groups()
        return f"CAT{cam}_{clip}"               # namespaced camera clip
    m = PLAIN.match(stem)
    if m:
        clip, _frame = m.groups()
        return f"P_{clip}"                       # plain clip
    raise ValueError(f"unparseable filename: {fname!r}")
