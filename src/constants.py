"""Canonical Evangelista 5-AU order + portable FGS decision constants (LAW).

Every module that is positionally coupled through this order (the npz y-matrix
writer, the CORN head list, the VLM column build, the kappa table, the Gate-6
severity collapse, the sum-pmf convolution) imports AU_ORDER from here instead
of re-declaring the list. tests/test_au_order.py pins the mirrors so a reorder
in any one site fails CI instead of silently mislabeling AUs.

POINT_DECISION_THRESHOLD + analgesia helpers are the SINGLE SOURCE for the
Evangelista >0.39 cut (portable method contract; dataset-agnostic). The 0-10
sum / 0.39 flag are ALWAYS computed in code (never by VLM or model output);
see src.model.decode (pmf/sum) and src.vlm.aggregate (fgs_sum/flag wrappers).
Provenance: Evangelista 2019 (sens 90.7% at >0.39 on human raters); Gate-4
pins decode->sum->flag against this const. No literal 0.39 anywhere else.

Leaf module by design: no third-party imports, safe for both the vlm stack and
the torch engine to depend on.
"""

AU_ORDER = ("ear", "orbital", "muzzle", "whiskers", "head")

# THE single portable 0.39 definition (Evangelista sum/10 cut, ~4/10).
# Imported by decode, wrapper/abstention (band center, distinct from 0.90 recall op),
# power calcs, all eval. Never re-declare literal elsewhere. Gate-4 unit test + 
# test_threshold_single_source.py enforce.
POINT_DECISION_THRESHOLD = 0.39
