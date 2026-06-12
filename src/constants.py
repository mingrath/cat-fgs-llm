"""Canonical Evangelista 5-AU order — THE single declaration (LAW).

Every module that is positionally coupled through this order (the npz y-matrix
writer, the CORN head list, the VLM column build, the kappa table, the Gate-6
severity collapse, the sum-pmf convolution) imports AU_ORDER from here instead
of re-declaring the list. tests/test_au_order.py pins the mirrors so a reorder
in any one site fails CI instead of silently mislabeling AUs.

Leaf module by design: no third-party imports, safe for both the vlm stack and
the torch engine to depend on.
"""

AU_ORDER = ("ear", "orbital", "muzzle", "whiskers", "head")
