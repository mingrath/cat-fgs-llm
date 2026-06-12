"""Single source of truth for the compute device (IMPLEMENTATION_PLAN §0.4).

Local target is Apple M4 (MPS). There is NO CUDA in this engine: CUDA lives ONLY
in notebooks/colab_*.ipynb (a !pip cell), never in src/. Never assume cuda here.
"""

import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  # never assume cuda locally
