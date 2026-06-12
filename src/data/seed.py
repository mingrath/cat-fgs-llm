"""Reproducibility seed (IMPLEMENTATION_PLAN §0.6).

One seed = 42, set once via seed_everything(); never re-seed ad-hoc. Every gate
inherits this. The literal default (42) mirrors configs/global.yaml.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
