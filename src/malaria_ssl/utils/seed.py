"""Global seeding for reproducibility."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
