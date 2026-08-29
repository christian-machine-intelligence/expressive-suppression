"""Tiny seed-setter shared by rollout/train entry points (from ICMI-018)."""
import os, random
import numpy as np
import torch

def set_global_seed(seed: int) -> None:
    """Seed all random sources for reproducible runs (per-seed variance studies)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
