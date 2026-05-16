"""Global randomness control utilities."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Set global random seeds for reproducible experiments.

    Parameters
    ----------
    seed:
        Integer seed used for Python, NumPy, PyTorch, CUDA, and
        ``PYTHONHASHSEED``.
    deterministic:
        When True, also requests deterministic PyTorch algorithms and cuDNN
        settings. This can improve reproducibility, but does not guarantee
        identical results across PyTorch versions, platforms, or CPU/GPU.

    Returns
    -------
    None
        This function mutates global random state and environment settings.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
