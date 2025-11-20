# utils/seed.py
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


# backward compatibility 이름
def set_global_seed(seed: int) -> None:
    set_random_seed(seed)
