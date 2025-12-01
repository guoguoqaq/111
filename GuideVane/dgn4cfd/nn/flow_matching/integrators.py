from typing import Callable
import torch
import numpy as np


def euler(
    rhs: Callable,
    y0:  torch.Tensor,
    t:   list,
) -> torch.Tensor:
    t = np.unique(t)
    for t0, t1 in zip(t[:-1], t[1:]):
        y0 = y0 + (t1 - t0) * rhs(t0, y0)
    return y0.clone()
