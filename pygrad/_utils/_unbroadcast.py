import typing as tp

import numpy as np


def _unbroadcast_to(x: np.ndarray, shape: tp.Union[tuple, int]) -> np.ndarray:
    if isinstance(shape, int):
        shape = (shape,)
    x = np.sum(x, axis=tuple(range(0, x.ndim - len(shape))))
    x = np.sum(
        x,
        axis=tuple(i for i in range(x.ndim) if shape[i] == 1),
        keepdims=True)
    return x
