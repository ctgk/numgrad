from typing import Tuple

import numpy as np

from pygrad._core._tensor import Tensor


def _numerical_grad(
        op: callable,
        *args: Tensor,
        epsilon: float = 1e-5,
        **kwargs) -> Tuple[np.ndarray]:
    dargs = []
    for arg in args:
        darg = np.zeros_like(arg._data)
        for i in range(arg.size):
            eps = np.zeros_like(arg._data)
            eps.ravel()[i] = epsilon
            args_p = [
                Tensor(a._data + (eps if a is arg else 0), a.dtype)
                for a in args
            ]
            args_m = [
                Tensor(a._data - (eps if a is arg else 0), a.dtype)
                for a in args
            ]
            out_p = op(*args_p, **kwargs)
            out_m = op(*args_m, **kwargs)
            darg.ravel()[i] = (
                np.sum(out_p._data) - np.sum(out_m._data)) / (2 * epsilon)
        dargs.append(darg)
    return tuple(dargs)
