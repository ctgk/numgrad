from typing import Tuple

import numpy as np

from pygrad._array import Array


def _numerical_grad(
        op: callable,
        *args: Array,
        epsilon: float = 1e-5,
        **kwargs) -> Tuple[np.ndarray]:
    dargs = []
    for arg in args:
        darg = np.zeros_like(arg.value)
        for i in range(arg.size):
            eps = np.zeros_like(arg.value)
            eps.ravel()[i] = epsilon
            args_p = [
                Array(a.value + (eps if a is arg else 0)) for a in args
            ]
            args_m = [
                Array(a.value - (eps if a is arg else 0)) for a in args
            ]
            out_p = op(*args_p, **kwargs)
            out_m = op(*args_m, **kwargs)
            darg.ravel()[i] = (
                np.sum(out_p.value) - np.sum(out_m.value)) / (2 * epsilon)
        dargs.append(darg)
    return tuple(dargs)
