from typing import Tuple

import numpy as np


def _numerical_grad(
    op: callable,
    *args: np.ndarray,
    epsilon: float = 1e-5,
    **kwargs,
) -> Tuple[np.ndarray]:
    args = tuple(np.asarray(a, dtype=float) for a in args)
    dargs = tuple(np.zeros_like(a, dtype=float) for a in args)
    for arg, darg in zip(args, dargs):
        for i in range(arg.size):
            eps = np.zeros_like(arg)
            eps.ravel()[i] = epsilon
            args_p = [
                a + (eps if a is arg else 0)
                for a in args
            ]
            args_m = [
                a - (eps if a is arg else 0)
                for a in args
            ]
            out_p = op(*args_p, **kwargs)
            out_m = op(*args_m, **kwargs)
            darg.ravel()[i] = (
                np.nansum(out_p) - np.nansum(out_m)) / (2 * epsilon)
            if np.isnan(darg.ravel()[i]):
                darg.ravel()[i] = 0
    return dargs
