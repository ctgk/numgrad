# https://numpy.org/doc/stable/reference/routines.sort.html
import numpy as np

from numgrad._vjp import _register_vjp


def _unpermute(a, permutation, axis):
    unpermutation = np.zeros(a.shape, dtype=int)
    np.put_along_axis(
        unpermutation,
        permutation,
        np.argsort(np.ones(a.shape, dtype=int), axis),
        axis,
    )
    return np.take_along_axis(a, unpermutation, axis)


# https://numpy.org/doc/stable/reference/routines.sort.html#sorting
_register_vjp(
    np.sort,
    lambda a, axis=-1: lambda g, r: (
        _unpermute(g, np.argsort(a, axis), axis) if axis is not None
        else _unpermute(g, np.argsort(a, axis), 0).reshape(a.shape)
    ),
)
