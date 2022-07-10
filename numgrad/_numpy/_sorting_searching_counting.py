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


def _sort_vjp(a, axis=-1):
    def vjp(g, r):
        if axis is None:
            return _unpermute(g, np.argsort(a, axis), 0).reshape(a.shape)
        return _unpermute(g, np.argsort(a, axis), axis)
    return vjp


# https://numpy.org/doc/stable/reference/routines.sort.html#sorting
_register_vjp(np.sort, _sort_vjp)
_register_vjp(np.msort, lambda a: _sort_vjp(a, 0))
