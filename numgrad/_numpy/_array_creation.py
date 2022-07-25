# https://numpy.org/doc/stable/reference/routines.array-creation.html

import numpy as np

from numgrad._vjp import _bind_vjp


def _linspace_vjp_start(g, r, start, stop, num=50, endpoint=True, axis=0):
    return np.sum(
        g * np.linspace(
            np.ones_like(start), np.zeros_like(stop),
            num, endpoint, axis=axis,
        ),
        axis,
    )


def _linspace_vjp_stop(g, r, start, stop, num=50, endpoint=True, axis=0):
    return np.sum(
        g * np.linspace(
            np.zeros_like(start), np.ones_like(stop),
            num, endpoint, axis=axis,
        ),
        axis,
    )


def _pad_to(a, shape):
    if a.shape == shape:
        return a
    return np.pad(a, tuple((0, shape[i] - a.shape[i]) for i in range(a.ndim)))


# https://numpy.org/doc/stable/reference/routines.array-creation.html#numerical-ranges
_bind_vjp(np.linspace, _linspace_vjp_start, _linspace_vjp_stop)

# https://numpy.org/doc/stable/reference/routines.array-creation.html#building-matrices
_bind_vjp(np.diag, lambda g, r, v, k=0: _pad_to(np.diag(g, k), v.shape))
_bind_vjp(np.diagflat, lambda g, r, v, k=0: np.diag(g, k).reshape(*v.shape))
_bind_vjp(np.tril, lambda g, r, _, k=0: np.tril(g, k))
_bind_vjp(np.triu, lambda g, r, _, k=0: np.triu(g, k))
_bind_vjp(
    np.vander,
    lambda g, r, x, N=None, increasing=False: (
        n := len(x) if N is None else N,
        np.sum(g[:, 1:] * r[:, :-1] * range(1, n), -1) if increasing
        else np.sum(g[:, :-1] * r[:, 1:] * range(1, n)[::-1], -1),
    )[1],
)
