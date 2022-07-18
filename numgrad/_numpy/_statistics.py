import numpy as np

from numgrad._utils._expand_to import _expand_to
from numgrad._vjp import _bind_vjp

# https://numpy.org/doc/stable/reference/routines.statistics.html#order-statistics
_bind_vjp(
    np.ptp,
    lambda a, axis=None, *, keepdims=False: lambda g, r: (
        g_expanded := _expand_to(g, a.shape, axis, keepdims),
        np.where(a == np.amax(a, axis=axis, keepdims=True), g_expanded, 0)
        + np.where(a == np.amin(a, axis=axis, keepdims=True), -g_expanded, 0),
    )[-1],
)

# https://numpy.org/doc/stable/reference/routines.statistics.html#averages-and-variances
_bind_vjp(
    np.mean,
    lambda a, axis=None, *, keepdims=False: lambda g, r: (
        _expand_to(g, a.shape, axis, keepdims) * g.size / a.size
    ),
)
_bind_vjp(
    np.std,
    lambda a, axis=None, *, ddof=0, keepdims=False: lambda g, r: (
        np.zeros_like(a) if a.size <= 1 else
        (
            g_over_r := _expand_to(g / r, a.ndim, axis, keepdims),
            d := a - a.mean(axis, keepdims=True),
            g_over_r * d / (a.size / r.size - ddof),
        )[-1]
    ),
)
_bind_vjp(
    np.var,
    lambda a, axis=None, *, ddof=0, keepdims=False: lambda g, r: (
        np.zeros_like(a) if a.size <= 1 else
        (
            g_expanded := _expand_to(g, a.ndim, axis, keepdims),
            d := a - a.mean(axis, keepdims=True),
            2 * g_expanded * d / (a.size / r.size - ddof),
        )[-1]
    ),
)
_bind_vjp(
    np.nanmean,
    lambda a, axis=None, *, keepdims=False: lambda g, r: (
        _expand_to(g, a.shape, axis, keepdims) / np.sum(
            ~np.isnan(a), axis, keepdims=True)
    ),
)
_bind_vjp(
    np.nanstd,
    lambda a, axis=None, *, ddof=0, keepdims=False: lambda g, r: (
        np.zeros_like(a) if a.size <= 1 else
        np.nan_to_num(_expand_to(g / r, a.ndim, axis, keepdims) * (
            a - np.nanmean(a, axis, keepdims=True)) / (
                np.sum(~np.isnan(a), axis, keepdims=True) - ddof))
    ),
)
_bind_vjp(
    np.nanvar,
    lambda a, axis=None, *, ddof=0, keepdims=False: lambda g, r: (
        np.zeros_like(a) if a.size <= 1 else
        2 * _expand_to(g, a.ndim, axis, keepdims) * (
            a - np.nanmean(a, axis, keepdims=True)) / (
                np.sum(~np.isnan(a), axis, keepdims=True) - ddof)
    ),
)
