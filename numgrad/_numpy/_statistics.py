# https://numpy.org/doc/stable/reference/routines.statistics.html

import numpy as np
from numpy.lib.stride_tricks import as_strided

from numgrad._numpy._linalg import _dot_nd_1d_vjp_a, _dot_nd_1d_vjp_b
from numgrad._utils._expand_to import _expand_to
from numgrad._vjp import _bind_vjp


def _correlate_vjp_a(g, r, a, v, mode='full'):
    w = {'full': v.size - 1, 'valid': 0, 'same': v.size // 2}[mode]
    a_pad = np.pad(a, w)
    s = a_pad.strides[0]
    a_col = as_strided(a_pad, [g.size, v.size], [s, s])
    da_col = _dot_nd_1d_vjp_a(g, r, a_col, v)
    da_col_pad = np.pad(da_col, ((0,), (g.size,)))
    da_col_strided = as_strided(
        da_col_pad.ravel()[g.size + w:],
        [g.size, a.size], [da_col_pad.strides[0] - s, s])
    da = da_col_strided.sum(0)
    return da


def _correlate_vjp_v(g, r, a, v, mode='full'):
    w = {'full': v.size - 1, 'valid': 0, 'same': v.size // 2}[mode]
    a_pad = np.pad(a, w)
    s = a_pad.strides[0]
    a_col = as_strided(a_pad, [g.size, v.size], [s, s])
    dv = _dot_nd_1d_vjp_b(g, r, a_col, v)
    return dv


# https://numpy.org/doc/stable/reference/routines.statistics.html#order-statistics
_bind_vjp(
    np.ptp,
    lambda g, r, a, axis=None, *, keepdims=False: (
        g_expanded := _expand_to(g, a.shape, axis, keepdims),
        np.where(a == np.amax(a, axis=axis, keepdims=True), g_expanded, 0)
        + np.where(a == np.amin(a, axis=axis, keepdims=True), -g_expanded, 0),
    )[-1],
)

# https://numpy.org/doc/stable/reference/routines.statistics.html#averages-and-variances
_bind_vjp(
    np.mean,
    lambda g, r, a, axis=None, *, keepdims=False: (
        _expand_to(g, a.shape, axis, keepdims) * g.size / a.size
    ),
)
_bind_vjp(
    np.std,
    lambda g, r, a, axis=None, *, ddof=0, keepdims=False: (
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
    lambda g, r, a, axis=None, *, ddof=0, keepdims=False: (
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
    lambda g, r, a, axis=None, *, keepdims=False: (
        _expand_to(g, a.shape, axis, keepdims) / np.sum(
            ~np.isnan(a), axis, keepdims=True)
    ),
)
_bind_vjp(
    np.nanstd,
    lambda g, r, a, axis=None, *, ddof=0, keepdims=False: (
        np.zeros_like(a) if a.size <= 1 else
        np.nan_to_num(_expand_to(g / r, a.ndim, axis, keepdims) * (
            a - np.nanmean(a, axis, keepdims=True)) / (
                np.sum(~np.isnan(a), axis, keepdims=True) - ddof))
    ),
)
_bind_vjp(
    np.nanvar,
    lambda g, r, a, axis=None, *, ddof=0, keepdims=False: (
        np.zeros_like(a) if a.size <= 1 else
        2 * _expand_to(g, a.ndim, axis, keepdims) * (
            a - np.nanmean(a, axis, keepdims=True)) / (
                np.sum(~np.isnan(a), axis, keepdims=True) - ddof)
    ),
)

# https://numpy.org/doc/stable/reference/routines.statistics.html#correlating
_bind_vjp(np.correlate, _correlate_vjp_a, _correlate_vjp_v)
