from functools import partial
from inspect import getfullargspec

import numpy as np
from numpy.lib.stride_tricks import as_strided

from numgrad._utils._expand_to import _expand_to
from numgrad._utils._unbroadcast import _unbroadcast_to
from numgrad._vjp import _register_vjp


def _get_prod_vjp(a, axis=None, *, keepdims=False, _prod=np.prod):
    return lambda g, r: (
        _expand_to(g, a.ndim, axis, keepdims)
        * (r if keepdims else _prod(a, axis, keepdims=True)) / a
    )


def _sum_vjp(g, a, axis, keepdims):
    return _expand_to(g, a.shape, axis, keepdims)


def _get_sum_vjp(a, axis=None, *, keepdims=False):
    return lambda g, r: _sum_vjp(g, a, axis, keepdims)


def _cumulate_inversely(cum_op, a, axis=None):
    return np.flip(cum_op(np.flip(a, axis), axis), axis)


def _get_cumprod_vjp(a, axis=None):

    def _cumprod_vjp(g, r):
        if a.ndim == 0:
            return g
        if axis is None:
            return _cumulate_inversely(np.cumsum, g * r).reshape(a.shape) / a
        return _cumulate_inversely(np.cumsum, g * r, axis) / a

    return _cumprod_vjp


def _get_cumsum_vjp(a, axis=None):

    def _cumsum_vjp(g, r):
        if a.ndim == 0:
            return g
        if axis is None:
            return _cumulate_inversely(np.cumsum, g, 0).reshape(a.shape)
        return _cumulate_inversely(np.cumsum, g, axis)

    return _cumsum_vjp


def _get_unary_vjp(df):

    def _unary_vjp(x):
        n = len(getfullargspec(df).args)
        return lambda g, r: df(*[a for a in (g, r, x)[:n]])

    return _unary_vjp


def _get_binary_vjps(df1, df2):

    def _binary_vjps(x1, x2):
        n1 = len(getfullargspec(df1).args)
        n2 = len(getfullargspec(df2).args)
        return (
            lambda g, r: _unbroadcast_to(
                df1(*[a for a in (g, r, x1, x2)[:n1]]), x1.shape),
            lambda g, r: _unbroadcast_to(
                df2(*[a for a in (g, r, x2, x1)[:n2]]), x2.shape),
        )

    return _binary_vjps


def _get_multi_vjps(*df):
    n = tuple(len(getfullargspec(d).args) for d in df)

    def _multi_vjps(*x):
        return tuple(
            lambda g, r: _unbroadcast_to(
                d(*[a for a in (g, r, *x)][:n_]), x_.shape)
            for d, x_, n_ in zip(df, x, n)
        )

    return _multi_vjps


def _get_vjp(*df):
    if len(df) == 1:
        return _get_unary_vjp(df[0])
    if len(df) == 2:
        return _get_binary_vjps(df[0], df[1])
    return _get_multi_vjps(*df)


def _get_reduce_finding_vjp(a, axis=None, *, keepdims=False):
    return lambda g, r: np.where(
        a == _expand_to(r, a.ndim, axis, keepdims),
        _expand_to(g, a.shape, axis, keepdims), 0,
    )


_matmul_nd_1d_vjp_x1 = lambda dy, x1, x2: np.broadcast_to(
    dy[..., None] * x2, x1.shape)
_matmul_nd_1d_vjp_x2 = lambda dy, x1, x2: _unbroadcast_to(
    dy[..., None] * x1, x2.shape)


def _convolve_vjp_x1(g, a, v, mode):
    w = {'full': v.size - 1, 'valid': 0, 'same': v.size // 2}[mode]
    a_pad = np.pad(a, w)
    s = a_pad.strides[0]
    a_col = as_strided(a_pad, [g.size, v.size], [s, s])
    da_col = _matmul_nd_1d_vjp_x1(g, a_col, v[::-1])
    da_col_pad = np.pad(da_col, ((0,), (g.size,)))
    da_col_strided = as_strided(
        da_col_pad.ravel()[g.size + w:],
        [g.size, a.size], [da_col_pad.strides[0] - s, s])
    da = da_col_strided.sum(0)
    return da


def _convolve_vjp_x2(g, a, v, mode):
    w = {'full': v.size - 1, 'valid': 0, 'same': v.size // 2}[mode]
    a_pad = np.pad(a, w)
    s = a_pad.strides[0]
    a_col = as_strided(a_pad, [g.size, v.size], [s, s])
    dv = _matmul_nd_1d_vjp_x2(g, a_col, v[::-1])[::-1]
    return dv


# https://numpy.org/doc/stable/reference/routines.math.html#trigonometric-functions
_hypot_vjp = lambda g, r, x: g * x / r
_register_vjp(np.sin, lambda x: lambda g, r: g * np.cos(x))
_register_vjp(np.cos, lambda x: lambda g, r: g * -np.sin(x))
_register_vjp(np.tan, lambda x: lambda g, r: g * (1 + np.square(r)))
_register_vjp(np.arcsin, lambda x: lambda g, r: g / np.cos(r))
_register_vjp(np.arccos, lambda x: lambda g, r: g / -np.sin(r))
_register_vjp(np.arctan, lambda x: lambda g, r: g * (np.cos(r) ** 2))
_register_vjp(np.hypot, _get_vjp(_hypot_vjp, _hypot_vjp))

# https://numpy.org/doc/stable/reference/routines.math.html#hyperbolic-functions
_register_vjp(np.sinh, lambda x: lambda g, r: g * np.cosh(x))
_register_vjp(np.cosh, lambda x: lambda g, r: g * np.sinh(x))
_register_vjp(np.tanh, lambda x: lambda g, r: g * (1 - np.square(r)))
_register_vjp(np.arcsinh, lambda x: lambda g, r: g / np.cosh(r))
_register_vjp(np.arccosh, lambda x: lambda g, r: g / np.sinh(r))
_register_vjp(np.arctanh, lambda x: lambda g, r: g / (1 - np.square(x)))

# https://numpy.org/doc/stable/reference/routines.math.html#sums-products-differences
_register_vjp(np.prod, _get_prod_vjp)
_register_vjp(np.sum, _get_sum_vjp)
_register_vjp(np.nanprod, partial(_get_prod_vjp, _prod=np.nanprod))
_register_vjp(np.nansum, _get_sum_vjp)
_register_vjp(np.cumprod, _get_cumprod_vjp)
_register_vjp(np.cumsum, _get_cumsum_vjp)
_register_vjp(np.nancumprod, _get_cumprod_vjp)
_register_vjp(np.nancumsum, _get_cumsum_vjp)

# https://numpy.org/doc/stable/reference/routines.math.html#exponents-and-logarithms
_logaddexp_vjp = lambda g, r, x: g * np.exp(x - r)
_logaddexp2_vjp = lambda g, r, x: g * np.exp2(x - r)
_register_vjp(np.exp, lambda x: lambda g, r: g * r)
_register_vjp(np.expm1, lambda x: lambda g, r: g * (r + 1))
_register_vjp(np.exp2, lambda x: lambda g, r: g * r * np.log(2))
_register_vjp(np.log, lambda x: lambda g, r: g / x)
_register_vjp(np.log10, lambda x: lambda g, r: g / (x * np.log(10)))
_register_vjp(np.log2, lambda x: lambda g, r: g / (x * np.log(2)))
_register_vjp(np.log1p, lambda x: lambda g, r: g / (1 + x))
_register_vjp(np.logaddexp, _get_vjp(_logaddexp_vjp, _logaddexp_vjp))
_register_vjp(np.logaddexp2, _get_vjp(_logaddexp2_vjp, _logaddexp2_vjp))

# https://numpy.org/doc/stable/reference/routines.math.html#arithmetic-operations
_mul_vjp = lambda g, r, x1, x2: g * x2
_div1_vjp = lambda g, r, x1, x2: g / x2
_div2_vjp = lambda g, r, x2, x1: g * x1 / -(x2 ** 2)
_pow1_vjp = lambda g, r, x1, x2: g * x2 * (x1 ** (x2 - 1))
_pow2_vjp = lambda g, r, x2, x1: None if np.any(x1 < 0) else g * r * np.log(x1)
_register_vjp(np.add, _get_binary_vjps(lambda g: g, lambda g: g))
_register_vjp(np.reciprocal, lambda x: lambda dy, y: dy * -(y ** 2))
_register_vjp(np.positive, lambda x: lambda dy, y: +dy)
_register_vjp(np.negative, lambda x: lambda dy, y: -dy)
_register_vjp(np.multiply, _get_vjp(_mul_vjp, _mul_vjp))
_register_vjp(np.divide, _get_vjp(_div1_vjp, _div2_vjp))
_register_vjp(np.power, _get_vjp(_pow1_vjp, _pow2_vjp))
_register_vjp(np.subtract, _get_vjp(lambda g: g, lambda g: -g))
_register_vjp(np.float_power, _get_vjp(_pow1_vjp, _pow2_vjp))

# https://numpy.org/doc/stable/reference/routines.math.html#extrema-finding
_finding_vjp = lambda g, r, x: np.where(x == r, g, 0)
_register_vjp(np.maximum, _get_vjp(_finding_vjp, _finding_vjp))
_register_vjp(np.fmax, _get_vjp(_finding_vjp, _finding_vjp))
_register_vjp(np.amax, _get_reduce_finding_vjp)
_register_vjp(np.nanmax, _get_reduce_finding_vjp)
_register_vjp(np.minimum, _get_vjp(_finding_vjp, _finding_vjp))
_register_vjp(np.fmin, _get_vjp(_finding_vjp, _finding_vjp))
_register_vjp(np.amin, _get_reduce_finding_vjp)
_register_vjp(np.nanmin, _get_reduce_finding_vjp)

# https://numpy.org/doc/stable/reference/routines.math.html#miscellaneous
_register_vjp(
    np.convolve,
    lambda a, v, mode='full': (
        lambda g, r: _convolve_vjp_x1(g, np.asarray(a), np.asarray(v), mode),
        lambda g, r: _convolve_vjp_x2(g, np.asarray(a), np.asarray(v), mode),
    ),
)
_register_vjp(
    np.clip,
    lambda a, a_min, a_max: (
        lambda g, r: _unbroadcast_to(
            g * np.logical_and(r != a_min, r != a_max), a.shape),
        lambda g, r: _unbroadcast_to(g * (r == a_min), a_min.shape),
        lambda g, r: _unbroadcast_to(g * (r == a_max), a_max.shape),
    ),
)
_register_vjp(np.sqrt, lambda x: lambda g, r: g * 0.5 / r)
_register_vjp(np.cbrt, lambda x: lambda g, r: g / (3 * r ** 2))
_register_vjp(np.square, lambda x: lambda g, r: g * 2 * x)
_register_vjp(np.absolute, lambda x: lambda g, r: g * np.sign(x))
_register_vjp(np.fabs, lambda x: lambda g, r: g * np.sign(x))
_register_vjp(
    np.nan_to_num,
    lambda x, copy=True, nan=0., posinf=None, neginf=None: (
        lambda g, r: np.where(np.isfinite(x), g, 0)),
)
