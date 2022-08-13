# https://numpy.org/doc/stable/reference/routines.math.html

from functools import partial

import numpy as np
from numpy.lib.stride_tricks import as_strided

from numgrad._utils._expand_to import _expand_to
from numgrad._utils._unbroadcast import _unbroadcast_to
from numgrad._vjp import _bind_vjp


def _prod_vjp(g, r, a, axis=None, *, keepdims=False, _prod=np.prod):
    return _expand_to(g, a.ndim, axis, keepdims) * (
        r if keepdims else _prod(a, axis, keepdims=True)) / a


def _sum_vjp(g, a, axis=None, *, keepdims=False):
    return _expand_to(g, a.shape, axis, keepdims)


def _cumulate_inversely(cum_op, a, axis=None):
    return np.flip(cum_op(np.flip(a, axis), axis), axis)


def _cumprod_vjp(g, r, a, axis=None):
    if a.ndim == 0:
        return g
    if axis is None:
        return _cumulate_inversely(np.cumsum, g * r).reshape(a.shape) / a
    return _cumulate_inversely(np.cumsum, g * r, axis) / a


def _cumsum_vjp(g, r, a, axis=None):
    if a.ndim == 0:
        return g
    if axis is None:
        return _cumulate_inversely(np.cumsum, g, 0).reshape(a.shape)
    return _cumulate_inversely(np.cumsum, g, axis)


def _diff_vjp(g, r, a, n=1, axis=-1, prepend=None, append=None):
    for _ in range(n):
        g = -np.diff(g, n=1, axis=axis, prepend=0, append=0)
    indices = range(
        0 if prepend is None else np.asarray(prepend).shape[axis],
        g.shape[axis] - (
            0 if append is None else np.asarray(append).shape[axis]),
    )
    return g.take(indices, axis)


def _diff_vjp_prepend(g, r, a, n=1, axis=-1, prepend=None, append=None):
    for _ in range(n):
        g = -np.diff(g, n=1, axis=axis, prepend=0, append=0)
    return g.take(range(prepend.shape[axis]), axis)


def _diff_vjp_append(g, r, a, n=1, axis=-1, prepend=None, append=None):
    for _ in range(n):
        g = -np.diff(g, n=1, axis=axis, prepend=0, append=0)
    return g.take(
        range(g.shape[axis] - prepend.shape[axis], g.shape[axis]), axis)


def _ediff1d_vjp(g, r, ary, to_end=None, to_begin=None):
    if to_end is None and to_begin is None:
        return -np.diff(g, append=0, prepend=0)
    s = slice(
        None if to_begin is None else np.asarray(to_begin).size,
        None if to_end is None else -np.asarray(to_end).size,
    )
    return -np.diff(g[s], append=0, prepend=0)


def _reduce_finding_vjp(g, r, a, axis=None, *, keepdims=False):
    return np.where(
        a == _expand_to(r, a.ndim, axis, keepdims),
        _expand_to(g, a.shape, axis, keepdims), 0,
    )


_matmul_nd_1d_vjp_x1 = lambda dy, x1, x2: np.broadcast_to(
    dy[..., None] * x2, x1.shape)
_matmul_nd_1d_vjp_x2 = lambda dy, x1, x2: _unbroadcast_to(
    dy[..., None] * x1, x2.shape)


def _convolve_vjp_x1(g, a, v, mode='full'):
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


def _convolve_vjp_x2(g, a, v, mode='full'):
    w = {'full': v.size - 1, 'valid': 0, 'same': v.size // 2}[mode]
    a_pad = np.pad(a, w)
    s = a_pad.strides[0]
    a_col = as_strided(a_pad, [g.size, v.size], [s, s])
    dv = _matmul_nd_1d_vjp_x2(g, a_col, v[::-1])[::-1]
    return dv


# https://numpy.org/doc/stable/reference/routines.math.html#trigonometric-functions
_bind_vjp(np.sin, lambda x: np.cos(x))
_bind_vjp(np.cos, lambda x: -np.sin(x))
_bind_vjp(np.tan, lambda r, _: 1 + np.square(r))
_bind_vjp(np.arcsin, lambda g, r, _: g / np.cos(r))
_bind_vjp(np.arccos, lambda g, r, _: g / -np.sin(r))
_bind_vjp(np.arctan, lambda r, _: np.cos(r) ** 2)
_bind_vjp(np.hypot, lambda r, x1, _: x1 / r, lambda r, _, x2: x2 / r)
_bind_vjp(
    np.arctan2,
    lambda r, _, x2: (np.cos(r) ** 2) / x2,
    lambda r, x1, x2: (np.cos(r) ** 2) * -x1 / (x2 ** 2),
)
_bind_vjp(np.degrees, lambda _: 180 / np.pi)
_bind_vjp(np.radians, lambda _: np.pi / 180)
_bind_vjp(np.rad2deg, lambda _: 180 / np.pi)
_bind_vjp(np.deg2rad, lambda _: np.pi / 180)

# https://numpy.org/doc/stable/reference/routines.math.html#hyperbolic-functions
_bind_vjp(np.sinh, lambda x: np.cosh(x))
_bind_vjp(np.cosh, lambda x: np.sinh(x))
_bind_vjp(np.tanh, lambda r, _: (1 - np.square(r)))
_bind_vjp(np.arcsinh, lambda g, r, _: g / np.cosh(r))
_bind_vjp(np.arccosh, lambda g, r, _: g / np.sinh(r))
_bind_vjp(np.arctanh, lambda g, x: g / (1 - np.square(x)))

# https://numpy.org/doc/stable/reference/routines.math.html#sums-products-differences
_bind_vjp(np.prod, _prod_vjp)
_bind_vjp(np.sum, _sum_vjp)
_bind_vjp(np.nanprod, partial(_prod_vjp, _prod=np.nanprod))
_bind_vjp(np.nansum, _sum_vjp)
_bind_vjp(np.cumprod, _cumprod_vjp)
_bind_vjp(np.cumsum, _cumsum_vjp)
_bind_vjp(np.nancumprod, _cumprod_vjp)
_bind_vjp(np.nancumsum, _cumsum_vjp)
_bind_vjp(
    np.diff,
    _diff_vjp,
    lambda: None,  # n
    lambda: None,  # axis
    _diff_vjp_prepend,
    _diff_vjp_append,
)
_bind_vjp(
    np.ediff1d,
    _ediff1d_vjp,
    lambda g, ary, to_end=None, to_begin=None: g[-to_end.size:].reshape(
        to_end.shape),
    lambda g, ary, to_end=None, to_begin=None: g[:to_begin.size].reshape(
        to_begin.shape),
)

# https://numpy.org/doc/stable/reference/routines.math.html#exponents-and-logarithms
_bind_vjp(np.exp, lambda r, _: r)
_bind_vjp(np.expm1, lambda r, _: r + 1)
_bind_vjp(np.exp2, lambda r, _: r * np.log(2))
_bind_vjp(np.log, lambda g, x: g / x)
_bind_vjp(np.log10, lambda g, x: g / (x * np.log(10)))
_bind_vjp(np.log2, lambda g, x: g / (x * np.log(2)))
_bind_vjp(np.log1p, lambda g, x: g / (1 + x))
_bind_vjp(
    np.logaddexp,
    lambda r, x1, _: np.exp(x1 - r),
    lambda r, _, x2: np.exp(x2 - r),
)
_bind_vjp(
    np.logaddexp2,
    lambda r, x1, _: np.exp2(x1 - r),
    lambda r, _, x2: np.exp2(x2 - r),
)

# https://numpy.org/doc/stable/reference/routines.math.html#arithmetic-operations
_bind_vjp(np.add, lambda g, *_: +g, lambda g, *_: +g)
_bind_vjp(np.reciprocal, lambda r, _: -(r ** 2))
_bind_vjp(np.positive, lambda g, _: +g)
_bind_vjp(np.negative, lambda g, _: -g)
_bind_vjp(np.multiply, lambda x1, x2: x2, lambda x1, x2: x1)
_bind_vjp(np.divide, lambda g, x1, x2: g / x2, lambda x1, x2: x1 / -(x2 ** 2))
_bind_vjp(
    np.power,
    lambda x1, x2: x2 * x1 ** (x2 - 1),
    lambda r, x1, x2: None if np.any(x1 < 0) else r * np.log(x1),
)
_bind_vjp(np.subtract, lambda g, *_: +g, lambda g, *_: -g)
_bind_vjp(
    np.float_power,
    lambda x1, x2: x2 * x1 ** (x2 - 1),
    lambda r, x1, x2: None if np.any(x1 < 0) else r * np.log(x1),
)
_bind_vjp(np.fmod, lambda g, *_: +g, lambda r, x1, x2: (r - x1) / x2)
_bind_vjp(np.mod, lambda g, *_: +g, lambda r, x1, x2: (r - x1) / x2)

# https://numpy.org/doc/stable/reference/routines.math.html#extrema-finding
_finding_vjp_x1 = lambda g, r, x1, x2: np.where(x1 == r, g, 0)
_finding_vjp_x2 = lambda g, r, x1, x2: np.where(x2 == r, g, 0)
_bind_vjp(np.maximum, _finding_vjp_x1, _finding_vjp_x2)
_bind_vjp(np.fmax, _finding_vjp_x1, _finding_vjp_x2)
_bind_vjp(np.amax, _reduce_finding_vjp)
_bind_vjp(np.nanmax, _reduce_finding_vjp)
_bind_vjp(np.minimum, _finding_vjp_x1, _finding_vjp_x2)
_bind_vjp(np.fmin, _finding_vjp_x1, _finding_vjp_x2)
_bind_vjp(np.amin, _reduce_finding_vjp)
_bind_vjp(np.nanmin, _reduce_finding_vjp)

# https://numpy.org/doc/stable/reference/routines.math.html#miscellaneous
_bind_vjp(np.convolve, _convolve_vjp_x1, _convolve_vjp_x2)
_bind_vjp(
    np.clip,
    lambda r, a, a_min, a_max: np.logical_and(r != a_min, r != a_max),
    lambda r, a, a_min, a_max: r == a_min,
    lambda r, a, a_min, a_max: r == a_max,
)
_bind_vjp(np.sqrt, lambda r, _: 0.5 / r)
_bind_vjp(np.cbrt, lambda g, r, _: g / (3 * r ** 2))
_bind_vjp(np.square, lambda x: 2 * x)
_bind_vjp(np.absolute, lambda x: np.sign(x))
_bind_vjp(np.fabs, lambda x: np.sign(x))
_bind_vjp(np.nan_to_num, lambda g, x, *_, **k: np.where(np.isfinite(x), g, 0))
