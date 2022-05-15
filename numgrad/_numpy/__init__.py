import numpy as np

from numgrad._utils._expand_to import _expand_to
from numgrad._utils._unbroadcast import _unbroadcast_to
from numgrad._variable import Variable
from numgrad._vjp import _register_vjp, differentiable


# https://numpy.org/doc/stable/reference/arrays.ndarray.html#special-methods
def _getitem_vjp(dy, y, x, key):
    dx = np.zeros_like(x)
    dx[key] = dy
    return dx


Variable.__getitem__ = differentiable(_getitem_vjp)(
    lambda self, key: self[key])
Variable.__getitem__.__doc__ = np.ndarray.__getitem__.__doc__


# https://numpy.org/doc/stable/reference/routines.array-creation.html#building-matrices
_register_vjp(
    np.diag,
    lambda dy, y, x, k=0: (
        dx := np.diag(dy, k=k),
        dx if x.shape == dx.shape else np.pad(
            dx, ((0, x.shape[0] - dx.shape[0]), (0, x.shape[1] - dx.shape[1])),
        ),
    )[1],
)
_register_vjp(
    np.diagflat,
    lambda dy, y, x, k=0: np.diag(dy, k=k).reshape(*x.shape),
)
_register_vjp(np.tril, lambda dy, y, _x, k=0: np.tril(dy, k))
_register_vjp(np.triu, lambda dy, y, _x, k=0: np.triu(dy, k))
_register_vjp(
    np.vander,
    lambda dy, y, x, N=None, increasing=False: (
        N := len(x) if N is None else N,
        np.sum(dy[:, 1:] * y[:, :-1] * range(1, N), -1) if increasing
        else np.sum(dy[:, :-1] * y[:, 1:] * range(1, N)[::-1], -1),
    )[1],
)


# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#changing-array-shape
_register_vjp(
    np.reshape,
    lambda dy, y, x, newshape, order=None: dy.reshape(*x.shape, order=order),
)
_register_vjp(
    np.ravel,
    lambda dy, y, x, order=None: dy.reshape(*x.shape, order=order),
)
Variable.flatten = differentiable(
    lambda dy, y, x: dy.reshape(x.shape))(lambda a: a.flatten())
Variable.flatten.__doc__ = np.ndarray.flatten.__doc__

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#transpose-like-operations
_register_vjp(
    np.moveaxis,
    lambda dy, y, x, source, destination: np.moveaxis(
        dy, source=destination, destination=source),
)
_register_vjp(
    np.swapaxes,
    lambda dy, y, x, axis1, axis2: np.swapaxes(dy, axis1, axis2),
)
_register_vjp(
    np.transpose,
    lambda dy, y, x, axes=None: (
        np.transpose(dy) if axes is None
        else np.transpose(dy, np.argsort(axes))
    ),
)

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#changing-number-of-dimensions
_register_vjp(
    np.broadcast_to,
    lambda dy, y, x, shape: _unbroadcast_to(dy, x.shape),
)
_register_vjp(np.expand_dims, lambda dy, y, x, axis: np.squeeze(dy, axis))
_register_vjp(np.squeeze, lambda dy, y, x, axis=None: (
    np.expand_dims(dy, [ax for ax, len_ in enumerate(x.shape) if len_ == 1])
    if axis is None else np.expand_dims(dy, axis)))

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#splitting-arrays
_register_vjp(
    np.split,
    lambda dy, y, x, indices_or_sections, axis=0: np.concatenate(
        dy, axis=axis),
)
_register_vjp(
    np.array_split,
    lambda dy, y, x, indices_or_sections, axis=0: np.concatenate(
        dy, axis=axis),
)
_register_vjp(
    np.dsplit,
    lambda dy, y, x, indices_or_sections: (
        np.concatenate(dy, axis=2)),
)
_register_vjp(
    np.hsplit,
    lambda dy, y, x, indices_or_sections: (
        np.concatenate(dy, axis=1)),
)
_register_vjp(
    np.vsplit,
    lambda dy, y, x, indices_or_sections: (
        np.concatenate(dy, axis=0)),
)

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#rearranging-elements
_register_vjp(np.flip, lambda dy, y, x, axis=None: np.flip(dy, axis))
_register_vjp(np.fliplr, lambda dy, y, x: np.fliplr(dy))
_register_vjp(np.flipud, lambda dy, y, x: np.flipud(dy))
_register_vjp(np.roll, lambda dy, y, x, shift, axis=None: np.roll(
    dy, -shift if isinstance(shift, int) else [-s for s in shift], axis))
_register_vjp(
    np.rot90,
    lambda dy, y, x, k=1, axes=(0, 1): np.rot90(dy, -k, axes),
)


# https://numpy.org/doc/stable/reference/routines.linalg.html
_matmul_0d_0d_vjp_x1 = lambda dy, x1, x2: dy * x2
_matmul_0d_0d_vjp_x2 = lambda dy, x1, x2: dy * x1

_matmul_1d_1d_vjp_x1 = lambda dy, x1, x2: dy * x2
_matmul_1d_1d_vjp_x2 = lambda dy, x1, x2: dy * x1
_matmul_1d_nd_vjp_x1 = lambda dy, x1, x2: _unbroadcast_to(
    (dy[..., None, :] * x2).sum(-1), x1.shape)
_matmul_1d_nd_vjp_x2 = lambda dy, x1, x2: np.broadcast_to(
    dy[..., None, :], x2.shape) * x1[:, None]

_matmul_nd_1d_vjp_x1 = lambda dy, x1, x2: np.broadcast_to(
    dy[..., None] * x2, x1.shape)
_matmul_nd_1d_vjp_x2 = lambda dy, x1, x2: _unbroadcast_to(
    dy[..., None] * x1, x2.shape)
_matmul_nd_nd_vjp_x1 = lambda dy, x1, x2: _unbroadcast_to(
    dy @ np.swapaxes(x2, -1, -2), x1.shape)
_matmul_nd_nd_vjp_x2 = lambda dy, x1, x2: _unbroadcast_to(
    np.swapaxes(x1, -1, -2) @ dy, x2.shape)

_dot_0d_0d_vjp_x1 = _matmul_0d_0d_vjp_x1
_dot_0d_0d_vjp_x2 = _matmul_0d_0d_vjp_x2
_dot_1d_1d_vjp_x1 = _matmul_1d_1d_vjp_x1
_dot_1d_1d_vjp_x2 = _matmul_1d_1d_vjp_x2
_dot_1d_nd_vjp_x1 = _matmul_1d_nd_vjp_x1
_dot_1d_nd_vjp_x2 = _matmul_1d_nd_vjp_x2
_dot_nd_1d_vjp_x1 = _matmul_nd_1d_vjp_x1
_dot_nd_1d_vjp_x2 = _matmul_nd_1d_vjp_x2
_dot_nd_nd_vjp_x1 = lambda dy, x1, x2: (
    dy[..., None] * np.moveaxis(x2, -2, -1)).sum(
        tuple(-i - 2 for i in range(x2.ndim - 1)))
_dot_nd_nd_vjp_x2 = lambda dy, x1, x2: np.swapaxes(
    np.tensordot(
        dy, x1,
        [range(-x1.ndim - x2.ndim + 2, -x2.ndim + 1), range(x1.ndim - 1)],
    ), -1, -2,
)

_inner_0d_0d_vjp_x1 = _matmul_0d_0d_vjp_x1
_inner_0d_0d_vjp_x2 = _matmul_0d_0d_vjp_x2
_inner_1d_1d_vjp_x1 = _matmul_1d_1d_vjp_x1
_inner_1d_1d_vjp_x2 = _matmul_1d_1d_vjp_x2
_inner_1d_nd_vjp_x1 = lambda dy, x1, x2: _unbroadcast_to(
    dy[..., None] * x2, x1.shape)
_inner_1d_nd_vjp_x2 = lambda dy, x1, x2: dy[..., None] * x1
_inner_nd_1d_vjp_x1 = _matmul_nd_1d_vjp_x1
_inner_nd_1d_vjp_x2 = _matmul_nd_1d_vjp_x2
_inner_nd_nd_vjp_x1 = lambda dy, x1, x2: (
    dy[..., None] * x2).sum(tuple(-i - 2 for i in range(x2.ndim - 1)))
_inner_nd_nd_vjp_x2 = lambda dy, x1, x2: np.tensordot(
    dy, x1, [range(-x1.ndim - x2.ndim + 2, -x2.ndim + 1), range(x1.ndim - 1)])

_register_vjp(
    np.dot,
    lambda dy, y, x1, x2: (
        x1 := np.asarray(x1),
        x2 := np.asarray(x2),
        d1 := 'n' if (x1.ndim > 1) else x1.ndim,
        d2 := 'n' if (x2.ndim > 1) else x2.ndim,
        eval(f'_dot_{d1}d_{d2}d_vjp_x1')(dy, x1, x2),
    )[-1],
    lambda dy, y, x1, x2: (
        x1 := np.asarray(x1),
        x2 := np.asarray(x2),
        d1 := 'n' if (x1.ndim > 1) else x1.ndim,
        d2 := 'n' if (x2.ndim > 1) else x2.ndim,
        eval(f'_dot_{d1}d_{d2}d_vjp_x2')(dy, x1, x2),
    )[-1],
)
_register_vjp(
    np.vdot,
    lambda dy, y, x1, x2: (dy * x2).reshape(x1.shape),
    lambda dy, y, x1, x2: (dy * x1).reshape(x2.shape),
)
_register_vjp(
    np.inner,
    lambda dy, y, x1, x2: (
        x1 := np.asarray(x1),
        x2 := np.asarray(x2),
        d1 := 'n' if (x1.ndim > 1) else x1.ndim,
        d2 := 'n' if (x2.ndim > 1) else x2.ndim,
        eval(f'_inner_{d1}d_{d2}d_vjp_x1')(dy, x1, x2),
    )[-1],
    lambda dy, y, x1, x2: (
        x1 := np.asarray(x1),
        x2 := np.asarray(x2),
        d1 := 'n' if (x1.ndim > 1) else x1.ndim,
        d2 := 'n' if (x2.ndim > 1) else x2.ndim,
        eval(f'_inner_{d1}d_{d2}d_vjp_x2')(dy, x1, x2),
    )[-1],
)
_register_vjp(
    np.outer,
    lambda dy, y, x1, x2: np.sum(dy * np.ravel(x2), -1).reshape(x1.shape),
    lambda dy, y, x1, x2: np.sum(dy * np.ravel(x1)[None, ...], -1).reshape(
        x2.shape),
)


_register_vjp(
    np.matmul,
    lambda dy, y, x1, x2: (
        x1 := np.asarray(x1),
        x2 := np.asarray(x2),
        d1 := 'n' if (x1.ndim > 1) else x1.ndim,
        d2 := 'n' if (x2.ndim > 1) else x2.ndim,
        eval(f'_matmul_{d1}d_{d2}d_vjp_x1')(dy, x1, x2),
    )[-1],
    lambda dy, y, x1, x2: (
        x1 := np.asarray(x1),
        x2 := np.asarray(x2),
        d1 := 'n' if (x1.ndim > 1) else x1.ndim,
        d2 := 'n' if (x2.ndim > 1) else x2.ndim,
        eval(f'_matmul_{d1}d_{d2}d_vjp_x2')(dy, x1, x2),
    )[-1],
)

# https://numpy.org/doc/stable/reference/routines.linalg.html#norms-and-other-numbers
_register_vjp(
    np.linalg.det,
    lambda dy, y, x: (dy * y)[..., None, None] * np.linalg.inv(
        np.swapaxes(x, -1, -2)),
)

# https://numpy.org/doc/stable/reference/routines.math.html#trigonometric-functions
_register_vjp(np.sin, lambda dy, y, x: dy * np.cos(x))
_register_vjp(np.cos, lambda dy, y, x: dy * -np.sin(x))
_register_vjp(np.tan, lambda dy, y, x: dy * (1 + np.square(y)))
_register_vjp(np.arcsin, lambda dy, y, x: dy / np.cos(y))
_register_vjp(np.arccos, lambda dy, y, x: dy / -np.sin(y))
_register_vjp(np.arctan, lambda dy, y, x: dy * (np.cos(y) ** 2))
_register_vjp(
    np.hypot,
    lambda dy, y, x1, x2: _unbroadcast_to(dy * x1 / y, x1.shape),
    lambda dy, y, x1, x2: _unbroadcast_to(dy * x2 / y, x2.shape),
)

# https://numpy.org/doc/stable/reference/routines.math.html#hyperbolic-functions
_register_vjp(np.sinh, lambda dy, y, x: dy * np.cosh(x))
_register_vjp(np.cosh, lambda dy, y, x: dy * np.sinh(x))
_register_vjp(np.tanh, lambda dy, y, x: dy * (1 - np.square(y)))
_register_vjp(np.arcsinh, lambda dy, y, x: dy / np.cosh(y))
_register_vjp(np.arccosh, lambda dy, y, x: dy / np.sinh(y))
_register_vjp(np.arctanh, lambda dy, y, x: dy / (1 - np.square(x)))

# https://numpy.org/doc/stable/reference/routines.math.html#sums-products-differences
_register_vjp(
    np.prod,
    lambda dy, y, x, axis=None, keepdims=False: (
        _expand_to(dy, x.ndim, axis, keepdims)
        * (y if keepdims else np.prod(x, axis, keepdims=True)) / x
    ),
)
_register_vjp(
    np.sum,
    lambda dy, y, x, axis=None, keepdims=False, **kwargs: _expand_to(
        dy, x.shape, axis, keepdims),
)
_register_vjp(
    np.nanprod,
    lambda dy, y, x, axis=None, keepdims=False, **kwargs: (
        _expand_to(dy, x.ndim, axis, keepdims)
        * (y if keepdims else np.nanprod(x, axis, keepdims=True)) / x
    ),
)
_register_vjp(
    np.nansum,
    lambda dy, y, x, axis=None, keepdims=False, **kwargs: _expand_to(
        dy, x.shape, axis, keepdims),
)
_register_vjp(
    np.cumprod,
    lambda dy, y, x, axis=None, **kwargs: (
        dy if x.ndim == 0 else np.flip(np.cumsum(np.flip((dy * y).reshape(
            *x.shape), axis), axis).reshape(*x.shape), axis) / x
    ),
)
_register_vjp(
    np.cumsum,
    lambda dy, y, x, axis=None, **kwargs: (
        dy if x.ndim == 0 else np.flip(np.cumsum(np.flip(
            dy.reshape(*x.shape), axis), axis).reshape(*x.shape), axis)
    ),
)
_register_vjp(
    np.nancumprod,
    lambda dy, y, x, axis=None, **kwargs: (
        dy if x.ndim == 0 else np.flip(np.cumsum(np.flip((dy * y).reshape(
            *x.shape), axis), axis).reshape(*x.shape), axis) / x
    ),
)
_register_vjp(
    np.nancumsum,
    lambda dy, y, x, axis=None, **kwargs: (
        dy if x.ndim == 0 else np.flip(np.cumsum(np.flip(
            dy.reshape(*x.shape), axis), axis).reshape(*x.shape), axis)
    ),
)

# https://numpy.org/doc/stable/reference/routines.math.html#exponents-and-logarithms
_register_vjp(np.exp, lambda dy, y, x: dy * y)
_register_vjp(np.expm1, lambda dy, y, x: dy * (y + 1))
_register_vjp(np.exp2, lambda dy, y, x: dy * y * np.log(2))
_register_vjp(np.log, lambda dy, y, x: dy / x)
_register_vjp(np.log10, lambda dy, y, x: dy / (x * np.log(10)))
_register_vjp(np.log2, lambda dy, y, x: dy / (x * np.log(2)))
_register_vjp(np.log1p, lambda dy, y, x: dy / (1 + x))
_register_vjp(
    np.logaddexp,
    lambda dy, y, x1, x2: _unbroadcast_to(dy * np.exp(x1 - y), x1.shape),
    lambda dy, y, x1, x2: _unbroadcast_to(dy * np.exp(x2 - y), x2.shape),
)
_register_vjp(
    np.logaddexp2,
    lambda dy, y, x1, x2: _unbroadcast_to(dy * np.exp2(x1 - y), x1.shape),
    lambda dy, y, x1, x2: _unbroadcast_to(dy * np.exp2(x2 - y), x2.shape),
)

# https://numpy.org/doc/stable/reference/routines.math.html#arithmetic-operations
_register_vjp(
    np.add,
    lambda dy, y, x1, x2: _unbroadcast_to(dy, x1.shape),
    lambda dy, y, x1, x2: _unbroadcast_to(dy, x2.shape),
)
_register_vjp(np.reciprocal, lambda dy, y, x: dy * -(y ** 2))
_register_vjp(np.positive, lambda dy, y, x: dy)
_register_vjp(np.negative, lambda dy, y, x: -dy)
_register_vjp(
    np.multiply,
    lambda dy, y, x1, x2: _unbroadcast_to(dy * x2, x1.shape),
    lambda dy, y, x1, x2: _unbroadcast_to(dy * x1, x2.shape),
)
_register_vjp(
    np.divide,
    lambda dy, y, x1, x2: _unbroadcast_to(dy / x2, x1.shape),
    lambda dy, y, x1, x2: _unbroadcast_to(dy * x1 / -(x2 ** 2), x2.shape),
)
_register_vjp(
    np.power,
    lambda dy, y, x1, x2: _unbroadcast_to(dy * x2 * y / x1, x1.shape),
    lambda dy, y, x1, x2: (
        None if np.any(np.asarray(x1) < 0)
        else _unbroadcast_to(dy * y * np.log(x1), x2.shape)
    ),
)
_register_vjp(
    np.subtract,
    lambda dy, y, x1, x2: _unbroadcast_to(dy, x1.shape),
    lambda dy, y, x1, x2: _unbroadcast_to(-dy, x2.shape),
)
_register_vjp(
    np.float_power,
    lambda dy, y, x1, x2: _unbroadcast_to(dy * x2 * y / x1, x1.shape),
    lambda dy, y, x1, x2: (
        None if np.any(np.asarray(x1) < 0)
        else _unbroadcast_to(dy * y * np.log(x1), x2.shape)
    ),
)

# https://numpy.org/doc/stable/reference/routines.math.html#extrema-finding
_register_vjp(
    np.maximum,
    lambda dy, y, x1, x2: _unbroadcast_to(
        np.where(np.asarray(x1) != np.asarray(y), 0, dy), x1.shape),
    lambda dy, y, x1, x2: _unbroadcast_to(
        np.where(np.asarray(x2) != np.asarray(y), 0, dy), x2.shape),
)
_register_vjp(
    np.fmax,
    lambda dy, y, x1, x2: _unbroadcast_to(
        np.where(np.asarray(x1) != np.asarray(y), 0, dy), x1.shape),
    lambda dy, y, x1, x2: _unbroadcast_to(
        np.where(np.asarray(x2) != np.asarray(y), 0, dy), x2.shape),
)
_register_vjp(
    np.amax,
    lambda dy, y, x, axis=None, keepdims=False, **kwargs: np.where(
        np.asarray(x) == (
            np.asarray(y) if keepdims else
            np.asarray(x).max(axis, keepdims=True)
        ),
        _expand_to(dy, x.shape, axis, keepdims), 0,
    ),
)
_register_vjp(
    np.nanmax,
    lambda dy, y, x, axis=None, keepdims=False: np.where(
        np.asarray(x) == (
            np.asarray(y) if keepdims else
            np.nanmax(np.asarray(x), axis, keepdims=True)
        ),
        _expand_to(dy, x.shape, axis, keepdims), 0,
    ),
)
_register_vjp(
    np.minimum,
    lambda dy, y, x1, x2: _unbroadcast_to(
        np.where(np.asarray(x1) != np.asarray(y), 0, dy), x1.shape),
    lambda dy, y, x1, x2: _unbroadcast_to(
        np.where(np.asarray(x2) != np.asarray(y), 0, dy), x2.shape),
)
_register_vjp(
    np.fmin,
    lambda dy, y, x1, x2: _unbroadcast_to(
        np.where(np.asarray(x1) != np.asarray(y), 0, dy), x1.shape),
    lambda dy, y, x1, x2: _unbroadcast_to(
        np.where(np.asarray(x2) != np.asarray(y), 0, dy), x2.shape),
)
_register_vjp(
    np.amin,
    lambda dy, y, x, axis=None, keepdims=False, **kwargs: np.where(
        np.asarray(x) == (
            np.asarray(y) if keepdims else
            np.asarray(x).min(axis, keepdims=True)
        ),
        _expand_to(dy, x.shape, axis, keepdims), 0,
    ),
)
_register_vjp(
    np.nanmin,
    lambda dy, y, x, axis=None, keepdims=False: np.where(
        np.asarray(x) == (
            np.asarray(y) if keepdims else
            np.nanmin(np.asarray(x), axis, keepdims=True)
        ),
        _expand_to(dy, x.shape, axis, keepdims), 0,
    ),
)

# https://numpy.org/doc/stable/reference/routines.math.html#miscellaneous
_register_vjp(np.sqrt, lambda dy, y, x: dy * 0.5 / y)
_register_vjp(np.cbrt, lambda dy, y, x: dy / (3 * y ** 2))
_register_vjp(np.square, lambda dy, y, x: dy * 2 * x)
_register_vjp(np.absolute, lambda dy, y, x: dy * np.sign(x))
_register_vjp(np.fabs, lambda dy, y, x: dy * np.sign(x))

# https://numpy.org/doc/stable/reference/random/legacy.html#functions-in-numpy-random
_register_vjp(
    np.random.exponential,
    lambda dy, y, scale, size=None: (
        dx := dy * y / scale,
        dx if size is None else _unbroadcast_to(dx, scale.shape),
    )[1],
    module_name='numpy.random', func_name='exponential',
)
_register_vjp(
    np.random.normal,
    lambda dy, y, loc, scale, size=None, **kwargs: _unbroadcast_to(
        dy, loc.shape),
    lambda dy, y, loc, scale, size=None: _unbroadcast_to(
        dy * (np.asarray(y) - np.asarray(loc)) / np.asarray(scale),
        scale.shape,
    ),
    module_name='numpy.random', func_name='normal',
)
_register_vjp(
    np.random.uniform,
    lambda dy, y, low, high, size=None: (
        y := np.asarray(y),
        low := np.asarray(low),
        high := np.asarray(high),
        u := (y - low) / (high - low),
        _unbroadcast_to(dy - dy * u, low.shape),
    )[-1],
    lambda dy, y, low, high, size=None: (
        y := np.asarray(y),
        low := np.asarray(low),
        high := np.asarray(high),
        u := (y - low) / (high - low),
        _unbroadcast_to(dy * u, high.shape),
    )[-1],
    module_name='numpy.random', func_name='uniform',
)


# https://numpy.org/doc/stable/reference/routines.statistics.html#averages-and-variances
_register_vjp(
    np.mean,
    lambda dy, y, x, axis=None, keepdims=False: (
        _expand_to(dy, x.shape, axis, keepdims) * dy.size / x.size
    ),
)
_register_vjp(
    np.nanmean,
    lambda dy, y, x, axis=None, *, keepdims=False: (
        _expand_to(dy, x.shape, axis, keepdims) / np.sum(
            ~np.isnan(x), axis, keepdims=True)
    ),
)


__all__ = []
