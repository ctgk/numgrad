import numpy as np

from numgrad._utils._expand_to import _expand_to
from numgrad._utils._unbroadcast import _unbroadcast_to
from numgrad._variable import Variable
from numgrad._vjp import _register_vjp, differentiable


# https://numpy.org/doc/stable/reference/arrays.ndarray.html#special-methods
def _getitem_vjp(dy, _y, x, key):
    dx = np.zeros_like(x)
    dx[key] = dy
    return dx


Variable.__getitem__ = differentiable(_getitem_vjp)(
    lambda self, key: self[key])
Variable.__getitem__.__doc__ = np.ndarray.__getitem__.__doc__


# https://numpy.org/doc/stable/reference/routines.array-creation.html#building-matrices
_register_vjp(np.diag, lambda dy, _y, x, k=0: _unbroadcast_to(
    np.diag(dy, k=k), x.shape))
_register_vjp(np.diagflat, lambda dy, _y, x, k=0: np.diag(dy, k=k).reshape(
    *x.shape))
_register_vjp(np.tril, lambda dy, _y, _x, k=0: np.tril(dy, k))
_register_vjp(np.triu, lambda dy, _y, _x, k=0: np.triu(dy, k))


# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#changing-array-shape
_register_vjp(np.reshape, lambda dy, _y, x, _newshape, order=None: dy.reshape(
    *x.shape, order=order))
_register_vjp(np.ravel, lambda dy, _y, x, order=None: dy.reshape(
    *x.shape, order=order))

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#transpose-like-operations
_register_vjp(np.moveaxis, lambda dy, _y, _x, source, destination: np.moveaxis(
    dy, source=destination, destination=source))
_register_vjp(np.swapaxes, lambda dy, _y, _x, axis1, axis2: np.swapaxes(
    dy, axis1, axis2))
_register_vjp(
    np.transpose,
    lambda dy, _y, _x, axes=None: (
        np.transpose(dy) if axes is None
        else np.transpose(dy, np.argsort(axes))
    ),
)


# https://numpy.org/doc/stable/reference/routines.linalg.html
def _matmul_vjp_x1(dy, _y, x1, x2):
    x1, x2 = np.asarray(x1), np.asarray(x2)
    if x2.ndim == 1:
        return np.broadcast_to(dy[..., None], x1.shape) * x2
    if x1.ndim == 1:
        return _unbroadcast_to((dy[..., None, :] * x2).sum(-1), x1.shape)
    return _unbroadcast_to(dy @ np.swapaxes(x2, -1, -2), x1.shape)


def _matmul_vjp_x2(dy, _y, x1, x2):
    x1, x2 = np.asarray(x1), np.asarray(x2)
    if x2.ndim == 1:
        return _unbroadcast_to(dy[..., None] * x1, x2.shape)
    if x1.ndim == 1:
        return np.broadcast_to(dy[..., None, :], x2.shape) * x1[:, None]
    return _unbroadcast_to(np.swapaxes(x1, -1, -2) @ dy, x2.shape)


_register_vjp(np.matmul, _matmul_vjp_x1, _matmul_vjp_x2)


# https://numpy.org/doc/stable/reference/routines.math.html#trigonometric-functions
_register_vjp(np.sin, lambda dy, _y, x: dy * np.cos(x))
_register_vjp(np.cos, lambda dy, _y, x: dy * -np.sin(x))
_register_vjp(np.tan, lambda dy, y, _x: dy * (1 + np.square(y)))
_register_vjp(np.arcsin, lambda dy, y, _x: dy / np.cos(y))
_register_vjp(np.arccos, lambda dy, y, _x: dy / -np.sin(y))
_register_vjp(np.arctan, lambda dy, y, _x: dy * (np.cos(y) ** 2))
_register_vjp(
    np.hypot,
    lambda dy, y, x1, _x2: _unbroadcast_to(dy * x1 / y, x1.shape),
    lambda dy, y, _x1, x2: _unbroadcast_to(dy * x2 / y, x2.shape),
)

# https://numpy.org/doc/stable/reference/routines.math.html#hyperbolic-functions
_register_vjp(np.sinh, lambda dy, _y, x: dy * np.cosh(x))
_register_vjp(np.cosh, lambda dy, _y, x: dy * np.sinh(x))
_register_vjp(np.tanh, lambda dy, y, _x: dy * (1 - np.square(y)))
_register_vjp(np.arcsinh, lambda dy, y, _x: dy / np.cosh(y))
_register_vjp(np.arccosh, lambda dy, y, _x: dy / np.sinh(y))
_register_vjp(np.arctanh, lambda dy, _y, x: dy / (1 - np.square(x)))

# https://numpy.org/doc/stable/reference/routines.math.html#sums-products-differences
_register_vjp(
    np.sum,
    lambda dy, _y, x, axis=None, keepdims=False, **kwargs: _expand_to(
        dy, x.shape, axis, keepdims),
)

# https://numpy.org/doc/stable/reference/routines.math.html#exponents-and-logarithms
_register_vjp(np.exp, lambda dy, y, _x: dy * y)
_register_vjp(np.expm1, lambda dy, y, _x: dy * (y + 1))
_register_vjp(np.exp2, lambda dy, y, _x: dy * y * np.log(2))
_register_vjp(np.log, lambda dy, _y, x: dy / x)
_register_vjp(np.log10, lambda dy, _y, x: dy / (x * np.log(10)))
_register_vjp(np.log2, lambda dy, _y, x: dy / (x * np.log(2)))
_register_vjp(np.log1p, lambda dy, _y, x: dy / (1 + x))
_register_vjp(
    np.logaddexp,
    lambda dy, y, x1, _x2: _unbroadcast_to(dy * np.exp(x1 - y), x1.shape),
    lambda dy, y, _x1, x2: _unbroadcast_to(dy * np.exp(x2 - y), x2.shape),
)
_register_vjp(
    np.logaddexp2,
    lambda dy, y, x1, _x2: _unbroadcast_to(dy * np.exp2(x1 - y), x1.shape),
    lambda dy, y, _x1, x2: _unbroadcast_to(dy * np.exp2(x2 - y), x2.shape),
)

# https://numpy.org/doc/stable/reference/routines.math.html#arithmetic-operations
_register_vjp(
    np.add,
    lambda dy, _y, x1, _x2: _unbroadcast_to(dy, x1.shape),
    lambda dy, _y, _x1, x2: _unbroadcast_to(dy, x2.shape),
)
_register_vjp(np.reciprocal, lambda dy, y, _x: dy * -(y ** 2))
_register_vjp(np.positive, lambda dy, _y, _x: dy)
_register_vjp(np.negative, lambda dy, _y, _x: -dy)
_register_vjp(
    np.multiply,
    lambda dy, _y, x1, x2: _unbroadcast_to(dy * x2, x1.shape),
    lambda dy, _y, x1, x2: _unbroadcast_to(dy * x1, x2.shape),
)
_register_vjp(
    np.divide,
    lambda dy, _y, x1, x2: _unbroadcast_to(dy / x2, x1.shape),
    lambda dy, _y, x1, x2: _unbroadcast_to(dy * x1 / -(x2 ** 2), x2.shape),
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
    lambda dy, _y, x1, _x2: _unbroadcast_to(dy, x1.shape),
    lambda dy, _y, _x1, x2: _unbroadcast_to(-dy, x2.shape),
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
    lambda dy, y, x1, _x2: _unbroadcast_to(
        np.where(np.asarray(x1) != np.asarray(y), 0, dy), x1.shape),
    lambda dy, y, _x1, x2: _unbroadcast_to(
        np.where(np.asarray(x2) != np.asarray(y), 0, dy), x2.shape),
)
_register_vjp(
    np.fmax,
    lambda dy, y, x1, _x2: _unbroadcast_to(
        np.where(np.asarray(x1) != np.asarray(y), 0, dy), x1.shape),
    lambda dy, y, _x1, x2: _unbroadcast_to(
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
    lambda dy, y, x1, _x2: _unbroadcast_to(
        np.where(np.asarray(x1) != np.asarray(y), 0, dy), x1.shape),
    lambda dy, y, _x1, x2: _unbroadcast_to(
        np.where(np.asarray(x2) != np.asarray(y), 0, dy), x2.shape),
)
_register_vjp(
    np.fmin,
    lambda dy, y, x1, _x2: _unbroadcast_to(
        np.where(np.asarray(x1) != np.asarray(y), 0, dy), x1.shape),
    lambda dy, y, _x1, x2: _unbroadcast_to(
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
_register_vjp(np.sqrt, lambda dy, y, _x: dy * 0.5 / y)
_register_vjp(np.cbrt, lambda dy, y, _x: dy / (3 * y ** 2))
_register_vjp(np.square, lambda dy, _y, x: dy * 2 * x)
_register_vjp(np.absolute, lambda dy, _y, x: dy * np.sign(x))
_register_vjp(np.fabs, lambda dy, _y, x: dy * np.sign(x))

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
    lambda dy, _y, loc, scale, size=None, **kwargs: (  # noqa: U100
        _unbroadcast_to(dy, loc.shape)),
    lambda dy, y, loc, scale, size=None: _unbroadcast_to(  # noqa: U100
        dy * (np.asarray(y) - np.asarray(loc)) / np.asarray(scale),
        scale.shape,
    ),
    module_name='numpy.random', func_name='normal',
)
_register_vjp(
    np.random.uniform,
    lambda dy, y, low, high, size=None: (  # noqa: U100
        u := (np.asarray(y) - np.asarray(low)) / (
            np.asarray(high) - np.asarray(low)),
        _unbroadcast_to(dy - dy * u, low.shape),
    )[1],
    lambda dy, y, low, high, size=None: (  # noqa: U100
        u := (np.asarray(y) - np.asarray(low)) / (
            np.asarray(high) - np.asarray(low)),
        _unbroadcast_to(dy * u, high.shape),
    )[1],
    module_name='numpy.random', func_name='uniform',
)


# https://numpy.org/doc/stable/reference/routines.statistics.html#averages-and-variances
_register_vjp(
    np.mean,
    lambda dy, _y, x, axis=None, keepdims=False: (
        _expand_to(dy, x.shape, axis, keepdims) * dy.size / x.size
    ),
)
_register_vjp(
    np.nanmean,
    lambda dy, _y, x, axis=None, *, keepdims=False: (
        nan_mask := np.isnan(x),
        np.where(
            nan_mask, 0,
            _expand_to(dy, x.shape, axis, keepdims) / np.sum(
                ~nan_mask, axis, keepdims=True),
        ),
    )[1],
)


__all__ = []
