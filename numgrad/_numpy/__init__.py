import numpy as np
from numpy.lib.stride_tricks import as_strided

from numgrad._numpy import (  # noqa: F401
    _linalg,
    _random,
    _sorting_searching_counting,
    _statistics,
)
from numgrad._utils._expand_to import _expand_to
from numgrad._utils._to_array import _to_array
from numgrad._utils._unbroadcast import _unbroadcast_to
from numgrad._variable import Variable
from numgrad._vjp import _register_vjp, custom_vjp


# https://numpy.org/doc/stable/reference/arrays.ndarray.html#special-methods
def _getitem_vjp(x, key):
    def vjp(dy, y):
        dx = np.zeros_like(x)
        dx[key] = dy
        return dx
    return vjp


Variable.__getitem__ = custom_vjp(_getitem_vjp)(
    lambda self, key: self[key])
Variable.__getitem__.__doc__ = np.ndarray.__getitem__.__doc__


# https://numpy.org/doc/stable/reference/routines.array-creation.html#numerical-ranges
_register_vjp(
    np.linspace,
    lambda start, stop, num=50, endpoint=True, axis=0: (
        lambda g, r: _unbroadcast_to(
            np.sum(
                g * np.linspace(
                    np.ones_like(start), np.zeros_like(stop), num,
                    endpoint=endpoint, axis=axis,
                ),
                axis=axis,
            ),
            start.shape,
        ),
        lambda g, r: _unbroadcast_to(
            np.sum(
                g * np.linspace(
                    np.zeros_like(start), np.ones_like(stop), num,
                    endpoint=endpoint, axis=axis,
                ),
                axis=axis,
            ),
            stop.shape,
        ),
    ),
)


# https://numpy.org/doc/stable/reference/routines.array-creation.html#building-matrices
_register_vjp(
    np.diag,
    lambda v, k=0: lambda g, r: (
        dv := np.diag(g, k=k),
        dv if v.shape == dv.shape else np.pad(
            dv, ((0, v.shape[0] - dv.shape[0]), (0, v.shape[1] - dv.shape[1])),
        ),
    )[1],
)
_register_vjp(
    np.diagflat,
    lambda v, k=0: lambda g, r: np.diag(g, k).reshape(*v.shape),
)
_register_vjp(np.tril, lambda v, k=0: lambda g, r: np.tril(g, k))
_register_vjp(np.triu, lambda v, k=0: lambda g, r: np.triu(g, k))
_register_vjp(
    np.vander,
    lambda x, N=None, increasing=False: lambda g, r: (
        n := len(x) if N is None else N,
        np.sum(g[:, 1:] * r[:, :-1] * range(1, n), -1) if increasing
        else np.sum(g[:, :-1] * r[:, 1:] * range(1, n)[::-1], -1),
    )[1],
)


# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#changing-array-shape
_register_vjp(
    np.reshape,
    lambda a, newshape, order=None: lambda g, r: g.reshape(
        *a.shape, order=order),
)
_register_vjp(
    np.ravel,
    lambda x, order=None: lambda g, r: g.reshape(*x.shape, order=order),
)
Variable.flatten = custom_vjp(
    lambda x: lambda g, r: g.reshape(x.shape))(lambda a: a.flatten())
Variable.flatten.__doc__ = np.ndarray.flatten.__doc__

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#transpose-like-operations
_register_vjp(
    np.moveaxis,
    lambda a, source, destination: lambda g, r: np.moveaxis(
        g, destination, source),
)
_register_vjp(
    np.swapaxes,
    lambda a, axis1, axis2: lambda g, r: np.swapaxes(g, axis1, axis2),
)
_register_vjp(
    np.transpose,
    lambda a, axes=None: lambda g, r: (
        np.transpose(g) if axes is None
        else np.transpose(g, np.argsort(axes))
    ),
)


# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#changing-number-of-dimensions
def _get_atleast_nd_vjps(*arys):
    return tuple(
        (
            lambda g, r, index=i: np.reshape(
                g[index] if isinstance(g, tuple) else g,
                arys[index].shape,
            )
        )
        for i, _ in enumerate(arys)
    )


_register_vjp(np.atleast_1d, _get_atleast_nd_vjps)
_register_vjp(np.atleast_2d, _get_atleast_nd_vjps)
_register_vjp(np.atleast_3d, _get_atleast_nd_vjps)
_register_vjp(
    np.broadcast_to,
    lambda array, shape: lambda g, r: _unbroadcast_to(g, array.shape),
)
_register_vjp(
    np.broadcast_arrays,
    lambda *args: tuple(
        (lambda g, r, index=i: _unbroadcast_to(g[index], args[index].shape))
        for i, _ in enumerate(args)
    ),
)
_register_vjp(np.expand_dims, lambda a, axis: lambda g, r: np.squeeze(g, axis))
_register_vjp(np.squeeze, lambda a, axis=None: lambda g, r: (
    np.expand_dims(g, [ax for ax, len_ in enumerate(a.shape) if len_ == 1])
    if axis is None else np.expand_dims(g, axis)))

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#changing-kind-of-array
_register_vjp(np.asarray, lambda a: lambda g, r: +g)
_register_vjp(np.asanyarray, lambda a: lambda g, r: +g)

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#joining-arrays
_register_vjp(
    np.concatenate,
    lambda arrays, axis=0: lambda g, r: (
        indices := np.cumsum([_to_array(a).shape[axis] for a in arrays]),
        np.split(g, indices[:-1], axis=axis),
    )[-1],
)
_register_vjp(
    np.stack,
    lambda arrays, axis=0: lambda g, r: [
        np.squeeze(g_, axis) for g_ in np.split(g, g.shape[axis], axis)
    ],
)
_register_vjp(
    np.vstack,
    lambda tup: lambda g, r: (
        indices := np.cumsum([a.shape[0] for a in np.atleast_2d(*tup)]),
        [
            _unbroadcast_to(g_, _to_array(a).shape) for g_, a in
            zip(np.split(g, indices[:-1], axis=0), tup)
        ],
    )[-1],
)
_register_vjp(
    np.hstack,
    lambda tup: lambda g, r: (
        axis := 0 if all(_to_array(a).ndim == 1 for a in tup) else 1,
        indices := np.cumsum([_to_array(a).shape[axis] for a in tup]),
        np.split(g, indices[:-1], axis=axis),
    )[-1],
)
_register_vjp(
    np.dstack,
    lambda tup: lambda g, r: (
        arys := [_to_array(a) for a in tup],
        arys_expanded := [
            a if a.ndim >= 3 else np.expand_dims(a, -1) if a.ndim == 2
            else np.expand_dims(a, (0, -1)) for a in arys
        ],
        indices := np.cumsum([a.shape[2] for a in arys_expanded]),
        [
            g_ if g_.ndim == a.ndim else np.squeeze(
                g_, {1: (0, -1), 2: -1}[a.ndim])
            for g_, a in zip(np.split(g, indices[:-1], axis=2), arys)
        ],
    )[-1],
)
_register_vjp(
    np.column_stack,
    lambda tup: lambda g, r: (
        arys := [_to_array(a) for a in tup],
        arys_expanded := [a if a.ndim > 1 else a[..., None] for a in arys],
        indices := np.cumsum([a.shape[1] for a in arys_expanded]),
        [
            g_ if a.ndim == 2 else g_[:, 0] for g_, a in
            zip(np.split(g, indices[:-1], axis=1), arys)
        ],
    )[-1],
)
_register_vjp(
    np.row_stack,
    lambda tup: lambda g, r: (
        indices := np.cumsum([a.shape[0] for a in np.atleast_2d(*tup)]),
        [
            _unbroadcast_to(g_, _to_array(a).shape) for g_, a in
            zip(np.split(g, indices[:-1], axis=0), tup)
        ],
    )[-1],
)

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#splitting-arrays
_register_vjp(
    np.split,
    lambda ary, indices_or_sections, axis=0: lambda g, r: np.concatenate(
        g, axis=axis),
)
_register_vjp(
    np.array_split,
    lambda ary, indices_or_sections, axis=0: lambda g, r: np.concatenate(
        g, axis=axis),
)
_register_vjp(
    np.dsplit,
    lambda ary, indices_or_sections: lambda g, r: np.concatenate(g, axis=2),
)
_register_vjp(
    np.hsplit,
    lambda ary, indices_or_sections: lambda g, r: np.concatenate(g, axis=1),
)
_register_vjp(
    np.vsplit,
    lambda ary, indices_or_sections: lambda g, r: np.concatenate(g, axis=0),
)

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#rearranging-elements
_register_vjp(np.flip, lambda m, axis=None: lambda g, r: np.flip(g, axis))
_register_vjp(np.fliplr, lambda m: lambda g, r: np.fliplr(g))
_register_vjp(np.flipud, lambda m: lambda g, r: np.flipud(g))
_register_vjp(np.roll, lambda a, shift, axis=None: lambda g, r: np.roll(
    g, -shift if isinstance(shift, int) else [-s for s in shift], axis))
_register_vjp(
    np.rot90, lambda m, k=1, axes=(0, 1): lambda g, r: np.rot90(g, -k, axes))


# # https://numpy.org/doc/stable/reference/routines.linalg.html
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

_dot_0d_0d_vjp_a = _matmul_0d_0d_vjp_x1
_dot_0d_0d_vjp_b = _matmul_0d_0d_vjp_x2
_dot_1d_1d_vjp_a = _matmul_1d_1d_vjp_x1
_dot_1d_1d_vjp_b = _matmul_1d_1d_vjp_x2
_dot_1d_nd_vjp_a = _matmul_1d_nd_vjp_x1
_dot_1d_nd_vjp_b = _matmul_1d_nd_vjp_x2
_dot_nd_1d_vjp_a = _matmul_nd_1d_vjp_x1
_dot_nd_1d_vjp_b = _matmul_nd_1d_vjp_x2
_dot_nd_nd_vjp_a = lambda dy, x1, x2: (
    dy[..., None] * np.moveaxis(x2, -2, -1)).sum(
        tuple(-i - 2 for i in range(x2.ndim - 1)))
_dot_nd_nd_vjp_b = lambda dy, x1, x2: np.swapaxes(
    np.tensordot(
        dy, x1,
        [range(-x1.ndim - x2.ndim + 2, -x2.ndim + 1), range(x1.ndim - 1)],
    ), -1, -2,
)

_inner_0d_0d_vjp_a = _matmul_0d_0d_vjp_x1
_inner_0d_0d_vjp_b = _matmul_0d_0d_vjp_x2
_inner_1d_1d_vjp_a = _matmul_1d_1d_vjp_x1
_inner_1d_1d_vjp_b = _matmul_1d_1d_vjp_x2
_inner_1d_nd_vjp_a = lambda dy, x1, x2: _unbroadcast_to(
    dy[..., None] * x2, x1.shape)
_inner_1d_nd_vjp_b = lambda dy, x1, x2: dy[..., None] * x1
_inner_nd_1d_vjp_a = _matmul_nd_1d_vjp_x1
_inner_nd_1d_vjp_b = _matmul_nd_1d_vjp_x2
_inner_nd_nd_vjp_a = lambda dy, x1, x2: (
    dy[..., None] * x2).sum(tuple(-i - 2 for i in range(x2.ndim - 1)))
_inner_nd_nd_vjp_b = lambda dy, x1, x2: np.tensordot(
    dy, x1, [range(-x1.ndim - x2.ndim + 2, -x2.ndim + 1), range(x1.ndim - 1)])

_register_vjp(
    np.dot,
    lambda a, b: (
        a := _to_array(a),
        b := _to_array(b),
        d1 := 'n' if (a.ndim > 1) else a.ndim,
        d2 := 'n' if (b.ndim > 1) else b.ndim,
        (
            lambda g, r: eval(f'_dot_{d1}d_{d2}d_vjp_a')(g, a, b),
            lambda g, r: eval(f'_dot_{d1}d_{d2}d_vjp_b')(g, a, b),
        ),
    )[-1],
)
_register_vjp(
    np.vdot,
    lambda a, b: (
        lambda g, r: (g * b).reshape(a.shape),
        lambda g, r: (g * a).reshape(b.shape),
    ),
)
_register_vjp(
    np.inner,
    lambda a, b: (
        a := _to_array(a),
        b := _to_array(b),
        d1 := 'n' if (a.ndim > 1) else a.ndim,
        d2 := 'n' if (b.ndim > 1) else b.ndim,
        (
            lambda g, r: eval(f'_inner_{d1}d_{d2}d_vjp_a')(g, a, b),
            lambda g, r: eval(f'_inner_{d1}d_{d2}d_vjp_b')(g, a, b),
        ),
    )[-1],
)
_register_vjp(
    np.outer,
    lambda a, b: (
        lambda g, r: np.sum(g * np.ravel(b), -1).reshape(a.shape),
        lambda g, r: np.sum(g * np.ravel(a)[None, ...], -1).reshape(b.shape),
    ),
)
_register_vjp(
    np.matmul,
    lambda x1, x2: (
        x1 := _to_array(x1),
        x2 := _to_array(x2),
        d1 := 'n' if (x1.ndim > 1) else x1.ndim,
        d2 := 'n' if (x2.ndim > 1) else x2.ndim,
        (
            lambda g, r: eval(f'_matmul_{d1}d_{d2}d_vjp_x1')(g, x1, x2),
            lambda g, r: eval(f'_matmul_{d1}d_{d2}d_vjp_x2')(g, x1, x2),
        ),
    )[-1],
)


# https://numpy.org/doc/stable/reference/routines.linalg.html#norms-and-other-numbers
_register_vjp(
    np.trace,
    lambda a, offset=0, axis1=0, axis2=1: lambda g, r: np.multiply(
        np.expand_dims(
            np.eye(a.shape[axis1], a.shape[axis2], k=offset),
            [i for i in range(a.ndim) if i not in (axis1, axis2)]),
        np.expand_dims(g, (axis1, axis2)),
    ),
)

# https://numpy.org/doc/stable/reference/routines.math.html#trigonometric-functions
_register_vjp(np.sin, lambda x: lambda g, r: g * np.cos(x))
_register_vjp(np.cos, lambda x: lambda g, r: g * -np.sin(x))
_register_vjp(np.tan, lambda x: lambda g, r: g * (1 + np.square(r)))
_register_vjp(np.arcsin, lambda x: lambda g, r: g / np.cos(r))
_register_vjp(np.arccos, lambda x: lambda g, r: g / -np.sin(r))
_register_vjp(np.arctan, lambda x: lambda g, r: g * (np.cos(r) ** 2))
_register_vjp(
    np.hypot,
    lambda x1, x2: (
        lambda g, r: _unbroadcast_to(g * x1 / r, x1.shape),
        lambda g, r: _unbroadcast_to(g * x2 / r, x2.shape),
    ),
)

# https://numpy.org/doc/stable/reference/routines.math.html#hyperbolic-functions
_register_vjp(np.sinh, lambda x: lambda g, r: g * np.cosh(x))
_register_vjp(np.cosh, lambda x: lambda g, r: g * np.sinh(x))
_register_vjp(np.tanh, lambda x: lambda g, r: g * (1 - np.square(r)))
_register_vjp(np.arcsinh, lambda x: lambda g, r: g / np.cosh(r))
_register_vjp(np.arccosh, lambda x: lambda g, r: g / np.sinh(r))
_register_vjp(np.arctanh, lambda x: lambda g, r: g / (1 - np.square(x)))

# https://numpy.org/doc/stable/reference/routines.math.html#sums-products-differences
_register_vjp(
    np.prod,
    lambda a, axis=None, *, keepdims=False: lambda g, r: (
        _expand_to(g, a.ndim, axis, keepdims)
        * (r if keepdims else np.prod(a, axis, keepdims=True)) / a
    ),
)
_register_vjp(
    np.sum,
    lambda a, axis=None, *, keepdims=False: lambda g, r: _expand_to(
        g, a.shape, axis, keepdims),
)
_register_vjp(
    np.nanprod,
    lambda a, axis=None, *, keepdims=False: lambda g, r: (
        _expand_to(g, a.ndim, axis, keepdims)
        * (r if keepdims else np.nanprod(a, axis, keepdims=True)) / a
    ),
)
_register_vjp(
    np.nansum,
    lambda a, axis=None, *, keepdims=False: lambda g, r: _expand_to(
        g, a.shape, axis, keepdims),
)
_register_vjp(
    np.cumprod,
    lambda a, axis=None: lambda g, r: (
        g if a.ndim == 0 else np.flip(np.cumsum(np.flip((g * r).reshape(
            *a.shape), axis), axis).reshape(*a.shape), axis) / a
    ),
)
_register_vjp(
    np.cumsum,
    lambda a, axis=None: lambda g, r: (
        g if a.ndim == 0 else np.flip(np.cumsum(np.flip(
            g.reshape(*a.shape), axis), axis).reshape(*a.shape), axis)
    ),
)
_register_vjp(
    np.nancumprod,
    lambda a, axis=None: lambda g, r: (
        g if a.ndim == 0 else np.flip(np.cumsum(np.flip((g * r).reshape(
            *a.shape), axis), axis).reshape(*a.shape), axis) / a
    ),
)
_register_vjp(
    np.nancumsum,
    lambda a, axis=None: lambda g, r: (
        g if a.ndim == 0 else np.flip(np.cumsum(np.flip(
            g.reshape(*a.shape), axis), axis).reshape(*a.shape), axis)
    ),
)

# https://numpy.org/doc/stable/reference/routines.math.html#exponents-and-logarithms
_register_vjp(np.exp, lambda x: lambda g, r: g * r)
_register_vjp(np.expm1, lambda x: lambda g, r: g * (r + 1))
_register_vjp(np.exp2, lambda x: lambda g, r: g * r * np.log(2))
_register_vjp(np.log, lambda x: lambda g, r: g / x)
_register_vjp(np.log10, lambda x: lambda g, r: g / (x * np.log(10)))
_register_vjp(np.log2, lambda x: lambda g, r: g / (x * np.log(2)))
_register_vjp(np.log1p, lambda x: lambda g, r: g / (1 + x))
_register_vjp(
    np.logaddexp,
    lambda x1, x2: (
        lambda g, r: _unbroadcast_to(g * np.exp(x1 - r), x1.shape),
        lambda g, r: _unbroadcast_to(g * np.exp(x2 - r), x2.shape),
    ),
)
_register_vjp(
    np.logaddexp2,
    lambda x1, x2: (
        lambda g, r: _unbroadcast_to(g * np.exp2(x1 - r), x1.shape),
        lambda g, r: _unbroadcast_to(g * np.exp2(x2 - r), x2.shape),
    ),
)

# https://numpy.org/doc/stable/reference/routines.math.html#arithmetic-operations
_register_vjp(
    np.add,
    lambda x1, x2: (
        lambda dy, y: _unbroadcast_to(dy, x1.shape),
        lambda dy, y: _unbroadcast_to(dy, x2.shape),
    ),
)
_register_vjp(np.reciprocal, lambda x: lambda dy, y: dy * -(y ** 2))
_register_vjp(np.positive, lambda x: lambda dy, y: +dy)
_register_vjp(np.negative, lambda x: lambda dy, y: -dy)
_register_vjp(
    np.multiply,
    lambda x1, x2: (
        lambda dy, y: _unbroadcast_to(dy * x2, x1.shape),
        lambda dy, y: _unbroadcast_to(dy * x1, x2.shape),
    ),
)
_register_vjp(
    np.divide,
    lambda x1, x2: (
        lambda dy, y: _unbroadcast_to(dy / x2, x1.shape),
        lambda dy, y: _unbroadcast_to(dy * x1 / -(x2 ** 2), x2.shape),
    ),
)
_register_vjp(
    np.power,
    lambda x1, x2: (
        lambda dy, y: _unbroadcast_to(dy * x2 * (x1 ** (x2 - 1)), x1.shape),
        lambda dy, y: None if np.any(x1 < 0) else _unbroadcast_to(
            dy * y * np.log(x1), x2.shape),
    ),
)
_register_vjp(
    np.subtract,
    lambda x1, x2: (
        lambda dy, y: _unbroadcast_to(dy, x1.shape),
        lambda dy, y: _unbroadcast_to(-dy, x2.shape),
    ),
)
_register_vjp(
    np.float_power,
    lambda x1, x2: (
        lambda dy, y: _unbroadcast_to(dy * x2 * (x1 ** (x2 - 1)), x1.shape),
        lambda dy, y: None if np.any(x1 < 0) else _unbroadcast_to(
            dy * y * np.log(x1), x2.shape),
    ),
)

# https://numpy.org/doc/stable/reference/routines.math.html#extrema-finding
_register_vjp(
    np.maximum,
    lambda x1, x2: (
        lambda g, r: _unbroadcast_to(np.where(x1 != r, 0, g), x1.shape),
        lambda g, r: _unbroadcast_to(np.where(x2 != r, 0, g), x2.shape),
    ),
)
_register_vjp(
    np.fmax,
    lambda x1, x2: (
        lambda g, r: _unbroadcast_to(np.where(x1 != r, 0, g), x1.shape),
        lambda g, r: _unbroadcast_to(np.where(x2 != r, 0, g), x2.shape),
    ),
)
_register_vjp(
    np.amax,
    lambda a, axis=None, *, keepdims=False: lambda g, r: np.where(
        a == _expand_to(r, a.ndim, axis, keepdims),
        _expand_to(g, a.shape, axis, keepdims), 0,
    ),
)
_register_vjp(
    np.nanmax,
    lambda a, axis=None, *, keepdims=False: lambda g, r: np.where(
        a == _expand_to(r, a.ndim, axis, keepdims),
        _expand_to(g, a.shape, axis, keepdims), 0,
    ),
)
_register_vjp(
    np.minimum,
    lambda x1, x2: (
        lambda g, r: _unbroadcast_to(np.where(x1 != r, 0, g), x1.shape),
        lambda g, r: _unbroadcast_to(np.where(x2 != r, 0, g), x2.shape),
    ),
)
_register_vjp(
    np.fmin,
    lambda x1, x2: (
        lambda g, r: _unbroadcast_to(np.where(x1 != r, 0, g), x1.shape),
        lambda g, r: _unbroadcast_to(np.where(x2 != r, 0, g), x2.shape),
    ),
)
_register_vjp(
    np.amin,
    lambda a, axis=None, *, keepdims=False: lambda g, r: np.where(
        a == _expand_to(r, a.ndim, axis, keepdims),
        _expand_to(g, a.shape, axis, keepdims), 0,
    ),
)
_register_vjp(
    np.nanmin,
    lambda a, axis=None, *, keepdims=False: lambda g, r: np.where(
        a == _expand_to(r, a.ndim, axis, keepdims),
        _expand_to(g, a.shape, axis, keepdims), 0,
    ),
)


# https://numpy.org/doc/stable/reference/routines.math.html#miscellaneous
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


# https://numpy.org/doc/stable/reference/routines.statistics.html#correlating
def _correlate_vjp_x1(g, a, v, mode):
    w = {'full': v.size - 1, 'valid': 0, 'same': v.size // 2}[mode]
    a_pad = np.pad(a, w)
    s = a_pad.strides[0]
    a_col = as_strided(a_pad, [g.size, v.size], [s, s])
    da_col = _matmul_nd_1d_vjp_x1(g, a_col, v)
    da_col_pad = np.pad(da_col, ((0,), (g.size,)))
    da_col_strided = as_strided(
        da_col_pad.ravel()[g.size + w:],
        [g.size, a.size], [da_col_pad.strides[0] - s, s])
    da = da_col_strided.sum(0)
    return da


def _correlate_vjp_x2(g, a, v, mode):
    w = {'full': v.size - 1, 'valid': 0, 'same': v.size // 2}[mode]
    a_pad = np.pad(a, w)
    s = a_pad.strides[0]
    a_col = as_strided(a_pad, [g.size, v.size], [s, s])
    dv = _matmul_nd_1d_vjp_x2(g, a_col, v)
    return dv


_register_vjp(
    np.correlate,
    lambda a, v, mode='valid': (
        lambda g, r: _correlate_vjp_x1(g, np.asarray(a), np.asarray(v), mode),
        lambda g, r: _correlate_vjp_x2(g, np.asarray(a), np.asarray(v), mode),
    ),
)


__all__ = []
