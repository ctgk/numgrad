# https://numpy.org/doc/stable/reference/routines.array-manipulation.html

import numpy as np

from numgrad._utils._to_array import _to_array
from numgrad._utils._unbroadcast import _unbroadcast_to
from numgrad._variable import Variable
from numgrad._vjp import _bind_vjp, _VJPIterator, custom_vjp


_atleast_nd_vjp_iterator = _VJPIterator(
    lambda g, r, *arys, _nth: np.reshape(
        g[_nth] if isinstance(g, tuple) else g,
        arys[_nth].shape,
    ),
)


# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#changing-array-shape
_bind_vjp(
    np.reshape,
    lambda g, r, a, newshape, order=None: g.reshape(*a.shape, order=order),
)
_bind_vjp(
    np.ravel,
    lambda g, r, x, order=None: g.reshape(*x.shape, order=order),
)
Variable.flatten = custom_vjp(
    lambda g, r, x: g.reshape(x.shape))(lambda a: a.flatten())
Variable.flatten.__doc__ = np.ndarray.flatten.__doc__


# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#transpose-like-operations
_bind_vjp(
    np.moveaxis,
    lambda g, r, a, source, destination: np.moveaxis(g, destination, source),
)
_bind_vjp(
    np.swapaxes,
    lambda g, r, a, axis1, axis2: np.swapaxes(g, axis1, axis2),
)
_bind_vjp(
    np.transpose,
    lambda g, r, a, axes=None: (
        np.transpose(g) if axes is None
        else np.transpose(g, np.argsort(axes))
    ),
)

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#changing-number-of-dimensions
_bind_vjp(np.atleast_1d, _atleast_nd_vjp_iterator)
_bind_vjp(np.atleast_2d, _atleast_nd_vjp_iterator)
_bind_vjp(np.atleast_3d, _atleast_nd_vjp_iterator)
_bind_vjp(
    np.broadcast_to,
    lambda g, r, array, shape: _unbroadcast_to(g, array.shape),
)
_bind_vjp(
    np.broadcast_arrays,
    _VJPIterator(
        lambda g, r, *args, _nth: _unbroadcast_to(g[_nth], args[_nth].shape),
    ),
)
_bind_vjp(np.expand_dims, lambda g, r, a, axis: np.squeeze(g, axis))
_bind_vjp(np.squeeze, lambda g, r, a, axis=None: (
    np.expand_dims(g, [ax for ax, len_ in enumerate(a.shape) if len_ == 1])
    if axis is None else np.expand_dims(g, axis)))

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#changing-kind-of-array
_bind_vjp(np.asarray, lambda g, r, _: +g)
_bind_vjp(np.asanyarray, lambda g, r, _: +g)

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#joining-arrays
_bind_vjp(
    np.concatenate,
    lambda g, r, arrays, axis=0: (
        indices := np.cumsum([_to_array(a).shape[axis] for a in arrays]),
        np.split(g, indices[:-1], axis=axis),
    )[-1],
)
_bind_vjp(
    np.stack,
    lambda g, r, arrays, axis=0: [
        np.squeeze(g_, axis) for g_ in np.split(g, g.shape[axis], axis)
    ],
)
_bind_vjp(
    np.vstack,
    lambda g, r, tup: (
        indices := np.cumsum([a.shape[0] for a in np.atleast_2d(*tup)]),
        [
            _unbroadcast_to(g_, _to_array(a).shape) for g_, a in
            zip(np.split(g, indices[:-1], axis=0), tup)
        ],
    )[-1],
)
_bind_vjp(
    np.hstack,
    lambda g, r, tup: (
        axis := 0 if all(_to_array(a).ndim == 1 for a in tup) else 1,
        indices := np.cumsum([_to_array(a).shape[axis] for a in tup]),
        np.split(g, indices[:-1], axis=axis),
    )[-1],
)
_bind_vjp(
    np.dstack,
    lambda g, r, tup: (
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
_bind_vjp(
    np.column_stack,
    lambda g, r, tup: (
        arys := [_to_array(a) for a in tup],
        arys_expanded := [a if a.ndim > 1 else a[..., None] for a in arys],
        indices := np.cumsum([a.shape[1] for a in arys_expanded]),
        [
            g_ if a.ndim == 2 else g_[:, 0] for g_, a in
            zip(np.split(g, indices[:-1], axis=1), arys)
        ],
    )[-1],
)
# _bind_vjp(np.row_stack, ...) np.row_stack is np.vstack

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#splitting-arrays
_bind_vjp(np.split, lambda g, r, ary, _, axis=0: np.concatenate(g, axis=axis))
_bind_vjp(np.array_split, lambda g, r, ary, _, axis=0: np.concatenate(g, axis))
_bind_vjp(np.dsplit, lambda g, r, ary, _: np.concatenate(g, axis=2))
_bind_vjp(np.hsplit, lambda g, r, ary, _: np.concatenate(g, axis=1))
_bind_vjp(np.vsplit, lambda g, r, ary, _: np.concatenate(g, axis=0))

# https://numpy.org/doc/stable/reference/routines.array-manipulation.html#rearranging-elements
_bind_vjp(np.flip, lambda g, r, m, axis=None: np.flip(g, axis))
_bind_vjp(np.fliplr, lambda g, r, m: np.fliplr(g))
_bind_vjp(np.flipud, lambda g, r, m: np.flipud(g))
_bind_vjp(np.roll, lambda g, r, a, shift, axis=None: np.roll(
    g, -shift if isinstance(shift, int) else [-s for s in shift], axis))
_bind_vjp(np.rot90, lambda g, r, m, k=1, axes=(0, 1): np.rot90(g, -k, axes))
