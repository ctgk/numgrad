# https://numpy.org/doc/stable/reference/routines.linalg.html

import numpy as np

from numgrad._vjp import _bind_vjp


def _t(x):
    return np.swapaxes(x, -1, -2)


def _dot_1d_1d_vjp_a(g, r, a, b):
    return g * b


def _dot_1d_1d_vjp_b(g, r, a, b):
    return g * a


def _dot_1d_nd_vjp_a(g, r, a, b):
    return (g[..., None, :] * b).sum(-1)


def _dot_1d_nd_vjp_b(g, r, a, b):
    return np.broadcast_to(g[..., None, :], b.shape) * a[:, None]


def _dot_nd_1d_vjp_a(g, r, a, b):
    return np.broadcast_to(g[..., None] * b, a.shape)


def _dot_nd_1d_vjp_b(g, r, a, b):
    return g[..., None] * a


def _dot_nd_nd_vjp_a(g, r, a, b):
    return (g[..., None] * np.moveaxis(b, -2, -1)).sum(
        tuple(-i - 2 for i in range(b.ndim - 1)))


def _dot_nd_nd_vjp_b(g, r, a, b):
    return np.swapaxes(
        np.tensordot(
            g, a,
            [range(-a.ndim - b.ndim + 2, -b.ndim + 1), range(a.ndim - 1)],
        ),
        -1, -2,
    )


def _dot_vjp_a(g, r, a, b):
    adim = 'n' if (a.ndim > 1) else '1'
    bdim = 'n' if (b.ndim > 1) else '1'
    return eval(f'_dot_{adim}d_{bdim}d_vjp_a')(g, r, a, b)


def _dot_vjp_b(g, r, a, b):
    adim = 'n' if (a.ndim > 1) else '1'
    bdim = 'n' if (b.ndim > 1) else '1'
    return eval(f'_dot_{adim}d_{bdim}d_vjp_b')(g, r, a, b)


def _inner_1d_nd_vjp_a(g, r, a, b):
    return g[..., None] * b


def _inner_1d_nd_vjp_b(g, r, a, b):
    return g[..., None] * a


def _inner_nd_nd_vjp_a(g, r, a, b):
    return (g[..., None] * b).sum(tuple(-i - 2 for i in range(b.ndim - 1)))


def _inner_nd_nd_vjp_b(g, r, a, b):
    return np.tensordot(
        g, a, [range(-a.ndim - b.ndim + 2, -b.ndim + 1), range(a.ndim - 1)])


def _inner_vjp_a(g, r, a, b):
    _inner_1d_1d_vjp_a = _dot_1d_1d_vjp_a  # noqa: F841
    _inner_nd_1d_vjp_a = _dot_nd_1d_vjp_a  # noqa: F841
    adim = 'n' if (a.ndim > 1) else '1'
    bdim = 'n' if (b.ndim > 1) else '1'
    return eval(f'_inner_{adim}d_{bdim}d_vjp_a')(g, r, a, b)


def _inner_vjp_b(g, r, a, b):
    _inner_1d_1d_vjp_b = _dot_1d_1d_vjp_b  # noqa: F841
    _inner_nd_1d_vjp_b = _dot_nd_1d_vjp_b  # noqa: F841
    adim = 'n' if (a.ndim > 1) else '1'
    bdim = 'n' if (b.ndim > 1) else '1'
    return eval(f'_inner_{adim}d_{bdim}d_vjp_b')(g, r, a, b)


def _matmul_nd_nd_vjp_a(g, r, a, b):
    return g @ _t(b)


def _matmul_nd_nd_vjp_b(g, r, a, b):
    return _t(a) @ g


def _matmul_vjp_a(g, r, a, b):
    _matmul_1d_1d_vjp_a = _dot_1d_1d_vjp_a  # noqa: F841
    _matmul_1d_nd_vjp_a = _dot_1d_nd_vjp_a  # noqa: F841
    _matmul_nd_1d_vjp_a = _dot_nd_1d_vjp_a  # noqa: F841
    adim = 'n' if (a.ndim > 1) else '1'
    bdim = 'n' if (b.ndim > 1) else '1'
    return eval(f'_matmul_{adim}d_{bdim}d_vjp_a')(g, r, a, b)


def _matmul_vjp_b(g, r, a, b):
    _matmul_1d_1d_vjp_b = _dot_1d_1d_vjp_b  # noqa: F841
    _matmul_1d_nd_vjp_b = _dot_1d_nd_vjp_b  # noqa: F841
    _matmul_nd_1d_vjp_b = _dot_nd_1d_vjp_b  # noqa: F841
    adim = 'n' if (a.ndim > 1) else '1'
    bdim = 'n' if (b.ndim > 1) else '1'
    return eval(f'_matmul_{adim}d_{bdim}d_vjp_b')(g, r, a, b)


# https://numpy.org/doc/stable/reference/routines.linalg.html#matrix-and-vector-products
_bind_vjp(np.dot, _dot_vjp_a, _dot_vjp_b)
_bind_vjp(
    np.vdot,
    lambda g, r, a, b: (g * b).reshape(a.shape),
    lambda g, r, a, b: (g * a).reshape(b.shape),
)
_bind_vjp(np.inner, _inner_vjp_a, _inner_vjp_b)
_bind_vjp(
    np.outer,
    lambda g, r, a, b: np.sum(g * np.ravel(b), -1).reshape(a.shape),
    lambda g, r, a, b: np.sum(g * np.ravel(a)[None, ...], -1).reshape(b.shape),
)
_bind_vjp(np.matmul, _matmul_vjp_a, _matmul_vjp_b)

# https://numpy.org/doc/stable/reference/routines.linalg.html#decompositions
_bind_vjp(
    np.linalg.cholesky,
    lambda g, r, a: (
        g_lower := np.tril(g),
        rgt := _t(r) @ g_lower,
        phi := 0.5 * (np.tril(rgt) + np.tril(rgt, -1)),
        s := np.linalg.solve(_t(r), phi @ np.linalg.inv(r)),
        0.5 * (s + _t(s)),
    )[-1],
)

# https://numpy.org/doc/stable/reference/routines.linalg.html#norms-and-other-numbers
_bind_vjp(
    np.linalg.det,
    lambda g, r, a: (g * r)[..., None, None] * np.linalg.inv(_t(a)),
)
_bind_vjp(
    np.linalg.slogdet,
    lambda g, r, a: g[1][..., None, None] * np.linalg.inv(_t(a)),
)
_bind_vjp(
    np.trace,
    lambda g, r, a, offset=0, axis1=0, axis2=1: np.multiply(
        np.expand_dims(
            np.eye(a.shape[axis1], a.shape[axis2], k=offset),
            [i for i in range(a.ndim) if i not in (axis1, axis2)]),
        np.expand_dims(g, (axis1, axis2)),
    ),
)

# https://numpy.org/doc/stable/reference/routines.linalg.html#solving-equations-and-inverting-matrices
_bind_vjp(
    np.linalg.solve,
    lambda g, r, a, b: (
        f := lambda x: x if a.ndim == b.ndim else x[..., None],
        -np.linalg.solve(_t(a), f(g)) @ _t(f(r)),
    )[-1],
    lambda g, r, a, b: (
        f := lambda x: x if a.ndim == b.ndim else x[..., None],
        np.squeeze(
            np.linalg.solve(_t(a), f(g)),
            tuple() if a.ndim == b.ndim else -1,
        ),
    )[-1],
)
_bind_vjp(
    np.linalg.inv,
    lambda g, r, a: -_t(np.linalg.solve(a, _t(np.linalg.solve(_t(a), g)))),
)
