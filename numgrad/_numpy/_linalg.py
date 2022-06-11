import numpy as np

from numgrad._utils._to_array import _to_array
from numgrad._vjp import _register_vjp


# https://numpy.org/doc/stable/reference/routines.linalg.html#decompositions
def _t(x):
    return np.swapaxes(x, -1, -2)


_register_vjp(
    np.linalg.cholesky,
    lambda a: lambda g, r: (
        g_lower := np.tril(g),
        rgt := _t(r) @ g_lower,
        phi := 0.5 * (np.tril(rgt) + np.tril(rgt, -1)),
        s := np.linalg.solve(_t(r), phi @ np.linalg.inv(r)),
        0.5 * (s + _t(s)),
    )[-1],
)


# https://numpy.org/doc/stable/reference/routines.linalg.html#norms-and-other-numbers
_register_vjp(
    np.linalg.det,
    lambda a: lambda g, r: (g * r)[..., None, None] * np.linalg.inv(_t(a)),
)
_register_vjp(
    np.linalg.slogdet,
    lambda a: lambda g, r: g[1][..., None, None] * np.linalg.inv(_t(a)),
)


# https://numpy.org/doc/stable/reference/routines.linalg.html#solving-equations-and-inverting-matrices
_register_vjp(
    np.linalg.solve,
    lambda a, b: (
        a := _to_array(a),
        b := _to_array(b),
        f := lambda x: x if a.ndim == b.ndim else x[..., None],
        (
            lambda g, r: -np.linalg.solve(_t(a), f(g)) @ _t(f(r)),
            lambda g, r: np.squeeze(
                np.linalg.solve(_t(a), f(g)),
                tuple() if a.ndim == b.ndim else -1,
            ),
        ),
    )[-1],
)
_register_vjp(
    np.linalg.inv,
    lambda a: lambda g, r: -_t(
        np.linalg.solve(a, _t(np.linalg.solve(_t(a), g)))),
)
