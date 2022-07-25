import numpy as np
import scipy.special as sp

from numgrad._utils._expand_to import _expand_to_if_array
from numgrad._vjp import _bind_vjp

# https://docs.scipy.org/doc/scipy/reference/special.html#raw-statistical-functions
_bind_vjp(sp.expit, lambda r, _: r * (1 - r))
_bind_vjp(sp.log_expit, lambda r, _: 1 - np.exp(r))

# https://docs.scipy.org/doc/scipy/reference/special.html#gamma-and-related-functions
_bind_vjp(sp.gamma, lambda r, x: r * sp.digamma(x))

# https://docs.scipy.org/doc/scipy/reference/special.html#other-special-functions
_bind_vjp(
    sp.softmax,
    lambda g, r, _, axis=None: (
        rg := r * g,
        rg - r * rg.sum(axis=axis, keepdims=True),
    )[1],
)
_bind_vjp(
    sp.log_softmax,
    lambda g, r, _, axis=None: g - np.exp(r) * g.sum(axis, keepdims=True),
)

# https://docs.scipy.org/doc/scipy/reference/special.html#convenience-functions
_bind_vjp(
    sp.logsumexp,
    lambda g, r, a, axis=None, *, keepdims=False, return_sign=False: (
        g := g[0] if return_sign else g,
        r := r[0] if return_sign else r,
        _expand_to_if_array(g, a.ndim, axis, keepdims) * np.exp(
            a - _expand_to_if_array(r, a.ndim, axis, keepdims)),
    )[-1],
)
