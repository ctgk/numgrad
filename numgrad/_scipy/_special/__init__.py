import numpy as np
import scipy.special as sp

from numgrad._utils._expand_to import _expand_to_if_array
from numgrad._vjp import _register_vjp

# https://docs.scipy.org/doc/scipy/reference/special.html#raw-statistical-functions
_register_vjp(sp.expit, lambda x: lambda g, r: g * r * (1 - r))
_register_vjp(sp.log_expit, lambda x: lambda g, r: g * (1 - np.exp(r)))

# https://docs.scipy.org/doc/scipy/reference/special.html#gamma-and-related-functions
_register_vjp(sp.gamma, lambda x: lambda g, r: g * r * sp.digamma(x))

# https://docs.scipy.org/doc/scipy/reference/special.html#other-special-functions
_register_vjp(
    sp.softmax,
    lambda x, axis=None: lambda g, r: (
        rg := r * g,
        rg - r * rg.sum(axis=axis, keepdims=True),
    )[1],
)
_register_vjp(
    sp.log_softmax,
    lambda x, axis=None: lambda g, r: (
        g - np.exp(r) * g.sum(axis, keepdims=True)
    ),
)

# https://docs.scipy.org/doc/scipy/reference/special.html#convenience-functions
_register_vjp(
    sp.logsumexp,
    lambda a, axis=None, *, keepdims=False, return_sign=False: lambda g, r: (
        g := g[0] if return_sign else g,
        r := r[0] if return_sign else r,
        _expand_to_if_array(g, a.ndim, axis, keepdims) * np.exp(
            a - _expand_to_if_array(r, a.ndim, axis, keepdims)),
    )[-1],
)
