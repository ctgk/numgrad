import numpy as np
import scipy.special as sp

from numgrad._utils._expand_to import _expand_to_if_array
from numgrad._vjp import _register_vjp

# https://docs.scipy.org/doc/scipy/reference/special.html#raw-statistical-functions
_register_vjp(sp.expit, lambda dy, y, _x: dy * y * (1 - y))
_register_vjp(sp.log_expit, lambda dy, y, _x: dy * (1 - np.exp(y)))

# https://docs.scipy.org/doc/scipy/reference/special.html#gamma-and-related-functions
_register_vjp(sp.gamma, lambda dy, y, x: dy * y * sp.digamma(x))

# https://docs.scipy.org/doc/scipy/reference/special.html#other-special-functions
_register_vjp(
    sp.softmax,
    lambda dy, y, _x, axis=None: (
        ydy := y * dy,
        ydy - y * ydy.sum(axis=axis, keepdims=True),
    )[1],
)
_register_vjp(
    sp.log_softmax,
    lambda dy, y, _x, axis=None: (
        dy - np.exp(y) * dy.sum(axis=axis, keepdims=True)
    ),
)

# https://docs.scipy.org/doc/scipy/reference/special.html#convenience-functions
_register_vjp(
    sp.logsumexp,
    lambda dy, y, x, axis=None, keepdims=False, return_sign=False: (
        _expand_to_if_array(dy, x.ndim, axis, keepdims)
        * np.exp(x - _expand_to_if_array(y, x.ndim, axis, keepdims))
        if return_sign is False else None
    ),
)
