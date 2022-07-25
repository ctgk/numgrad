# https://numpy.org/doc/stable/reference/random/index.html

import numpy as np

from numgrad._utils._unbroadcast import _unbroadcast_to
from numgrad._vjp import _bind_vjp


# https://numpy.org/doc/stable/reference/random/legacy.html#functions-in-numpy-random
_bind_vjp(
    np.random.exponential,
    lambda g, r, scale, size=None: (
        dx := g * r / scale,
        dx if size is None else _unbroadcast_to(dx, scale.shape),
    )[1],
    module_name='numpy.random', func_name='exponential',
)
_bind_vjp(
    np.random.normal,
    lambda g, r, loc, scale, size=None: +g,
    lambda g, r, loc, scale, size=None: g * (r - loc) / scale,
    module_name='numpy.random', func_name='normal',
)
_bind_vjp(
    np.random.uniform,
    lambda g, r, low, high, size=None: g * (high - r) / (high - low),
    lambda g, r, low, high, size=None: g * (r - low) / (high - low),
    module_name='numpy.random', func_name='uniform',
)
