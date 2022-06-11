import numpy as np

from numgrad._utils._unbroadcast import _unbroadcast_to
from numgrad._vjp import _register_vjp


# https://numpy.org/doc/stable/reference/random/legacy.html#functions-in-numpy-random
_register_vjp(
    np.random.exponential,
    lambda scale, size=None: lambda g, r: (
        dx := g * r / scale,
        dx if size is None else _unbroadcast_to(dx, scale.shape),
    )[1],
    module_name='numpy.random', func_name='exponential',
)
_register_vjp(
    np.random.normal,
    lambda loc, scale, size=None: (
        lambda g, r: _unbroadcast_to(g, loc.shape),
        lambda g, r: _unbroadcast_to(g * (r - loc) / scale, scale.shape),
    ),
    module_name='numpy.random', func_name='normal',
)
_register_vjp(
    np.random.uniform,
    lambda low, high, size=None: (
        lambda g, r: _unbroadcast_to(g * (high - r) / (high - low), low.shape),
        lambda g, r: _unbroadcast_to(g * (r - low) / (high - low), high.shape),
    ),
    module_name='numpy.random', func_name='uniform',
)
