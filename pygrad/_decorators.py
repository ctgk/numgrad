import typing as tp

import numpy as np

from pygrad._config import config
from pygrad._tensor import _ndarray_views, Tensor


_PATCHED_FUNCTION: tp.Dict[callable, callable] = {}
_REGISTERED_GRADIENT_FUNCTION: tp.Dict[callable, callable] = {}


def register_gradient(forward: callable) -> callable:
    """Register a gradient function of a forward function.

    Parameters
    ----------
    forward : callable
        Forward function to register gradient function for.

    Returns
    -------
    callable
        Registered gradient function.
    """
    if not isinstance(forward, np.ufunc):

        def patched(*args, **kwargs):
            if any(isinstance(a, Tensor) for a in args):
                out = forward(*_ndarray_views(*args), **kwargs).view(Tensor)
                if config._graph is not None:
                    config._graph._add_node(out, forward, *args, **kwargs)
                return out
            return forward(*args, **kwargs)

        _PATCHED_FUNCTION[forward] = patched

    def decorator(grad_func):
        _REGISTERED_GRADIENT_FUNCTION[forward] = grad_func
        return grad_func

    return decorator
