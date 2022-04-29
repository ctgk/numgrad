import typing as tp

import numpy as np

from pygrad._config import config
from pygrad._variable import _ndarray_views, Variable


_PATCHED_FUNCTION: tp.Dict[callable, callable] = {}
_REGISTERED_GRADIENT_FUNCTION: tp.Dict[callable, callable] = {}


def register_gradient(
    forward: callable,
    *,
    method: tp.Optional[str] = None,
) -> callable:
    """Register a gradient function of a forward function.

    Parameters
    ----------
    forward : callable
        Forward function to register gradient function for.
    method : tp.Optional[str], optional
        Method of `forward`, by default None

    Returns
    -------
    callable
        Registered gradient function.
    """
    if not isinstance(forward, np.ufunc):

        def patched(*args, **kwargs):
            if any(isinstance(a, Variable) for a in args):
                out = forward(*_ndarray_views(*args), **kwargs).view(Variable)
                if config._graph is not None:
                    config._graph._add_node(out, forward, *args, **kwargs)
                return out
            return forward(*args, **kwargs)

        _PATCHED_FUNCTION[forward] = patched

    def decorator(grad_func):
        if method is not None:
            _REGISTERED_GRADIENT_FUNCTION[getattr(forward, method)] = grad_func
        else:
            _REGISTERED_GRADIENT_FUNCTION[forward] = grad_func
        return grad_func

    return decorator
