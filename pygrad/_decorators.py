import typing as tp

import numpy as np

from pygrad._config import config
from pygrad._variable import _ndarray_views, Variable


_PATCHED_FUNCTION: tp.Dict[callable, tp.Tuple[str, str, callable]] = {}
_REGISTERED_GRADIENT_FUNCTION: tp.Dict[callable, callable] = {}


def register_gradient(
    forward: callable,
    *,
    method: tp.Optional[str] = None,
    module_name: str = None,
    function_name: str = None,
) -> callable:
    """Register a gradient function of a forward function.

    Parameters
    ----------
    forward : callable
        Forward function to register gradient function for.
    method : tp.Optional[str], optional
        Method of `forward`, by default None
    module_name : str, optional
        Name of module that have forward function to patch, by default None
    function_name : str, optional
        Name of forward function to patch, by default None

    Returns
    -------
    callable
        Registered gradient function.
    """
    if not isinstance(forward, np.ufunc):

        def patched(*args, **kwargs):
            if any(isinstance(a, Variable) for a in args):
                out = Variable(forward(*_ndarray_views(*args), **kwargs))
                if config._graph is not None:
                    config._graph._add_node(out, forward, *args, **kwargs)
                return out
            return forward(*args, **kwargs)

        _PATCHED_FUNCTION[forward] = (
            module_name if module_name is not None else '.'.join(
                m for m in forward.__module__.split('.')
                if not m.startswith('_')
            ),
            forward.__name__ if function_name is None else function_name,
            patched,
        )

    def decorator(grad_func):
        if method is not None:
            _REGISTERED_GRADIENT_FUNCTION[getattr(forward, method)] = grad_func
        else:
            _REGISTERED_GRADIENT_FUNCTION[forward] = grad_func
        return grad_func

    return decorator
