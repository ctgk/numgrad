import functools
import inspect
import itertools
import typing as tp

import numpy as np

from numgrad._config import config
from numgrad._variable import _ndarray_args, _ndarray_kwargs, Variable


def _register_gradient(
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
                result = forward(
                    *_ndarray_args(*args), **_ndarray_kwargs(**kwargs))
                if config._graph is not None:
                    result = Variable(result)
                    config._graph._add_node(result, forward, *args, **kwargs)
                return result
            return forward(*args, **kwargs)

        config._patched_function[forward] = (
            module_name if module_name is not None else '.'.join(
                m for m in forward.__module__.split('.')
                if not m.startswith('_')
            ),
            forward.__name__ if function_name is None else function_name,
            patched,
        )

    def decorator(grad_func):
        if method is not None:
            config._registered_gradient_function[
                getattr(forward, method)] = grad_func
        else:
            config._registered_gradient_function[forward] = grad_func
        return grad_func

    return decorator


def differentiable(grad_func: callable):
    """Make a function differentiable.

    Parameters
    ----------
    grad_func : callable
        Gradient function whose parameters are parameters of the function
        followed by gradient_of_output and output.

    Examples
    --------
    >>> def custom_gradient(doutput, output, x):
    ...     return 3 * doutput  # note that this is wrong.
    ...
    >>> @differentiable(custom_gradient)
    ... def twice(x):
    ...     return 2 * x
    ...
    >>> twice(np.array([4, 2]))
    array([8, 4])
    >>> a = ng.Variable([4, 2])
    >>> with ng.Graph() as g:
    ...     b = twice(a)
    ...
    >>> b
    Variable([8., 4.])
    >>> g.gradient(b, [a])[0]  # custom gradient is used
    array([3., 3.])
    """

    def decorator(forward):
        config._registered_gradient_function[forward] = grad_func

        @functools.wraps(forward)
        def wrapped_forward(*args, **kwargs):
            result = forward(
                *_ndarray_args(*args), **_ndarray_kwargs(**kwargs))
            if (
                config._graph is not None
                and any(
                    isinstance(a, Variable) for a
                    in itertools.chain(args, kwargs.values())
                )
            ):
                result = Variable(result)
                config._graph._add_node(result, forward, *args, **kwargs)
            return result

        wrapped_forward.__signature__ = inspect.signature(forward)
        return wrapped_forward

    return decorator
