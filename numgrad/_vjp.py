import functools
import inspect
import itertools

import numpy as np

from numgrad._config import config
from numgrad._variable import _ndarray_args, _ndarray_kwargs, Variable


def _register_vjp(
    forward: callable,
    func_to_vjp: callable,
    module_name: str = None,
    func_name: str = None,
):

    config._func2vjps[forward] = func_to_vjp

    if isinstance(forward, np.ufunc):
        return
    if (
        hasattr(forward, '__code__')
        and '__array_function__' in repr(forward.__code__)
    ):
        return

    def patched(*args, **kwargs):
        result = forward(
            *_ndarray_args(*args), **_ndarray_kwargs(**kwargs))
        if any(
            isinstance(a, Variable) for a
            in itertools.chain(args, kwargs.values())
        ):
            result = Variable._postprocess(result, forward, *args, **kwargs)
        return result

    config._patched_function[forward] = (
        module_name if module_name is not None else '.'.join(
            m for m in forward.__module__.split('.')
            if not m.startswith('_')
        ),
        forward.__name__ if func_name is None else func_name,
        patched,
    )


def custom_vjp(func_to_vjp: callable) -> callable:
    """Return wrapper function to make a custom differentiable function.

    Parameters
    ----------
    func_to_vjp : callable
        Function to return vector-jacobian-product function(s).

    Returns
    -------
    callable
        Wrapper function which make argument function differentiable.

    Examples
    --------
    >>> # note that this is a custom gradient
    >>> @custom_vjp(lambda x: lambda g, r: g * 3)
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
    >>> g.backward(b, [a])[0]  # custom gradient is used
    array([3., 3.])
    """

    def decorator(forward):
        config._func2vjps[forward] = func_to_vjp

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
