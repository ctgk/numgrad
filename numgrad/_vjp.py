from collections.abc import Iterable
import functools
import inspect
import itertools

import numpy as np

from numgrad._config import config
from numgrad._variable import _ndarray_args, _ndarray_kwargs, Variable


def _wrap_vjp(vjp: callable) -> callable:
    vjp_args = inspect.getfullargspec(vjp).args
    if 'g' not in vjp_args:
        vjp = functools.partial(
            lambda vjp, g, *args, **kwargs: g * vjp(*args, **kwargs),
            vjp,
        )
    if 'r' not in vjp_args:
        vjp = functools.partial(
            lambda vjp, g, r, *args, **kwargs: vjp(g, *args, **kwargs),
            vjp,
        )
    return vjp


class _VJPIterator:

    def __init__(self, vjp: callable) -> None:
        self._vjp = vjp
        self._index = 0

    def __len__(self):
        return 0

    def __iter__(self):
        return _VJPIterator(self._vjp)

    def __next__(self):
        vjp = functools.partial(self._vjp, _nth=self._index)
        self._index += 1
        return vjp


def _bind_vjp(
    forward: callable,
    *vjps: callable,
    module_name: str = None,
    func_name: str = None,
):

    if len(vjps) == 1 and isinstance(vjps[0], Iterable):
        config._func2vjps[forward] = vjps[0]
    else:
        config._func2vjps[forward] = tuple(_wrap_vjp(vjp) for vjp in vjps)

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


def custom_vjp(*vjps: callable) -> callable:
    """Return wrapper function to make a custom differentiable function.

    Parameters
    ----------
    vjps : callable
        Function(s) to return vector-jacobian-product for each argument.

    Returns
    -------
    callable
        Wrapper function which make argument function differentiable.

    Examples
    --------
    >>> # note that this is a custom gradient
    >>> @custom_vjp(lambda g, r, x: g * 3)
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
        if len(vjps) == 1 and isinstance(vjps[0], Iterable):
            config._func2vjps[forward] = vjps[0]
        else:
            config._func2vjps[forward] = tuple(_wrap_vjp(vjp) for vjp in vjps)

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
