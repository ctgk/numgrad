import functools
import inspect
import typing as tp

import numpy as np

from pygrad._core._config import config
from pygrad._core._dtypes import DataType
from pygrad._core._node import _Node
from pygrad._core._tensor import Tensor, TensorLike


class _DifferentiableOperator(_Node):

    def __init__(self, function: callable) -> None:
        super().__init__(name=function.__name__.lstrip('_'))
        self._function = function
        self._args: tp.Tuple[Tensor] = None
        self._grad_func: callable = None
        self._child: Tensor = None

    def _get_out_dtype(*args: TensorLike) -> tp.Type[DataType]:
        out_dtype = None
        for arg in args:
            if isinstance(arg, Tensor):
                out_dtype = arg.dtype
                break
        else:
            return config.dtype
        for arg in args:
            if isinstance(arg, Tensor):
                if out_dtype != arg.dtype:
                    raise ValueError(
                        'Mix of multiple data types: '
                        f'{out_dtype} and {arg.dtype}.',
                    )
        return out_dtype

    def _check_args(self, *args: TensorLike) -> tp.Tuple[Tensor]:
        dtype = self._get_out_dtype(*args)
        args = tuple(
            arg if isinstance(arg, Tensor) else Tensor(arg, dtype)
            for arg in args
        )
        for arg in args:
            arg._children.append(self)
        return args

    def __call__(self, *args: TensorLike, **kwargs) -> Tensor:
        self._args: tp.Tuple[Tensor] = self._check_args(*args)
        has_variable = any(a._is_variable for a in self._args)
        out_data, self._grad_func = self._function(
            *tuple(a._data for a in self._args),
            **kwargs,
        )
        out = Tensor(
            data=out_data,
            dtype=out_data.dtype,
            is_variable=has_variable,
            name=None if self._name is None else self._name + '.out',
            _parent=self if has_variable else None,
        )
        if has_variable:
            self._child = out
            if config._graph is not None:
                config._graph._operations.append(self)
        return out

    def backward(self, dout: np.ndarray):
        dargs = self._grad_func(dout)
        if not isinstance(dargs, tuple):
            dargs = (dargs,)
        for arg, darg in zip(self._args, dargs):
            if arg.is_variable:
                arg.backward(darg)


def differentiable_operator(func: callable) -> callable:
    """Decorate numpy function and convert it to differentiable one.

    Function to decorate must have positional parameters and key word only
    parameters whose type must be `TensorLike` object and others respectively.

    Parameters
    ----------
    func : callable
        Function that returns pair of output and backward function.

    Returns
    -------
    callable
        Differentiable function.

    Examples
    --------
    >>> @differentiable_operator
    ... def twice(x):
    ...     def grad(dout):
    ...         return 2 * dout
    ...     return 2 * x, grad
    >>> a = Tensor([-1, 0, 1], is_variable=True)
    >>> b = twice(a)
    >>> b.backward()
    >>> b
    Tensor([-2.,  0.,  2.])
    >>> a.grad
    array([2., 2., 2.])
    """

    @functools.wraps(func)
    def wrapped_function(*args, **kwargs):
        operator = _DifferentiableOperator(func)
        return operator(*args, **kwargs)

    wrapped_function.__signature__ = inspect.signature(func)
    return wrapped_function
