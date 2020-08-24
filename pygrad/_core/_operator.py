import abc
from typing import Tuple

import numpy as np

from pygrad._core._array import Array
from pygrad._core._config import config
from pygrad._core._node import _Node
from pygrad._core._types import DataType


class _Operator(_Node):

    def __init__(self, *args: Array, name: str = None):
        args = tuple(
            arg if isinstance(arg, Array) else Array(arg, dtype)
            for arg, dtype in zip(args, self._input_dtypes(*args))
        )
        super().__init__(*args, name=name)
        self._differentiable = any(arg._is_variable for arg in args)
        self._check_dtype()

    def _input_dtypes(self, *args: Array) -> Tuple[DataType]:
        for arg in args:
            if isinstance(arg, Array):
                dtype = arg.dtype
                break
        else:
            dtype = config.dtype
        return (dtype,) * len(args)

    def _check_dtype(self):
        if len(self._args) == 0:
            return
        dtype = self._args[0].dtype
        for arg in self._args:
            if dtype is not arg.dtype:
                raise ValueError(
                    'All array\'s dtype must be the same, but were '
                    f'{tuple(arg.dtype for arg in self._args)}')

    @property
    def _args(self) -> Tuple[Array]:
        return self._parents

    def forward(self) -> Array:
        return Array(
            self._forward_numpy(*tuple(arg._data for arg in self._args)),
            is_variable=self._differentiable,
            name=None if self._name is None else self._name + '.out',
            _parent=self,
        )

    def backward(self, delta: np.ndarray) -> Array:
        dargs = self._backward_numpy(
            delta, *tuple(arg._data for arg in self._args))
        dargs = dargs if isinstance(dargs, tuple) else (dargs,)
        for arg, darg in zip(self._args, dargs):
            if darg is not None:
                arg.backward(_grad=darg)

    @abc.abstractmethod
    def _forward_numpy(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _backward_numpy(self, *args, **kwargs):
        pass
