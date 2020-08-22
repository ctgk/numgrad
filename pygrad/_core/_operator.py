import abc
from typing import Tuple

import numpy as np

from pygrad._core._array import Array
from pygrad._core._config import config
from pygrad._core._errors import DifferentiationError
from pygrad._core._node import _Node
from pygrad._core._types import DataType


class _Operator(_Node):

    def __init__(self, *args: Array, name: str = None):
        args = tuple(
            arg if isinstance(arg, Array) else Array(arg, dtype)
            for arg, dtype in zip(args, self._input_dtypes(*args))
        )
        super().__init__(*args, name=name)
        self._is_differentiable = any(arg._is_differentiable for arg in args)
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
                raise ValueError('All array\'s dtype must be the same.')

    @property
    def _args(self) -> Tuple[Array]:
        return self._parents

    def forward(self) -> Array:
        return Array(
            self._forward_numpy(*tuple(arg.value for arg in self._args)),
            is_differentiable=self._is_differentiable,
            **{
                k: v + '.out' for k, v in zip(['name'], [self._name])
                if v is not None
            },
            _parent=self,
        )

    def backward(self, delta: np.ndarray) -> Array:
        dargs = self._backward_numpy(
            delta, *tuple(arg.value for arg in self._args))
        dargs = dargs if isinstance(dargs, tuple) else (dargs,)
        for arg, darg in zip(self._args, dargs):
            if darg is not None:
                try:
                    arg.backward(_grad=darg)
                except DifferentiationError:
                    pass

    @abc.abstractmethod
    def _forward_numpy(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _backward_numpy(self, *args, **kwargs):
        pass
