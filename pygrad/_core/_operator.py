import abc
from typing import Tuple

import numpy as np

from pygrad._core._array import Array
from pygrad._core._config import config
# from pygrad._core._node import _Node
from pygrad._core._types import DataType


class _Operator(abc.ABC):

    def __init__(self, *args: Array, name: str = None):
        self._args = tuple(
            arg if isinstance(arg, Array) else Array(arg, dtype)
            for arg, dtype in zip(args, self._input_dtypes(*args))
        )
        # super().__init__(*args, name=name)
        if name is not None:
            for ng_char in (',', '(', ')'):
                if ng_char in name:
                    raise ValueError(
                        f'NG character {ng_char} contained'
                        f' in arg \'name\', {name}.')
        self._name = name
        self._differentiable = any(arg._is_variable for arg in self._args)
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

    def forward(self) -> Array:
        if config._graph is None:
            return Array(
                self._forward_numpy(*tuple(arg._data for arg in self._args)),
                is_variable=self._differentiable,
                name=None if self._name is None else self._name + '.out',
            )
        else:
            out = Array(
                self._forward_numpy(*tuple(arg._data for arg in self._args)),
                is_variable=self._differentiable,
                name=None if self._name is None else self._name + '.out',
                # _parent=self,
            )
            config._graph._add_array_op(out, self)
            return out

    def backward(self, delta: np.ndarray):
        dargs = self._backward_numpy(
            delta, *tuple(arg._data for arg in self._args))
        dargs = dargs if isinstance(dargs, tuple) else (dargs,)
        return dargs

    @abc.abstractmethod
    def _forward_numpy(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _backward_numpy(self, *args, **kwargs):
        pass
