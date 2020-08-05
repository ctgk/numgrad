from typing import Tuple

import numpy as np

from pygrad._array import Array
from pygrad._errors import DifferentiationError
from pygrad._node import _Node


class _Operator(_Node):

    def __init__(self, *args: Array, name: str = None):
        args = tuple(
            arg if isinstance(arg, Array) else Array(arg) for arg in args)
        super().__init__(*args, name=name)
        self._is_differentiable = any(arg._is_differentiable for arg in args)
        assert(callable(getattr(self, '_forward_numpy')))
        assert(callable(getattr(self, '_backward_numpy')))

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
            try:
                arg.backward(_grad=darg)
            except DifferentiationError:
                pass
