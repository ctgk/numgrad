from typing import Tuple, Type

import numpy as np

from pygrad._errors import DifferentiationError
from pygrad._node import _Node
from pygrad._types import DataType, _to_pygrad_type
from pygrad._type_check import _typecheck_args


class Array(_Node):

    @_typecheck_args
    def __init__(
            self,
            value: object,
            dtype: Type[DataType] = None,
            is_differentiable: bool = False,
            *,
            name: str = None,
            **kwargs):
        if is_differentiable and '_parent' in kwargs:
            super().__init__(kwargs['_parent'], name=name)
        else:
            super().__init__(name=name)
        self._parent = kwargs.get(
            '_parent', None) if is_differentiable else None
        self._value = np.asarray(value, dtype=dtype)
        self._is_differentiable: bool = is_differentiable
        self._num_backwards: int = 0
        self._grad = None

    def __repr__(self) -> str:
        repr_ = repr(self.value)
        if self._name is not None:
            repr_ = repr_[:-1] + f', name={self._name})'
        return repr_

    @property
    def value(self) -> np.ndarray:
        return self._value

    @property
    def dtype(self) -> DataType:
        return _to_pygrad_type(self._value.dtype)

    @property
    def ndim(self) -> int:
        return self._value.ndim

    @property
    def size(self) -> int:
        return self._value.size

    @property
    def shape(self) -> Tuple[int]:
        return self._value.shape

    @property
    def is_differentiable(self) -> bool:
        return self._is_differentiable

    def clear_grad(self):
        self._num_backwards = 0
        self.grad = None

    def backward(self, **kwargs):
        if not self._is_differentiable:
            raise DifferentiationError(
                'Cannot call backward() method of non-differentiable array.')
        if len(self._children) == 0:
            grad = kwargs.get('_grad', np.ones_like(self.value, self.dtype))
        else:
            grad = kwargs.get('_grad')
            if self._grad is None:
                self._grad = grad
            else:
                self._grad += grad
            self._num_backwards += 1
        if ((self._parent is not None)
                and (self._num_backwards == len(self.children))):
            self._parent.backward(self.grad)
