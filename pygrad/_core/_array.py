from typing import Iterable, Tuple, Type, Union

import numpy as np

from pygrad._core._errors import DifferentiationError
from pygrad._core._node import _Node
from pygrad._core._types import DataType, _to_pygrad_type
from pygrad._utils._typecheck import _typecheck


class Array(_Node):

    @_typecheck()
    def __init__(
            self,
            value: object,
            dtype: Type[DataType] = None,
            is_differentiable: bool = False,
            *,
            name: str = None,
            **kwargs):
        """Construct array object.

        Parameters
        ----------
        value : object
            Value of this array.
        dtype : Type[DataType], optional
            Desired data type, by default None
        is_differentiable : bool, optional
            Set True if you want to compute gradient of this array,
            by default False
        name : str, optional
            Name of this array, by default None
        """
        if is_differentiable and '_parent' in kwargs:
            super().__init__(kwargs['_parent'], name=name)
        else:
            super().__init__(name=name)
        self._parent = kwargs.get(
            '_parent', None) if is_differentiable else None
        self._value = np.asarray(value, dtype=dtype)
        if is_differentiable and 'float' not in repr(self._value.dtype):
            raise DifferentiationError(
                'Non-floating array is not differentiable.')
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

    @property
    def grad(self) -> np.ndarray:
        if self._grad is None or (self._num_backwards != len(self._children)):
            raise ValueError('This object does not have a valid gradient.')
        return self._grad

    def clear_grad(self):
        self._children = []
        self._num_backwards = 0
        self._grad = None

    def backward(self, **kwargs):
        if not self._is_differentiable:
            raise DifferentiationError(
                'Cannot call backward() method of non-differentiable array.')
        if len(self._children) == 0:
            grad = kwargs.get('_grad', np.ones_like(self.value, self.dtype))
        else:
            grad = kwargs.get('_grad')
            self._num_backwards += 1
        if self._grad is None:
            self._grad = grad
        else:
            self._grad += grad
        if ((self._parent is not None)
                and (self._num_backwards == len(self._children))):
            self._parent.backward(self._grad)

    def __neg__(self):
        raise NotImplementedError

    def sum(self,
            axis: Union[int, Iterable[int], None] = None,
            keepdims: bool = False,
            *,
            name: str = None):
        raise NotImplementedError

    def mean(
            self,
            axis: Union[int, Iterable[int], None] = None,
            keepdims: bool = False,
            *,
            name: str = None):
        raise NotImplementedError
