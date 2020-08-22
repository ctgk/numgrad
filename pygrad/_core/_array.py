import typing as tp

import numpy as np

from pygrad._core._errors import DifferentiationError
from pygrad._core._node import _Node
from pygrad._core._types import DataType, _to_pygrad_type
from pygrad._utils._typecheck import _typecheck


class Array(_Node):

    @_typecheck()
    def __init__(
            self,
            data: object,
            dtype: tp.Type[DataType] = None,
            is_variable: bool = False,
            *,
            name: tp.Union[str, None] = None,
            **kwargs):
        """Construct array object.

        Parameters
        ----------
        data : object
            Data of this array.
        dtype : Type[DataType], optional
            Desired data type, by default None
        is_variable : bool, optional
            Set True if you want to compute gradient of this array,
            by default False
        name : str, optional
            Name of this array, by default None
        """
        if is_variable and '_parent' in kwargs:
            super().__init__(kwargs['_parent'], name=name)
        else:
            super().__init__(name=name)
        self._parent = kwargs.get(
            '_parent', None) if is_variable else None
        self._data = np.asarray(data, dtype=dtype)
        if is_variable and 'float' not in repr(self._data.dtype):
            raise DifferentiationError(
                'Non-floating array is not differentiable.')
        self._is_variable: bool = is_variable
        self._num_backwards: int = 0
        self._grad = None

    def __repr__(self) -> str:
        repr_ = repr(self.data)
        if self._name is not None:
            repr_ = repr_[:-1] + f', name={self._name})'
        return repr_

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def dtype(self) -> DataType:
        return _to_pygrad_type(self._data.dtype)

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def shape(self) -> tp.Tuple[int]:
        return self._data.shape

    @property
    def is_variable(self) -> bool:
        return self._is_variable

    @property
    def grad(self) -> np.ndarray:
        if self._grad is None or (self._num_backwards != len(self._children)):
            raise ValueError('This object does not have a valid gradient.')
        return self._grad

    @_typecheck()
    def astype(self, dtype: tp.Type[DataType]):
        return Array(self._data, dtype=dtype)

    def clear_grad(self):
        self._children = []
        self._num_backwards = 0
        self._grad = None

    def backward(self, **kwargs):
        if not self._is_variable:
            raise DifferentiationError(
                'Cannot call backward() method of non-differentiable array.')
        if len(self._children) == 0:
            grad = kwargs.get('_grad', np.ones_like(self.data))
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
            axis: tp.Union[int, tp.Iterable[int], None] = None,
            keepdims: bool = False,
            *,
            name: str = None):
        raise NotImplementedError

    def mean(
            self,
            axis: tp.Union[int, tp.Iterable[int], None] = None,
            keepdims: bool = False,
            *,
            name: str = None):
        raise NotImplementedError
