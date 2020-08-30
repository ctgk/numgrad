import typing as tp

import numpy as np

from pygrad._core._errors import DifferentiationError
from pygrad._core._types import DataType, _to_pygrad_type
from pygrad._utils._typecheck import _typecheck


class Array(object):
    __array_ufunc__ = None

    @_typecheck()
    def __init__(
            self,
            data: object,
            dtype: tp.Type[DataType] = None,
            is_variable: bool = False,
            *,
            name: tp.Union[str, None] = None):
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
        if name is not None:
            for ng_char in (',', '(', ')'):
                if ng_char in name:
                    raise ValueError(
                        f'NG character {ng_char} contained'
                        f' in arg \'name\', {name}.')
        self._name = name
        self._data = np.asarray(data, dtype=dtype)
        if is_variable and 'float' not in repr(self._data.dtype):
            raise DifferentiationError(
                'Non-floating array is not differentiable.')
        self._is_variable: bool = is_variable
        self._num_backwards: int = 0
        self._grad = None
        self._graph = None

    def __repr__(self) -> str:
        repr_ = repr(self.data)
        if self._name is not None:
            repr_ = repr_[:-1] + f', name={self._name})'
        return repr_

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    @_typecheck()
    def data(self, value: np.ndarray):
        if self._graph is not None:
            raise ValueError('Cannot set data to output of operator')
        if value.shape != self._data.shape:
            raise ValueError('Inappropriate shape of the array')
        if value.dtype != self._data.dtype:
            raise ValueError('Inappropriate data type of the array')
        self._data = value

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
        if self._grad is None:
            raise ValueError('This gradient is empty.')
        return self._grad

    @_typecheck()
    def astype(self, dtype: tp.Type[DataType]):
        return Array(self._data, dtype=dtype)

    def clear_grad(self):
        self._grad = None

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
