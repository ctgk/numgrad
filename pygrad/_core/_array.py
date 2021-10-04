import typing as tp

import numpy as np

from pygrad._core._errors import DifferentiationError
from pygrad._core._types import _to_pygrad_type, DataType
from pygrad._utils._typecheck import _typecheck


class Array(object):
    """Array class."""

    __array_ufunc__ = None

    @_typecheck()
    def __init__(
        self,
        data: object,
        dtype: tp.Type[DataType] = None,
        is_variable: bool = False,
        *,
        name: tp.Union[str, None] = None,
    ):
        """Initialize array object.

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
        """Return representation.

        Returns
        -------
        str
            Representation of the object.
        """
        repr_ = repr(self.data)
        if self._name is not None:
            repr_ = repr_[:-1] + f', name={self._name})'
        return repr_

    @property
    def name(self) -> str:
        """Return name of the array.

        Returns
        -------
        str
            Name of the array.
        """
        return self._name

    @property
    def data(self) -> np.ndarray:
        """Return numpy array.

        Returns
        -------
        np.ndarray
            Numpy array.
        """
        return self._data

    @data.setter
    @_typecheck()
    def data(self, value: np.ndarray):
        """Set numpy data.

        Parameters
        ----------
        value : np.ndarray
            Numpy data.
        """
        if self._graph is not None:
            raise ValueError('Cannot set data to output of operator')
        if value.shape != self._data.shape:
            raise ValueError('Inappropriate shape of the array')
        if value.dtype != self._data.dtype:
            raise ValueError('Inappropriate data type of the array')
        self._data = value

    @property
    def dtype(self) -> DataType:
        """Return data type of the array.

        Returns
        -------
        DataType
            Data type of the array.
        """
        return _to_pygrad_type(self._data.dtype)

    @property
    def ndim(self) -> int:
        """Return dimensionality of the array.

        Returns
        -------
        int
            Dimensionality of the array.
        """
        return self._data.ndim

    @property
    def size(self) -> int:
        """Return number of elements in the array.

        Returns
        -------
        int
            Number of elements in the array.
        """
        return self._data.size

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        """Return shape of the array.

        Returns
        -------
        tp.Tuple[int, ...]
            Shape of the array.
        """
        return self._data.shape

    @property
    def is_variable(self) -> bool:
        """Return true if the array is variable.

        Returns
        -------
        bool
            True if the array is variable, otherwise it is constant.
        """
        return self._is_variable

    @property
    def grad(self) -> np.ndarray:
        """Return derivative with respect to the array.

        Returns
        -------
        np.ndarray
            Derivative with respect to the array.
        """
        if self._grad is None:
            raise ValueError('This gradient is empty.')
        return self._grad

    @_typecheck()
    def astype(self, dtype: tp.Type[DataType]):
        """Return array with specified data type.

        Parameters
        ----------
        dtype : tp.Type[DataType]
            Data type.

        Returns
        -------
        Array
            Array with specified data type.
        """
        return Array(self._data, dtype=dtype)

    def clear_grad(self):
        """Clear derivative."""
        self._grad = None

    def __neg__(self):
        """Return negated array."""
        raise NotImplementedError

    def sum(  # noqa: A003
        self,
        axis: tp.Union[int, tp.Iterable[int], None] = None,
        keepdims: bool = False,
        *,
        name: str = None,
    ):
        """Return sum of the array along specified axes.

        Parameters
        ----------
        axis : tp.Union[int, tp.Iterable[int], None], optional
            Axis to sum along, by default None
        keepdims : bool, optional
            Keep dimensionality of the array if true, by default False
        name : str, optional
            Name of the operation, by default None
        """
        raise NotImplementedError

    def mean(
        self,
        axis: tp.Union[int, tp.Iterable[int], None] = None,
        keepdims: bool = False,
        *,
        name: str = None,
    ):
        """Return mean of the array along specified axes.

        Parameters
        ----------
        axis : tp.Union[int, tp.Iterable[int], None], optional
            Axis to mean along, by default None
        keepdims : bool, optional
            Keep dimensionality of the array if true, by default False
        name : str, optional
            Name of the operation, by default None
        """
        raise NotImplementedError
