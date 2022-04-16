from numbers import Number
import os
import typing as tp

import numpy as np

from pygrad._core._config import config
from pygrad._core._differentiation_error import DifferentiationError
from pygrad._core._dtypes import _to_pygrad_type, DataType
from pygrad._core._node import _Node
from pygrad._utils._typecheck import _typecheck


TensorLike = tp.Union[Number, tp.List, tp.Tuple, np.ndarray]
TypeLike = tp.Union[tp.Type[DataType], tp.Type[int], tp.Type[float], np.dtype]


class Tensor(_Node):
    """Tensor object.

    Examples
    --------
    >>> a = gd.Tensor(1, dtype=gd.Float32, is_variable=True, name='a')
    >>> a
    Tensor(1., dtype=float32)
    >>> a.dtype
    <class 'pygrad.Float32'>
    >>> a.is_variable
    True
    >>> a.name
    'a'
    """

    __array_ufunc__ = None

    @_typecheck()
    def __init__(  # noqa: D107
        self,
        data: TensorLike,
        dtype: tp.Optional[TypeLike] = None,
        *,
        is_variable: bool = False,
        name: tp.Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(name=name)
        self._data = np.asarray(
            data,
            dtype=dtype if dtype is not None else config.dtype,
        )
        self._grad = None
        self._num_grad_accumulation = 0
        self._children = []
        self._parent = kwargs.get('_parent', None)
        self._is_variable = is_variable

    def numpy(self) -> np.ndarray:
        """Return a numpy array of the tensor.

        Returns
        -------
        np.ndarray
            Numpy array of the tensor.
        """
        return self._data

    @property
    def grad(self) -> np.ndarray:  # noqa: D102
        if self._is_valid_grad():
            return self._grad
        if not self._is_variable:
            return None
        if self._num_grad_accumulation == 0:
            raise ValueError(
                'Please call `gd.Tensor.backward()` method of a leaf node.')
        raise ValueError(
            'There may be multiple leaf nodes under the node '
            f'(name={self._name}), please remove unnecessary operations.',
        )

    @property
    def dtype(self) -> tp.Type[DataType]:  # noqa: D102
        return _to_pygrad_type(self._data.dtype)

    @property
    def ndim(self) -> int:  # noqa: D102
        return self._data.ndim

    @property
    def size(self) -> int:  # noqa: D102
        return self._data.size

    @property
    def shape(self) -> tp.Tuple[int, ...]:  # noqa: D102
        return self._data.shape

    @property
    def is_variable(self) -> bool:  # noqa: D102
        return self._is_variable

    def __repr__(self) -> str:  # noqa: D105
        old = 'array'
        new = 'Tensor'
        lines = repr(self._data).splitlines()
        lines[0] = lines[0].replace(old, new)
        for index in range(1, len(lines)):
            if len(lines[index]) > 0:
                lines[index] = ' ' * (len(new) - len(old)) + lines[index]
        return os.linesep.join(lines)

    @_typecheck()
    def astype(self, dtype: tp.Type[DataType]) -> 'Tensor':  # noqa: D102
        return Tensor(self._data, dtype=dtype)

    def backward(self, dout: TensorLike = 1.):  # noqa: D102
        if self._is_variable is False:
            raise DifferentiationError(
                'An attempt to backpropagate gradient from a constant object',
            )
        self._accumulate_grad(dout)
        if self._parent is None:
            return
        if self._backprop_to_parent():
            self._parent.backward(self._grad)

    def clear(self):  # noqa: D102
        self._children = []
        self._grad = None
        self._num_grad_accumulation = 0

    def _is_valid_grad(self) -> bool:
        if len(self._children) == 0 and self._num_grad_accumulation == 1:
            return True
        if len(self._children) == self._num_grad_accumulation:
            return True
        return False

    def _accumulate_grad(self, dout: TensorLike):
        if self._grad is None:
            self._grad = np.ones_like(self._data) * dout
        else:
            self._grad += dout
        self._num_grad_accumulation += 1

    def _backprop_to_parent(self) -> bool:
        if self._is_valid_grad():
            return True
        if len(self._children) < self._num_grad_accumulation:
            raise AssertionError(
                'Number of gradient accumulation '
                'exceeds the number of children.',
            )
        return False  # len(self._children) > self._num_grad_accumulation


TensorLike = tp.Union[Number, tp.List, tp.Tuple, np.ndarray, Tensor]
