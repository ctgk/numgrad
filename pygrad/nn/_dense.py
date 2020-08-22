import typing as tp

import numpy as np

from pygrad._core._array import Array
from pygrad._core._config import config
from pygrad._core._module import Module
from pygrad._core._types import DataType
from pygrad._math._matmul import matmul
from pygrad._utils._typecheck import _typecheck


class Dense(Module):
    """Densely connected layer

    Examples
    --------
    >>> import pygrad as pg; import numpy as np
    >>> d = Dense(2, 3)
    >>> d(np.random.normal(size=(4, 2))).shape
    (4, 3)
    """

    @_typecheck()
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            dtype: tp.Union[tp.Type[DataType], None] = None):
        """Constrct densely connected layer

        Parameters
        ----------
        in_features : int
            Dimesionality of input features
        out_features : int
            Dimesionality of output features
        bias : bool, optional
            Whether to add bias or not, by default True
        dtype : tp.Union[tp.Type[DataType], None], optional
            Desired data types of weight and bias, by default None
        """
        super().__init__()
        dtype = dtype if dtype is not None else config.dtype
        self._in_features = in_features
        self._out_features = out_features
        v = 1 / np.sqrt(in_features)
        self.weight = Array(
            np.random.uniform(-v, v, (in_features, out_features)),
            dtype=dtype,
            is_variable=True)
        if bias:
            self.bias = Array(
                np.random.uniform(-v, v, (out_features,)),
                dtype=dtype,
                is_variable=True)

    def __call__(self, x: Array, **kwargs) -> Array:
        x = matmul(x, self.weight)
        if hasattr(self, 'bias'):
            x = x + self.bias
        return x
