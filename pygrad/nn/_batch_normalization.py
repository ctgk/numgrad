import typing as tp

import numpy as np

from pygrad._core._array import Array
from pygrad._core._config import config
from pygrad._core._module import Module
from pygrad._core._types import DataType
from pygrad._math._mean import mean
from pygrad._math._sqrt import sqrt
from pygrad._math._square import square
from pygrad._utils._typecheck import _typecheck


class BatchNormalization(Module):
    """Batch normalization layer

    Examples
    --------
    >>> import pygrad as gd; import numpy as np
    >>> bn = gd.nn.BatchNormalization(2)
    >>> a = gd.Array([[0, 1], [2, 3], [3, 4]], gd.Float64)
    >>> b = bn(a, update_emas=True)
    >>> b.shape
    (3, 2)
    >>> np.allclose(gd.mean(b, 0).data, 0)
    True
    """

    @_typecheck()
    def __init__(
            self,
            size: tp.Union[int, tp.Iterable[int]],
            momentum: float = 0.9,
            gamma: float = 1.,
            beta: float = 0.,
            dtype: tp.Union[tp.Type[DataType], None] = None):
        super().__init__()
        dtype = dtype if dtype is not None else config.dtype
        self._momentum = momentum
        self.mean_ema = np.zeros(size, dtype=dtype)
        self.var_ema = np.ones(size, dtype=dtype)
        self.gamma = Array(gamma, dtype=dtype, is_variable=True)
        self.beta = Array(beta, dtype=dtype, is_variable=True)

    def _update_emas(self, m, v):
        self.mean_ema *= self._momentum
        self.mean_ema += (1 - self._momentum) * m
        self.var_ema *= self._momentum
        self.var_ema += (1 - self._momentum) * v

    @_typecheck(exclude_args=('x',))
    def __call__(self, x: Array, *, update_emas: bool = False) -> Array:
        if update_emas:
            axes_reduce = tuple(i for i in range(x.ndim - self.mean_ema.ndim))
            m = mean(x, axis=axes_reduce)
            xc = x - m
            v = mean(square(xc), axis=axes_reduce)
            std = sqrt(v + np.finfo(v.dtype).eps)
            self._update_emas(m.data, v.data)
            return self.beta + self.gamma * xc / std
        else:
            xc = x - self.mean_ema
            return self.beta + self.gamma * xc / np.sqrt(
                self.var_ema + np.finfo(self.var_ema.dtype).eps)
