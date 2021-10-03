import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Softplus(_Operator):

    def __init__(self, x: Array, name: str = None):
        super().__init__(x, name=name)

    @staticmethod
    def _forward_numpy(x):
        return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))

    @staticmethod
    def _backward_numpy(delta, x):
        return (np.tanh(0.5 * x) * 0.5 + 0.5) * delta


@_typecheck(exclude_args=('x',))
def softplus(x: Array, *, name: str = None) -> Array:
    return _Softplus(x, name=name).forward()
