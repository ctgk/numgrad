"""Simple gradient computation library for Python."""

from pygrad import distributions, nn, optimizers, random, stats
from pygrad._core._config import config
from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._differentiation_error import DifferentiationError
from pygrad._core._dtypes import (  # noqa: I101
    DataType,
    Int8, Int16, Int32, Int64,  # noqa: I100
    Float16, Float32, Float64,
)
from pygrad._core._graph import Graph
from pygrad._core._module import Module
from pygrad._core._tensor import Tensor
from pygrad._manipulation._getitem import _getitem_from_tensor
from pygrad._manipulation._reshape import reshape
from pygrad._manipulation._transpose import transpose
from pygrad._math._add import add
from pygrad._math._cos import cos
from pygrad._math._cosh import cosh
from pygrad._math._divide import divide
from pygrad._math._exp import exp
from pygrad._math._gamma import gamma
from pygrad._math._log import log
from pygrad._math._logsumexp import logsumexp
from pygrad._math._matmul import matmul
from pygrad._math._max import max
from pygrad._math._mean import mean
from pygrad._math._min import min
from pygrad._math._multiply import multiply
from pygrad._math._negate import negate
from pygrad._math._sin import sin
from pygrad._math._sinh import sinh
from pygrad._math._sqrt import sqrt
from pygrad._math._square import square
from pygrad._math._subtract import subtract
from pygrad._math._sum import sum
from pygrad._math._tan import tan
from pygrad._math._tanh import tanh


def _reshape(x: Tensor, *newshape):
    return reshape(x, newshape)


def _transpose(x: Tensor, *axes):
    return transpose(x, axes) if axes else transpose(x)


Tensor.__getitem__ = _getitem_from_tensor
Tensor.__add__ = add
Tensor.__matmul__ = matmul
Tensor.__mul__ = multiply
Tensor.__neg__ = negate
Tensor.__sub__ = subtract
Tensor.__truediv__ = divide
Tensor.__radd__ = add
Tensor.__rmatmul__ = lambda x, y: matmul(y, x)
Tensor.__rmul__ = multiply
Tensor.__rsub__ = lambda x, y: subtract(y, x)
Tensor.__rtruediv__ = lambda x, y: divide(y, x)
Tensor.max = max
Tensor.mean = mean
Tensor.min = min
Tensor.reshape = _reshape
Tensor.sum = sum
Tensor.transpose = _transpose
Tensor.T = property(lambda self: transpose(self))


_classes = [
    DifferentiationError,
    Graph,
    Module,
    Tensor,
    DataType, Int8, Int16, Int32, Int64, Float16, Float32, Float64,
]

_functions = [
    add,
    cos,
    cosh,
    differentiable_operator,
    divide,
    exp,
    gamma,
    log,
    logsumexp,
    matmul,
    max,
    mean,
    min,
    multiply,
    negate,
    sin,
    sinh,
    sqrt,
    square,
    subtract,
    sum,
    tan,
    tanh,
]

_modules = [
    distributions,
    nn,
    optimizers,
    random,
    stats,
]

for _cls in _classes:
    _cls.__module__ = __name__


__all__ = (
    [_cls.__name__ for _cls in _classes]
    + [_func.__name__ for _func in _functions]
    + [_mod.__name__ for _mod in _modules]
    + [
        'config',
    ]
)
