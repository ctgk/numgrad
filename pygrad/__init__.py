from pygrad._core._array import Array
from pygrad._core._config import config
from pygrad._core._errors import DifferentiationError
from pygrad._core._graph import Graph
from pygrad._core._module import Module
from pygrad._core._types import (
    DataType, Int8, Int16, Int32, Int64, Float16, Float32, Float64, Float128
)

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

from pygrad import distributions, nn, optimizers, random, stats


def _reshape(x: Array, *newshape):
    return reshape(x, newshape)


def _transpose(x: Array, *axes):
    return transpose(x, axes) if axes else transpose(x)


Array.__add__ = add
Array.__matmul__ = matmul
Array.__mul__ = multiply
Array.__neg__ = negate
Array.__sub__ = subtract
Array.__truediv__ = divide
Array.__radd__ = add
Array.__rmatmul__ = lambda x, y: matmul(y, x)
Array.__rmul__ = multiply
Array.__rsub__ = lambda x, y: subtract(y, x)
Array.__rtruediv__ = lambda x, y: divide(y, x)
Array.max = max
Array.mean = mean
Array.min = min
Array.reshape = _reshape
Array.sum = sum
Array.transpose = _transpose
Array.T = property(lambda self: transpose(self))


_classes = [
    Array,
    DifferentiationError,
    Graph,
    Module,
    DataType, Int8, Int16, Int32, Int64, Float16, Float32, Float64, Float128,
]

for _cls in _classes:
    _cls.__module__ = 'pygrad'


__all__ = [_cls.__name__ for _cls in _classes] + [
    'config',

    'reshape',
    'transpose',

    'add',
    'cos',
    'cosh',
    'divide',
    'exp',
    'gamma',
    'log',
    'logsumexp',
    'matmul',
    'max',
    'mean',
    'min',
    'multiply',
    'negate',
    'sin',
    'sinh',
    'sqrt',
    'square',
    'subtract',
    'sum',
    'tan',
    'tanh',

    'distributions', 'nn', 'optimizers', 'random', 'stats',
]
