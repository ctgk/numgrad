"""Neural network module."""

from pygrad.nn._conv2d import Conv2D, conv2d
from pygrad.nn._conv2d_transpose import conv2d_transpose, Conv2DTranspose
from pygrad.nn._dense import Dense
from pygrad.nn._dropout import Dropout, dropout
from pygrad.nn._flatten import Flatten
from pygrad.nn._leaky_relu import leaky_relu, LeakyReLU
from pygrad.nn._max_pool2d import max_pool2d, MaxPool2D
from pygrad.nn._relu import ReLU, relu
from pygrad.nn._reshape import Reshape
from pygrad.nn._sequential import Sequential
from pygrad.nn._softplus import softplus


_classes = [
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    LeakyReLU,
    MaxPool2D,
    ReLU,
    Reshape,
    Sequential,
]


_functions = [
    conv2d,
    conv2d_transpose,
    dropout,
    leaky_relu,
    max_pool2d,
    relu,
    softplus,
]

for _cls in _classes:
    _cls.__module__ = __name__


__all__ = (
    [_cls.__name__ for _cls in _classes]
    + [_func.__name__ for _func in _functions]
)


del _cls
del _classes
del _functions
