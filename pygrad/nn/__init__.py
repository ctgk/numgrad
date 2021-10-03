from pygrad.nn._conv2d import conv2d, Conv2D
from pygrad.nn._conv2d_transpose import conv2d_transpose, Conv2DTranspose
from pygrad.nn._dense import Dense
from pygrad.nn._dropout import dropout, Dropout
from pygrad.nn._flatten import Flatten
from pygrad.nn._leaky_relu import leaky_relu, LeakyReLU
from pygrad.nn._maxpool2d import max_pool2d, MaxPool2D
from pygrad.nn._relu import relu, ReLU
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

for _cls in _classes:
    _cls.__module__ = 'pygrad.nn'


__all__ = [_cls.__name__ for _cls in _classes] + [
    'conv2d',
    'conv2d_transpose',
    'dropout',
    'leaky_relu',
    'max_pool2d',
    'relu',
    'softplus',
]
