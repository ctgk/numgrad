from pygrad.nn._dense import Dense

from pygrad.nn._conv2d import conv2d, Conv2D
from pygrad.nn._conv2d_transpose import conv2d_transpose, Conv2DTranspose
from pygrad.nn._dropout import dropout
from pygrad.nn._leaky_relu import leaky_relu
from pygrad.nn._maxpool2d import max_pool2d, MaxPool2D
from pygrad.nn._relu import relu


_classes = [
    Conv2D,
    Conv2DTranspose,
    Dense,
    MaxPool2D,
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
]
