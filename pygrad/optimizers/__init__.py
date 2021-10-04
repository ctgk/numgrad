"""Optimizers module."""

from pygrad.optimizers._adam import Adam
from pygrad.optimizers._gradient import Gradient
from pygrad.optimizers._momentum import Momentum
from pygrad.optimizers._optimizer import Optimizer
from pygrad.optimizers._rmsprop import RMSProp

_classes = [
    Adam,
    Gradient,
    Momentum,
    Optimizer,
    RMSProp,
]


for _cls in _classes:
    _cls.__module__ = 'pygrad.optimizers'


__all__ = [_cls.__name__ for _cls in _classes]
