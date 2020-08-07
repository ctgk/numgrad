import typing as tp

import numpy as np

from pygrad._array import Array
from pygrad.optimizers._gradient import Gradient
from pygrad._type_check import _typecheck_args


class Momentum(Gradient):

    @_typecheck_args
    def __init__(
            self,
            parameters: tp.Iterable[Array],
            learning_rate: float = 0.001,
            momentum: float = 0.9):
        super().__init__(parameters, learning_rate=learning_rate)
        self._check_in_range('momentum', momentum, 0, 1)
        self._momentum = momentum
        self._inertia = [np.zeros_like(p.value) for p in parameters]

    def _update(self, learning_rate: float):
        for param, inertia in zip(self._parameters, self._inertia):
            inertia += (1 - self._momentum) * (
                learning_rate * param.grad - inertia)
            param._value += inertia
