import typing as tp

import numpy as np

from pygrad._array import Array
from pygrad.optimizers._gradient import Gradient
from pygrad._type_check import _typecheck_args


class RMSProp(Gradient):

    @_typecheck_args
    def __init__(
            self,
            parameters: tp.Iterable[Array],
            learning_rate: float = 0.001,
            rho: float = 0.9):
        super().__init__(parameters, learning_rate=learning_rate)
        self._check_in_range('rho', rho, 0, 1)
        self._rho = rho
        self._mean_squared_grad = [np.zeros_like(p.value) for p in parameters]

    def _update(self, learning_rate: float):
        for param, msg in zip(self._parameters, self._mean_squared_grad):
            g = param.grad
            msg += (1 - self._rho) * (g ** 2 - msg)
            param._value += learning_rate * g / (
                np.sqrt(msg) + np.finfo(g.dtype).eps)
