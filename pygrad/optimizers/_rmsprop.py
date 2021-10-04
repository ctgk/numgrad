import typing as tp

import numpy as np

from pygrad._core._array import Array
from pygrad._core._module import Module
from pygrad._utils._typecheck import _typecheck
from pygrad.optimizers._gradient import Gradient


class RMSProp(Gradient):
    """RMSProp optimizer."""

    @_typecheck(exclude_args=('parameters', 'learning_rate'))
    def __init__(
        self,
        parameters: tp.Union[Module, tp.Iterable[Array]],
        learning_rate: float = 0.001,
        rho: float = 0.9,
    ):
        """Initialize RMSProp optimizer.

        Parameters
        ----------
        parameters : tp.Union[Module, tp.Iterable[Array]]
            Parameters to optimize
        learning_rate : float, optional
            Learning rate of update, by default 0.001
        rho : float, optional
            Parameter of update, by default 0.9
        """
        super().__init__(parameters, learning_rate=learning_rate)
        self._check_in_range('rho', rho, 0, 1)
        self._rho = rho
        self._mean_squared_grad = [np.zeros_like(p._data) for p in parameters]

    def _update(self, learning_rate: float):
        for param, msg in zip(self._parameters, self._mean_squared_grad):
            g = param.grad
            msg += (1 - self._rho) * (g ** 2 - msg)
            param._data += learning_rate * g / (
                np.sqrt(msg) + np.finfo(g.dtype).eps)
