import typing as tp

import numpy as np

from pygrad._core._module import Module
from pygrad._core._tensor import Tensor
from pygrad._utils._typecheck import _typecheck
from pygrad.optimizers._gradient import Gradient


class Momentum(Gradient):
    """Momentum optimizer."""

    @_typecheck(exclude_args=('parameters', 'learning_rate'))
    def __init__(
        self,
        parameters: tp.Union[Module, tp.List[Tensor], tp.Tuple[Tensor]],
        learning_rate: float = 0.001,
        momentum: float = 0.9,
    ):
        """Initialize momentum optimizer.

        Parameters
        ----------
        parameters : tp.Union[Module, tp.List[Tensor], tp.Tuple[Tensor]]
            Parameters to optimize.
        learning_rate : float, optional
            Learning rate, by default 0.001
        momentum : float, optional
            Momentum of update, by default 0.9
        """
        super().__init__(parameters, learning_rate=learning_rate)
        self._check_in_range('momentum', momentum, 0, 1)
        self._momentum = momentum
        self._inertia = [np.zeros_like(p._data) for p in parameters]

    def _update(self, learning_rate: float):
        for param, inertia in zip(self._parameters, self._inertia):
            inertia += (
                (1 - self._momentum) * (learning_rate * param.grad - inertia)
            )
            param._data += inertia
