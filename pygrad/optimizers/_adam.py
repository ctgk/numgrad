import typing as tp

import numpy as np

from pygrad._core._array import Array
from pygrad.optimizers._gradient import Gradient
from pygrad._utils._typecheck import _typecheck


class Adam(Gradient):
    """Adam optimizer
    """

    @_typecheck(exclude=('parameters', 'learning_rate'))
    def __init__(
            self,
            parameters: tp.Iterable[Array],
            learning_rate: float = 0.001,
            beta1: float = 0.9,
            beta2: float = 0.999):
        """Construct Adam optimizer

        Parameters
        ----------
        parameters : tp.Iterable[Array]
            Parameters to be optimized
        learning_rate : float, optional
            Learning rate of this optimizer, by default 0.001
        beta1 : float, optional
            Exponential decay rate for the 1st moment estimates, by default 0.9
        beta2 : float, optional
            Exponential decay rate for the 2nd moment estimates,
            by default 0.999
        """
        super().__init__(parameters, learning_rate=learning_rate)
        self._check_in_range('beta1', beta1, 0, 1)
        self._check_in_range('beta2', beta2, 0, 1)
        self._beta1 = beta1
        self._beta2 = beta2
        self._moment1 = [np.zeros_like(p._data) for p in parameters]
        self._moment2 = [np.zeros_like(p._data) for p in parameters]

    def _update(self, learning_rate: float):
        alpha = (
            learning_rate
            * (1 - self._beta2 ** self._n_iter) ** 2
            / (1 - self._beta1 ** self._n_iter))
        for param, m1, m2 in zip(
                self._parameters, self._moment1, self._moment2):
            g = param.grad
            m1 += (1 - self._beta1) * (g - m1)
            m2 += (1 - self._beta2) * (g ** 2 - m2)
            param._data += alpha * m1 / (np.sqrt(m2) + np.finfo(g.dtype).eps)
