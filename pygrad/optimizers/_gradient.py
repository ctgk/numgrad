import typing as tp

from pygrad._core._array import Array
from pygrad.optimizers._optimizer import Optimizer
from pygrad._utils._typecheck import _typecheck


class Gradient(Optimizer):
    r"""Gradient descent (ascent) algorithm.

    Given learning rate :math:`\alpha` and value to optimize
    :math:`\mathcal{L}`, this algorithm updates each parameter :math:`\theta`
    using the following rule:

    .. math::
        \theta \leftarrow
            \theta + \alpha {\partial\mathcal{L}\over\partial\theta}

    Examples
    --------
    >>> import pygrad as pg
    >>> theta = pg.Array(10., is_variable=True)
    >>> optimizer = pg.optimizers.Gradient([theta], 0.1)
    >>> optimizer.minimize(theta)
    >>> theta
    array(9.9)
    >>> optimizer.minimize(theta)
    >>> theta
    array(9.8)
    >>> optimizer.minimize(theta)
    >>> theta
    array(9.7)
    >>> optimizer.maximize(theta)
    >>> theta
    array(9.8)
    """

    @_typecheck()
    def __init__(
            self,
            parameters: tp.Iterable[Array],
            learning_rate: float = 1e-3):
        super().__init__(parameters)
        self._check_in_range('learning_rate', learning_rate, 0, None)
        self._learning_rate = learning_rate

    def _update(self, learning_rate: float):
        for p in self._parameters:
            p._data += learning_rate * p.grad

    @_typecheck()
    def minimize(self, loss: Array = None, clear_grad: bool = True):
        """Small updation of each parameter to minimize the given loss.

        Parameters
        ----------
        loss : Array
            Loss value to minimize, default is None which assumes that
            backward() method of loss value has been already called.
        clear_grad : bool
            Clear gradient of parameters after updation if True,
            default is True
        """
        with self._increment_count_calc_grad_clear_grad(loss, clear_grad):
            self._update(-self._learning_rate)

    @_typecheck()
    def maximize(self, score: Array = None, clear_grad: bool = True):
        """Small updation of each parameter to maximize the given score.

        Parameters
        ----------
        score : Array
            Score value to maximize, default is None which assumes that
            backward() method of score value has been already called.
        clear_grad : bool
            Clear gradient of parameters after updation if True,
            default is True
        """
        with self._increment_count_calc_grad_clear_grad(score, clear_grad):
            self._update(self._learning_rate)
