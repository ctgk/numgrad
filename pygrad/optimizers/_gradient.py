import typing as tp

from pygrad._core._module import Module
from pygrad._core._tensor import Tensor
from pygrad._utils._typecheck import _typecheck
from pygrad.optimizers._optimizer import Optimizer


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
    >>> theta = gd.Tensor(10., is_variable=True)
    >>> optimizer = gd.optimizers.Gradient([theta], 0.1)
    >>> loss = theta * 1
    >>> optimizer.minimize(loss)
    >>> theta
    Tensor(9.9)
    >>> theta.clear(); loss = theta * 1
    >>> optimizer.minimize(loss)
    >>> theta
    Tensor(9.8)
    >>> theta.clear(); loss = theta * 1
    >>> optimizer.minimize(loss)
    >>> theta
    Tensor(9.7)
    >>> theta.clear(); loss = theta * 1
    >>> optimizer.maximize(loss)
    >>> theta
    Tensor(9.8)
    """

    @_typecheck()
    def __init__(
        self,
        parameters: tp.Union[Module, tp.Tuple[Tensor], tp.List[Tensor]],
        learning_rate: float = 1e-3,
    ):
        """Initialize gradient optimizer.

        Parameters
        ----------
        parameters : tp.Union[Module, tp.Tuple[Tensor], tp.List[Tensor]]
            Parameter to optimize.
        learning_rate : float, optional
            Learning rate of the update, by default 1e-3
        """
        super().__init__(parameters)
        self.learning_rate = learning_rate

    @property
    def learning_rate(self) -> float:
        """Return learning rate.

        Returns
        -------
        float
            Learning rate.
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        self._check_in_range('learning_rate', value, 0, None)
        self._learning_rate = value

    def _update(self, learning_rate: float):
        for p in self._parameters:
            p._data += learning_rate * p.grad

    def _minimize(self, loss: Tensor):
        self._increment_count_calc_grad(loss)
        self._update(-self._learning_rate)

    def _maximize(self, score: Tensor):
        self._increment_count_calc_grad(score)
        self._update(self._learning_rate)
