import typing as tp

from pygrad._core._array import Array
from pygrad._core._graph import Graph
from pygrad._core._module import Module
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
    >>> import pygrad as gd
    >>> theta = gd.Array(10., is_variable=True)
    >>> with gd.Graph() as g:
    ...     loss = theta * 1
    >>> optimizer = gd.optimizers.Gradient([theta], 0.1)
    >>> g.forward(); optimizer.minimize(g)
    >>> theta
    array(9.9)
    >>> g.forward(); optimizer.minimize(g)
    >>> theta
    array(9.8)
    >>> g.forward(); optimizer.minimize(g)
    >>> theta
    array(9.7)
    >>> g.forward(); optimizer.maximize(g)
    >>> theta
    array(9.8)
    """

    @_typecheck()
    def __init__(
            self,
            parameters: tp.Union[Module, tp.Iterable[Array]],
            learning_rate: float = 1e-3):
        super().__init__(parameters)
        self.learning_rate = learning_rate

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        self._check_in_range('learning_rate', value, 0, None)
        self._learning_rate = value

    def _update(self, learning_rate: float):
        for p in self._parameters:
            p._data += learning_rate * p.grad

    @_typecheck()
    def minimize(self, graph: Graph = None, clear_grad: bool = True):
        """Small updation of each parameter to minimize the given loss.

        Parameters
        ----------
        graph : Graph
            Graph whose terminal node is loss value to minimize,
            default is None which assumes that backward() method of loss value
            has been already called.
        clear_grad : bool
            Clear gradient of parameters after updation if True,
            default is True
        """
        with self._increment_count_calc_grad_clear(graph, clear_grad):
            self._update(-self._learning_rate)

    @_typecheck()
    def maximize(self, graph: Graph = None, clear_grad: bool = True):
        """Small updation of each parameter to maximize the given score.

        Parameters
        ----------
        graph : Graph
            Graph whose terminal node is score value to maximize,
            default is None which assumes that backward() method of score value
            has been already called.
        clear_grad : bool
            Clear gradient of parameters after updation if True,
            default is True
        """
        with self._increment_count_calc_grad_clear(graph, clear_grad):
            self._update(self._learning_rate)
