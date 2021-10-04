import typing as tp

from pygrad._utils._typecheck import _typecheck
from pygrad.distributions._distribution import Distribution
from pygrad.stats._exponential import Exponential as ExpStats


class Exponential(Distribution):
    r"""Exponential distribution.

    .. math::
        {\rm Exp}(x|\beta) = {1\over\beta}e^{-{x\over\beta}}
    """

    @_typecheck()
    def __init__(
        self,
        rv: str = 'x',
        name: str = 'Exp',
        *,
        conditions: tp.Union[tp.List[str], None] = None,
    ):
        """Initialize an exponential distribution.

        Parameters
        ----------
        rv : str, optional
            Name of the random variable, by default 'x'
        name : str, optional
            Name of the distribution, by default 'Exp'
        conditions : tp.Union[tp.List[str], None], optional
            Name of the conditional random variables, by default None
        """
        super().__init__(rv=rv, name=name, conditions=conditions)

    @staticmethod
    def forward() -> ExpStats:
        """Return statistics of the distribution.

        Returns
        -------
        ExpStats
            Statistics of the distribution.
        """
        return ExpStats(1)
