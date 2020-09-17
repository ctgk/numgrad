import typing as tp

from pygrad._utils._typecheck import _typecheck
from pygrad.distributions._distribution import Distribution
from pygrad.stats._normal import Normal as NormalStats


class Normal(Distribution):
    """Normal distribution aka Gaussian distribution

    Examples
    --------
    >>> import pygrad as gd; import numpy as np; np.random.seed(0)
    >>> n = gd.distributions.Normal(rv='x')
    >>> n
    N(x)
    >>> n.logpdf(1)
    array(-1.41893853)
    >>> n.sample()['x']
    array(1.76405235)
    >>> n.sample()['x']
    array(0.40015721)
    """

    @_typecheck()
    def __init__(
            self,
            rv: str = 'x',
            name: str = 'N',
            *,
            conditions: tp.Union[tp.List[str], None] = None):
        super().__init__(rv=rv, name=name, conditions=conditions)

    @staticmethod
    def forward() -> NormalStats:
        return NormalStats(0, 1)
