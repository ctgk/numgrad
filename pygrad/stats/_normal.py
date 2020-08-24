import typing as tp

import numpy as np

from pygrad._core._array import Array
from pygrad._math._log import log
from pygrad._math._square import square
from pygrad._utils._typecheck import _typecheck
from pygrad.random._normal import normal
from pygrad.stats._distribution import Distribution


_ln2pi_half = 0.5 * np.log(2 * np.pi)


class Normal(Distribution):
    """Normal distribution aka Gaussian distribution

    Examples
    --------
    >>> import pygrad as gd; import numpy as np; np.random.seed(0)
    >>> n = gd.stats.Normal(rv='x')
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
            size: tp.Union[int, tp.Iterable[int], None] = None):
        super().__init__(rv=rv, name=name)
        self._size = size

    @staticmethod
    def forward():
        return {'loc': 0, 'scale': 1}

    def _logpdf(self, x: Array, loc: Array, scale: Array) -> Array:
        return -0.5 * (square((x - loc) / scale)) - log(scale) - _ln2pi_half

    def _sample(self, loc: Array, scale: Array) -> Array:
        return normal(loc, scale, size=self._size)
