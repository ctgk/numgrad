from pygrad._core._array import Array
from pygrad._utils._typecheck import _typecheck
from pygrad.random._gumbel_sigmoid import gumbel_sigmoid
from pygrad.stats._bernoulli import Bernoulli


class RelaxedBernoulli(Bernoulli):
    """Relaxed bernoulli distribution

    Examples
    --------
    >>> import pygrad as gd; import numpy as np; np.random.seed(0)
    >>> b = gd.stats.RelaxedBernoulli()
    >>> b
    Bern(x)
    >>> b.logpdf(1)
    array(-0.69314718)
    >>> b.sample()['x']
    array(1.)
    >>> b.sample()['x']
    array(1.1964751e-07)
    """

    @_typecheck()
    def __init__(
            self,
            temperature: float = 1e-2,
            rv: str = 'x',
            name: str = 'Bern'):
        super().__init__(rv=rv, name=name)
        self._temperature = temperature

    def _sample(self, logits: Array) -> Array:
        return gumbel_sigmoid(logits, temperature=self._temperature)
