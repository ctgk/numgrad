from pygrad._utils._typecheck import _typecheck
from pygrad.distributions._bernoulli import Bernoulli
from pygrad.stats._relaxed_bernoulli import (
    RelaxedBernoulli as RelaxedBernoulliStats)


class RelaxedBernoulli(Bernoulli):
    """Relaxed bernoulli distribution

    Examples
    --------
    >>> import pygrad as gd; import numpy as np; np.random.seed(0)
    >>> b = gd.distributions.RelaxedBernoulli()
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

    def forward(self) -> RelaxedBernoulliStats:
        return RelaxedBernoulliStats(logits=0, temperature=self._temperature)
