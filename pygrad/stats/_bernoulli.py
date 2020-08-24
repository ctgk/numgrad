from pygrad._core._array import Array
from pygrad._utils._typecheck import _typecheck
from pygrad.random._bernouilli import bernoulli
from pygrad.stats._distribution import Distribution
from pygrad.stats._sigmoid import sigmoid
from pygrad.stats._sigmoid_cross_entropy import sigmoid_cross_entropy


class Bernoulli(Distribution):
    """Bernoulli distribution

    Examples
    --------
    >>> import pygrad as gd; import numpy as np; np.random.seed(111)
    >>> b = gd.stats.Bernoulli()
    >>> b
    Bern(x)
    >>> b.logpdf(1)
    array(-0.69314718)
    >>> b.sample()['x']
    array(0)
    >>> b.sample()['x']
    array(1)
    """

    @_typecheck()
    def __init__(
            self,
            rv: str = 'x',
            name: str = 'Bern'):
        super().__init__(rv=rv, name=name)

    @staticmethod
    def forward():
        return {'logits': 0}

    def _logpdf(self, x: Array, logits: Array) -> Array:
        return -sigmoid_cross_entropy(x, logits)

    def _sample(self, logits: Array) -> Array:
        return bernoulli(sigmoid(logits))
