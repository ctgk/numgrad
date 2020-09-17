from pygrad._core._array import Array
from pygrad.random._bernouilli import bernoulli
from pygrad.stats._statistics import Statistics
from pygrad.stats._sigmoid import sigmoid
from pygrad.stats._sigmoid_cross_entropy import sigmoid_cross_entropy


class Bernoulli(Statistics):
    """Statistics of bernoulli distribution

    Examples
    --------
    >>> import pygrad as gd; import numpy as np; np.random.seed(111)
    >>> b = gd.stats.Bernoulli(logits=0)
    >>> b.logpdf(1)
    array(-0.69314718)
    >>> b.sample()
    array(0)
    >>> b.sample()
    array(1)
    """

    def __init__(self, logits: Array):
        super().__init__()
        self._logits = logits

    @property
    def logits(self) -> Array:
        return self._logits

    def logpdf(self, x):
        return -sigmoid_cross_entropy(x, self._logits)

    def sample(self):
        return bernoulli(sigmoid(self._logits))
