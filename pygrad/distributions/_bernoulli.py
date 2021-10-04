from pygrad._utils._typecheck import _typecheck
from pygrad.distributions._distribution import Distribution
from pygrad.stats._bernoulli import Bernoulli as BernoulliStats


class Bernoulli(Distribution):
    """Bernoulli distribution.

    Examples
    --------
    >>> import pygrad as gd; import numpy as np; np.random.seed(111)
    >>> b = gd.distributions.Bernoulli()
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
        name: str = 'Bern',
    ):
        """Initialize Bernoulli distribution.

        Parameters
        ----------
        rv : str, optional
            Name of the random variable, by default 'x'
        name : str, optional
            Name of the distribution, by default 'Bern'
        """
        super().__init__(rv=rv, name=name)

    @staticmethod
    def forward() -> BernoulliStats:
        """Return statistics of Bernoulli distribution.

        Returns
        -------
        BernoulliStats
            Statistics of Bernoulli distribution.
        """
        return BernoulliStats(logits=0)
