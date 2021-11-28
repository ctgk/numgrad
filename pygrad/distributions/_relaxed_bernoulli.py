from pygrad._utils._typecheck import _typecheck
from pygrad.distributions._bernoulli import Bernoulli
from pygrad.stats._relaxed_bernoulli import (
    RelaxedBernoulli as RelaxedBernoulliStats)


class RelaxedBernoulli(Bernoulli):
    """Relaxed bernoulli distribution.

    Examples
    --------
    >>> np.random.seed(0)
    >>> b = gd.distributions.RelaxedBernoulli()
    >>> b
    Bern(x)
    >>> b.logpdf(1)
    Tensor(-0.69314718)
    >>> b.sample()['x']
    Tensor(1.)
    >>> b.sample()['x']
    Tensor(1.1964751e-07)
    """

    @_typecheck()
    def __init__(
        self,
        temperature: float = 1e-2,
        rv: str = 'x',
        name: str = 'Bern',
    ):
        """Initialize relaxed Bernoulli distribution.

        Parameters
        ----------
        temperature : float, optional
            Temperature parameter of relaxation, by default 1e-2
        rv : str, optional
            Name of the random parameter, by default 'x'
        name : str, optional
            Name of the distribution, by default 'Bern'
        """
        super().__init__(rv=rv, name=name)
        self._temperature = temperature

    def forward(self) -> RelaxedBernoulliStats:
        """Return statistics of the distribution.

        Returns
        -------
        RelaxedBernoulliStats
            Statistics of the distribution.
        """
        return RelaxedBernoulliStats(logits=0, temperature=self._temperature)
