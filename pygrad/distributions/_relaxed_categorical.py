from pygrad._utils._typecheck import _typecheck
from pygrad.distributions._categorical import Categorical
from pygrad.stats._relaxed_categorical import (
    RelaxedCategorical as RelaxedCategoricalStats,
)


class RelaxedCategorical(Categorical):
    """Relaxed categorical distribution.

    Examples
    --------
    >>> np.random.seed(1)
    >>> c = gd.distributions.RelaxedCategorical(5)
    >>> c
    Cat(x)
    >>> c.logpdf([1, 0, 0, 0, 0])
    Tensor(-1.60943791)
    >>> c.sample()['x']
    Tensor([0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000,
            5.95945705e-315])
    >>> c.sample()['x']
    Tensor([1.00000000e+00, 1.63781861e-33, 7.77986202e-65, 1.80763229e-72,
            5.66643102e-91])
    """

    @_typecheck()
    def __init__(
        self,
        n_classes: int = None,
        temperature: float = 1e-2,
        rv: str = 'x',
        name: str = 'Cat',
    ):
        """Initialize the distribution.

        Parameters
        ----------
        n_classes : int, optional
            Number of the classes, by default None
        temperature : float, optional
            Relaxation parameter, by default 1e-2
        rv : str, optional
            Name of the random variable, by default 'x'
        name : str, optional
            Name of the distribution, by default 'Cat'
        """
        super().__init__(n_classes, rv=rv, name=name)
        self._temperature = temperature

    def forward(self) -> RelaxedCategoricalStats:
        """Return statistics of the distribution.

        Returns
        -------
        RelaxedCategoricalStats
            Statistics of the distribution.
        """
        return RelaxedCategoricalStats(
            logits=[0] * self._n_classes, temperature=self._temperature)
