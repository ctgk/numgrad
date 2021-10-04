from pygrad._utils._typecheck import _typecheck
from pygrad.distributions._distribution import Distribution
from pygrad.stats._categorical import Categorical as CategoricalStats


class Categorical(Distribution):
    r"""Categorical distribution class.

    .. math::
        {\rm Cat}({\boldsymbol x}) = \prod_{k=0}^{K-1} \mu_k^{x_k}
    """

    @_typecheck()
    def __init__(
        self,
        n_classes: int = None,
        rv: str = 'x',
        name: str = 'Cat',
    ):
        """Initialize categorical distribution.

        Parameters
        ----------
        n_classes : int, optional
            Number of classes, by default None
        rv : str, optional
            Name of random variable, by default 'x'
        name : str, optional
            Name of the distribution, by default 'Cat'
        """
        super().__init__(rv=rv, name=name)
        self._n_classes = n_classes

    def forward(self) -> CategoricalStats:
        """Return statistics of the distribution.

        Returns
        -------
        CategoricalStats
            Statistics of the distribution.
        """
        return CategoricalStats(logits=[0] * self._n_classes)
