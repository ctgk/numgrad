from pygrad._core._array import Array
from pygrad.random._categorical import categorical
from pygrad.stats._softmax_cross_entropy import softmax_cross_entropy
from pygrad.stats._statistics import Statistics


class Categorical(Statistics):
    r"""Statistics of categorical distribution.

    .. math::
        {\rm Cat}({\boldsymbol x}) = \prod_{k=0}^{K-1} \mu_k^{x_k}
    """

    def __init__(self, logits: Array):
        """Initialize statistics of categorical distribution.

        Parameters
        ----------
        logits : Array
            Logits parameter of categorical distribution.
        """
        super().__init__()
        self._logits = logits

    @property
    def logits(self) -> Array:
        """Return logits parameter of the categorical distribution.

        Returns
        -------
        Array
            Logits parameter of the categorical distribution.
        """
        return self._logits

    def logpdf(self, x):
        """Return logarithm of probability density (mass) function.

        Parameters
        ----------
        x : Array
            Observed data.

        Returns
        -------
        Array
            Logarithm of probability density (mass) function.
        """
        return -softmax_cross_entropy(x, self._logits)

    def sample(self):
        """Return random sample according to the statistics.

        Returns
        -------
        Array
            Random sample.
        """
        return categorical(self._logits)
