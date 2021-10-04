from pygrad._core._array import Array
from pygrad._math._log import log
from pygrad.random._exponential import exponential
from pygrad.stats._statistics import Statistics


class Exponential(Statistics):
    r"""Statistics of exponential distribution.

    .. math::
        p(x|\beta) = {1\over\beta}e^{-{x\over\beta}}
    """

    def __init__(self, scale: Array):
        """Initialize statistics of the distribution.

        Parameters
        ----------
        scale : Array
            Scale parameter.
        """
        super().__init__()
        self._scale = scale

    @property
    def scale(self) -> Array:
        """Return scale parameter.

        Returns
        -------
        Array
            Scale parameter.
        """
        return self._scale

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
        return -x / self._scale - log(self._scale)

    def sample(self):
        """Return random sample according to the statistics.

        Returns
        -------
        Array
            Random sample.
        """
        return exponential(self._scale)
