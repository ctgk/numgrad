import abc

from pygrad._core._array import Array


class Statistics(abc.ABC):
    """Statistics of a probability distribution."""

    @abc.abstractmethod
    def logpdf(self, x: Array) -> Array:
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
        pass

    @abc.abstractmethod
    def sample(self) -> Array:
        """Return random sample according to the statistics.

        Returns
        -------
        Array
            Random sample.
        """
        pass
