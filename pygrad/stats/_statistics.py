import abc

from pygrad._core._tensor import Tensor, TensorLike


class Statistics(abc.ABC):
    """Statistics of a probability distribution."""

    @abc.abstractmethod
    def logpdf(self, x: TensorLike) -> Tensor:
        """Return logarithm of probability density (mass) function.

        Parameters
        ----------
        x : TensorLike
            Observed data.

        Returns
        -------
        Tensor
            Logarithm of probability density (mass) function.
        """
        pass

    @abc.abstractmethod
    def sample(self) -> Tensor:
        """Return random sample according to the statistics.

        Returns
        -------
        Tensor
            Random sample.
        """
        pass
