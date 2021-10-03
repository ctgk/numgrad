import abc

from pygrad._core._array import Array


class Statistics(abc.ABC):
    """Statistics of a probability distribution
    """

    @abc.abstractmethod
    def logpdf(self, x: Array) -> Array:
        pass

    @abc.abstractmethod
    def sample(self) -> Array:
        pass
