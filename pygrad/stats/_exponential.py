from pygrad._core._array import Array
from pygrad._math._log import log
from pygrad.random._exponential import exponential
from pygrad.stats._statistics import Statistics


class Exponential(Statistics):
    r"""Statistics of exponential distribution

    .. math::
        p(x|\beta) = {1\over\beta}e^{-{x\over\beta}}
    """

    def __init__(self, scale: Array):
        super().__init__()
        self._scale = scale

    @property
    def scale(self) -> Array:
        return self._scale

    def logpdf(self, x):
        return -x / self._scale - log(self._scale)

    def sample(self):
        return exponential(self._scale)
