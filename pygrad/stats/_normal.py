import numpy as np

from pygrad._core._array import Array
from pygrad._math._log import log
from pygrad._math._square import square
from pygrad.random._normal import normal
from pygrad.stats._statistics import Statistics


_ln2pi_hf = 0.5 * np.log(2 * np.pi)


class Normal(Statistics):
    """Statistics of a normal distribution
    """

    def __init__(self, loc: Array, scale: Array):
        super().__init__()
        self._loc = loc
        self._scale = scale

    @property
    def loc(self) -> Array:
        return self._loc

    @property
    def scale(self) -> Array:
        return self._scale

    def logpdf(self, x) -> Array:
        return (
            -0.5 * (square((x - self._loc) / self._scale))
            - log(self._scale) - _ln2pi_hf)

    def sample(self) -> Array:
        return normal(self._loc, self._scale)
