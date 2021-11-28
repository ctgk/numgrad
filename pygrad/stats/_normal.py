import numpy as np

from pygrad._core._tensor import Tensor, TensorLike
from pygrad._math._log import log
from pygrad._math._square import square
from pygrad.random._normal import normal
from pygrad.stats._statistics import Statistics


_ln2pi_hf = 0.5 * np.log(2 * np.pi)


class Normal(Statistics):
    """Statistics of a normal distribution."""

    def __init__(self, loc: TensorLike, scale: TensorLike):
        """Initialize the statistics object.

        Parameters
        ----------
        loc : TensorLike
            Location parameter.
        scale : TensorLike
            Scale parameter.
        """
        super().__init__()
        self._loc = loc
        self._scale = scale

    @property
    def loc(self) -> TensorLike:
        """Return location parameter of the statistics.

        Returns
        -------
        TensorLike
            Location parameter.
        """
        return self._loc

    @property
    def scale(self) -> TensorLike:
        """Return scale parameter of the statistics.

        Returns
        -------
        TensorLike
            Scale parameter.
        """
        return self._scale

    def logpdf(self, x: TensorLike) -> Tensor:
        """Return logarithm of pdf.

        Parameters
        ----------
        x : TensorLike
            Observed data

        Returns
        -------
        Tensor
            Logarithm of pdf.
        """
        return (
            -0.5 * (square((x - self._loc) / self._scale))
            - log(self._scale) - _ln2pi_hf)

    def sample(self) -> Tensor:
        """Return random sample.

        Returns
        -------
        Tensor
            Random sample.
        """
        return normal(self._loc, self._scale)
