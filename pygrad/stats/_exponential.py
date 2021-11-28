from pygrad._core._tensor import Tensor, TensorLike
from pygrad._math._log import log
from pygrad.random._exponential import exponential
from pygrad.stats._statistics import Statistics


class Exponential(Statistics):
    r"""Statistics of exponential distribution.

    .. math::
        p(x|\beta) = {1\over\beta}e^{-{x\over\beta}}
    """

    def __init__(self, scale: TensorLike):
        """Initialize statistics of the distribution.

        Parameters
        ----------
        scale : TensorLike
            Scale parameter.
        """
        super().__init__()
        self._scale = scale

    @property
    def scale(self) -> Tensor:
        """Return scale parameter.

        Returns
        -------
        TensorLike
            Scale parameter.
        """
        return self._scale

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
        return -x / self._scale - log(self._scale)

    def sample(self) -> Tensor:
        """Return random sample according to the statistics.

        Returns
        -------
        Tensor
            Random sample.
        """
        return exponential(self._scale)
