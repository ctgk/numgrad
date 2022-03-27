import numpy as np

from pygrad._core._tensor import Tensor, TensorLike
from pygrad._math._log import log
from pygrad._math._square import square
from pygrad._utils._typecheck import _typecheck
from pygrad.distributions._distribution import BasicDistribution
from pygrad.random._normal import normal


_ln2pi_half = 0.5 * np.log(2 * np.pi)


class Normal(BasicDistribution):
    r"""Normal distribution aka Gaussian distribution.

    .. math::
        {\mathcal N}(x; \mu, \sigma^2) &= {1\over\sqrt{2\pi\sigma^2}}
            {\rm exp}\left\{-{1\over2\sigma^2}(x-\mu)^2\right\}\\
        H[x] &= {1\over2}\ln\sigma^2 + {1\over2}(1+\ln2\pi)

    Examples
    --------
    >>> n = gd.distributions.Normal(loc=0, scale=1, notation='N(x)')
    >>> n
    N(x)
    >>> n.logp(1)
    Tensor(-1.41893853)
    """

    @_typecheck()
    def __init__(
        self,
        loc: TensorLike,
        scale: TensorLike,
        *,
        notation: str = 'N(x)',
    ):
        """Initialize normal distribution.

        Parameters
        ----------
        loc : TensorLike
            Location parameter.
        scale : TensorLike
            Scale parameter.
        notation : str, optional
            Notation of the distribution, by default 'N(x)'
        """
        super().__init__(notation=notation)
        self._loc = loc
        self._scale = scale

    @property
    def loc(self) -> TensorLike:
        """Return location parameter of the normal distribution.

        Returns
        -------
        TensorLike
            Location parameter of the normal distribution.
        """
        return self._loc

    @property
    def scale(self) -> TensorLike:
        """Return scale parameter of the normal distribution.

        Returns
        -------
        TensorLike
            Scale parameter of the normal distribution.
        """
        return self._scale

    def _entropy(self) -> Tensor:
        return log(self._scale) + 0.5 + _ln2pi_half

    def _logp(self, observed: TensorLike) -> Tensor:
        return (
            -0.5 * (square((observed - self._loc) / self._scale))
            - log(self._scale) - _ln2pi_half
        )

    def _sample(self) -> Tensor:
        return normal(loc=self._loc, scale=self._scale)
