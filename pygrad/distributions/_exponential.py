import typing as tp

from pygrad._core._tensor import Tensor, TensorLike
from pygrad._math._log import log
from pygrad._utils._typecheck import _typecheck
from pygrad.distributions._distribution import BasicDistribution
from pygrad.random._exponential import exponential


class Exponential(BasicDistribution):
    r"""Exponential distribution.

    .. math::
        \begin{cases}
        {\rm Exp}(x;\lambda) = \lambda e^{-\lambda x} & ({\rm rate}) \\
        {\rm Exp}(x;\beta) = {1\over\beta}e^{-{x\over\beta}} & ({\rm scale})
        \end{cases}

    Examples
    --------
    >>> p = gd.distributions.Exponential(scale=1.)
    >>> p
    Exp(x)
    >>> p.entropy()
    Tensor(1.)
    >>> p.logp(1)
    Tensor(-1.)
    >>> p.sample()  # doctest: +SKIP
    Tensor(0.10042737)
    """

    @_typecheck()
    def __init__(
        self,
        *,
        scale: tp.Optional[TensorLike] = None,
        rate: tp.Optional[TensorLike] = None,
        notation: str = 'Exp(x)',
    ):
        """Initialize an exponential distribution.

        Parameters
        ----------
        scale : tp.Optional[TensorLike]
            Scale parameter. Larger the value is, larger the value of sample
            from the distribution be. A user must pass only one of `scale` or
            `rate`.
        rate : tp.Optional[TensorLike]
            Rate parameter. Larger the value is, smaller the value of sample
            from the distribution be. A user must pass only one of `scale` or
            `rate`.
        notation : str, optional
            Notation of the distribution, by default 'Exp(x)'
        """
        super().__init__(notation=notation)
        if scale is None and rate is None:
            raise ValueError('Please pass either one of `scale` or `rate`.')
        if scale is not None and rate is not None:
            raise ValueError(
                'Please pass only one of `scale` or `rate`, not both of them.')
        self._scale = scale
        self._rate = rate

    def _entropy(self) -> Tensor:
        if self._scale is not None:
            return 1 + log(self._scale)
        return 1 - log(self._rate)

    def _logp(self, observed: TensorLike) -> Tensor:
        if self._scale is not None:
            return -observed / self._scale - log(self._scale)
        return -observed * self._rate + log(self._rate)

    def _sample(self) -> Tensor:
        if self._scale is not None:
            return exponential(self._scale)
        return exponential(1 / self._rate)
