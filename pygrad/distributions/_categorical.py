from pygrad._core._differentiation_error import DifferentiationError
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad.distributions._distribution import BasicDistribution
from pygrad.stats._softmax import softmax
from pygrad.stats._softmax_cross_entropy import softmax_cross_entropy


class Categorical(BasicDistribution):
    r"""Categorical distribution class.

    .. math::
        {\rm Cat}({\boldsymbol x}) = \prod_{k=0}^{K-1} \mu_k^{x_k}

    Examples
    --------
    >>> np.random.seed(1111)
    >>> p = gd.distributions.Categorical(logits=[0, 0, 0], notation='Cat(x)')
    >>> p
    Cat(x)
    >>> p.logp([1, 0, 0])
    Tensor(-1.09861229)
    """

    @_typecheck()
    def __init__(
        self,
        logits: TensorLike,
        *,
        notation: str = 'Cat(x)',
    ):
        """Initialize categorical distribution.

        Parameters
        ----------
        logits : TensorLike
            Log probability(s)
        notation : str, optional
            Notation of the distribution, by default 'Cat(x)'
        """
        super().__init__(notation=notation)
        self._logits = logits

    @property
    def logits(self) -> TensorLike:
        """Return log probability parameter of the categorical distribution.

        Returns
        -------
        TensorLike
            Log probability parameter of the categorical distribution.
        """
        return self._logits

    def _entropy(self) -> Tensor:
        mu = softmax(self._logits)
        return softmax_cross_entropy(mu, self._logits)

    def _logp(self, observed: TensorLike) -> Tensor:
        return -softmax_cross_entropy(observed, self._logits)

    def _sample(self) -> Tensor:
        raise DifferentiationError(
            'Sampling from categorical distribution is not differentiable. '
            'Please use `RelaxedCategorical` if you want an approximation '
            'of differentiable sampling from categorical distribution.',
        )
