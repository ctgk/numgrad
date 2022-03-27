from pygrad._core._differentiation_error import DifferentiationError
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad.distributions._distribution import BasicDistribution
from pygrad.stats._sigmoid import sigmoid
from pygrad.stats._sigmoid_cross_entropy import sigmoid_cross_entropy


class Bernoulli(BasicDistribution):
    r"""Bernoulli distribution.

    .. math::
        {\rm Bern}(x; \mu) = \mu^x(1-\mu)^{(1-x)}

    Examples
    --------
    >>> np.random.seed(111)
    >>> b = gd.distributions.Bernoulli(logits=0, notation='Bern(x)')
    >>> b
    Bern(x)
    >>> b.logp(1)
    Tensor(-0.69314718)
    >>> b.sample()
    Traceback (most recent call last):
    ...
    pygrad.DifferentiationError: ...
    """

    @_typecheck()
    def __init__(
        self,
        logits: TensorLike,
        *,
        notation: str = 'Bern(x)',
    ):
        """Initialize Bernoulli distribution.

        Parameters
        ----------
        logits : TensorLike
            Log probability(s) parameter of the distribution.
        notation : str, optional
            Notation of the distribution, by default 'Bern(x)'
        """
        super().__init__(notation=notation)
        self._logits = logits

    @property
    def logits(self) -> TensorLike:
        """Return log probability(s) parameter of the distribution.

        Returns
        -------
        TensorLike
            Log probability(s) parameter of the distribution.
        """
        return self._logits

    def _entropy(self) -> Tensor:
        mu = sigmoid(self._logits)
        return sigmoid_cross_entropy(mu, self._logits)

    def _logp(self, observed) -> Tensor:
        return -sigmoid_cross_entropy(observed, self._logits)

    def _sample(self) -> Tensor:
        raise DifferentiationError(
            'Sampling from bernoulli distribution is not differentiable. '
            'Please use `RelaxedBernoulli` if you want an approximation of '
            'differentiable sampling from bernoulli distribution.',
        )
