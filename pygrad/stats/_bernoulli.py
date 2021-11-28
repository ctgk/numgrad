from pygrad._core._differentiation_error import DifferentiationError
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad.stats._sigmoid_cross_entropy import sigmoid_cross_entropy
from pygrad.stats._statistics import Statistics


class Bernoulli(Statistics):
    """Statistics of bernoulli distribution.

    Examples
    --------
    >>> b = gd.stats.Bernoulli(logits=0)
    >>> b.logpdf(1)
    Tensor(-0.69314718)
    """

    @_typecheck()
    def __init__(self, logits: TensorLike):
        """Initialize the statistics.

        Parameters
        ----------
        logits : TensorLike
            Logits parameter.
        """
        super().__init__()
        self._logits = logits

    @property
    def logits(self) -> TensorLike:
        """Return logits parameter.

        Returns
        -------
        TensorLike
            Logits parameter.
        """
        return self._logits

    @_typecheck()
    def logpdf(self, x: TensorLike) -> Tensor:
        """Return logarithm of pdf given observed data.

        Parameters
        ----------
        x : TensorLike
            Observed data.

        Returns
        -------
        Tensor
            Logarithm of pdf.
        """
        return -sigmoid_cross_entropy(x, self._logits)

    def sample(self):  # noqa: D102
        raise DifferentiationError(
            'Sampling from bernoulli distribution is not differentiable. '
            'Please use `RelaxedBernoulli` if you want an approximation of '
            'differentiable sampling from bernoulli distribution.',
        )
