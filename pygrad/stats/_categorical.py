from pygrad._core._differentiation_error import DifferentiationError
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad.stats._softmax_cross_entropy import softmax_cross_entropy
from pygrad.stats._statistics import Statistics


class Categorical(Statistics):
    r"""Statistics of categorical distribution.

    .. math::
        {\rm Cat}({\boldsymbol x}) = \prod_{k=0}^{K-1} \mu_k^{x_k}
    """

    @_typecheck()
    def __init__(self, logits: TensorLike):
        """Initialize statistics of categorical distribution.

        Parameters
        ----------
        logits : TensorLike
            Logits parameter of categorical distribution.
        """
        super().__init__()
        self._logits = logits

    @property
    def logits(self) -> TensorLike:
        """Return logits parameter of the categorical distribution.

        Returns
        -------
        TensorLike
            Logits parameter of the categorical distribution.
        """
        return self._logits

    @_typecheck()
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
        return -softmax_cross_entropy(x, self._logits)

    def sample(self):  # noqa: D102
        raise DifferentiationError(
            'Sampling from categorical distribution is not differentiable. '
            'Please use `RelaxedCategorical` if you want an approximation '
            'of differentiable sampling from categorical distribution.',
        )
