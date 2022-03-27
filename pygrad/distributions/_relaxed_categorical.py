from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad.distributions._categorical import Categorical
from pygrad.random._gumbel_softmax import gumbel_softmax


class RelaxedCategorical(Categorical):
    """Relaxed categorical distribution.

    Examples
    --------
    >>> c = gd.distributions.RelaxedCategorical(logits=[0] * 5)
    >>> c
    Cat(x)
    >>> c.logp([1, 0, 0, 0, 0])
    Tensor(-1.60943791)
    >>> c.sample()  # doctest: +SKIP
    Tensor([0., 0., 1., 0., 0.])
    >>> c.sample()  # doctest: +SKIP
    Tensor([1., 0., 0., 0., 0.])
    """

    @_typecheck()
    def __init__(
        self,
        logits: TensorLike,
        temperature: float = 1e-2,
        *,
        notation: str = 'Cat(x)',
    ):
        """Initialize relaxed categorical distribution.

        Parameters
        ----------
        logits : TensorLike
            Log probabilities.
        temperature : float, optional
            Temperature parameter of the relaxation, by default 1e-2
        notation : str, optional
            Notation of the distribution, by default 'Cat(x)'
        """
        super().__init__(logits=logits, notation=notation)
        self._temperature = temperature

    def _sample(self) -> Tensor:
        return gumbel_softmax(self._logits, self._temperature)
