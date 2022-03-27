from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad.distributions._bernoulli import Bernoulli
from pygrad.random._gumbel_sigmoid import gumbel_sigmoid


class RelaxedBernoulli(Bernoulli):
    """Relaxed bernoulli distribution.

    Examples
    --------
    >>> b = gd.distributions.RelaxedBernoulli(logits=0, temperature=1e-2)
    >>> b
    Bern(x)
    >>> b.logp(1)
    Tensor(-0.69314718)
    >>> b.sample()  # doctest: +SKIP
    Tensor(1.)
    >>> b.sample()  # doctest: +SKIP
    Tensor(1.1964751e-07)
    """

    @_typecheck()
    def __init__(
        self,
        logits: TensorLike,
        temperature: float = 1e-2,
        *,
        notation: str = 'Bern(x)',
    ):
        """Initialize relaxed Bernoulli distribution.

        Parameters
        ----------
        logits : TensorLike
            Log probability(s).
        temperature : float, optional
            Temperature parameter of the relaxation, by default 1e-2
        notation : str, optional
            Notation of the distribution, by default 'Bern(x)'
        """
        super().__init__(logits=logits, notation=notation)
        self._temperature = temperature

    def _sample(self) -> Tensor:
        return gumbel_sigmoid(self._logits, self._temperature)
