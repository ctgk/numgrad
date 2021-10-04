from pygrad._core._array import Array
from pygrad._utils._typecheck import _typecheck
from pygrad.random._gumbel_sigmoid import gumbel_sigmoid
from pygrad.stats._bernoulli import Bernoulli


class RelaxedBernoulli(Bernoulli):
    """Class for statistics of relaxed Bernoulli distribution."""

    @_typecheck(exclude_args=('logits',))
    def __init__(self, logits: Array, temperature: float = 1e-3) -> None:
        """Initialize statistics of relaxed Bernoulli distribution.

        Parameters
        ----------
        logits : Array
            Logits parameter of the relaxed Bernoulli distribution.
        temperature : float, optional
            Temperature parameter of the relaxation, by default 1e-3
        """
        super().__init__(logits)
        self._temperature = temperature

    @property
    def temperature(self) -> float:
        """Return temperature parameter of the relaxed Bernoulli distribution.

        Returns
        -------
        float
            Temperature parameter of the relaxed Bernoulli distribution.
        """
        return self._temperature

    def sample(self) -> Array:
        """Return random sample from the relaxed Bernoulli distribution.

        Returns
        -------
        Array
            Random sample from the relaxed Bernoulli distribution.
        """
        return gumbel_sigmoid(self._logits, self._temperature)
