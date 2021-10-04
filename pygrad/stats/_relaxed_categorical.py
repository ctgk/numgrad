from pygrad._core._array import Array
from pygrad._utils._typecheck import _typecheck
from pygrad.random._gumbel_softmax import gumbel_softmax
from pygrad.stats._categorical import Categorical


class RelaxedCategorical(Categorical):
    """Class for statistics of relaxed categorical distribution."""

    @_typecheck(exclude_args=('logits',))
    def __init__(self, logits: Array, temperature: float = 1e-3):
        """Initialize the statistics object.

        Parameters
        ----------
        logits : Array
            Logits parameter.
        temperature : float, optional
            Relaxation parameter, by default 1e-3
        """
        super().__init__(logits)
        self._temperature = temperature

    @property
    def temperature(self) -> float:
        """Return relaxation parameter.

        Returns
        -------
        float
            Relaxation parameter.
        """
        return self._temperature

    def sample(self) -> Array:
        """Return random sample.

        Returns
        -------
        Array
            Random sample.
        """
        return gumbel_softmax(self._logits, self._temperature)
