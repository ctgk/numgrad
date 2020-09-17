from pygrad._core._array import Array
from pygrad._utils._typecheck import _typecheck
from pygrad.random._gumbel_sigmoid import gumbel_sigmoid
from pygrad.stats._bernoulli import Bernoulli


class RelaxedBernoulli(Bernoulli):

    @_typecheck(exclude_args=('logits',))
    def __init__(self, logits: Array, temperature: float = 1e-3) -> Array:
        super().__init__(logits)
        self._temperature = temperature

    @property
    def temperature(self) -> float:
        return self._temperature

    def sample(self) -> Array:
        return gumbel_sigmoid(self._logits, self._temperature)
