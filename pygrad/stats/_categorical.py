from pygrad._core._array import Array
from pygrad.random._categorical import categorical
from pygrad.stats._statistics import Statistics
from pygrad.stats._softmax_cross_entropy import softmax_cross_entropy


class Categorical(Statistics):
    r"""Statistics of categorical distribution

    .. math::
        {\rm Cat}({\boldsymbol x}) = \prod_{k=0}^{K-1} \mu_k^{x_k}
    """

    def __init__(self, logits: Array):
        super().__init__()
        self._logits = logits

    @property
    def logits(self) -> Array:
        return self._logits

    def logpdf(self, x):
        return -softmax_cross_entropy(x, self._logits)

    def sample(self):
        return categorical(self._logits)
