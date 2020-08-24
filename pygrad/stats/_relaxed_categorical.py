from pygrad._core._array import Array
from pygrad._utils._typecheck import _typecheck
from pygrad.random._gumbel_softmax import gumbel_softmax
from pygrad.stats._distribution import Distribution
from pygrad.stats._softmax_cross_entropy import softmax_cross_entropy


class RelaxedCategorical(Distribution):
    """Relaxed categorical distribution

    Examples
    --------
    >>> import pygrad as gd; import numpy as np; np.random.seed(1)
    >>> c = gd.stats.RelaxedCategorical(5)
    >>> c
    Cat(x)
    >>> c.logpdf([1, 0, 0, 0, 0])
    array(-1.60943791)
    >>> c.sample()['x']
    array([0.00000000e+000, 0.00000000e+000, 1.00000000e+000, 0.00000000e+000,
           5.95945705e-315])
    >>> c.sample()['x']
    array([1.00000000e+00, 1.63781861e-33, 7.77986202e-65, 1.80763229e-72,
           5.66643102e-91])
    """

    @_typecheck()
    def __init__(
            self,
            n_classes: int = None,
            temperature: float = 1e-2,
            rv: str = 'x',
            name: str = 'Cat'):
        super().__init__(rv=rv, name=name)
        self._n_classes = n_classes
        self._temperature = temperature

    def forward(self):
        return {'logits': [0] * self._n_classes}

    def _logpdf(self, x: Array, logits: Array) -> Array:
        return -softmax_cross_entropy(x, logits)

    def _sample(self, logits: Array) -> Array:
        return gumbel_softmax(logits, temperature=self._temperature)
