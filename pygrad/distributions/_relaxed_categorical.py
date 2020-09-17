from pygrad._core._array import Array
from pygrad._utils._typecheck import _typecheck
from pygrad.distributions._categorical import Categorical
from pygrad.random._gumbel_softmax import gumbel_softmax


class RelaxedCategorical(Categorical):
    """Relaxed categorical distribution

    Examples
    --------
    >>> import pygrad as gd; import numpy as np; np.random.seed(1)
    >>> c = gd.distributions.RelaxedCategorical(5)
    >>> c
    Cat(x)
    >>> c.logpdf([1, 0, 0, 0, 0])
    array(-1.60943791)
    >>> c.sample()['x']
    array([0., 0., 1., 0., 0.])
    >>> c.sample()['x']
    array([1., 0., 0., 0., 0.])
    """

    @_typecheck()
    def __init__(
            self,
            n_classes: int = None,
            temperature: float = 1e-2,
            rv: str = 'x',
            name: str = 'Cat'):
        super().__init__(n_classes, rv=rv, name=name)
        self._temperature = temperature

    def _sample(self, logits: Array) -> Array:
        return gumbel_softmax(logits, temperature=self._temperature)
