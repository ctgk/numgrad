import numpy as np
import scipy.special as sp

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _GumbelSoftmax(_Operator):

    def __init__(
            self,
            logits,
            temperature: float,
            axis: int = -1,
            name: str = None):
        super().__init__(logits, name=name)
        self._temperature = temperature
        self._axis = axis

    def _forward_numpy(self, logits):
        g = np.random.gumbel(size=logits.shape).astype(logits.dtype)
        self.output = sp.softmax(
            (logits + g) / self._temperature, axis=self._axis)
        return self.output

    def _backward_numpy(self, delta, logits):
        dx = self.output * delta
        dx -= self.output * dx.sum(axis=self._axis, keepdims=True)
        dlogits = dx / self._temperature
        return dlogits


@_typecheck(exclude_args=('logits',))
def gumbel_softmax(
    logits: Array,
    temperature: float = 1e-3,
    axis: int = -1,
    *,
    name: str = None,
) -> Array:
    """Return random sample from gumbel softmax distribution.

    Parameters
    ----------
    logits : Array
        Logit of probabilities.
    temperature : float, optional
        Temperature parameter for smoothing softmax activation, by default 1e-3
    axis : int, optional
        Axis of probabilities, by default -1
    name : str, optional
        The name of the operation, by default None

    Returns
    -------
    Array
        Samples from gumbel softmax distribution.
    """
    return _GumbelSoftmax(logits, temperature, axis=axis, name=name).forward()
