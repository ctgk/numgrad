import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _GumbelSigmoid(_Operator):

    def __init__(self, logits, temperature: float, name: str = None):
        super().__init__(logits, name=name)
        self._temperature = temperature

    def _forward_numpy(self, logits):
        dg = np.random.gumbel(size=logits) - np.random.gumbel(size=logits)
        a = (logits + dg) / self._temperature
        self.output = np.tanh(0.5 * a) * 0.5 + 0.5
        return self.output

    def _backward_numpy(self, delta, logits):
        da = delta * self.output * (1 - self.output)
        dlogits = da / self._temperature
        return dlogits


@_typecheck(exclude=('logits',))
def gumbel_sigmoid(
        logits: Array,
        temperature: float = 1e-3,
        *,
        name: str = None) -> Array:
    """Return random samples from gumbel sigmoid distributions.

    Parameters
    ----------
    logits : Array
        Logit of probabilities
    temperature : float, optional
        Smoothing parameter of sigmoid activations, by default 1e-3
    name : str, optional
        The name of the operation, by default None

    Returns
    -------
    Array
        Samples from gumbel sigmoid distributions
    """
    return _GumbelSigmoid(logits, temperature, name=name).forward()
