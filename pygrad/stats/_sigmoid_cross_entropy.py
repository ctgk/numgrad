import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


class _SigmoidCrossEntropy(_Operator):

    def __init__(self, labels, logits, name=None):
        super().__init__(labels, logits, name=name)

    @staticmethod
    def _forward_numpy(labels, logits):
        return (
            np.maximum(logits, 0)
            - labels * logits
            + np.log1p(np.exp(-np.abs(logits)))
        )

    def _backward_numpy(self, delta, labels, logits):
        probs = np.tanh(logits * 0.5) * 0.5 + 0.5
        dlabels = _unbroadcast_to(
            delta * np.arctanh(1 - 2 * probs) * 2, labels.shape
        ) if self._args[0].is_variable else None
        dlogits = _unbroadcast_to(
            delta * (probs - labels), logits.shape
        ) if self._args[1].is_variable else None
        return dlabels, dlogits


@_typecheck(exclude=('labels', 'logits'))
def sigmoid_cross_entropy(
        labels: Array,
        logits: Array,
        *,
        name: str = None) -> Array:
    r"""Return cross entropy of sigmoid of logits relative to given labels.

    .. math::
        f(t, a) = t \ln\sigma(a) + (1 - t)\ln(1 - \sigma(a))

    Parameters
    ----------
    labels : Array
        Target probability distribution
    logits : Array
        Logits of probabilities.
    name : str, optional
        Name of the operation, by default None

    Returns
    -------
    Array
        Cross entropy of sigmoid of logits relative to the labels.

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.stats.sigmoid_cross_entropy([0, 1], [0.5, 100])
    array([9.74076984e-01, 3.72007598e-44])
    """
    return _SigmoidCrossEntropy(labels, logits, name=name).forward()
