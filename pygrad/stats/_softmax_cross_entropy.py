import numpy as np
import scipy.special as sp

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


class _SoftmaxCrossEntropy(_Operator):

    def __init__(self, labels, logits, axis: int = -1, name=None):
        super().__init__(labels, logits, name=name)
        self._axis = axis

    def _forward_numpy(self, labels, logits):
        self.log_softmax = sp.log_softmax(logits, axis=self._axis)
        return -(labels * self.log_softmax).sum(axis=self._axis)

    def _backward_numpy(self, delta, labels, logits):
        delta = np.expand_dims(delta, self._axis)
        if self._args[0].is_variable:
            dlabels = _unbroadcast_to(-delta * self.log_softmax, labels.shape)
        else:
            dlabels = None
        if self._args[1].is_variable:
            probs = np.exp(self.log_softmax)
            dlogits = _unbroadcast_to(delta * (probs - labels), logits.shape)
        else:
            dlogits = None
        return dlabels, dlogits


@_typecheck(exclude_args=('labels', 'logits'))
def softmax_cross_entropy(
        labels: Array,
        logits: Array,
        axis: int = -1,
        *,
        name: str = None) -> Array:
    r"""Return cross entropy of softmax of logits relative to given labels.

    .. math::
        f({\boldsymbol p}, {\boldsymbol a}) =
            -\Sigma_{i=0}^{N-1} p_i \ln{e^{a_i}\over\Sigma_{n=0}^{N-1}e^{a_n}}

    Parameters
    ----------
    labels : Array
        Target probability distribution along the given axis.
        Typically one-of-k coding format along the given axis.
    logits : Array
        Logits of probailities along the given axis.
    axis : int, optional
        Axis of distribution, by default -1
    name : str, optional
        Name of the operation, by default None

    Returns
    -------
    Array
        Cross entropy of softmax of logits relative to given labels.

    Examples
    --------
    >>> import pygrad as gd
    >>> gd.stats.softmax_cross_entropy([1, 0, 0], [0, 1, -1])
    array(1.40760596)
    >>> gd.stats.softmax_cross_entropy([1, 0, 0], [10, -10, -10])
    array(4.12230738e-09)
    """
    return _SoftmaxCrossEntropy(labels, logits, axis=axis, name=name).forward()
