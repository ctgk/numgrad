import numpy as np
import scipy.special as sp

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


@_typecheck()
@differentiable_operator
def _softmax_cross_entropy(
    labels: TensorLike,
    logits: TensorLike,
    *,
    axis: int = -1,
    keepdims: bool = False,
):
    log_softmax = sp.log_softmax(logits, axis=axis)

    def grad(dout):
        if not keepdims:
            dout = np.expand_dims(dout, axis)
        proba = np.exp(log_softmax)
        dlabels = _unbroadcast_to(-dout * log_softmax, labels.shape)
        dlogits = _unbroadcast_to(dout * (proba - labels), logits.shape)
        return dlabels, dlogits

    out = -(labels * log_softmax).sum(axis=axis, keepdims=keepdims)
    return out, grad


def softmax_cross_entropy(
    labels: TensorLike,
    logits: TensorLike,
    axis: int = -1,
    keepdims: bool = False,
) -> Tensor:
    r"""Return cross entropy of softmax of logits relative to given labels.

    .. math::
        f({\boldsymbol p}, {\boldsymbol a}) =
            -\Sigma_{i=0}^{N-1} p_i \ln{e^{a_i}\over\Sigma_{n=0}^{N-1}e^{a_n}}

    Parameters
    ----------
    labels : TensorLike
        Target probability distribution along the given axis.
        Typically one-of-k coding format along the given axis.
    logits : TensorLike
        Logits of probabilities along the given axis.
    axis : int, optional
        Axis of distribution, by default -1
    keepdims : bool, optional
        True to keep dimensionality of the resulting tensor, otherwise false.

    Returns
    -------
    Tensor
        Cross entropy of softmax of logits relative to given labels.

    Examples
    --------
    >>> gd.stats.softmax_cross_entropy([1, 0, 0], [0, 1, -1])
    Tensor(1.40760596)
    >>> gd.stats.softmax_cross_entropy([1, 0, 0], [10, -10, -10])
    Tensor(4.12230738e-09)
    """
    return _softmax_cross_entropy(labels, logits, axis=axis, keepdims=keepdims)
