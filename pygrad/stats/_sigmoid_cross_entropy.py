import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


@_typecheck()
@differentiable_operator
def _sigmoid_cross_entropy(labels: TensorLike, logits: TensorLike):
    def grad(dout):
        proba = np.tanh(logits * 0.5) * 0.5 + 0.5
        dlabels = _unbroadcast_to(
            dout * np.arctanh(1 - 2 * proba) * 2,
            labels.shape,
        )
        dlogits = _unbroadcast_to(
            dout * (proba - labels),
            logits.shape,
        )
        return dlabels, dlogits

    out = (
        np.maximum(logits, 0)
        - labels * logits
        + np.log1p(np.exp(-np.abs(logits)))
    )
    return out, grad


def sigmoid_cross_entropy(
    labels: TensorLike,
    logits: TensorLike,
) -> Tensor:
    r"""Return cross entropy of sigmoid of logits relative to given labels.

    .. math::
        f(t, a) = t \ln\sigma(a) + (1 - t)\ln(1 - \sigma(a))

    Parameters
    ----------
    labels : TensorLike
        Target probability distribution
    logits : TensorLike
        Logits of probabilities.

    Returns
    -------
    Tensor
        Cross entropy of sigmoid of logits relative to the labels.

    Examples
    --------
    >>> import pygrad as gd
    >>> gd.stats.sigmoid_cross_entropy([0, 1], [0.5, 100])
    Tensor([9.74076984e-01, 3.72007598e-44])
    """
    return _sigmoid_cross_entropy(labels, logits)
