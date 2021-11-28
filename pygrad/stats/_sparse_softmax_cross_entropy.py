import typing as tp

import numpy as np
import scipy.special as sp

from pygrad._core._config import config
from pygrad._core._differentiable_operator import _DifferentiableOperator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


def _sparse_softmax_cross_entropy(
    labels: TensorLike,
    logits: TensorLike,
    *,
    axis: int = -1,
    keepdims: bool = False,
):
    num_classes = logits.shape[axis]
    log_softmax = sp.log_softmax(logits, axis=axis)
    labels_dense = np.eye(num_classes)[labels.ravel()].reshape(
        *labels.shape, num_classes)
    if (axis != -1) or (axis != labels.ndim - 1):
        labels_dense = np.moveaxis(labels_dense, -1, axis)

    def grad(dout):
        if not keepdims:
            dout = np.expand_dims(dout, axis)
        proba = np.exp(log_softmax)
        dlogits = _unbroadcast_to(dout * (proba - labels_dense), logits.shape)
        return None, dlogits

    out = -(labels_dense * log_softmax).sum(axis=axis, keepdims=keepdims)
    return out, grad


class _SparseSoftmaxCrossEntropy(_DifferentiableOperator):

    def __init__(self) -> None:
        super().__init__(_sparse_softmax_cross_entropy)

    @staticmethod
    def _get_out_dtype(
        labels_sparse: Tensor,
        logits: Tensor,
    ):
        return logits.dtype

    def _check_args(
        self,
        labels_sparse: TensorLike,
        logits: TensorLike,
    ) -> tp.Tuple[Tensor]:
        if not isinstance(labels_sparse, Tensor):
            labels_sparse = Tensor(labels_sparse, dtype=config.int)
        if not isinstance(logits, Tensor):
            logits = Tensor(logits, dtype=config.dtype)
        for arg in (labels_sparse, logits):
            arg._children.append(self)
        return labels_sparse, logits

    @_typecheck()
    def __call__(
        self,
        labels_sparse: TensorLike,
        logits: TensorLike,
        *,
        axis: int = -1,
        keepdims: bool = False,
    ):
        return super().__call__(
            labels_sparse,
            logits,
            axis=axis,
            keepdims=keepdims,
        )


def sparse_softmax_cross_entropy(
    labels: TensorLike,
    logits: TensorLike,
    axis: int = -1,
    keepdims: bool = False,
) -> Tensor:
    """Return cross entropy of softmax of logits relative to given labels.

    Parameters
    ----------
    labels : TensorLike
        Target labels
    logits : TensorLike
        Logits of probabilities along the given axis
    axis : int, optional
        Axis of distributions, by default -1
    keepdims : bool, optional
        True to keep dimensionality of the resulting tensor, otherwise false.

    Returns
    -------
    Tensor
        Cross entropy

    Examples
    --------
    >>> import pygrad as gd
    >>> gd.stats.sparse_softmax_cross_entropy([2, 0], [[0, 1, -1], [5, -2, 0]])
    Tensor([2.40760596, 0.00762072])
    """
    return _SparseSoftmaxCrossEntropy().__call__(
        labels, logits, axis=axis, keepdims=keepdims)
