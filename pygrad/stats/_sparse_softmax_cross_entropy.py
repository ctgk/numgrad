import typing as tp

import numpy as np
import scipy.special as sp

from pygrad._core._array import Array
from pygrad._core._config import config
from pygrad._core._operator import _Operator
from pygrad._core._types import _is_float, _is_int
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


class _SparseSoftmaxCrossEntropy(_Operator):

    def __init__(self, labels, logits, axis: int = -1, name=None):
        super().__init__(labels, logits, name=name)
        self._axis = axis

    def _input_dtypes(self, *args):
        return (config.int, config.dtype)

    def _check_dtype(self):
        if not _is_int(self._args[0].dtype):
            raise ValueError(
                f'Arg \'labels\' must be int type, not {self._args[0].dtype}')
        if not _is_float(self._args[1].dtype):
            raise ValueError(
                'Arg \'logits\' must be float type, '
                f'not {self._args[1].dtype}')

    def _forward_numpy(self, labels, logits):
        n_classes = logits.shape[self._axis]
        self._log_softmax = sp.log_softmax(logits, axis=self._axis)
        self._labels = np.eye(n_classes)[labels.ravel()].reshape(
            *labels.shape, n_classes)
        if (self._axis != -1) or (self._axis != labels.ndim - 1):
            self._labels = np.moveaxis(self._labels, -1, self._axis)
        return -(self._labels * self._log_softmax).sum(axis=self._axis)

    def _backward_numpy(self, delta, labels, logits):
        delta = np.expand_dims(delta, self._axis)
        probs = np.exp(self._log_softmax)
        dlogits = _unbroadcast_to(delta * (probs - self._labels), logits.shape)
        return None, dlogits


@_typecheck(exclude_args=('labels', 'logits'))
def sparse_softmax_cross_entropy(
        labels: Array,
        logits: Array,
        axis: int = -1,
        *,
        name: tp.Union[str, None] = None) -> Array:
    """Return cross entropy of softmax of logits relative to given labels

    Parameters
    ----------
    labels : Array
        Target labels
    logits : Array
        Logits of probabilities along the given axis
    axis : int, optional
        Axis of distributions, by default -1
    name : tp.Union[str, None], optional
        The name of the operation, by default None

    Returns
    -------
    Array
        Cross entropy

    Examples
    --------
    >>> import pygrad as gd
    >>> gd.stats.sparse_softmax_cross_entropy([2, 0], [[0, 1, -1], [5, -2, 0]])
    array([2.40760596, 0.00762072])
    """
    return _SparseSoftmaxCrossEntropy(
        labels, logits, axis=axis, name=name).forward()
