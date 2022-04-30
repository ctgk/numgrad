import numpy as np
import scipy.special as sp

from pygrad._decorators import _register_gradient


@_register_gradient(sp.logsumexp)
def _logsumexp_gradient(
    doutput, output, x, axis=None, keepdims=False, return_sign=False,
):
    if return_sign:
        raise NotImplementedError(
            'Cannot compute gradient of `scipy.special.logsumexp` '
            'with `return_sign=True`')
    if all((
        isinstance(doutput, np.ndarray),
        (not keepdims),
        (axis is not None),
    )):
        axis_positive = []
        if isinstance(axis, int):
            axis = (axis,)
        for ax in axis:
            if ax < 0:
                axis_positive.append(x.ndim + ax)
            else:
                axis_positive.append(ax)
        for ax in sorted(axis_positive):
            doutput = np.expand_dims(doutput, ax)
            output = np.expand_dims(output, ax)
    doutput = np.broadcast_to(doutput, x.shape)
    output = np.broadcast_to(output, x.shape)
    return doutput * np.exp(x - output)


@_register_gradient(sp.log_softmax)
def _log_softmax_gradient(dy, y, x, axis=None):
    return dy - np.exp(y) * dy.sum(axis=axis, keepdims=True)


@_register_gradient(sp.softmax)
def _softmax_gradient(dy, y, x, axis=None):
    dx = y * dy
    dx -= y * dx.sum(axis=axis, keepdims=True)
    return dx


@_register_gradient(sp.gamma)
def _gamma_gradient(do, o, x):
    return sp.digamma(x) * o * do
