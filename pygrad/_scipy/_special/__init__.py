import numpy as np
import scipy.special as sp

from pygrad._decorators import register_gradient


@register_gradient(sp.log_softmax)
def _log_softmax_gradient(dy, y, x, axis=None):
    return dy - np.exp(y) * dy.sum(axis=axis, keepdims=True)
