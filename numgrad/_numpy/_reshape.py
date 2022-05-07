import numpy as np

from numgrad._decorators import _register_gradient


@_register_gradient(np.reshape)
def _reshape_gradient(dy, _y, x, _newshape, order=None):
    """Gradient of np.reshape which suports __array_function__."""
    return dy.reshape(*x.shape, order=order)
