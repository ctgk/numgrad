import numpy as np

from numgrad._decorators import _register_gradient


@_register_gradient(np.transpose)
def _transpose_gradient(dy, _y, _x, axes=None):
    """Gradient of np.transpose which suports __array_function__."""
    if axes is None:
        return np.transpose(dy)
    return np.transpose(dy, np.argsort(axes))
