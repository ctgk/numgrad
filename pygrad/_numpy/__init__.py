import numpy as np

from pygrad._decorators import register_gradient
from pygrad._utils._unbroadcast import _unbroadcast_to


@register_gradient(np.add)
def _add_gradient(doutput, output, x, y):
    return (
        _unbroadcast_to(doutput, x.shape) if hasattr(x, 'shape') else None,
        _unbroadcast_to(doutput, y.shape) if hasattr(y, 'shape') else None,
    )


@register_gradient(np.square)
def _square_gradient(dy, y, x):
    return 2 * x * dy
