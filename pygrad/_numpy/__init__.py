import numpy as np

from pygrad._decorators import register_gradient


@register_gradient(np.square)
def _square_gradient(dy, y, x):
    return 2 * x * dy
