import numpy as np


from pygrad._decorators import register_gradient
from pygrad._utils._unbroadcast import _unbroadcast_to


@register_gradient(
    np.random.exponential,
    module_name='numpy.random',
    function_name='exponential',
)
def _exponential_gradient(do, o, scale, size=None):
    return _unbroadcast_to(o / scale * do, scale.shape)
