import numpy as np


from numgrad._decorators import _register_gradient
from numgrad._utils._unbroadcast import _unbroadcast_to


@_register_gradient(
    np.random.exponential,
    module_name='numpy.random',
    function_name='exponential',
)
def _exponential_gradient(
    do,
    o,
    scale,
    size=None,  # noqa: U100
):
    return _unbroadcast_to(o / scale * do, scale.shape)


@_register_gradient(
    np.random.normal,
    module_name='numpy.random',
    function_name='normal',
)
def _normal_gradient(
    do,
    o,
    loc,
    scale,
    size=None,  # noqa: U100
):
    loc, scale = np.asarray(loc), np.asarray(scale)
    dloc = _unbroadcast_to(do, loc.shape)
    dscale = _unbroadcast_to((o - loc) / scale * do, scale.shape)
    return dloc, dscale


@_register_gradient(
    np.random.uniform,
    module_name='numpy.random',
    function_name='uniform',
)
def _uniform_gradient(
    do,
    o,
    low,
    high,
    size=None,  # noqa: U100
):
    low, high = np.asarray(low), np.asarray(high)
    u = (o - low) / (high - low)
    du = do * u
    dlow = _unbroadcast_to(do - du, low.shape)
    dhigh = _unbroadcast_to(du, high.shape)
    return dlow, dhigh
