import numpy as np

from numgrad._config import config
from numgrad._variable import Variable


def _to_array(a):
    if not isinstance(a, (Variable, np.ndarray)):
        return np.asarray(a, config.dtype)
    return a


def _to_array_or_number(a):
    if isinstance(a, Variable):
        return a
    if np.isscalar(a):
        return config.dtype(a)
    return _to_array(a)
