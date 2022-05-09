import numpy as np

from numgrad._variable import Variable


def _isscalar(a):
    if isinstance(a, Variable):
        a = a._data
    return np.isscalar(a)
