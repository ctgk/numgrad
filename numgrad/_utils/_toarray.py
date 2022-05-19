import numpy as np

from numgrad._variable import Variable


def _toarray(a):
    if not isinstance(a, (Variable, np.ndarray)):
        return np.asarray(a)
    return a
