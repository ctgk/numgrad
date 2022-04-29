import numpy as np
import pytest
import scipy.special as sp

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('function, x', [
    (lambda a: a + [[1, 2], [3, 4]], gd.Variable([1, 2])),
    (np.square, gd.Variable([2, -1])),
    (lambda a: sp.log_softmax(a), gd.Variable([0.5, 0, -0.5])),
    (
        lambda a: sp.log_softmax(a, axis=-1),
        gd.Variable([[0.5, 0, -0.5], [0, 1, 2]]),
    ),
    (
        lambda a: sp.log_softmax(a, axis=(0, 2)),
        gd.Variable(np.random.rand(2, 3, 4)),
    ),
])
def test_gradient(function, x):
    with gd.Graph() as g:
        y = function(x)
    dx_actual = g.gradient(y, [x])[0]
    dx_expected = _numerical_grad(function, x)[0]
    assert np.allclose(dx_expected, dx_actual)


if __name__ == '__main__':
    pytest.main([__file__])
