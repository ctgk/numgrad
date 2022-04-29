import numpy as np
import pytest
import scipy.special as sp

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('function, x', [
    # signs
    (np.positive, gd.Variable(-3)),
    (lambda a: +a, gd.Variable(-3)),
    (np.negative, gd.Variable(-3)),
    (lambda a: -a, gd.Variable(-3)),

    # four basic operations
    (lambda a: np.add(a, [[1, 2], [3, 4]]), gd.Variable([1, 2])),
    (lambda a: a + [[1, 2], [3, 4]], gd.Variable([1, 2])),
    (lambda a: np.subtract(a, [[1, 2], [3, 4]]), gd.Variable([1, 2])),
    (lambda a: a - [[1, 2], [3, 4]], gd.Variable([1, 2])),
    (lambda a: np.multiply(a, [[1, 2], [3, 4]]), gd.Variable([1, 2])),
    (lambda a: a * [[1, 2], [3, 4]], gd.Variable([1, 2])),
    (lambda a: np.divide(a, [[1, 2], [3, 4]]), gd.Variable([1, 2])),
    (lambda a: a / [[1, 2], [3, 4]], gd.Variable([1, 2])),

    # powers
    (np.square, gd.Variable([2, -1])),
    (np.sqrt, gd.Variable([3, 0.5])),

    # trigonometric functions
    (np.cos, gd.Variable(np.random.uniform(-10, 10, (3, 2)))),
    (np.sin, gd.Variable(np.random.uniform(-10, 10, (2, 5)))),
    (np.tan, gd.Variable(np.random.uniform(-10, 10, (4, 1)))),

    # hyperbolic functions
    (np.cosh, gd.Variable(np.random.uniform(-10, 10, (3, 4)))),
    (np.sinh, gd.Variable(np.random.uniform(-10, 10, (1, 5)))),
    (np.tanh, gd.Variable(np.random.uniform(-10, 10, (4, 2)))),

    # exp & log
    (np.exp, gd.Variable([-1, -0.2, 0.5])),
    (np.log, gd.Variable([1, 0.2, 0.5, 2])),

    (lambda a: sp.log_softmax(a), gd.Variable([0.5, 0, -0.5])),
    (
        lambda a: sp.log_softmax(a, axis=-1),
        gd.Variable([[0.5, 0, -0.5], [0, 1, 2]]),
    ),
    (
        lambda a: sp.log_softmax(a, axis=(0, 2)),
        gd.Variable(np.random.rand(2, 3, 4)),
    ),

    # gamma function
    (sp.gamma, gd.Variable([1, 0.5, 3.3])),

    # stats
    (np.max, gd.Variable(9)),
    (np.max, gd.Variable([1, 2])),
    (lambda a: a.max(axis=1), gd.Variable(np.random.rand(2, 3) * 10)),
    (
        lambda a: a.max(axis=(0, 2), keepdims=True),
        gd.Variable(np.random.rand(2, 4, 3) * 10),
    ),
    (np.min, gd.Variable(9)),
    (np.min, gd.Variable([1, 2])),
    (lambda a: a.min(axis=1), gd.Variable(np.random.rand(2, 3) * 10)),
    (
        lambda a: a.min(axis=(0, 2), keepdims=True),
        gd.Variable(np.random.rand(2, 4, 3) * 10),
    ),
    (lambda a: np.mean(a), gd.Variable(-1)),
    (lambda a: np.mean(a), gd.Variable([-1, 1])),
    (lambda a: a.mean(axis=1), gd.Variable(np.random.rand(3, 2))),
    (
        lambda a: np.mean(a, axis=(0, 2), keepdims=True),
        gd.Variable(np.random.rand(4, 2, 3)),
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
