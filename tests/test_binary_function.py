import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.mark.parametrize('function, x, y', [
    (lambda a, b: a + b, gd.Variable([[1, 2]]), gd.Variable([[1], [2]])),
    (np.add, gd.Variable([[1, 2]]), gd.Variable([[1], [2]])),
    (lambda a, b: a - b, gd.Variable([[1, 2]]), gd.Variable([[1], [2]])),
    (np.subtract, gd.Variable([[1, 2]]), gd.Variable([[1], [2]])),
    (lambda a, b: a * b, gd.Variable([[1, 2]]), gd.Variable([[1], [2]])),
    (np.multiply, gd.Variable([[1, 2]]), gd.Variable([[1], [2]])),
    (lambda a, b: a / b, gd.Variable([[1, 2]]), gd.Variable([[1], [2]])),
    (np.divide, gd.Variable([[1, 2]]), gd.Variable([[1], [2]])),
    (lambda a, b: a @ b, [1, 2], [1, 2]),
    (lambda a, b: np.matmul(a, b), [1, 2], [[1, 2], [3, 4]]),
    (lambda a, b: a @ b, [[1, 2], [3, 4]], [1, 2]),
    (lambda a, b: np.matmul(a, b), [[1, 2], [3, 4]], [[1, 2], [3, 4]]),
    (lambda a, b: a @ b, np.random.rand(3, 4, 2), [[1, 2], [3, 4]]),
    (np.hypot, np.random.normal(size=(3,)), np.random.normal(size=(4, 1))),
    (lambda a, b: (np.random.seed(0), np.random.normal(a, b))[1], 0, 1),
    (lambda a, b: (np.random.seed(0), np.random.uniform(a, b))[1], 0, 1),
])
def test_gradient(function, x, y):
    x, y = gd.Variable(x), gd.Variable(y)
    with gd.Graph() as g:
        output = function(x, y)
    dx_actual, dy_actual = g.gradient(output, [x, y])
    dx_expected, dy_expected = _numerical_grad(function, x, y)
    assert np.allclose(dx_expected, dx_actual)
    assert np.allclose(dy_expected, dy_actual)
