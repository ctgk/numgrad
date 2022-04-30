import numpy as np
import pytest
import scipy.special as sp

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('function, x', [
    # signs
    (np.positive, -3),
    (lambda a: +a, -3),
    (np.negative, -3),
    (lambda a: -a, -3),

    # four basic operations
    (lambda a: np.add(a, [[1, 2], [3, 4]]), [1, 2]),
    (lambda a: a + [[1, 2], [3, 4]], [1, 2]),
    (lambda a: np.subtract(a, [[1, 2], [3, 4]]), [1, 2]),
    (lambda a: a - [[1, 2], [3, 4]], [1, 2]),
    (lambda a: np.multiply(a, [[1, 2], [3, 4]]), [1, 2]),
    (lambda a: a * [[1, 2], [3, 4]], [1, 2]),
    (lambda a: np.divide(a, [[1, 2], [3, 4]]), [1, 2]),
    (lambda a: a / [[1, 2], [3, 4]], [1, 2]),
    (lambda a: a @ [1, 2], [1, 2]),
    (lambda a: np.matmul(a, [1, 2]), [[1, 2], [3, 4]]),
    (lambda a: a @ [[1, 2], [3, 4]], [1, 2]),
    (lambda a: np.matmul(a, [[1, 2], [3, 4]]), [[1, 2], [3, 4]]),
    (lambda a: a @ [[1, 2], [3, 4]], np.random.rand(3, 4, 2)),

    # powers
    (np.square, [2, -1]),
    (np.sqrt, [3, 0.5]),

    # trigonometric functions
    (np.cos, np.random.uniform(-10, 10, (3, 2))),
    (np.sin, np.random.uniform(-10, 10, (2, 5))),
    (np.tan, np.random.uniform(-10, 10, (4, 1))),

    # hyperbolic functions
    (np.cosh, np.random.uniform(-10, 10, (3, 4))),
    (np.sinh, np.random.uniform(-10, 10, (1, 5))),
    (np.tanh, np.random.uniform(-10, 10, (4, 2))),

    # exp & log
    (np.exp, [-1, -0.2, 0.5]),
    (np.log, [1, 0.2, 0.5, 2]),

    (lambda a: sp.expit(a), [1, 0, -1]),
    (lambda a: sp.log_expit(a), [1, 0, -1]),
    (lambda a: sp.log_softmax(a), [0.5, 0, -0.5]),
    (
        lambda a: sp.log_softmax(a, axis=-1),
        [[0.5, 0, -0.5], [0, 1, 2]],
    ),
    (lambda a: sp.logsumexp(a), -1),
    (lambda a: sp.logsumexp(a), [-1, 1]),
    (lambda a: sp.logsumexp(a, axis=1), np.random.rand(3, 2)),
    (
        lambda a: sp.logsumexp(a, axis=(0, 2), keepdims=True),
        np.random.rand(4, 2, 3),
    ),
    (lambda a: sp.log_softmax(a, axis=(0, 2)), np.random.rand(2, 3, 4)),
    (lambda a: sp.softmax(a), -1),
    (lambda a: sp.softmax(a), [-1, 1]),
    (lambda a: sp.softmax(a, axis=1), np.random.rand(3, 2)),
    (lambda a: sp.softmax(a, axis=(0, 2)), np.random.rand(4, 2, 3)),
    (lambda a: sp.softmax(a), np.random.rand(4, 2, 3)),

    # gamma function
    (sp.gamma, [1, 0.5, 3.3]),

    # stats
    (np.max, 9),
    (np.max, [1, 2]),
    (lambda a: a.max(axis=1), np.random.rand(2, 3) * 10),
    (lambda a: a.max(axis=(0, 2), keepdims=True), np.random.rand(2, 4, 3)),
    (np.min, 9),
    (np.min, [1, 2]),
    (lambda a: a.min(axis=1), np.random.rand(2, 3) * 10),
    (lambda a: a.min(axis=(0, 2), keepdims=True), np.random.rand(2, 4, 3)),
    (lambda a: np.mean(a), -1),
    (lambda a: np.mean(a), [-1, 1]),
    (lambda a: a.mean(axis=1), np.random.rand(3, 2)),
    (lambda a: np.mean(a, (0, 2), keepdims=True), np.random.rand(4, 2, 3)),
    (lambda a: np.sum(a), -1),
    (lambda a: np.sum(a), [-1, 1]),
    (lambda a: a.sum(axis=1), np.random.rand(3, 2)),
    (lambda a: np.sum(a, (0, 2), keepdims=True), np.random.rand(4, 2, 3)),

    # random
    (lambda a: (np.random.seed(0), np.random.exponential(a))[1], [1, 10]),
    (
        lambda a: (
            np.random.seed(0), np.random.exponential(a, size=(5, 2)),
        )[1],
        [1, 10],
    ),
    (lambda a: (np.random.seed(0), np.random.normal(a, 1))[1], [-1, 1]),
    (
        lambda a: (np.random.seed(0), np.random.normal(a, 1, size=(3, 2)))[1],
        [-1, 1],
    ),
    (lambda a: (np.random.seed(0), np.random.normal(0, a))[1], [1, 5]),
    (
        lambda a: (np.random.seed(0), np.random.normal(0, a, size=(4, 2)))[1],
        [1, 5],
    ),
    (lambda a: (np.random.seed(0), np.random.uniform(a, 10))[1], [-1, 1]),
    (
        lambda a: (
            np.random.seed(0),
            np.random.uniform(-10, a, size=(3, 2)),
        )[1],
        [-1, 1],
    ),
])
def test_gradient(function, x):
    x = gd.Variable(x)
    assert type(function(x)) != gd.Variable
    with gd.Graph() as g:
        y = function(x)
    assert type(y) == gd.Variable
    assert type(function(x)) != gd.Variable
    dx_actual = g.gradient(y, [x])[0]
    dx_expected = _numerical_grad(function, x)[0]
    assert np.allclose(dx_expected, dx_actual)


@pytest.mark.xfail
def test_gradient_error():
    a = gd.Variable([0, 0.5])
    with gd.Graph() as g:
        b = np.argsort(a)
    with pytest.raises(Exception):
        g.gradient(b, [a])[0]


if __name__ == '__main__':
    pytest.main([__file__])
