import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.mark.parametrize('x, y', [
    ([[1, -1, 5]], [[1], [2], [3]]),
    ([[1, -1, 5]], [1, 2, 3]),
    ([1, -1, 5], [[1], [2], [3]]),
])
def test_matmul_forward(x, y):
    actual = gd.matmul(x, y)
    assert np.allclose(actual.data, np.matmul(x, y))


@pytest.mark.parametrize('x, y', [
    (
        gd.Array(np.random.rand(3, 4), is_variable=True),
        gd.Array(np.random.rand(4, 6), is_variable=True)
    ),
    (
        gd.Array(np.random.rand(4), is_variable=True),
        gd.Array(np.random.rand(4, 6), is_variable=True)
    ),
    (
        gd.Array(np.random.rand(3, 4), is_variable=True),
        gd.Array(np.random.rand(4), is_variable=True)
    ),
])
def test_matmul_numerical_grad(x, y):
    (x @ y).backward()
    dx, dy = _numerical_grad(gd.matmul, x, y, epsilon=1e-3)
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)
    assert np.allclose(dy, y.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
