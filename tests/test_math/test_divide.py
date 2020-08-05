import numpy as np
import pytest

import pygrad as pg
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(9)


@pytest.mark.parametrize('x, y, name', [
    ([1, -1, 5], 3, 'div'),
    (2, [[1, 2], [-3, 4]], 'divide'),
])
def test_divide_forward(x, y, name):
    actual = pg.divide(x, y, name=name)
    assert np.allclose(actual.value, np.divide(x, y))
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, y', [
    (
        pg.Array(np.random.rand(5, 1, 3), is_differentiable=True),
        pg.Array(np.random.rand(2, 3), is_differentiable=True)
    ),
])
def test_divide_numerical_grad(x, y):
    (x / y).backward()
    dx, dy = _numerical_grad(pg.divide, x, y, epsilon=1e-3)
    assert np.allclose(dx, x.grad, rtol=1e-2, atol=1e-2)
    assert np.allclose(dy, y.grad, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
