import numpy as np
import pytest

import pygrad as pg
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, name, expected', [
    ([1, -1, 5], 'sigmoid', 1 / (1 + np.exp([-1, 1, -5]))),
])
def test_forward(x, name, expected):
    actual = pg.stats.sigmoid(x, name=name)
    assert np.allclose(actual.value, expected)
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x', [
    pg.Array(np.random.rand(2, 3), is_differentiable=True),
    pg.Array(np.random.rand(4, 2, 3), is_differentiable=True),
])
def test_backward(x):
    pg.stats.sigmoid(x).backward()
    dx = (pg.stats.sigmoid(x.value) * (1 - pg.stats.sigmoid(x.value))).value
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


@pytest.mark.parametrize('x', [
    pg.Array(np.random.rand(2, 3), is_differentiable=True),
    pg.Array(np.random.rand(4, 2, 3), is_differentiable=True),
])
def test_numerical_grad(x):
    pg.stats.sigmoid(x).backward()
    dx = _numerical_grad(pg.stats.sigmoid, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
