import numpy as np
import pytest

import pygrad as pg
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, alpha, name, expected', [
    ([1, -1, 5], 0.2, 'leaky_relu', [1, -0.2, 5]),
    ([1, -1, 5], 0.1, 'leaky_relu', [1, -0.1, 5]),
])
def test_forward(x, alpha, name, expected):
    actual = pg.nn.leaky_relu(x, alpha, name=name)
    assert np.allclose(actual.value, expected)
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, alpha', [
    (pg.Array(np.random.rand(2, 3), is_differentiable=True), 0.1),
    (pg.Array(np.random.rand(4, 2, 3), is_differentiable=True), 0.5),
])
def test_backward(x, alpha):
    pg.nn.leaky_relu(x, alpha).backward()
    dx = (x.value > 0) + alpha * (x.value < 0)
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


@pytest.mark.parametrize('x, alpha', [
    (pg.Array(np.random.rand(2, 3), is_differentiable=True), 0.2),
    (pg.Array(np.random.rand(4, 2, 3), is_differentiable=True), 0.5),
])
def test_numerical_grad(x, alpha):
    pg.nn.leaky_relu(x, alpha).backward()
    dx = _numerical_grad(lambda a: pg.nn.leaky_relu(a, alpha), x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
