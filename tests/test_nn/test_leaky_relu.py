import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, alpha, name, expected', [
    ([1, -1, 5], 0.2, 'leaky_relu', [1, -0.2, 5]),
    ([1, -1, 5], 0.1, 'leaky_relu', [1, -0.1, 5]),
])
def test_forward(x, alpha, name, expected):
    actual = gd.nn.leaky_relu(x, alpha, name=name)
    assert np.allclose(actual.data, expected)
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, alpha', [
    (gd.Array(np.random.rand(2, 3), is_variable=True), 0.1),
    (gd.Array(np.random.rand(4, 2, 3), is_variable=True), 0.5),
])
def test_backward(x, alpha):
    with gd.Graph() as g:
        gd.nn.leaky_relu(x, alpha)
    g.backward()
    dx = (x.data > 0) + alpha * (x.data < 0)
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


@pytest.mark.parametrize('x, alpha', [
    (gd.Array(np.random.rand(2, 3), is_variable=True), 0.2),
    (gd.Array(np.random.rand(4, 2, 3), is_variable=True), 0.5),
])
def test_numerical_grad(x, alpha):
    with gd.Graph() as g:
        gd.nn.leaky_relu(x, alpha)
    g.backward()
    dx = _numerical_grad(lambda a: gd.nn.leaky_relu(a, alpha), x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
