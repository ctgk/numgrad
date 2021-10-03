import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, y, name', [
    ([1, -1, 5], 3, 'add'),
])
def test_forward(x, y, name):
    actual = gd.add(x, y, name=name)
    assert np.allclose(actual.data, np.add(x, y))
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, y, expected_dx, expected_dy', [
    (
        gd.Array([1., -1, 5], is_variable=True),
        gd.Array(3., is_variable=True), np.ones(3), 3
    ),
])
def test_backward(x, y, expected_dx, expected_dy):
    with gd.Graph() as g:
        (x + y)
    g.backward()
    if expected_dx is not None:
        assert np.allclose(x.grad, expected_dx)
    if expected_dy is not None:
        assert np.allclose(y.grad, expected_dy)


@pytest.mark.parametrize('x, y', [
    (
        gd.Array(np.random.rand(5, 1, 3), is_variable=True),
        gd.Array(np.random.rand(2, 3), is_variable=True)
    ),
])
def test_numerical_grad(x, y):
    with gd.Graph() as g:
        (x + y)
    g.backward()
    dx, dy = _numerical_grad(gd.add, x, y, epsilon=1e-3)
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)
    assert np.allclose(dy, y.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
