import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, name, expected', [
    ([1, -1, 5], 'sigmoid', 1 / (1 + np.exp([-1, 1, -5]))),
])
def test_forward(x, name, expected):
    actual = gd.stats.sigmoid(x, name=name)
    assert np.allclose(actual.data, expected)
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x', [
    gd.Array(np.random.rand(2, 3), is_variable=True),
    gd.Array(np.random.rand(4, 2, 3), is_variable=True),
])
def test_backward(x):
    with gd.Graph() as g:
        gd.stats.sigmoid(x)
    g.backward()
    dx = (gd.stats.sigmoid(x.data) * (1 - gd.stats.sigmoid(x.data))).data
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


@pytest.mark.parametrize('x', [
    gd.Array(np.random.rand(2, 3), is_variable=True),
    gd.Array(np.random.rand(4, 2, 3), is_variable=True),
])
def test_numerical_grad(x):
    with gd.Graph() as g:
        gd.stats.sigmoid(x)
    g.backward()
    dx = _numerical_grad(gd.stats.sigmoid, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
