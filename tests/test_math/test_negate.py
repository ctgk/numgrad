import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, name, expected', [
    ([1, -1, 5], 'negate', [-1, 1, -5]),
])
def test_forward(x, name, expected):
    actual = gd.negate(x, name=name)
    assert np.allclose(actual.data, expected)
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, expected', [
    (gd.Array([1., -1, 5], is_variable=True), [-1, -1, -1]),
])
def test_backward(x, expected):
    with gd.Graph() as g:
        gd.negate(x)
    g.backward()
    assert np.allclose(x.grad, expected)


@pytest.mark.parametrize('x', [
    gd.Array(np.random.rand(2, 3), is_variable=True),
    gd.Array(np.random.rand(4, 2, 3), is_variable=True),
])
def test_numerical_grad(x):
    with gd.Graph() as g:
        gd.negate(x)
    g.backward()
    dx = _numerical_grad(gd.negate, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
