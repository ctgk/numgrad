import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, expected', [
    ([1, 2, 5], np.log([1, 2, 5])),
])
def test_forward(x, expected):
    actual = gd.log(x)
    assert np.allclose(actual.data, expected)


@pytest.mark.parametrize('x, expected', [
    (gd.Array([1., 2, 5], is_variable=True), [1, 0.5, 0.2]),
])
def test_backward(x, expected):
    with gd.Graph() as g:
        gd.log(x)
    g.backward()
    assert np.allclose(x.grad, expected)


@pytest.mark.parametrize('x', [
    gd.Array(np.random.rand(2, 3) + 1, is_variable=True),
    gd.Array(np.random.rand(4, 2, 3) + 10, is_variable=True),
])
def test_numerical_grad(x):
    with gd.Graph() as g:
        gd.log(x)
    g.backward()
    dx = _numerical_grad(gd.log, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
