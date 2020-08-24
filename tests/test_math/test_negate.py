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


@pytest.mark.parametrize('x, dy, expected', [
    (gd.Array([1., -1, 5], is_variable=True), None, [-1, -1, -1]),
    (gd.Array([-7., 3], is_variable=True), np.array([1., -2]), [-1, 2]),
])
def test_backward(x, dy, expected):
    if dy is None:
        gd.negate(x).backward()
    else:
        gd.negate(x).backward(_grad=dy)
    assert np.allclose(x.grad, expected)


@pytest.mark.parametrize('x', [
    gd.Array(np.random.rand(2, 3), is_variable=True),
    gd.Array(np.random.rand(4, 2, 3), is_variable=True),
])
def test_numerical_grad(x):
    gd.negate(x).backward()
    dx = _numerical_grad(gd.negate, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
