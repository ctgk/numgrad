import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, axes, name', [
    ([[1, 2, 3]], (1, 0), 'transpose'),
])
def test_forward(x, axes, name):
    actual = gd.transpose(x, axes, name=name)
    assert np.allclose(actual.data, np.transpose(x, axes))
    assert actual.shape == np.transpose(x, axes).shape
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, axes', [
    (
        gd.Array(np.random.rand(2, 3, 4, 2), is_variable=True),
        (1, 0, 3, 2)
    ),
    (
        gd.Array(np.random.rand(2, 3, 4, 2), is_variable=True),
        (2, 0, 3, 1)
    ),
    (
        gd.Array(np.random.rand(2, 3, 4), is_variable=True),
        (1, 0, 2)
    ),
])
def test_numerical_grad(x, axes):
    with gd.Graph() as g:
        x.transpose(*axes)
    g.backward()
    dx = _numerical_grad(lambda x: gd.transpose(x, axes), x)
    assert np.allclose(dx, x.grad)


@pytest.mark.parametrize('x', [
    gd.Array(np.random.rand(2, 3, 4, 2), is_variable=True),
    gd.Array(np.random.rand(2, 3, 4, 2), is_variable=True),
    gd.Array(np.random.rand(2, 3, 4), is_variable=True),
    gd.Array(np.random.rand(5, 2), is_variable=True),
])
def test_numerical_grad_2(x):
    with gd.Graph() as g:
        x.T
    g.backward()
    dx = _numerical_grad(lambda x: x.T, x)
    assert np.allclose(dx, x.grad)


if __name__ == "__main__":
    pytest.main([__file__])
