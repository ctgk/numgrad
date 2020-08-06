import numpy as np
import pytest

import pygrad as pg
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, axes, name', [
    ([[1, 2, 3]], (1, 0), 'transpose'),
])
def test_forward(x, axes, name):
    actual = pg.transpose(x, axes, name=name)
    assert np.allclose(actual.value, np.transpose(x, axes))
    assert actual.shape == np.transpose(x, axes).shape
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, axes', [
    (
        pg.Array(np.random.rand(2, 3, 4, 2), is_differentiable=True),
        (1, 0, 3, 2)
    ),
    (
        pg.Array(np.random.rand(2, 3, 4, 2), is_differentiable=True),
        (2, 0, 3, 1)
    ),
    (
        pg.Array(np.random.rand(2, 3, 4), is_differentiable=True),
        (1, 0, 2)
    ),
])
def test_numerical_grad(x, axes):
    x.transpose(*axes).backward()
    dx = _numerical_grad(lambda x: pg.transpose(x, axes), x)
    assert np.allclose(dx, x.grad)


@pytest.mark.parametrize('x', [
    pg.Array(np.random.rand(2, 3, 4, 2), is_differentiable=True),
    pg.Array(np.random.rand(2, 3, 4, 2), is_differentiable=True),
    pg.Array(np.random.rand(2, 3, 4), is_differentiable=True),
    pg.Array(np.random.rand(5, 2), is_differentiable=True),
])
def test_numerical_grad_2(x):
    x.T.backward()
    dx = _numerical_grad(lambda x: x.T, x)
    assert np.allclose(dx, x.grad)


if __name__ == "__main__":
    pytest.main([__file__])
