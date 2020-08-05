import numpy as np
import pytest

import pygrad as pg
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, y, name', [
    ([1, -1, 5], 3, 'multiply'),
])
def test_forward(x, y, name):
    actual = pg.multiply(x, y, name=name)
    assert np.allclose(actual.value, np.multiply(x, y))
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, y, dz, expected_dx, expected_dy', [
    (
        pg.Array([1., -1, 5], is_differentiable=True),
        pg.Array(3., is_differentiable=True), None,
        np.ones(3) * 3, 5
    ),
    (
        5, pg.Array([-7., 3], is_differentiable=True), [1, -2],
        None, 5 * np.array([1, -2])
    ),
])
def test_backward(x, y, dz, expected_dx, expected_dy):
    if dz is not None:
        (x * y).backward(_grad=dz)
    else:
        (x * y).backward()
    if expected_dx is not None:
        assert np.allclose(x.grad, expected_dx)
    if expected_dy is not None:
        assert np.allclose(y.grad, expected_dy)


@pytest.mark.parametrize('x, y', [
    (
        pg.Array(np.random.rand(5, 1, 3), is_differentiable=True),
        pg.Array(np.random.rand(2, 3), is_differentiable=True)
    ),
])
def test_numerical_grad(x, y):
    (x * y).backward()
    dx, dy = _numerical_grad(pg.multiply, x, y, epsilon=1e-3)
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)
    assert np.allclose(dy, y.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
