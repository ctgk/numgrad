import numpy as np
import pytest

import pygrad as pg
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, expected', [
    ([1, 2, 5], np.log([1, 2, 5])),
])
def test_forward(x, expected):
    actual = pg.log(x)
    assert np.allclose(actual.value, expected)


@pytest.mark.parametrize('x, dy, expected', [
    (pg.Array([1., 2, 5], is_differentiable=True), None, [1, 0.5, 0.2]),
    (
        pg.Array([7., 3], is_differentiable=True), [1, -2],
        np.array([1, -2]) / np.array([7, 3])
    ),
])
def test_backward(x, dy, expected):
    if dy is None:
        pg.log(x).backward()
    else:
        pg.log(x).backward(_grad=dy)
    assert np.allclose(x.grad, expected)


@pytest.mark.parametrize('x', [
    pg.Array(np.random.rand(2, 3) + 1, is_differentiable=True),
    pg.Array(np.random.rand(4, 2, 3) + 10, is_differentiable=True),
])
def test_numerical_grad(x):
    pg.log(x).backward()
    dx = _numerical_grad(pg.log, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
