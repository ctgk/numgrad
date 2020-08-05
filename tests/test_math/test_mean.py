import numpy as np
import pytest

import pygrad as pg
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, axis, keepdims, name, error', [
    ([1, 2, 3], 0, True, 'mean', 'NoError'),
    ([1, 2, 3], 'a', True, 'mean', TypeError),
    ([1, 2, 3], 0, 1, 'mean', TypeError),
    ([1, 2, 3], 0, True, 1, TypeError),
    ([1, 2, 3], 0, True, 's,m', ValueError),
])
def test_forward_error(x, axis, keepdims, name, error):
    if error == 'NoError':
        pg.mean(x, axis, keepdims, name=name)
    else:
        with pytest.raises(error):
            pg.mean(x, axis, keepdims, name=name)


@pytest.mark.parametrize('x, axis, keepdims, name, expected', [
    ([1, 3, 0, -2], None, True, 'mean', np.array([0.5])),
])
def test_forward(x, axis, keepdims, name, expected):
    actual = pg.mean(x, axis, keepdims, name=name)
    assert np.allclose(actual.value, expected)
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, axis, keepdims, dy, expected', [
    (
        pg.Array([1., 3, -2], is_differentiable=True),
        None, True, np.array([2]), np.array([2 / 3] * 3)
    ),
    (
        pg.Array([[2., 1], [-2, 5]], is_differentiable=True),
        0, False, np.array([1, 2]),
        np.array([[1, 2], [1, 2]]) * 0.5,
    ),
    (
        pg.Array([[2., 1], [-2, 5]], is_differentiable=True),
        1, False, np.array([1, 2]),
        np.array([[1, 1], [2, 2]]) * 0.5,
    ),
    (
        pg.Array([[2., 1], [-2, 5]], is_differentiable=True),
        1, True, np.array([[1], [2]]),
        np.array([[1, 1], [2, 2]]) * 0.5,
    ),
])
def test_backward(x, axis, keepdims, dy, expected):
    y = pg.mean(x, axis, keepdims)
    y.backward(_grad=dy)
    assert np.allclose(x.grad, expected)


@pytest.mark.parametrize('x, axis, keepdims', [
    (pg.Array(np.random.rand(2, 3), is_differentiable=True), 1, False),
    (pg.Array(np.random.rand(4, 2, 3), is_differentiable=True), 0, True),
])
def test_numerical_grad(x, axis, keepdims):
    x.mean().backward()
    dx = _numerical_grad(pg.mean, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
