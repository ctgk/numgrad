import numpy as np
import pytest

import pygrad as pg
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, name, expected', [
    (
        [1, -1, 5],
        'log_softmax',
        np.log(np.exp([1, -1, 5]) / np.exp([1, -1, 5]).sum())
    ),
])
def test_forward(x, name, expected):
    actual = pg.stats.log_softmax(x, name=name)
    assert np.allclose(actual.value, expected)
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, axis', [
    (pg.Array(np.random.rand(2, 3), is_differentiable=True), 0),
    (pg.Array(np.random.rand(4, 2, 3), is_differentiable=True), 0),
    (pg.Array(np.random.rand(4, 2, 3), is_differentiable=True), 1),
    (pg.Array(np.random.rand(4, 2, 3), is_differentiable=True), 2),
])
def test_numerical_grad(x, axis):
    pg.square(pg.stats.log_softmax(x, axis)).backward()
    dx = _numerical_grad(
        lambda a: pg.square(
            pg.stats.log_softmax(
                a, axis)), x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
