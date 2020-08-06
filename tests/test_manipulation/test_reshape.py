import numpy as np
import pytest

import pygrad as pg
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, newshape, name', [
    ([1, 2, 3], (-1, 1), 'reshape'),
])
def test_forward(x, newshape, name):
    actual = pg.reshape(x, newshape, name=name)
    assert np.allclose(actual.value, np.reshape(x, newshape))
    assert actual.shape == np.reshape(x, newshape).shape
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, newshape, dy', [
    (pg.Array([1., 2, 3, 4, 5, 6], is_differentiable=True), (2, 3), None),
    (
        pg.Array([1., 2, 3, 4, 5, 6], is_differentiable=True), (2, 3),
        np.array([[-1., 2, 4], [2, 0, -9]])
    ),
])
def test_backward(x, newshape, dy):
    if dy is None:
        x.reshape(*newshape).backward()
        dy = np.ones(newshape, dtype=x.dtype)
    else:
        x.reshape(*newshape).backward(_grad=dy)
    assert np.allclose(x.grad, dy.reshape(x.shape))


@pytest.mark.parametrize('x, newshape', [
    (pg.Array(np.random.rand(2, 3, 4), is_differentiable=True), (-1, 6)),
])
def test_numerical_grad(x, newshape):
    x.reshape(*newshape).backward()
    dx = _numerical_grad(lambda x: pg.reshape(x, newshape), x)
    assert np.allclose(dx, x.grad)


if __name__ == "__main__":
    pytest.main([__file__])
