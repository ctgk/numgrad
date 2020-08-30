import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, newshape, name', [
    ([1, 2, 3], (-1, 1), 'reshape'),
])
def test_forward(x, newshape, name):
    actual = gd.reshape(x, newshape, name=name)
    assert np.allclose(actual.data, np.reshape(x, newshape))
    assert actual.shape == np.reshape(x, newshape).shape
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, newshape', [
    (gd.Array([1., 2, 3, 4, 5, 6], is_variable=True), (2, 3)),
])
def test_backward(x, newshape):
    with gd.Graph() as g:
        x.reshape(*newshape)
    g.backward()
    dy = np.ones(newshape, dtype=x.dtype)
    assert np.allclose(x.grad, dy.reshape(x.shape))


@pytest.mark.parametrize('x, newshape', [
    (gd.Array(np.random.rand(2, 3, 4), is_variable=True), (-1, 6)),
])
def test_numerical_grad(x, newshape):
    with gd.Graph() as g:
        x.reshape(*newshape)
    g.backward()
    dx = _numerical_grad(lambda x: gd.reshape(x, newshape), x)
    assert np.allclose(dx, x.grad)


if __name__ == "__main__":
    pytest.main([__file__])
