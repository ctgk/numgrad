import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, name, expected', [
    (
        [1, -1, 5],
        'log_softmax',
        np.log(np.exp([1, -1, 5]) / np.exp([1, -1, 5]).sum()),
    ),
])
def test_forward(x, name, expected):
    actual = gd.stats.log_softmax(x, name=name)
    assert np.allclose(actual.data, expected)
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, axis', [
    (gd.Array(np.random.rand(2, 3), is_variable=True), 0),
    (gd.Array(np.random.rand(4, 2, 3), is_variable=True), 0),
    (gd.Array(np.random.rand(4, 2, 3), is_variable=True), 1),
    (gd.Array(np.random.rand(4, 2, 3), is_variable=True), 2),
])
def test_numerical_grad(x, axis):
    with gd.Graph() as g:
        gd.square(gd.stats.log_softmax(x, axis))
    g.backward()
    dx = _numerical_grad(
        lambda a: gd.square(
            gd.stats.log_softmax(
                a, axis)), x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
