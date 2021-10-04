import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.mark.parametrize('x, size, strides, pad', [
    (
        gd.Array(
            np.random.uniform(
                -2, 2, (2, 4, 4, 1)).astype(np.int).astype(np.float),
            is_variable=True),
        (2, 2),
        None,
        0,
    ),
])
def test_forward_backward(x, size, strides, pad):
    with gd.Graph() as g:
        gd.nn.max_pool2d(x, size, strides, pad)
    g.backward()


@pytest.mark.parametrize('x, size, strides, pad', [
    (
        gd.Array(np.random.rand(1, 4, 4, 2), is_variable=True),
        (2, 2),
        None,
        0,
    ),
    (
        gd.Array(np.random.rand(2, 4, 4, 2), is_variable=True),
        (3, 3),
        (1, 1),
        1,
    ),
])
def test_numerical_grad(x, size, strides, pad):
    with gd.Graph() as g:
        gd.nn.max_pool2d(x, size, strides, pad)
    g.backward()
    dx = _numerical_grad(lambda a: gd.nn.max_pool2d(a, size, strides, pad), x)
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
