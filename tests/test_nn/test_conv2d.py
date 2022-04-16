import numpy as np
import pytest
from scipy.signal import correlate2d

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.mark.parametrize('x, w', [
    (np.random.rand(1, 5, 5, 1), np.random.rand(3, 3, 1, 1)),
])
def test_forward(x, w):
    actual = gd.nn.conv2d(x, w)
    assert np.allclose(
        np.squeeze(actual.numpy()),
        correlate2d(np.squeeze(x), np.squeeze(w), mode='valid'))


@pytest.mark.parametrize('x, w, stride, pad', [
    (
        gd.Tensor(np.random.rand(1, 4, 4, 2), is_variable=True),
        gd.Tensor(np.random.rand(3, 3, 2, 5), is_variable=True),
        (1, 1),
        (0, 0),
    ),
    (
        gd.Tensor(np.random.rand(3, 7, 7, 1), is_variable=True),
        gd.Tensor(np.random.rand(5, 5, 1, 2), is_variable=True),
        (1, 1),
        (0, 0),
    ),
])
def test_numerical_grad(x, w, stride, pad):
    gd.nn.conv2d(x, w, stride, pad).backward()
    dx, dw = _numerical_grad(
        lambda a, b: gd.nn.conv2d(a, b, stride, pad), x, w)
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)
    assert np.allclose(dw, w.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
