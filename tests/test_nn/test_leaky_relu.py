import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, alpha, expected', [
    ([1, -1, 5], 0.2, [1, -0.2, 5]),
    ([1, -1, 5], 0.1, [1, -0.1, 5]),
])
def test_forward(x, alpha, expected):
    actual = gd.nn.leaky_relu(x, alpha)
    assert np.allclose(actual.numpy(), expected)
    assert actual.name == 'leaky_relu.out'


@pytest.mark.parametrize('x, alpha', [
    (gd.Tensor(np.random.rand(2, 3), is_variable=True), 0.1),
    (gd.Tensor(np.random.rand(4, 2, 3), is_variable=True), 0.5),
])
def test_backward(x, alpha):
    gd.nn.leaky_relu(x, alpha).backward()
    dx = (x.numpy() > 0) + alpha * (x.numpy() < 0)
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


@pytest.mark.parametrize('x, alpha', [
    (gd.Tensor(np.random.rand(2, 3), is_variable=True), 0.2),
    (gd.Tensor(np.random.rand(4, 2, 3), is_variable=True), 0.5),
])
def test_numerical_grad(x, alpha):
    gd.nn.leaky_relu(x, alpha).backward()
    dx = _numerical_grad(lambda a: gd.nn.leaky_relu(a, alpha), x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
