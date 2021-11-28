import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.fixture(params=[
    {'x': [1, -1, 5], 'y': 3},
    {'x': np.random.uniform(1, 10, (2, 3)), 'y': np.random.uniform(1, 10, 3)},
])
def parameter_data(request):
    return request.param


@pytest.fixture(params=[
    {'function': gd.add},
    {'function': gd.divide},
    {'function': gd.multiply},
    {'function': gd.subtract},
])
def parameter_function(request):
    return request.param


def test_backward(parameter_data, parameter_function):
    x = gd.Tensor(parameter_data['x'], is_variable=True)
    y = gd.Tensor(parameter_data['y'], is_variable=True)
    f = parameter_function['function']
    out = f(x, y)
    out.backward()
    dx, dy = _numerical_grad(f, x, y, epsilon=1e-3)
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)
    assert np.allclose(dy, y.grad, rtol=0, atol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__])
