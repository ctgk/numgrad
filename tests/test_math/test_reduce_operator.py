import numpy as np
import pytest
import scipy.special as sp

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.fixture(params=[
    {'data': 5},
    {'data': np.random.uniform(-1, 1, 2)},
    {'data': np.random.uniform(-1, 1, (2, 3))},
    {'data': np.random.uniform(-1, 1, (4, 2, 5))},
    {'data': np.random.uniform(-1, 1, (3, 1, 2, 4))},
])
def parameter_data(request):
    return request.param


@pytest.fixture(params=[
    {'function': gd.logsumexp},
    {'function': gd.max},
    {'function': gd.mean},
    {'function': gd.min},
    {'function': gd.sum},
])
def parameter_function(request):
    return request.param


def test_backward_reduce_all(parameter_data, parameter_function):
    x = gd.Tensor(parameter_data['data'], is_variable=True)
    f = parameter_function['function']
    g = eval('np.' + f.__name__) if f.__name__ != 'logsumexp' else sp.logsumexp
    try:
        g(x)
    except Exception:
        return
    y = f(x)
    y.backward()
    dx = _numerical_grad(f, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


def test_backward_reduce_all_keepdims(parameter_data, parameter_function):
    x = gd.Tensor(parameter_data['data'], is_variable=True)
    f = parameter_function['function']
    g = eval('np.' + f.__name__) if f.__name__ != 'logsumexp' else sp.logsumexp
    try:
        g(x, keepdims=True)
    except Exception:
        return
    y = f(x, keepdims=True)
    y.backward()
    dx = _numerical_grad(lambda a: f(a, keepdims=True), x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


def test_backward_reduce_one_axis(parameter_data, parameter_function):
    x = gd.Tensor(parameter_data['data'], is_variable=True)
    if x.ndim == 0:
        axis = 0
    else:
        axis = np.random.choice(x.ndim)
    f = parameter_function['function']
    g = eval('np.' + f.__name__) if f.__name__ != 'logsumexp' else sp.logsumexp
    try:
        g(x, axis)
    except Exception:
        return
    y = f(x, axis)
    y.backward()
    dx = _numerical_grad(lambda a: f(a, axis), x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


def test_backward_reduce_one_axis_keepdims(parameter_data, parameter_function):
    x = gd.Tensor(parameter_data['data'], is_variable=True)
    if x.ndim == 0:
        axis = 0
    else:
        axis = np.random.choice(x.ndim)
    f = parameter_function['function']
    g = eval('np.' + f.__name__) if f.__name__ != 'logsumexp' else sp.logsumexp
    try:
        g(x, axis, keepdims=True)
    except Exception:
        return
    y = f(x, axis, keepdims=True)
    y.backward()
    dx = _numerical_grad(lambda a: f(a, axis, keepdims=True), x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


def test_backward_reduce_two_axis(parameter_data, parameter_function):
    x = gd.Tensor(parameter_data['data'], is_variable=True)
    if x.ndim < 2:
        return
    else:
        axis = tuple(np.random.choice(x.ndim, size=2, replace=False).tolist())
    f = parameter_function['function']
    g = eval('np.' + f.__name__) if f.__name__ != 'logsumexp' else sp.logsumexp
    try:
        g(x, axis)
    except Exception:
        return
    y = f(x, axis)
    y.backward()
    dx = _numerical_grad(lambda a: f(a, axis), x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


def test_backward_reduce_two_axis_keepdims(parameter_data, parameter_function):
    x = gd.Tensor(parameter_data['data'], is_variable=True)
    if x.ndim < 2:
        return
    else:
        axis = tuple(np.random.choice(x.ndim, size=2, replace=False).tolist())
    f = parameter_function['function']
    g = eval('np.' + f.__name__) if f.__name__ != 'logsumexp' else sp.logsumexp
    try:
        g(x, axis, keepdims=True)
    except Exception:
        return
    y = f(x, axis, keepdims=True)
    y.backward()
    dx = _numerical_grad(lambda a: f(a, axis, keepdims=True), x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__])
