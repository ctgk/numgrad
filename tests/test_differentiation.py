import warnings

import numpy as np
import pytest

import numgrad as ng
from numgrad._utils._numerical_grad import _numerical_grad


def _test_graph_backward(f: callable, *args):
    args = tuple(ng.Variable(a) for a in args)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with ng.Graph() as g:
            y = f(*args)
        print([node.function for node in g._node_list])
        assert type(y) == ng.Variable
        dargs_actual = g.backward(y, args)
        dargs_expected = _numerical_grad(f, *args)
        for arg, actual, expected in zip(args, dargs_actual, dargs_expected):
            assert type(arg._data) == type(actual)
            if isinstance(actual, np.ndarray):
                assert arg.shape == actual.shape
            assert np.allclose(expected, actual)


def _test_egrad(f: callable, *args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dargs_actual = ng.elementwise_grad(f)(*args)
    if not isinstance(dargs_actual, tuple):
        dargs_actual = (dargs_actual,)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        dargs_expected = _numerical_grad(f, *args)

    for arg, actual, expected in zip(args, dargs_actual, dargs_expected):
        if np.isscalar(arg):
            assert type(actual) == ng.config.dtype
        else:
            assert type(actual) == np.ndarray
        assert np.allclose(expected, actual)


def _test_graph_backward_custom_grad(f: callable, *args):
    args = tuple(ng.Variable(a) for a in args)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with ng.Graph() as g:
            y = f(*args)
        print([node.function for node in g._node_list])
        assert type(y) == ng.Variable
        dy = np.random.uniform(-10, 10)
        dargs_actual = g.backward(y, args, target_grad=dy)
        dargs_expected = _numerical_grad(f, *args)
        for arg, actual, expected in zip(args, dargs_actual, dargs_expected):
            assert type(arg._data) == type(actual)
            if isinstance(actual, np.ndarray):
                assert arg.shape == actual.shape
            assert np.allclose(expected * dy, actual)


def test_computational_graph_backward_error():
    a = ng.Variable([0, 0.5])
    with ng.Graph() as g:
        b = np.argsort(a)
    with pytest.raises(Exception):
        g.backward(b, [a])[0]


@pytest.mark.parametrize('function, args, kwargs, expect', [
    (np.square, (-2,), {}, -4.),
    (lambda a=3, b=-4: np.sqrt(a * a + b * b), (3,), {}, 0.6),
    (lambda a, b=-4: np.sqrt(a * a + b * b), (-3,), {}, -0.6),
    (lambda a, b=-4: np.sqrt(a * a + b * b), (-3, 4), {}, (-0.6, 0.8)),
    (lambda a, b=-4: np.sqrt(a * a + b * b), (-3,), dict(b=4), -0.6),
    (lambda *args: np.sqrt(sum(a * a for a in args)), (3, 4), {}, (0.6, 0.8)),
    (
        lambda a, *args: np.sqrt(a * a + sum(a * a for a in args)),
        (3, 4), {}, (0.6, 0.8),
    ),
    (
        lambda *args, **kwargs: np.sqrt(
            sum(a * a for a in args) + sum(a * a for a in kwargs.values())),
        (1, np.sqrt(8)), dict(a=4),
        ((0.2, np.sqrt(8) / 5)),
    ),
])
def test_grad(function, args, kwargs, expect):
    actual = ng.grad(function)(*args, **kwargs)
    if expect is None:
        assert actual is None
    elif isinstance(expect, tuple):
        assert len(actual) == len(expect)
        for a, e in zip(actual, expect):
            assert np.allclose(a, e)
    else:
        assert np.allclose(actual, expect)


@pytest.mark.parametrize('dfunc, args, kwargs, expect', [
    (ng.grad(ng.grad(lambda a: a ** 3)), (-2,), {}, -12),
    (
        ng.elementwise_grad(ng.elementwise_grad(np.sin)),
        ([0, 1, 2],), {}, -np.sin([0, 1, 2]),
    ),
    (
        ng.elementwise_grad(ng.elementwise_grad(
            ng.elementwise_grad(lambda a: a ** 4))),
        ([0, -1, 2],), {}, [0, -24, 48],
    ),
])
def test_higher_order_derivatives(dfunc, args, kwargs, expect):
    actual = dfunc(*args, **kwargs)
    if isinstance(expect, tuple):
        assert len(actual) == len(expect)
        for a, e in zip(actual, expect):
            assert np.allclose(a, e)
    else:
        assert np.allclose(actual, expect)


@pytest.mark.parametrize('function, args, kwargs, expect', [
    (lambda a=3, b=-4: np.sqrt(a * a + b * b), (), {}, ValueError),
    (lambda a, b=-4: np.sqrt(a * a + b * b), (), dict(a=3), ValueError),
    (lambda a, b: np.sqrt(a * a + b * b), ([1, 2], 3), {}, ValueError),
])
def test_grad_error(function, args, kwargs, expect):
    with pytest.raises(expect):
        ng.grad(function)(*args, **kwargs)


if __name__ == '__main__':
    pytest.main([__file__])
