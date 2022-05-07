import numpy as np
import pytest
import scipy.special as sp

import numflow as nf
from numflow._utils._numerical_grad import _numerical_grad


np.random.seed(0)


indexings = [
    (lambda a: a[0], np.array([3, 1, 9])),
    (lambda a: a[::2], np.array([3, 1, 9])),
    (lambda a: a[np.array([0])], np.random.rand(4, 2, 3)),
]
array_manipulation = [
    (lambda a: a.reshape(2, 3), np.arange(6)),
    (lambda a: a.reshape(-1, 3), np.arange(6)),
    (lambda a: np.reshape(a, (3, -1)), np.arange(6)),
    (lambda a: np.transpose(a), np.random.rand(2, 3)),
    (lambda a: a.transpose(), np.random.rand(2, 3)),
    (lambda a: np.transpose(a, (0, 2, 1)), np.random.rand(2, 3, 4)),
    (lambda a: a.transpose(0, 2, 1), np.random.rand(2, 3, 4)),
    (lambda a: a.T, np.random.rand(2, 3)),
]
linear_algebra = [
    (lambda a: a @ [1, 2], [1, 2]),
    (lambda a: np.matmul(a, [1, 2]), [[1, 2], [3, 4]]),
    (lambda a: a @ [[1, 2], [3, 4]], [1, 2]),
    (lambda a: np.matmul(a, [[1, 2], [3, 4]]), [[1, 2], [3, 4]]),
    (lambda a: a @ [[1, 2], [3, 4]], np.random.rand(3, 4, 2)),
    (lambda a, b: a @ b, ([1, 2], [1, 2])),
    (lambda a, b: np.matmul(a, b), ([1, 2], [[1, 2], [3, 4]])),
    (lambda a, b: a @ b, ([[1, 2], [3, 4]], [1, 2])),
    (lambda a, b: np.matmul(a, b), ([[1, 2], [3, 4]], [[1, 2], [3, 4]])),
    (lambda a, b: a @ b, (np.random.rand(3, 4, 2), [[1, 2], [3, 4]])),
]
trigonometrics = [
    (np.cos, np.random.uniform(-10, 10, (3, 2))),
    (np.sin, np.random.uniform(-10, 10, (2, 5))),
    (np.tan, np.random.uniform(-10, 10, (4, 1))),
    (np.arcsin, np.random.uniform(-1, 1, (3, 4))),
    (np.arccos, np.random.uniform(-1, 1, (3, 4))),
    (np.arctan, np.random.uniform(-10, 10, (5, 3))),
    (lambda a: np.hypot(a, 4), 3),
    (lambda a: np.hypot([2, 1], a), [[1], [-2]]),
    (np.hypot, (np.random.normal(size=(3,)), np.random.normal(size=(4, 1)))),
]
hyperbolics = [
    (np.cosh, np.random.uniform(-10, 10, (3, 4))),
    (np.sinh, np.random.uniform(-10, 10, (1, 5))),
    (np.tanh, np.random.uniform(-10, 10, (4, 2))),
    (np.arcsinh, np.random.uniform(-10, 10, (4, 2, 3))),
    (np.arccosh, np.random.uniform(1, 10, (5, 2))),
    (np.arctanh, np.random.uniform(-1, 1, (2,))),
]
sum_products_differences = [
    (lambda a: np.sum(a), -1),
    (lambda a: np.sum(a), [-1, 1]),
    (lambda a: a.sum(axis=1), np.random.rand(3, 2)),
    (lambda a: np.sum(a, (0, 2), keepdims=True), np.random.rand(4, 2, 3)),
]
exponents_logarithms = [
    (np.exp, [-1, -0.2, 0.5, 2]),
    (np.expm1, [-1, -0.2, 0.5, 2]),
    (np.exp2, [-1, -0.2, 0.5, 2]),
    (np.log, [1, 0.2, 0.5, 2]),
    (np.log10, [1, 0.2, 0.5, 2]),
    (np.log2, [1, 0.2, 0.5, 2]),
    (np.log1p, [1, 0.2, 0.5, 2, -0.9]),
    (lambda a: np.logaddexp(a, [1, 2]), np.random.rand(4, 2)),
    (lambda a: np.logaddexp([1, 2], a), np.random.rand(4, 2)),
    (
        np.logaddexp,
        (np.random.normal(size=(3, 4)), np.random.normal(size=(5, 1, 4))),
    ),
    (lambda a: np.logaddexp2(a, [1, 2]), np.random.rand(4, 2)),
    (lambda a: np.logaddexp2([1, 2], a), np.random.rand(4, 2)),
    (
        np.logaddexp2,
        (np.random.normal(size=(3, 4)), np.random.normal(size=(5, 1, 4))),
    ),
]
arithmetics = [
    (np.positive, -3),
    (lambda a: +a, -3),
    (np.negative, -3),
    (lambda a: -a, -3),
    (np.reciprocal, [1, -2]),
    (lambda a: np.add(a, [[1, 2], [3, 4]]), [1, 2]),
    (lambda a: a + [[1, 2], [3, 4]], [1, 2]),
    (lambda a, b: a + b, ([[1, 2]], [[1], [2]])),
    (np.add, ([[1, 2]], [[1], [2]])),
    (lambda a: np.subtract(a, [[1, 2], [3, 4]]), [1, 2]),
    (lambda a: a - [[1, 2], [3, 4]], [1, 2]),
    (lambda a, b: a - b, ([[1, 2]], [[1], [2]])),
    (np.subtract, ([[1, 2]], [[1], [2]])),
    (lambda a: np.multiply(a, [[1, 2], [3, 4]]), [1, 2]),
    (lambda a: a * [[1, 2], [3, 4]], [1, 2]),
    (lambda a: np.float64(1) * a, [1, 2]),
    (lambda a, b: a * b, ([[1, 2]], [[1], [2]])),
    (np.multiply, ([[1, 2]], [[1], [2]])),
    (lambda a: np.divide(a, [[1, 2], [3, 4]]), [1, 2]),
    (lambda a: a / [[1, 2], [3, 4]], [1, 2]),
    (lambda a, b: a / b, ([[1, 2]], [[1], [2]])),
    (np.divide, ([[1, 2]], [[1], [2]])),
    (np.true_divide, ([[1, 2]], [[1], [2]])),
    (lambda a: np.power(a, [[1], [-2]]), [[1, 2]]),
    (np.power, ([[1, 2]], [[1], [-2]])),
    (lambda a: np.float_power(a, [[1], [-2]]), [[1, 2]]),
    (np.float_power, ([[1, 2]], [[1], [-2]])),
]
extrema_finding = [
    (np.maximum, (3, -1)),
    (np.maximum, (0.5, np.random.rand(3, 2))),
    (np.maximum, (np.random.rand(4, 3), 0.5)),
    (np.maximum, (np.random.rand(2, 3, 4), np.random.rand(1, 4))),
    (np.fmax, (np.nan, 3)),
    (np.fmax, (3, np.nan)),
    (np.fmax, ([1, np.nan, -1], [[-0.5], [0.5]])),
    # (np.fmax, ([1, np.nan, -1], [[-0.5], [np.nan]])),
    (np.amax, 9),
    (np.amax, [1, 2]),
    (np.max, 9),
    (np.max, [1, 2]),
    (lambda a: a.max(axis=1), np.random.rand(2, 3) * 10),
    (lambda a: a.max(axis=(0, 2), keepdims=True), np.random.rand(2, 4, 3)),
    # (lambda a: np.nanmax(a), np.nan),
    (lambda a: np.nanmax(a), [np.nan, 1]),
    (lambda a: np.nanmax(a, axis=0, keepdims=True), [np.nan, 1]),
    (np.minimum, (3, -1)),
    (np.minimum, (0.5, np.random.rand(3, 2))),
    (np.minimum, (np.random.rand(4, 3), 0.5)),
    (np.minimum, (np.random.rand(2, 3, 4), np.random.rand(1, 4))),
    (np.fmin, (np.nan, 3)),
    (np.fmin, (3, np.nan)),
    (np.fmin, ([1, np.nan, -1], [[-0.5], [0.5]])),
    # (np.fmin, ([1, np.nan, -1], [[-0.5], [np.nan]])),
    (np.amin, 9),
    (np.amin, [1, 2]),
    (np.min, 9),
    (np.min, [1, 2]),
    (lambda a: a.min(axis=1), np.random.rand(2, 3) * 10),
    (lambda a: a.min(axis=(0, 2), keepdims=True), np.random.rand(2, 4, 3)),
    # (lambda a: np.nanmax(a), np.nan),
    (lambda a: np.nanmax(a), [np.nan, 1]),
    (lambda a: np.nanmax(a, axis=0, keepdims=True), [np.nan, 1]),
    (lambda a: np.nanmin(a), [np.nan, 1]),
    (lambda a: np.nanmin(a, axis=0, keepdims=True), [np.nan, 1]),
]
miscellaneous = [
    (np.sqrt, [3, 0.5]),
    (np.cbrt, [3, 0.5]),
    (np.square, [2, -1]),
    (np.abs, [2, -1]),
    (np.absolute, [2, -1]),
    (np.fabs, [2, -1]),
]
random = [
    (lambda a: (np.random.seed(0), np.random.exponential(a))[1], [1, 10]),
    (
        lambda a: (
            np.random.seed(0), np.random.exponential(a, size=(5, 2)),
        )[1],
        [1, 10],
    ),
    (lambda a: (np.random.seed(0), np.random.normal(a, 1))[1], [-1, 1]),
    (
        lambda a: (np.random.seed(0), np.random.normal(a, 1, size=(3, 2)))[1],
        [-1, 1],
    ),
    (lambda a: (np.random.seed(0), np.random.normal(0, a))[1], [1, 5]),
    (
        lambda a: (np.random.seed(0), np.random.normal(0, a, size=(4, 2)))[1],
        [1, 5],
    ),
    (lambda a, b: (np.random.seed(0), np.random.normal(a, b))[1], (0, 1)),
    (lambda a: (np.random.seed(0), np.random.uniform(a, 10))[1], [-1, 1]),
    (
        lambda a: (
            np.random.seed(0),
            np.random.uniform(-10, a, size=(3, 2)),
        )[1],
        [-1, 1],
    ),
    (lambda a, b: (np.random.seed(0), np.random.uniform(a, b))[1], (0, 1)),
]
statistics = [
    (lambda a: np.mean(a), -1),
    (lambda a: np.mean(a), [-1, 1]),
    (lambda a: a.mean(axis=1), np.random.rand(3, 2)),
    (lambda a: np.mean(a, (0, 2), keepdims=True), np.random.rand(4, 2, 3)),
    (lambda a: np.nanmean(a), 1),
    (lambda a: np.nanmean(a), np.nan),
    (lambda a: np.nanmean(a), [1, np.nan, -3]),
    (lambda a: np.nanmean(a), [[1, np.nan, -3], [np.nan, np.nan, 5]]),
    (lambda a: np.nanmean(a, 1), [[1, np.nan, -3], [np.nan, np.nan, 5]]),
]
scipy_specials = [
    (sp.gamma, [1, 0.5, 3.3]),
    (lambda a: sp.expit(a), [1, 0, -1]),
    (lambda a: sp.log_expit(a), [1, 0, -1]),
    (lambda a: sp.log_softmax(a), [0.5, 0, -0.5]),
    (
        lambda a: sp.log_softmax(a, axis=-1),
        [[0.5, 0, -0.5], [0, 1, 2]],
    ),
    (lambda a: sp.log_softmax(a, axis=(0, 2)), np.random.rand(2, 3, 4)),
    (lambda a: sp.logsumexp(a), -1),
    (lambda a: sp.logsumexp(a), [-1, 1]),
    (lambda a: sp.logsumexp(a, axis=1), np.random.rand(3, 2)),
    (
        lambda a: sp.logsumexp(a, axis=(0, 2), keepdims=True),
        np.random.rand(4, 2, 3),
    ),
    (lambda a: sp.softmax(a), -1),
    (lambda a: sp.softmax(a), [-1, 1]),
    (lambda a: sp.softmax(a, axis=1), np.random.rand(3, 2)),
    (lambda a: sp.softmax(a, axis=(0, 2)), np.random.rand(4, 2, 3)),
    (lambda a: sp.softmax(a), np.random.rand(4, 2, 3)),
]


@pytest.fixture(params=[
    *(  # numpy
        *(  # array objects
            *indexings,
        ),
        *(  # routines
            *array_manipulation,
            *linear_algebra,
            *(  # mathematical functions
                *trigonometrics,
                *hyperbolics,
                *sum_products_differences,
                *exponents_logarithms,
                *arithmetics,
                *extrema_finding,
                *miscellaneous,
            ),
            *random,
            *statistics,
        ),
    ),
    *scipy_specials,
])
def parameters(request):
    return request.param


def test_computation_graph_gradient(parameters):
    f = parameters[0]
    args = parameters[1] if isinstance(
        parameters[1], tuple) else (parameters[1],)
    args = tuple(nf.Variable(a) for a in args)

    return_type_of_function = type(f(*args))
    assert return_type_of_function != nf.Variable
    with nf.Graph() as g:
        assert len(g._node_list) == 0
        y = f(*args)
    assert len(g._node_list) == 1
    print(g._node_list[0].function)
    assert type(y) == nf.Variable
    if return_type_of_function == float:
        assert type(y._data) == nf.config.dtype
    else:
        assert type(y._data) == return_type_of_function

    assert type(f(*args)) == return_type_of_function
    dargs_actual = g.gradient(y, args)
    dargs_expected = _numerical_grad(f, *args)
    for arg, actual, expected in zip(args, dargs_actual, dargs_expected):
        assert type(arg._data) == type(actual)
        assert np.allclose(expected, actual)

    with nf.Graph() as g:
        y = np.nanmean(f(*args))
    dargs_actual = g.gradient(y, args)
    dargs_expected = _numerical_grad(lambda *a: np.nanmean(f(*a)), *args)
    for actual, expected in zip(dargs_actual, dargs_expected):
        assert np.allclose(expected, actual)


def test_gradient_error():
    a = nf.Variable([0, 0.5])
    with nf.Graph() as g:
        b = np.argsort(a)
    with pytest.raises(Exception):
        g.gradient(b, [a])[0]


if __name__ == '__main__':
    pytest.main([__file__])
