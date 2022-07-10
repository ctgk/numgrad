import warnings

import numpy as np
import pytest
import scipy.special as sp

import numgrad as ng
from numgrad._utils._numerical_grad import _numerical_grad

from tests.test_numpy.test_mathematical import test_differentiation_mathematical  # noqa: I100, I202, E501
from tests.test_numpy.test_sorting_searching_counting import test_differentiation_sorting_searching_counting  # noqa: I100, I202, E501
from tests.test_numpy.test_statistics import test_differentiation_statistics  # noqa: I100, I202, E501


np.random.seed(0)


indexings = [
    (lambda a: a[0], np.array([3, 1, 9])),
    (lambda a: a[::2], np.array([3, 1, 9])),
    (lambda a: a[np.array([0])], np.random.rand(4, 2, 3)),
]
array_creation = [
    (lambda a: np.linspace(a, 10), 0),
    (lambda a: np.linspace(-2, a), 0),
    (lambda a, b: np.linspace(a, b, 10), ([-2, 3], [4, 5])),
    (lambda a, b: np.linspace(a, b, 10, endpoint=False), ([-2, 3], [4, 5])),
    (lambda a, b: np.linspace(a, b, 10, axis=1), ([-2, 3], [[4], [5]])),
    (lambda a, b: np.linspace(a, b, 10, axis=-1), ([-2, 3], [[4], [5]])),
    (lambda a: np.diag(a), np.array([1, 2])),
    (lambda a: np.diag(a, k=1), np.array([1, 2])),
    (lambda a: np.diag(a, k=-2), np.array([1, 2])),
    (lambda a: np.diag(a), np.random.rand(3, 3)),
    (lambda a: np.diag(a), np.random.rand(2, 3)),
    (lambda a: np.diag(a, k=1), np.random.rand(3, 3)),
    (lambda a: np.diag(a, k=-1), np.random.rand(2, 3)),
    (lambda a: np.diagflat(a), np.array([[1, 2], [3, 4]])),
    (lambda a: np.diagflat(a), np.array([[1, 2, 3], [3, 4, 5]])),
    (lambda a: np.diagflat(a, k=2), np.array([[1, 2, 3]])),
    (lambda a: np.tril(a), np.random.rand(3, 3)),
    (lambda a: np.tril(a, k=1), np.random.rand(3, 3)),
    (lambda a: np.tril(a, k=-2), np.random.rand(3, 3)),
    (lambda a: np.tril(a), np.random.rand(3, 5)),
    (lambda a: np.tril(a), np.random.rand(5, 3)),
    (lambda a: np.triu(a), np.random.rand(3, 3)),
    (lambda a: np.triu(a, k=1), np.random.rand(3, 3)),
    (lambda a: np.triu(a, k=-2), np.random.rand(3, 3)),
    (lambda a: np.triu(a), np.random.rand(3, 5)),
    (lambda a: np.triu(a), np.random.rand(5, 3)),
    (lambda a: np.vander(a), np.array([-1, 2, -3])),
    (lambda a: np.vander(a, 0), np.array([-1, 2, -3])),
    (lambda a: np.vander(a, 1), np.array([-1, 2, -3])),
    (lambda a: np.vander(a, 2), np.array([-1, 2, -3])),
    (lambda a: np.vander(a, 4, True), np.array([-1, 2, -3])),
]
array_manipulation = [
    (lambda a: a.reshape(2, 3), np.arange(6)),
    (lambda a: a.reshape(-1, 3), np.arange(6)),
    (lambda a: np.reshape(a, (3, -1)), np.arange(6)),
    (lambda a: a.ravel(), np.random.rand(2, 3)),
    (lambda a: np.ravel(a), np.random.rand(2, 3)),
    (lambda a: a.flatten(), np.random.rand(3, 2)),
    (lambda a: np.moveaxis(a, 0, 2), np.random.rand(4, 2, 3)),
    (lambda a: np.moveaxis(a, 0, -1), np.random.rand(4, 2, 3)),
    (lambda a: np.moveaxis(a, 1, 1), np.random.rand(4, 2, 3)),
    (lambda a: np.moveaxis(a, -2, -1), np.random.rand(4, 2, 3)),
    (lambda a: np.swapaxes(a, 0, 1), np.array([[1, 2, 3]])),
    (lambda a: a.swapaxes(0, 2), np.arange(8).reshape(2, 2, 2)),
    (lambda a: np.transpose(a), np.random.rand(2, 3)),
    (lambda a: a.transpose(), np.random.rand(2, 3)),
    (lambda a: np.transpose(a, (0, 2, 1)), np.random.rand(2, 3, 4)),
    (lambda a: a.transpose(0, 2, 1), np.random.rand(2, 3, 4)),
    (lambda a: a.T, np.random.rand(2, 3)),
    (lambda *a: np.atleast_1d(*a), 5),
    (lambda *a: sum(np.atleast_1d(*a)), (0, 5)),
    (lambda *a: sum(np.atleast_1d(*a)), ([1], 5)),
    (lambda *a: sum(np.atleast_1d(*a)), ([1], [[5, 2]])),
    (lambda *a: np.atleast_2d(*a), 5),
    (lambda *a: sum(np.atleast_2d(*a)), (0, 5)),
    (lambda *a: sum(np.atleast_2d(*a)), ([1], 5)),
    (lambda *a: sum(np.atleast_2d(*a)), ([1], [[5, 2]])),
    (lambda *a: np.atleast_3d(*a), 5),
    (lambda *a: sum(np.atleast_3d(*a)), (0, 5)),
    (lambda *a: sum(np.atleast_3d(*a)), ([1], 5)),
    (lambda *a: sum(np.atleast_3d(*a)), ([1], [[5, 2]])),
    (lambda a: np.broadcast_to(a, 4), 5),
    (lambda a: np.broadcast_to(a, 4), np.array([5])),
    (lambda a: np.broadcast_to(a, (4, 2)), np.array([5])),
    (lambda a: np.broadcast_to(a, (4, 2)), np.array([[5], [4], [3], [2]])),
    (lambda a: np.multiply(*np.broadcast_arrays(a, [[0, 1]])), [[2], [3]]),
    (lambda *a: np.multiply(*np.broadcast_arrays(*a)), ([[2], [3]], [[0, 1]])),
    (lambda a: np.expand_dims(a, 0), 1),
    (lambda a: np.expand_dims(a, 0), np.array([1, 2])),
    (lambda a: np.expand_dims(a, 1), np.array([1, 2])),
    (lambda a: np.expand_dims(a, (0, 1)), np.array([1, 2])),
    (lambda a: np.expand_dims(a, (2, 0)), np.array([1, 2])),
    (lambda a: np.squeeze(a), 1),
    (lambda a: np.squeeze(a), np.random.rand(1, 3, 1)),
    (lambda a: a.squeeze(0), np.random.rand(1, 3, 1)),
    (lambda a: np.squeeze(a, 2), np.random.rand(1, 3, 1)),
    (lambda a: a.squeeze(), np.random.rand(1, 1)),
    (lambda a: np.asarray(a), 0),
    (lambda a: np.asarray(a), [0]),
    (lambda a: np.asanyarray(a), 0),
    (lambda a: np.asanyarray(a), [0]),
    (lambda *a: np.concatenate(a), ([0], [1])),
    (lambda a: np.concatenate([a, [[1], [2]]], axis=1), np.random.rand(2, 3)),
    (lambda *a: np.stack(a), ([1], [2])),
    (
        lambda *a: np.stack(a, axis=1),
        tuple(np.random.rand(3, 4) for _ in range(5)),
    ),
    (lambda a: np.vstack([a, [[4], [5], [6]]]), [[1], [2], [3]]),
    (lambda *a: np.vstack(a), ([1, 2, 3], [4, 5, 6])),
    (lambda *a: np.vstack(a), (np.ones((2, 3, 4)), np.zeros((5, 3, 4)))),
    (lambda a: np.hstack([a, [[4], [5], [6]]]), [[1, 1], [2, 2], [3, 3]]),
    (lambda *a: np.hstack(a), ([1, 2, 3], [4, 5])),
    (lambda *a: np.hstack(a), (np.ones((2, 3, 4)), np.zeros((2, 5, 4)))),
    (lambda a: np.dstack([a, [[4], [5], [6]]]), [[1], [2], [3]]),
    (lambda *a: np.dstack(a), ([1, 2, 3], np.random.rand(1, 3, 1))),
    (lambda *a: np.dstack(a), (np.ones((2, 3, 4)), np.zeros((2, 3, 1)))),
    (lambda *a: np.column_stack(a), ([1, 2, 3], [2, 3, 4])),
    (lambda *a: np.column_stack(a), ([1, 2, 3], [[2], [3], [4]])),
    (lambda *a: np.row_stack(a), ([1, 2, 3], [[2, 3, 4], [5, 6, 7]])),
    (lambda a: sum(np.split(a, 2)), np.random.rand(2, 3, 4)),
    (
        lambda a: sum(r.prod() for r in np.split(a, [3, 6], 2)),
        np.random.rand(2, 2, 4),
    ),
    (lambda a: sum(np.split(a, 3, axis=-1)), np.random.rand(2, 2, 4, 6)),
    (lambda a: sum(np.array_split(a, 1)), np.random.rand(2, 3, 4)),
    (
        lambda a: sum(r.prod() for r in np.array_split(a, [3, 6], 2)),
        np.random.rand(2, 2, 4),
    ),
    (lambda a: sum(np.array_split(a, 3, axis=-1)), np.random.rand(2, 2, 4, 6)),
    (lambda a: sum(np.dsplit(a, 2)), np.random.rand(2, 3, 4)),
    (
        lambda a: sum(r.prod() for r in np.dsplit(a, np.array([3, 6]))),
        np.random.rand(2, 2, 4),
    ),
    (lambda a: sum(np.dsplit(a, 2)), np.random.rand(2, 2, 4, 6)),
    (lambda a: sum(np.hsplit(a, 2)), np.random.rand(3, 2, 4)),
    (
        lambda a: sum(r.prod() for r in np.hsplit(a, np.array([3, 6]))),
        np.random.rand(2, 4, 2),
    ),
    (lambda a: sum(np.hsplit(a, 2)), np.random.rand(2, 2, 4, 6)),
    (lambda a: sum(np.vsplit(a, 2)), np.random.rand(2, 3, 4)),
    (
        lambda a: sum(r.prod() for r in np.vsplit(a, np.array([3, 6]))),
        np.random.rand(4, 2, 2),
    ),
    (lambda a: sum(np.vsplit(a, 2)), np.random.rand(2, 2, 4, 6)),
    (lambda a: np.flip(a), np.random.rand(2, 2, 2)),
    (lambda a: np.flip(a, 0), np.random.rand(2, 2, 2)),
    (lambda a: np.flip(a, 1), np.random.rand(2, 2, 2)),
    (lambda a: np.flip(a, (0, 2)), np.random.rand(2, 2, 2)),
    (lambda a: np.fliplr(a), np.diag([1, 2, 3])),
    (lambda a: np.fliplr(a), np.random.rand(2, 3, 5)),
    (lambda a: np.flipud(a), np.diag([1, 2, 3])),
    (lambda a: np.flipud(a), np.random.rand(2, 3, 5)),
    (lambda a: np.roll(a, 2), np.arange(10)),
    (lambda a: np.roll(a, -3), np.arange(10)),
    (lambda a: np.roll(a, 1), np.arange(10).reshape(2, 5)),
    (lambda a: np.roll(a, -1), np.arange(10).reshape(2, 5)),
    (lambda a: np.roll(a, 1, axis=0), np.arange(10).reshape(2, 5)),
    (lambda a: np.roll(a, -1, axis=1), np.arange(10).reshape(2, 5)),
    (lambda a: np.roll(a, (2, 1), axis=(1, 0)), np.arange(10).reshape(2, 5)),
    (lambda a: np.rot90(a), [[1, 2, 3], [3, 4, 5]]),
    (lambda a: np.rot90(a, 2), [[np.nan, 2, 3], [3, -1, 2]]),
    (lambda a: np.rot90(a, 1, (1, 2)), np.arange(12).reshape(2, 3, 2)),
]
linear_algebra = [
    (lambda a, b: np.dot(a, b), (1, 2)),
    (lambda a, b: np.dot(a, b), ([1, 2], [-1, 1])),
    (lambda a, b: np.dot(a, b), ([1, 2], np.random.rand(4, 2, 3))),
    (lambda a, b: np.dot(a, b), (np.random.rand(3, 2), [1, 2])),
    (
        lambda a, b: np.dot(a, b),
        (np.random.rand(3, 2, 4), np.random.rand(5, 4, 1)),
    ),
    (lambda a: np.vdot(a, 2), 1),
    (lambda a: np.vdot(2, a), 1),
    (lambda a, b: np.vdot(a, b), (1, 2)),
    (lambda a, b: np.vdot(a, b), ([1, 2], [2, 3])),
    (lambda a, b: np.vdot(a, b), (np.random.rand(2, 6), np.random.rand(4, 3))),
    (lambda a, b: np.inner(a, b), (2, 3)),
    (lambda a, b: np.inner(a, b), ([2, 3], [-1, 2])),
    (lambda a, b: np.inner(a, b), (np.random.rand(4, 3, 2), [-1, 2])),
    (lambda a, b: np.inner(a, b), ([-1, 2], np.random.rand(4, 3, 2))),
    (
        lambda a, b: np.inner(a, b),
        (np.random.rand(2, 3, 2), np.random.rand(4, 2)),
    ),
    (lambda a, b: np.outer(a, b), (1, 2)),
    (lambda a, b: np.outer(a, b), ([1, 2, 3], [-1, 0, 1])),
    (
        lambda a, b: a @ np.outer(b, [1, 2, 3]),
        (np.random.rand(2, 3), [-1, 0, 1]),
    ),
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
    (
        lambda a: np.linalg.cholesky(0.5 * (a + np.swapaxes(a, -1, -2))),
        np.eye(2),
    ),
    (
        lambda a: np.linalg.cholesky(0.5 * (a + np.swapaxes(a, -1, -2))),
        np.random.rand(2, 3, 3) + np.eye(3),
    ),
    (lambda a: np.linalg.det(a), np.eye(2)),
    (lambda a: np.linalg.det(a), np.random.rand(2, 3, 3) + np.eye(3) * 10),
    (lambda a: np.linalg.slogdet(a)[1], np.eye(2)),
    (
        lambda a: np.linalg.slogdet(a)[1],
        np.random.rand(2, 3, 3) + np.eye(3) * 10,
    ),
    (
        lambda a: np.multiply(*np.linalg.slogdet(a)),
        np.random.rand(2, 3, 3) + np.eye(3) * -10,
    ),
    (lambda a: np.trace(a), np.eye(2)),
    (lambda a: np.trace(a, 1, 1, 2), np.random.rand(2, 3, 4)),
    (lambda a, b: np.linalg.solve(a, b), (np.eye(2), np.eye(2))),
    (lambda a, b: np.linalg.solve(a, b), (np.eye(2), np.ones(2))),
    (
        lambda a, b: np.linalg.solve(a, b),
        (np.random.rand(2, 3, 3) + np.eye(3), np.random.rand(2, 3)),
    ),
    (
        lambda a, b: np.linalg.solve(a, b),
        (np.random.rand(2, 3, 3) + np.eye(3), np.random.rand(2, 3, 5)),
    ),
    (lambda a: np.linalg.inv(a), np.random.rand(2, 3, 3) + np.eye(3)),
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
    (lambda a: np.multiply(*sp.logsumexp(a, return_sign=True)), [-1, 1]),
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
            *array_creation,
            *array_manipulation,
            *linear_algebra,
            *test_differentiation_mathematical,
            *random,
            *test_differentiation_sorting_searching_counting,
            *test_differentiation_statistics,
        ),
    ),
    *scipy_specials,
])
def parameters(request):
    return request.param


def test_computational_graph_backward(parameters):
    f = parameters[0]
    args = parameters[1] if isinstance(
        parameters[1], tuple) else (parameters[1],)
    args = tuple(ng.Variable(a) for a in args)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return_type_of_function = type(f(*args))
        assert return_type_of_function != ng.Variable
        with ng.Graph() as g:
            y = f(*args)
        print([node.function for node in g._node_list])
        assert type(y) == ng.Variable
        if return_type_of_function == float:
            assert type(y._data) == ng.config.dtype
        else:
            assert type(y._data) == return_type_of_function

        assert type(f(*args)) == return_type_of_function
        dargs_actual = g.backward(y, args)
        dargs_expected = _numerical_grad(f, *args)
        for arg, actual, expected in zip(args, dargs_actual, dargs_expected):
            assert type(arg._data) == type(actual)
            if isinstance(actual, np.ndarray):
                assert arg.shape == actual.shape
            assert np.allclose(expected, actual)

        dy = np.random.uniform(-10, 10)
        dargs_with_dy = g.backward(y, args, target_grad=dy)
        for arg, actual, expected in zip(args, dargs_with_dy, dargs_expected):
            assert type(arg._data) == type(actual)
            assert np.allclose(dy * expected, actual)


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
