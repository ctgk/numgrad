import numpy as np
import pytest

from tests.test_differentiation import (  # noqa:I202
    _test_egrad,
    _test_graph_backward,
    _test_graph_backward_custom_grad,
)


@pytest.fixture(params=[
    # https://numpy.org/doc/stable/reference/routines.array-manipulation.html#changing-array-shape
    (lambda a: a.reshape(2, 3), np.arange(6)),
    (lambda a: a.reshape(-1, 3), np.arange(6)),
    (lambda a: np.reshape(a, (3, -1)), np.arange(6)),
    (lambda a: a.ravel(), np.random.rand(2, 3)),
    (lambda a: np.ravel(a), np.random.rand(2, 3)),
    (lambda a: a.flatten(), np.random.rand(3, 2)),

    # https://numpy.org/doc/stable/reference/routines.array-manipulation.html#transpose-like-operations
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

    # https://numpy.org/doc/stable/reference/routines.array-manipulation.html#changing-number-of-dimensions
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

    # https://numpy.org/doc/stable/reference/routines.array-manipulation.html#joining-arrays
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

    # https://numpy.org/doc/stable/reference/routines.array-manipulation.html#splitting-arrays
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

    # https://numpy.org/doc/stable/reference/routines.array-manipulation.html#rearranging-elements
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
])
def parameters(request):
    return request.param


def test_differentiation(parameters):
    f = parameters[0]
    args = parameters[1] if isinstance(
        parameters[1], tuple) else (parameters[1],)
    _test_graph_backward(f, *args)
    _test_graph_backward_custom_grad(f, *args)
    _test_egrad(f, *args)


if __name__ == '__main__':
    pytest.main([__file__])
