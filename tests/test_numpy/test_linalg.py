import numpy as np
import pytest

from tests.test_differentiation import (  # noqa:I202
    _test_egrad,
    _test_graph_backward,
    _test_graph_backward_custom_grad,
)


@pytest.fixture(params=[
    # https://numpy.org/doc/stable/reference/routines.linalg.html#matrix-and-vector-products
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

    # https://numpy.org/doc/stable/reference/routines.linalg.html#decompositions
    (
        lambda a: np.linalg.cholesky(0.5 * (a + np.swapaxes(a, -1, -2))),
        np.eye(2),
    ),
    (
        lambda a: np.linalg.cholesky(0.5 * (a + np.swapaxes(a, -1, -2))),
        np.random.rand(2, 3, 3) + np.eye(3),
    ),

    # https://numpy.org/doc/stable/reference/routines.linalg.html#norms-and-other-numbers
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

    # https://numpy.org/doc/stable/reference/routines.linalg.html#solving-equations-and-inverting-matrices
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
