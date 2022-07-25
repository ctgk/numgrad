import numpy as np
import pytest

from tests.test_differentiation import (  # noqa:I202
    _test_egrad,
    _test_graph_backward,
    _test_graph_backward_custom_grad,
)


@pytest.fixture(params=[
    # https://numpy.org/doc/stable/reference/routines.array-creation.html#numerical-ranges
    (lambda a: np.linspace(a, 10), 0),
    (lambda a: np.linspace(-2, a), 0),
    (lambda a, b: np.linspace(a, b, 10), ([-2, 3], [4, 5])),
    (lambda a, b: np.linspace(a, b, 10, endpoint=False), ([-2, 3], [4, 5])),
    (lambda a, b: np.linspace(a, b, 10, axis=1), ([-2, 3], [[4], [5]])),
    (lambda a, b: np.linspace(a, b, 10, axis=-1), ([-2, 3], [[4], [5]])),

    # https://numpy.org/doc/stable/reference/routines.array-creation.html#building-matrices
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
