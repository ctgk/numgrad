import numpy as np
import pytest
import scipy.special as sp

from tests.test_differentiation import (  # noqa:I202
    _test_egrad,
    _test_graph_backward,
    _test_graph_backward_custom_grad,
)


@pytest.fixture(params=[
    # https://docs.scipy.org/doc/scipy/reference/special.html#raw-statistical-functions
    (lambda a: sp.expit(a), [1, 0, -1]),
    (lambda a: sp.log_expit(a), [1, 0, -1]),

    # https://docs.scipy.org/doc/scipy/reference/special.html#gamma-and-related-functions
    (sp.gamma, [1, 0.5, 3.3]),

    # https://docs.scipy.org/doc/scipy/reference/special.html#other-special-functions
    (lambda a: sp.softmax(a), -1),
    (lambda a: sp.softmax(a), [-1, 1]),
    (lambda a: sp.softmax(a, axis=1), np.random.rand(3, 2)),
    (lambda a: sp.softmax(a, axis=(0, 2)), np.random.rand(4, 2, 3)),
    (lambda a: sp.softmax(a), np.random.rand(4, 2, 3)),
    (lambda a: sp.log_softmax(a), [0.5, 0, -0.5]),
    (
        lambda a: sp.log_softmax(a, axis=-1),
        [[0.5, 0, -0.5], [0, 1, 2]],
    ),
    (lambda a: sp.log_softmax(a, axis=(0, 2)), np.random.rand(2, 3, 4)),

    # https://docs.scipy.org/doc/scipy/reference/special.html#convenience-functions
    (lambda a: sp.logsumexp(a), -1),
    (lambda a: np.multiply(*sp.logsumexp(a, return_sign=True)), [-1, 1]),
    (lambda a: sp.logsumexp(a, axis=1), np.random.rand(3, 2)),
    (
        lambda a: sp.logsumexp(a, axis=(0, 2), keepdims=True),
        np.random.rand(4, 2, 3),
    ),
])
def parameters(request):
    return request.param


def test_graph_backward(parameters):
    f = parameters[0]
    args = parameters[1] if isinstance(
        parameters[1], tuple) else (parameters[1],)
    _test_egrad(f, *args)
    _test_graph_backward(f, *args)
    _test_graph_backward_custom_grad(f, *args)


if __name__ == '__main__':
    pytest.main([__file__])
