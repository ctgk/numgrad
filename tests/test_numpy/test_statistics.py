import numpy as np
import pytest

from tests.test_differentiation import (  # noqa:I202
    _test_egrad,
    _test_graph_backward,
    _test_graph_backward_custom_grad,
)


@pytest.fixture(params=[
    # https://numpy.org/doc/stable/reference/routines.statistics.html#order-statistics
    (lambda a: np.ptp(a), [-1, 1]),
    (lambda a: np.ptp(a, axis=1), [[-1, 1], [-2, 3]]),
    (lambda a: np.ptp(a, axis=0, keepdims=True), [[-1, 1], [-2, 3]]),

    # https://numpy.org/doc/stable/reference/routines.statistics.html#averages-and-variances
    (lambda a: np.mean(a), -1),
    (lambda a: np.mean(a), [-1, 1]),
    (lambda a: a.mean(axis=1), np.random.rand(3, 2)),
    (lambda a: np.mean(a, (0, 2), keepdims=True), np.random.rand(4, 2, 3)),
    (lambda a: np.std(a), -1),
    (lambda a: np.std(a), [-1, 1]),
    (lambda a: a.std(axis=1), np.random.rand(3, 2)),
    (lambda a: np.std(a, (0, 2), keepdims=True), np.random.rand(4, 2, 3)),
    (lambda a: np.var(a), -1),
    (lambda a: np.var(a), [-1, 1]),
    (lambda a: a.var(axis=1), np.random.rand(3, 2)),
    (lambda a: np.var(a, (0, 2), keepdims=True), np.random.rand(4, 2, 3)),
    (lambda a: np.nanmean(a), 1),
    (lambda a: np.nanmean(a), np.nan),
    (lambda a: np.nanmean(a), [1, np.nan, -3]),
    (lambda a: np.nanmean(a), [[1, np.nan, -3], [np.nan, np.nan, 5]]),
    (lambda a: np.nanmean(a, 1), [[1, np.nan, -3], [np.nan, np.nan, 5]]),
    (lambda a: np.nanstd(a), 1),
    (lambda a: np.nanstd(a), np.nan),
    (lambda a: np.nanstd(a), [1, np.nan, -3]),
    (lambda a: np.nanstd(a), [[1, np.nan, -3], [np.nan, np.nan, 5]]),
    (lambda a: np.nanstd(a, 1), [[1, np.nan, -3], [np.nan, np.nan, 5]]),
    (lambda a: np.nanvar(a), 1),
    (lambda a: np.nanvar(a), np.nan),
    (lambda a: np.nanvar(a), [1, np.nan, -3]),
    (lambda a: np.nanvar(a), [[1, np.nan, -3], [np.nan, np.nan, 5]]),
    (lambda a: np.nanvar(a, 1), [[1, np.nan, -3], [np.nan, np.nan, 5]]),
    (lambda a: np.correlate(a, [0, 1, 0.5], mode='full'), [1, 2, 3]),
    (lambda a: np.correlate(a, [0, 1, 0.5], mode='same'), [1, 2, 3]),
    (lambda a: np.correlate(a, [0, 1, 0.5], mode='valid'), [1, 2, 3]),
    (lambda a, v: np.correlate(a, v, mode='full'), ([1, 2, 3], [0, 1, 0.5])),
    (lambda a, v: np.correlate(a, v, mode='same'), ([1, 2, 3], [0, 1, 0.5])),
    (lambda a, v: np.correlate(a, v, mode='valid'), ([1, 2, 3], [0, 1, 0.5])),
    (lambda a, v: np.correlate(a, v, mode='valid'), ([1, 2, 3, 4], [2, 1, 3])),
    (lambda a, v: np.correlate(a, v, mode='full'), ([1, 2, 3], [1, 0.5])),
    (lambda a, v: np.correlate(a, v, mode='same'), ([1, 2, 3], [1, 0.5])),
    (lambda a, v: np.correlate(a, v, mode='valid'), ([1, 2, 3], [1, 0.5])),
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
