import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('labels, logits, axis', [
    (
        gd.Array([[0., 1, 0], [1, 0, 1]], is_variable=True),
        gd.Array(np.random.uniform(-9, 9, (2, 3)), is_variable=True),
        0
    ),
    (
        gd.Array([[0., 1, 0], [1, 0, 0]], is_variable=True),
        gd.Array(np.random.uniform(-9, 9, (2, 3)), is_variable=True),
        1
    ),
])
def test_numerical_grad_1(labels, logits, axis):
    with gd.Graph() as g:
        gd.stats.softmax_cross_entropy(labels, logits, axis)
    g.backward()
    dlabels, dlogits = _numerical_grad(
        lambda x, y: gd.stats.softmax_cross_entropy(x, y, axis),
        labels, logits)
    assert np.allclose(dlabels, labels.grad, rtol=0, atol=1e-2)
    assert np.allclose(dlogits, logits.grad, rtol=0, atol=1e-2)


@pytest.mark.parametrize('labels, logits, axis', [
    (
        gd.Array([[0., 1, 0], [1, 0, 1]]),
        gd.Array(np.random.uniform(-9, 9, (2, 3)), is_variable=True),
        0
    ),
    (
        gd.Array([[0., 1, 0], [1, 0, 0]]),
        gd.Array(np.random.uniform(-9, 9, (2, 3)), is_variable=True),
        1
    ),
])
def test_numerical_grad_2(labels, logits, axis):
    with gd.Graph() as g:
        gd.stats.softmax_cross_entropy(labels, logits, axis)
    g.backward()
    dlogits = _numerical_grad(
        lambda a: gd.stats.softmax_cross_entropy(
            labels, a, axis), logits)[0]
    assert np.allclose(dlogits, logits.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
