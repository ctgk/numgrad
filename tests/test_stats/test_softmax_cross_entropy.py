import numpy as np
import pytest

import pygrad as pg
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('labels, logits, axis', [
    (
        pg.Array([[0., 1, 0], [1, 0, 1]], is_variable=True),
        pg.Array(np.random.uniform(-9, 9, (2, 3)), is_variable=True),
        0
    ),
    (
        pg.Array([[0., 1, 0], [1, 0, 0]], is_variable=True),
        pg.Array(np.random.uniform(-9, 9, (2, 3)), is_variable=True),
        1
    ),
])
def test_numerical_grad_1(labels, logits, axis):
    pg.stats.softmax_cross_entropy(labels, logits, axis).backward()
    dlabels, dlogits = _numerical_grad(
        lambda x, y: pg.stats.softmax_cross_entropy(x, y, axis),
        labels, logits)
    assert np.allclose(dlabels, labels.grad, rtol=0, atol=1e-2)
    assert np.allclose(dlogits, logits.grad, rtol=0, atol=1e-2)


@pytest.mark.parametrize('labels, logits, axis', [
    (
        pg.Array([[0., 1, 0], [1, 0, 1]]),
        pg.Array(np.random.uniform(-9, 9, (2, 3)), is_variable=True),
        0
    ),
    (
        pg.Array([[0., 1, 0], [1, 0, 0]]),
        pg.Array(np.random.uniform(-9, 9, (2, 3)), is_variable=True),
        1
    ),
])
def test_numerical_grad_2(labels, logits, axis):
    pg.stats.softmax_cross_entropy(labels, logits, axis).backward()
    dlogits = _numerical_grad(
        lambda a: pg.stats.softmax_cross_entropy(
            labels, a, axis), logits)[0]
    assert np.allclose(dlogits, logits.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
