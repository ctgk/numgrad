import numpy as np
import pytest

import pygrad as pg
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('labels, logits, axis', [
    (
        pg.Array([1, 0, 1]),
        pg.Array(np.random.uniform(-9, 9, (2, 3)), is_differentiable=True),
        0
    ),
    (
        pg.Array([1, 0]),
        pg.Array(np.random.uniform(-9, 9, (2, 3)), is_differentiable=True),
        1
    ),
    (
        pg.Array([[1, 0], [2, 1]]),
        pg.Array(np.random.uniform(-9, 9, (2, 2, 4)), is_differentiable=True),
        -1
    )
])
def test_numerical_grad(labels, logits, axis):
    pg.stats.sparse_softmax_cross_entropy(labels, logits, axis).backward()
    dlogits = _numerical_grad(
        lambda a: pg.stats.sparse_softmax_cross_entropy(
            labels, a, axis), logits)[0]
    assert np.allclose(dlogits, logits.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
