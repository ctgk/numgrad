from unittest.mock import patch

import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('logits, temperature', [
    (
        gd.Array(np.random.rand(2, 3), is_variable=True),
        0.1,
    ),
    (
        gd.Array(np.random.rand(4, 2, 3), is_variable=True),
        1e-3,
    ),
    (
        gd.Array(np.random.rand(3, 2), is_variable=True),
        0.01,
    ),
])
def test_numerical_grad(logits, temperature):
    g = np.random.gumbel(size=logits.shape)
    with patch('numpy.random.gumbel', return_value=g):
        with gd.Graph() as g:
            gd.random.gumbel_sigmoid(logits, temperature)
        g.backward()
        dlogits = _numerical_grad(
            lambda x: gd.random.gumbel_sigmoid(x, temperature),
            logits)[0]
    assert np.allclose(dlogits, logits.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
