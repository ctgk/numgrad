import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.mark.parametrize('x', [
    gd.Array(np.random.uniform(-10, 10, (2, 3)), is_variable=True),
    gd.Array(np.random.uniform(-10, 10, (4, 2, 3)), is_variable=True),
])
def test_numerical_grad(x):
    with gd.Graph() as g:
        gd.tan(x)
    g.backward()
    dx = _numerical_grad(gd.tan, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
