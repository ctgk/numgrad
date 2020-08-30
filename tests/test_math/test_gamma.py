import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.mark.parametrize('x', [
    gd.Array(np.random.uniform(1, 9, (2, 3)), is_variable=True),
])
def test_numerical_grad(x):
    with gd.Graph() as g:
        gd.gamma(x)
    g.backward()
    dx = _numerical_grad(lambda x: gd.gamma(x), x)[0]
    print(dx)
    print(x.grad)
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
