import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.mark.parametrize('x, axis, keepdims', [
    (
        gd.Array(np.random.uniform(-9, 9, (2, 3)), is_variable=True),
        None, False,
    ),
    (
        gd.Array(np.random.uniform(-9, 9, (4, 2, 3)), is_variable=True),
        1, False,
    ),
    (
        gd.Array(np.random.uniform(-9, 9, (4, 2, 3)), is_variable=True),
        (0, -1), True,
    ),
])
def test_numerical_grad(x, axis, keepdims):
    args = {
        k: v for k, v in zip(('axis', 'keepdims'), (axis, keepdims))
        if v is not None
    }
    with gd.Graph() as g:
        gd.min(x, **args)
    g.backward()
    dx = _numerical_grad(lambda x: gd.min(x, **args), x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
