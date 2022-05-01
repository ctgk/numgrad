import numpy as np
import pytest

import numflow as nf
from numflow._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.mark.parametrize('function, x, y', [
])
def test_gradient(function, x, y):
    x, y = nf.Variable(x), nf.Variable(y)
    with nf.Graph() as g:
        output = function(x, y)
    dx_actual, dy_actual = g.gradient(output, [x, y])
    dx_expected, dy_expected = _numerical_grad(function, x, y)
    assert np.allclose(dx_expected, dx_actual)
    assert np.allclose(dy_expected, dy_actual)
