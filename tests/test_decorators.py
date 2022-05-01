import numpy as np
import pytest

import numflow as nf


def test_differentiable():

    def custom_gradient(do, _o, _x):
        return 3 * do

    @nf.differentiable(custom_gradient)
    def twice(x):
        return 2 * x

    a = nf.Variable([4, 2])
    with nf.Graph() as g:
        b = twice(a)
    assert np.allclose(g.gradient(b, [a])[0], np.array([3, 3]))


if __name__ == '__main__':
    pytest.main([__file__])
