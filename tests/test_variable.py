import numpy as np
import pytest

import pygrad as gd


def test_init_error():
    with pytest.raises(ValueError):
        gd.Variable(1, dtype=int)


def test_init():
    gd.Variable([1, 2])


def test_init_pass_dtype():
    assert gd.Variable(1, np.float32).dtype == np.float32


def test_default_dtype():
    assert gd.Variable(1).dtype == float


def test_non_default_dtype():
    gd.config.dtype = np.float64
    assert gd.Variable(1).dtype == np.float64
    gd.config.dtype = float


def test_ufunc():
    a = gd.Variable([0, 1])
    assert type(a + 0) == gd.Variable


@pytest.mark.xfail
def test_view():
    a = np.array([1, 2]).view(gd.Variable)
    assert a.dtype == float


if __name__ == '__main__':
    pytest.main([__file__])
