import numpy as np
import pytest

import numflow as nf


def test_init_error():
    with pytest.raises(ValueError):
        nf.Variable(1, dtype=int)


def test_init():
    nf.Variable([1, 2])


def test_init_pass_dtype():
    assert nf.Variable(1, np.float32).dtype == np.float32


def test_default_dtype():
    assert nf.Variable(1).dtype == float


def test_non_default_dtype():
    nf.config.dtype = np.float64
    assert nf.Variable(1).dtype == np.float64
    nf.config.dtype = float


def test_ufunc():
    a = nf.Variable([0, 1])
    assert type(a + 0) == np.ndarray


@pytest.mark.xfail
def test_view():
    a = np.array([1, 2]).view(nf.Variable)
    assert a.dtype == float


if __name__ == '__main__':
    pytest.main([__file__])
