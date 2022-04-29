import numpy as np
import pytest

import pygrad as gd


def test_init_error():
    with pytest.raises(ValueError):
        gd.Tensor(1, dtype=int)


def test_init():
    gd.Tensor([1, 2])


def test_init_pass_dtype():
    assert gd.Tensor(1, np.float32).dtype == np.float32


def test_default_dtype():
    assert gd.Tensor(1).dtype == float


def test_non_default_dtype():
    gd.config.dtype = np.float64
    assert gd.Tensor(1).dtype == np.float64
    gd.config.dtype = float


def test_ufunc():
    a = gd.Tensor([0, 1])
    assert type(a + 0) == gd.Tensor


if __name__ == '__main__':
    pytest.main([__file__])
