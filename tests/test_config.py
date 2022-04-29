import numpy as np
import pytest

import pygrad as gd


def test_default_dtype():
    assert gd.config.dtype == float


def test_set_dtype_error():
    with pytest.raises(ValueError):
        gd.config.dtype = int


def test_set_dtype():
    assert gd.config.dtype == float
    gd.config.dtype = np.float32
    assert gd.config.dtype == np.float32
    gd.config.dtype = float
    assert gd.config.dtype == float


if __name__ == '__main__':
    pytest.main([__file__])
