import numpy as np
import pytest

import numgrad as ng


def test_default_dtype():
    assert ng.config.dtype == np.float64


def test_set_dtype_error():
    with pytest.raises(ValueError):
        ng.config.dtype = int


def test_set_dtype():
    assert ng.config.dtype == np.float64
    ng.config.dtype = np.float32
    assert ng.config.dtype == np.float32
    ng.config.dtype = np.float64
    assert ng.config.dtype == np.float64


if __name__ == '__main__':
    pytest.main([__file__])
