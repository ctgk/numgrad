import numpy as np
import pytest

import numflow as nf


def test_default_dtype():
    assert nf.config.dtype == float


def test_set_dtype_error():
    with pytest.raises(ValueError):
        nf.config.dtype = int


def test_set_dtype():
    assert nf.config.dtype == float
    nf.config.dtype = np.float32
    assert nf.config.dtype == np.float32
    nf.config.dtype = float
    assert nf.config.dtype == float


if __name__ == '__main__':
    pytest.main([__file__])
