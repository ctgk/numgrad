import numpy as np
import pytest

import pygrad


@pytest.mark.parametrize('type_', [
    pygrad.Int8,
    pygrad.Int16,
    pygrad.Int32,
    pygrad.Int64,
    pygrad.Float16,
    pygrad.Float32,
    pygrad.Float64,
    pygrad.Float128,
])
def test_issubclass_of_datatype(type_):
    assert issubclass(type_, pygrad.DataType)


@pytest.mark.parametrize('obj, dtype, expected', [
    (1, pygrad.Int8, np.int8),
    ([1, 2], pygrad.Int16, np.int16),
    ([1, 2], pygrad.Int32, np.int32),
    ([1, 2], pygrad.Int64, np.int64),
    ([1, 2], pygrad.Float16, np.float16),
    ([1, 2], pygrad.Float32, np.float32),
    ([1, 2], pygrad.Float64, np.float64),
    ([1, 2], pygrad.Float128, np.float128),
])
def test_numpy_asarray(obj, dtype, expected):
    a = np.asarray(obj, dtype=dtype)
    assert a.dtype == expected


if __name__ == "__main__":
    pytest.main([__file__])
