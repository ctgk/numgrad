import numpy as np
import pytest

import numgrad as ng


def test_init_error():
    with pytest.raises(ValueError):
        ng.Variable(1, dtype=int)


def test_init():
    ng.Variable([1, 2])


def test_init_pass_dtype():
    assert ng.Variable(1, np.float32).dtype == np.float32


def test_default_dtype():
    assert ng.Variable(1).dtype == np.float64


def test_non_default_dtype():
    ng.config.dtype = np.float32
    assert ng.Variable(1).dtype == np.float32
    ng.config.dtype = np.float64


def test_ufunc():
    a = ng.Variable([0, 1])
    assert type(a + 0) == np.ndarray


@pytest.mark.parametrize('self, method, args', [
    (ng.Variable([1, -1]), '__iadd__', 1),
    (ng.Variable([1, -1]), '__isub__', 1),
    (ng.Variable([1, -1]), '__imul__', 2),
    (ng.Variable([1, -1]), '__itruediv__', 2),
])
def test_inplace(self, method, args):
    if not isinstance(args, tuple):
        args = (args,)
    expect_id = id(self)
    expect_id_of_data = id(self._data)
    getattr(self, method)(*args)
    assert expect_id == id(self)
    assert expect_id_of_data == id(self._data)

    with pytest.raises(ValueError):
        with ng.Graph():
            getattr(self, method)(*args)


if __name__ == '__main__':
    pytest.main([__file__])
