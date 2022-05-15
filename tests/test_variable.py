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


@pytest.mark.parametrize('function, expect', [
    (lambda: np.asarray(ng.Variable([0, 1])), np.array([0., 1.])),
    (lambda: 0. in ng.Variable(0.), TypeError),
    (lambda: 0. in ng.Variable([0.]), True),
    (lambda: 1. in ng.Variable([[0., 1.], [2., 3.]]), True),
    (lambda: -1. not in ng.Variable([[0., 1.], [2., 3.]]), True),
    (lambda: float(ng.Variable(-1)), -1.),
    (lambda: float(ng.Variable([0, -1])), TypeError),
    (lambda: int(ng.Variable(-1)), -1),
    (lambda: int(ng.Variable([0, -1])), TypeError),
    (lambda: len(ng.Variable(-1)), TypeError),
    (lambda: len(ng.Variable([0, -1])), 2),
    (lambda: ng.Variable(0.).item(), 0.),
    (lambda: ng.Variable([0.]).item(), 0.),
    (lambda: ng.Variable([0., 1.]).item(), ValueError),
    (lambda: ng.Variable(1).ndim, 0),
    (lambda: ng.Variable([0, 1]).ndim, 1),
    (lambda: ng.Variable(0).shape, tuple()),
    (lambda: ng.Variable([0, 1]).shape, (2,)),
    (lambda: ng.Variable(0).size, 1),
    (lambda: ng.Variable([0, 1]).size, 2),
    (lambda: ng.Variable(0.).tolist(), 0.),
    (lambda: ng.Variable([0., 1.]).tolist(), [0., 1.]),
    (lambda: ng.Variable([[0., 1.], [2., 3.]]).tolist(), [[0., 1.], [2., 3.]]),
])
def test_method_and_property(function, expect):
    if isinstance(expect, type) and issubclass(expect, Exception):
        with pytest.raises(expect):
            function()
    elif isinstance(expect, np.ndarray):
        assert np.allclose(expect, function())
    else:
        assert function() == expect


@pytest.mark.parametrize('self, method, args', [
    (ng.Variable([1, -1]), '__iadd__', 1),
    (ng.Variable([1, -1]), '__isub__', 1),
    (ng.Variable([1, -1]), '__imul__', 2),
    (ng.Variable([1, -1]), '__itruediv__', 2),
    (ng.Variable([1, -1]), '__setitem__', (0, 2)),
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
