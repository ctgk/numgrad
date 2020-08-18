import numpy as np
import pytest

import pygrad as pg


np.random.seed(0)


@pytest.mark.parametrize('in_features, out_features, bias, dtype, error', [
    (2, 3, True, pg.Float64, None),
])
def test_init_error(in_features, out_features, bias, dtype, error):
    if error is None:
        pg.nn.Dense(in_features, out_features, bias, dtype)
    else:
        with pytest.raises(error):
            pg.nn.Dense(in_features, out_features, bias, dtype)


@pytest.mark.parametrize('in_features, out_features, dtype', [
    (2, 3, None),
    (4, 10, pg.Float32),
])
def test_init_with_bias(in_features, out_features, dtype):
    dense = pg.nn.Dense(in_features, out_features, True, dtype)
    assert dense.weight.shape == (in_features, out_features)
    assert dense.weight.dtype == pg.config.dtype if dtype is None else dtype
    assert dense.bias.shape == (out_features,)
    assert dense.bias.dtype == pg.config.dtype if dtype is None else dtype


@pytest.mark.parametrize('in_features, out_features, dtype', [
    (2, 3, None),
    (4, 10, pg.Float32),
])
def test_init_without_bias(in_features, out_features, dtype):
    dense = pg.nn.Dense(in_features, out_features, False, dtype)
    assert dense.weight.shape == (in_features, out_features)
    assert dense.weight.dtype == pg.config.dtype if dtype is None else dtype
    assert not hasattr(dense, 'bias')


@pytest.mark.parametrize('in_features, out_features, bias, dtype', [
    (2, 3, True, None),
    (4, 10, False, pg.Float32),
])
def test_call(in_features, out_features, bias, dtype):
    x = pg.random.normal(0, 1, size=(10, in_features))
    if dtype is not None:
        x = x.astype(dtype)
    dense = pg.nn.Dense(in_features, out_features, bias=bias, dtype=dtype)
    actual = dense(x)
    if bias:
        assert np.allclose(
            x.value @ dense.weight.value + dense.bias.value, actual.value)
    else:
        assert np.allclose(x.value @ dense.weight.value, actual.value)


@pytest.mark.parametrize('in_features, out_features, bias, dtype', [
    (2, 3, True, None),
    (4, 10, False, pg.Float32),
])
def test_train(in_features, out_features, bias, dtype):
    x = pg.random.normal(0, 1, size=(1, in_features))
    y = pg.random.normal(0, 1, size=(1, out_features))
    if dtype is not None:
        x = x.astype(dtype)
        y = y.astype(dtype)
    dense = pg.nn.Dense(in_features, out_features, bias=bias, dtype=dtype)
    optimizer = pg.optimizers.Gradient(dense.trainables(), 1e-1)
    for _ in range(100):
        loss = pg.square(dense(x) - y).sum()
        optimizer.minimize(loss)
        print(loss)
    assert np.isclose(loss.value, 0)


if __name__ == "__main__":
    pytest.main([__file__])
