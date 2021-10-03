import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('labels, logits', [
    (
        gd.Array(np.random.rand(2, 3), is_variable=True),
        gd.Array(np.random.uniform(-9, 9, (4, 2, 3)), is_variable=True)
    ),
])
def test_numerical_grad_1(labels, logits):
    with gd.Graph() as g:
        gd.stats.sigmoid_cross_entropy(labels, logits)
    g.backward()
    dlabels, dlogits = _numerical_grad(
        gd.stats.sigmoid_cross_entropy, labels, logits)
    print(dlabels)
    print(labels.grad)
    assert np.allclose(dlabels, labels.grad, rtol=0, atol=1e-2)
    assert np.allclose(dlogits, logits.grad, rtol=0, atol=1e-2)


@pytest.mark.parametrize('labels, logits', [
    (
        gd.Array([0., 1, 1]),
        gd.Array(np.random.uniform(-9, 9, (4, 2, 3)), is_variable=True)
    ),
])
def test_numerical_grad_2(labels, logits):
    with gd.Graph() as g:
        gd.stats.sigmoid_cross_entropy(labels, logits)
    g.backward()
    dlogits = _numerical_grad(
        lambda a: gd.stats.sigmoid_cross_entropy(
            labels, a), logits)[0]
    assert np.allclose(dlogits, logits.grad, rtol=0, atol=1e-2)


def test_xor():
    x = gd.Array([[1, 1], [1, -1], [-1, 1], [-1, -1]], gd.Float32)
    y = gd.Array([1, 0, 0, 1], gd.Float32)
    w1 = gd.Array(np.random.normal(size=(2, 10)), gd.Float32, True)
    b1 = gd.Array([0] * 10, gd.Float32, True)
    w2 = gd.Array(np.random.normal(size=(10,)), gd.Float32, True)
    b2 = gd.Array(0, gd.Float32, True)

    def predict_logit(x, w1, b1, w2, b2):
        x = gd.tanh(x @ w1 + b1)
        return x @ w2 + b2

    def loss_func(w1, b1, w2, b2):
        return gd.stats.sigmoid_cross_entropy(
            y, predict_logit(x, w1, b1, w2, b2), name='sce').sum()

    optimizer = gd.optimizers.Gradient([w1, b1, w2, b2], 1e-1)
    with gd.Graph() as g:
        loss_func(w1, b1, w2, b2)
    for i in range(101):
        g.forward()
        g.backward()
        if i % 20 == 0:
            dw1, db1, dw2, db2 = _numerical_grad(
                loss_func, w1, b1, w2, b2, epsilon=1e-3)
            print(dw1)
            print(w1.grad)
            assert np.allclose(w1.grad, dw1, rtol=0, atol=1e-2)
            assert np.allclose(b1.grad, db1, rtol=0, atol=1e-2)
            assert np.allclose(w2.grad, dw2, rtol=0, atol=1e-2)
            assert np.allclose(b2.grad, db2, rtol=0, atol=1e-2)
        optimizer.minimize(clear_grad=True)
    proba = gd.stats.sigmoid(predict_logit(x, w1, b1, w2, b2))
    assert np.allclose(proba.data, [1, 0, 0, 1], rtol=0, atol=1e-1)


if __name__ == "__main__":
    pytest.main([__file__])
