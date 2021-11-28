import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.mark.parametrize('labels, logits', [
    (
        gd.Tensor(np.random.rand(2, 3), is_variable=True),
        gd.Tensor(np.random.uniform(-9, 9, (4, 2, 3)), is_variable=True),
    ),
])
def test_sigmoid_cross_entropy_1(labels, logits):
    gd.stats.sigmoid_cross_entropy(labels, logits).backward()
    dlabels, dlogits = _numerical_grad(
        gd.stats.sigmoid_cross_entropy, labels, logits)
    assert np.allclose(dlabels, labels.grad, rtol=0, atol=1e-2)
    assert np.allclose(dlogits, logits.grad, rtol=0, atol=1e-2)


@pytest.mark.parametrize('labels, logits', [
    (
        gd.Tensor([0., 1, 1]),
        gd.Tensor(np.random.uniform(-9, 9, (4, 2, 3)), is_variable=True),
    ),
])
def test_sigmoid_cross_entropy_2(labels, logits):
    gd.stats.sigmoid_cross_entropy(labels, logits).backward()
    dlogits = _numerical_grad(
        lambda a: gd.stats.sigmoid_cross_entropy(
            labels, a), logits)[0]
    assert np.allclose(dlogits, logits.grad, rtol=0, atol=1e-2)


@pytest.mark.parametrize('labels, logits, axis', [
    (
        gd.Tensor([[0., 1, 0], [1, 0, 1]], is_variable=True),
        gd.Tensor(np.random.uniform(-9, 9, (2, 3)), is_variable=True),
        0,
    ),
    (
        gd.Tensor([[0., 1, 0], [1, 0, 0]], is_variable=True),
        gd.Tensor(np.random.uniform(-9, 9, (2, 3)), is_variable=True),
        1,
    ),
])
def test_softmax_cross_entropy_1(labels, logits, axis):
    gd.stats.softmax_cross_entropy(labels, logits, axis).backward()
    dlabels, dlogits = _numerical_grad(
        lambda x, y: gd.stats.softmax_cross_entropy(x, y, axis),
        labels, logits)
    assert np.allclose(dlabels, labels.grad, rtol=0, atol=1e-2)
    assert np.allclose(dlogits, logits.grad, rtol=0, atol=1e-2)


@pytest.mark.parametrize('labels, logits, axis', [
    (
        gd.Tensor([[0., 1, 0], [1, 0, 1]]),
        gd.Tensor(np.random.uniform(-9, 9, (2, 3)), is_variable=True),
        0,
    ),
    (
        gd.Tensor([[0., 1, 0], [1, 0, 0]]),
        gd.Tensor(np.random.uniform(-9, 9, (2, 3)), is_variable=True),
        1,
    ),
])
def test_softmax_cross_entropy_2(labels, logits, axis):
    gd.stats.softmax_cross_entropy(labels, logits, axis).backward()
    dlogits = _numerical_grad(
        lambda a: gd.stats.softmax_cross_entropy(
            labels, a, axis), logits)[0]
    assert np.allclose(dlogits, logits.grad, rtol=0, atol=1e-2)


@pytest.mark.parametrize('labels, logits, axis', [
    (
        [1, 0, 1],
        gd.Tensor(np.random.uniform(-9, 9, (2, 3)), is_variable=True),
        0,
    ),
    (
        gd.Tensor([1, 0], dtype=int),
        gd.Tensor(np.random.uniform(-9, 9, (2, 3)), is_variable=True),
        1,
    ),
    (
        gd.Tensor([[1, 0], [2, 1]], dtype=int),
        gd.Tensor(np.random.uniform(-9, 9, (2, 2, 4)), is_variable=True),
        -1,
    ),
])
def test_sparse_softmax_cross_entropy(labels, logits, axis):
    gd.stats.sparse_softmax_cross_entropy(labels, logits, axis).backward()
    dlogits = _numerical_grad(
        lambda a: gd.stats.sparse_softmax_cross_entropy(
            labels, a, axis), logits)[0]
    assert np.allclose(dlogits, logits.grad, rtol=0, atol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__])
