import numpy as np
import pytest
import scipy.special as sp

import pygrad as gd


np.random.seed(0)


def test_forward():
    logits = [0, 0.5, -0.5]
    p = sp.softmax(logits)
    actual = gd.random.categorical(
        gd.Array(logits, is_variable=True), size=(10000, 3))
    assert actual.is_variable is False
    assert np.allclose(p, np.mean(actual.data, axis=0), rtol=0, atol=0.01)


if __name__ == "__main__":
    pytest.main([__file__])
