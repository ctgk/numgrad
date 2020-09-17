import numpy as np
import pytest

import pygrad as gd


def test_ema():
    a = np.arange(24).reshape(4, 3, 2)
    bn = gd.nn.BatchNormalization((3, 2))
    for _ in range(100):
        bn(a, update_emas=True)
    assert np.allclose(bn.mean_ema, np.mean(a, 0), rtol=0, atol=1e-2)
    assert np.allclose(bn.var_ema, np.var(a, 0), rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
