import pytest

import pygrad as gd


def test_version():
    assert isinstance(gd.__version__, str)
    assert gd.__version__.count('.') == 2


if __name__ == '__main__':
    pytest.main([__file__])
