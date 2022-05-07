import pytest

import numgrad as ng


def test_version():
    assert isinstance(ng.__version__, str)
    assert ng.__version__.count('.') == 2


if __name__ == '__main__':
    pytest.main([__file__])
