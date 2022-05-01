import pytest

import numflow as nf


def test_version():
    assert isinstance(nf.__version__, str)
    assert nf.__version__.count('.') == 2


if __name__ == '__main__':
    pytest.main([__file__])
