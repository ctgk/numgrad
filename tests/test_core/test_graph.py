import pytest

import pygrad as gd


def test_enter_exit():
    assert gd.config.graph is None
    with gd.Graph() as g:
        assert gd.config.graph is g
    assert gd.config.graph is None


def test_enter_exit_error():
    with gd.Graph():
        with pytest.raises(ValueError):
            with gd.Graph():
                pass


if __name__ == '__main__':
    pytest.main([__file__])
