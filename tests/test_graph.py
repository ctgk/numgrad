import numpy as np
import pytest

import pygrad as gd


def test_enter_exit():
    assert gd.config._graph is None
    with gd.Graph() as g:
        assert gd.config._graph is g
    assert gd.config._graph is None


def test_enter_exit_error():
    with gd.Graph():
        with pytest.raises(ValueError):
            with gd.Graph():
                pass


def test_automatic_operation_storing():
    a = gd.Tensor(-1)
    with gd.Graph() as g:
        b = np.square(a)
    assert len(g._node_list) == 1
    assert g._node_list[0].result is b
    assert g._node_list[0].inputs[0] is a


if __name__ == '__main__':
    pytest.main([__file__])
