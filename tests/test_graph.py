import numpy as np
import pytest

import numflow as nf


def test_enter_exit():
    assert nf.config._graph is None
    with nf.Graph() as g:
        assert nf.config._graph is g
    assert nf.config._graph is None


def test_enter_exit_error():
    with nf.Graph():
        with pytest.raises(ValueError):
            with nf.Graph():
                pass


def test_automatic_operation_storing():
    a = nf.Variable(-1)
    with nf.Graph() as g:
        b = np.square(a)
    assert len(g._node_list) == 1
    assert g._node_list[0].result is b
    assert g._node_list[0].inputs[0] is a


@pytest.mark.parametrize('function, args, expect', [
    (lambda a, b: a + b, (1, 1), int),
    (lambda a, b: a + b, (nf.Variable(1), 1), nf.Variable),
    (lambda a, b: a + b, (1, nf.Variable(1)), nf.Variable),
    (lambda a, b: a + b, (np.float64(1), nf.Variable(1)), nf.Variable),
])
def test_result_type(function, args, expect):
    if not isinstance(args, tuple):
        args = (args,)
    with nf.Graph():
        result = function(*args)
    assert type(result) == expect


if __name__ == '__main__':
    pytest.main([__file__])
