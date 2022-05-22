import numpy as np
import pytest

import numgrad as ng


def test_enter_exit():
    assert ng.config._graph is None
    with ng.Graph() as g:
        assert ng.config._graph is g
    assert ng.config._graph is None


def test_enter_exit_error():
    with ng.Graph():
        with pytest.raises(ValueError):
            with ng.Graph():
                pass


def test_automatic_operation_storing():
    a = ng.Variable(-1)
    with ng.Graph() as g:
        b = np.square(a)
    assert len(g._node_list) == 1
    assert g._node_list[0].result is b
    assert g._node_list[0].inputs[0] is a


def test_multiple_graphs():
    a = ng.Variable(-1)
    with ng.Graph(_allow_multiple_graphs=True) as g1:
        b = np.square(a)
        with ng.Graph(_allow_multiple_graphs=True) as g2:
            c = np.square(b)
    assert len(g1._node_list) == 2
    assert len(g2._node_list) == 1
    assert g1._node_list[0].result is b
    assert g1._node_list[0].inputs[0] is a
    assert g1._node_list[1].result is c
    assert g1._node_list[1].inputs[0] is b


@pytest.mark.parametrize('function, args, expect', [
    (lambda a, b: a + b, (1, 1), int),
    (lambda a, b: a + b, (ng.Variable(1), 1), ng.Variable),
    (lambda a, b: a + b, (1, ng.Variable(1)), ng.Variable),
    (lambda a, b: a + b, (np.float64(1), ng.Variable(1)), ng.Variable),
])
def test_result_type(function, args, expect):
    if not isinstance(args, tuple):
        args = (args,)
    with ng.Graph():
        result = function(*args)
    assert type(result) == expect


if __name__ == '__main__':
    pytest.main([__file__])
