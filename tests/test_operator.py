import pytest

from pygrad._core._operator import _Operator


@pytest.mark.parametrize('args, name', [
    ((1, 2), None),
])
def test_operator_init_error(args, name):
    with pytest.raises(AttributeError):
        _Operator(*args, name)


if __name__ == "__main__":
    pytest.main([__file__])
