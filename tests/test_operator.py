import pytest

from pygrad._core._operator import _Operator


@pytest.mark.parametrize('args, name, error', [
    ((1, 2), None, TypeError),
])
def test_operator_init_error(args, name, error):
    with pytest.raises(error):
        _Operator(*args, name)


if __name__ == "__main__":
    pytest.main([__file__])
