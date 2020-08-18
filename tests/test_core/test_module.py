import pytest


import pygrad as pg


def test_init_error_1():
    class Derived(pg.Module):
        def __init__(self):
            self.a = 1

    with pytest.raises(TypeError):
        Derived()


def test_init_error_2():
    class Derived(pg.Module):
        def __init__(self):
            self.a = 1

        def __call__(self):
            pass

    with pytest.raises(RuntimeError):
        Derived()


def test_init_error_3():
    class Derived(pg.Module):
        def __init__(self):
            super().__init__()
            self.a = 1

        def __call__(self):
            pass

    Derived()


if __name__ == "__main__":
    pytest.main([__file__])
