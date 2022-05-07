# noqa: D100

import numpy as np
import pytest

import numgrad as ng


@pytest.fixture(autouse=True)
def _add_np(doctest_namespace):
    doctest_namespace['np'] = np


@pytest.fixture(autouse=True)
def _add_ng(doctest_namespace):
    doctest_namespace['ng'] = ng
