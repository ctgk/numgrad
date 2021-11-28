# noqa: D100

import numpy as np
import pytest

import pygrad as gd


@pytest.fixture(autouse=True)
def _add_np(doctest_namespace):
    doctest_namespace['np'] = np


@pytest.fixture(autouse=True)
def _add_gd(doctest_namespace):
    doctest_namespace['gd'] = gd
