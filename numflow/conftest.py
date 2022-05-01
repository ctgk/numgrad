# noqa: D100

import numpy as np
import pytest

import numflow as nf


@pytest.fixture(autouse=True)
def _add_np(doctest_namespace):
    doctest_namespace['np'] = np


@pytest.fixture(autouse=True)
def _add_nf(doctest_namespace):
    doctest_namespace['nf'] = nf
