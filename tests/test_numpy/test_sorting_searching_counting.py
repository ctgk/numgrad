import numpy as np


test_differentiation_sorting_searching_counting = [
    (lambda a: np.sort(a), [[3, 2, 1], [4, 6, -1]]),
    (lambda a: np.sort(a, axis=0), [[3, 2, 1], [4, 6, -1]]),
    (lambda a: np.sort(a, axis=None), [[3, 2, 1], [4, 6, -1]]),
]
