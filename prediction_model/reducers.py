"""Contains reducer (cost) functions for finding
distances between true value and got value

Inherit ``AbstractReducer`` and then override
``__call__`` method
"""
from abc import ABC, abstractmethod

import numpy as np


class AbstractReducer(ABC):
    @abstractmethod
    def __call__(self, d, y):
        pass


class EuclideanReducer(AbstractReducer):
    """Calculate Euclidean (L1) distance between two vectors::

        E = (d - y) ^ 2
    """
    def __call__(self, d, y):
        return np.sum((d - y) ** 2)


class MinkowskiReducer(AbstractReducer):
    """Calculate Minkowski (L1) distance between two vectors::

        E = (|d - y| ^ p) ^ 1/p

    :param p: power of distance (just look on the formula)
    """
    def __init__(self, p=1):
        self.p = p

    def __call__(self, d, y):
        return np.sum(np.abs(d - y) ** self.p) ** (1 / self.p)
