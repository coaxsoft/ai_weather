"""Here should be classes that post-process any data, for example, weights

In active development currently
"""
from abc import ABC, abstractmethod

import numpy as np


class AbstractPostprocessor(ABC):
    @abstractmethod
    def __call__(self, value):
        pass


class MaxWeightPostprocessor(ABC):
    """Postprocessor that zerofy all elements in ``value`` ndarray
    except max element. In the position of max element it assigns 1.

    For example::

        [3, 6, 1] -> [0, 1, 0]
    """
    def __call__(self, value: np.ndarray):
        indices = value.argmax()
        array = np.zeros(shape=value.shape)
        array[indices] = 1
        return array
