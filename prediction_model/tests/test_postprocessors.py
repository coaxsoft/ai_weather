import unittest

import numpy as np

from prediction_model.postprocessors import MaxWeightPostprocessor


class TestMaxClassPostprocessor(unittest.TestCase):
    def test_call(self):
        post = MaxWeightPostprocessor()
        value = np.array([6, 6, 2, 3]).reshape((4, 1))
        new_value = post(value)
        self.assertEqual(new_value.sum(), new_value.max())
