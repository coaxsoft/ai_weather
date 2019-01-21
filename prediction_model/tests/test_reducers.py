import unittest

from prediction_model.reducers import EuclideanReducer, MinkowskiReducer


class TestReducers(unittest.TestCase):
    def test_euclidean_reducer(self):
        reducer = EuclideanReducer()
        self.assertEqual(reducer(2, 1), 1)

    def test_minkowski_reducer(self):
        reducer = MinkowskiReducer()
        self.assertEqual(reducer(2, 1), 1)