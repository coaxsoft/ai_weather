import unittest

import numpy
import pandas

import settings
from prediction_model import to_vector, reverse_norma, max_or_null, to_list, none_to_zero
from prediction_model.estimators import StandardEstimator, cv
from prediction_model.converters import CSVDataConverter, MongoIntersectConverter
from prediction_model.preprocessors import Word2ClassPreprocessor
from prediction_model.reducers import MinkowskiReducer, EuclideanReducer

from prediction_model.tests import TEST_WEATHER_PATH


MONGO_URI = settings.MONGO_URI
MONGO_DB = 'weather_db_test'


CHECK_WORDS = {
    'sun': 1,
    'cloud': 2,
    'rain': 3,
    'shower': 4,
    'thunderstorm': 5,
    'fog': 6,
    'snow': 7,
}

KEY_MAP = {
    't_min': (
        ('temperature', 'min'),
    ),
    't_max': (
        ('temperature', 'max'),
    ),
    'class': (
        ('description',),
        (Word2ClassPreprocessor(CHECK_WORDS),)
    ),
}


class TestPredModel(unittest.TestCase):
    def test_to_vector(self):
        s: pandas.Series = pandas.Series(data=[3, 4, 5, 6, 7, 8, 1, 2, 3, 4])
        vec = to_vector(s)
        self.assertEqual(vec.shape, (len(s), 1))

    def test_normalization(self):
        vec = numpy.array([2, 2]).reshape((2, 1))
        no = reverse_norma(vec)
        for v in no:
            self.assertEqual(v, 0.5)

    def test_max_or_null(self):
        vec = numpy.array([3, 4, 1, 1, 2]).reshape((5, 1))
        self.assertEqual(numpy.sum(max_or_null(vec)), numpy.max(vec))

    def test_to_list(self):
        conv = CSVDataConverter(TEST_WEATHER_PATH)
        est = StandardEstimator(*conv.convert(KEY_MAP))
        est.reduce()
        data = to_list(est.produce(conv.predict)['data'])
        for v in data.values():
            self.assertTrue(isinstance(v, list))

    def test_none_to_zero(self):
        vec = numpy.array([None, None]).reshape((2, 1))
        none_to_zero(vec)
        self.assertEqual(numpy.sum(vec), 0)

    def test_none_to_zero_copy(self):
        vec = numpy.array([None, None])
        new = none_to_zero(vec, True)
        self.assertEqual(numpy.sum(new), 0)
        for v in vec:
            self.assertEqual(v, None)

    def test_cv(self):
        real = {
            'data': {
                'temperature': numpy.array([1, 3, 5, 7, 6, 4]).reshape(1, 6),
            },
            'slots': ['one', 'two'],
            'labels': [1, 3, 4, 5, 6, 7]
        }
        data = real.copy()
        res = cv(data, real)
        for v in res['data'].values():
            self.assertEqual(numpy.sum(v), 0)


class TestEstimator(unittest.TestCase):
    def test_estimator_type_error_raise(self):
        conv = CSVDataConverter(TEST_WEATHER_PATH)
        pred, real = conv.convert(KEY_MAP)
        del pred['slots']
        with self.assertRaises(KeyError):
            StandardEstimator(pred, real)

    def test_estimator_reduce_l2(self):
        real_weights = [0.66470588, 0.33529412]
        conv = CSVDataConverter(TEST_WEATHER_PATH)
        est = StandardEstimator(*conv.convert(KEY_MAP))
        est.reduce()
        for d, r in zip(est.weights['temperature'].T[0], real_weights):
            self.assertAlmostEqual(d, r)

    def test_estimator_reduce_l1(self):
        real_weights = [0.57777778, 0.42222222]
        conv = CSVDataConverter(TEST_WEATHER_PATH)
        est = StandardEstimator(*conv.convert(KEY_MAP))
        est.reduce(MinkowskiReducer())
        for d, r in zip(est.weights['temperature'].T[0], real_weights):
            self.assertAlmostEqual(d, r)

    def test_from_converter(self):
        real_weights = [0.57777778, 0.42222222]
        est = StandardEstimator.from_converter(CSVDataConverter(TEST_WEATHER_PATH), KEY_MAP)
        est.reduce(MinkowskiReducer())
        for d, r in zip(est.weights['temperature'].T[0], real_weights):
            self.assertAlmostEqual(d, r)

    def test_estimator_produce(self):
        test_temp_list = numpy.array([-3, 7, -4, 6]).reshape(2, 2)
        data = {
            'data': {
                'temperature': test_temp_list,
            },
            'slots': ['one', 'two'],
            'labels': [i for i in range(test_temp_list.shape[1])]
        }
        conv = CSVDataConverter(TEST_WEATHER_PATH)
        est = StandardEstimator(*conv.convert(KEY_MAP))
        est.reduce(EuclideanReducer())
        w = est.weights['temperature']
        res = est.produce(data)['data']['temperature']

        test_res = w.T.dot(test_temp_list)
        self.assertEqual(test_res[0, 0], res[0, 0])
        self.assertEqual(test_res[0, 1], res[0, 1])

    def test_estimator_produce_index_error(self):
        test_temp_list = numpy.array([-3, 7, -4, 6]).reshape(1, 4)
        data = {
            'data': {
                'temperature': test_temp_list,
            },
            'slots': ['one', 'two'],
            'labels': [1, 3, 4, 5, 6, 7]
        }
        real = {
            'data': {
                'temperature': numpy.array([1, 3, 5, 7, 6, 4]).reshape(1, 6),
            },
            'slots': ['one', 'two'],
            'labels': [1, 3, 4, 5, 6, 7]
        }
        est = StandardEstimator(data, real)

        with self.assertRaises(IndexError):
            est.reduce(EuclideanReducer())


class TestEstimatorMongo(unittest.TestCase):
    def test_weights(self):
        converter = MongoIntersectConverter.from_mongo(mongo_uri=MONGO_URI,
                                                       mongo_db=MONGO_DB,
                                                       data_collections=('gismeteo',
                                                                         'meteoprog'),
                                                       real_collection='actual_weather',
                                                       city='Ivano-Frankivsk',
                                                       country='Ukraine',
                                                       distance=0,
                                                       limit=200)
        est = StandardEstimator.from_converter(converter, KEY_MAP)
        est.reduce()
        w = est.get_weights()
        for v in w.values():
            self.assertEqual(v.shape, (2, 1))
