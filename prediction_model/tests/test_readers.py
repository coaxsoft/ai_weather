import unittest

import datetime

import settings
from prediction_model.converters import MongoIntersectConverter
from prediction_model.estimators import StandardEstimator
from prediction_model.io.writers import MongoWriter
from prediction_model.io.readers import MongoReader
from prediction_model.preprocessors import Word2ClassPreprocessor

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


class TestMongoReader(unittest.TestCase):
    def setUp(self):
        self.city = 'Ivano-Frankivsk'
        self.country = 'Ukraine'
        self.distance = 0

        converter = MongoIntersectConverter.from_mongo(mongo_uri=MONGO_URI,
                                                       mongo_db=MONGO_DB,
                                                       data_collections=('gismeteo',
                                                                         'meteoprog'),
                                                       real_collection='actual_weather',
                                                       city=self.city,
                                                       country=self.country,
                                                       distance=self.distance,
                                                       limit=200)
        self.est = StandardEstimator.from_converter(converter, KEY_MAP)

    def test_read_weights(self):
        writer = MongoWriter(MONGO_URI, MONGO_DB, {'city': self.city,
                                                   'country': self.country,
                                                   'forecast_distance': self.distance})
        reader = MongoReader(MONGO_URI, MONGO_DB)
        self.est.reduce()
        writer.write_weights(self.est)
        today = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        reader.search(self.city, self.country, self.distance, today)
        weights = reader.read_weights()

        for k, v in weights.items():
            self.assertTrue(v.shape, (len(v), 1))
            for l, r in zip(v.T[0], self.est.get_weights()[k].T[0]):
                self.assertEqual(l, r)

        writer.mongo[writer.weights_collection].delete_one({'updated': today})

    def test_read_weights_error(self):
        reader = MongoReader(MONGO_URI, MONGO_DB)
        with self.assertRaises(TypeError):
            reader.read_weights()

    def test_read_weights_empty(self):
        reader = MongoReader(MONGO_URI, MONGO_DB)
        reader.search(self.city, self.country, -1)
        w = reader.read_weights()
        self.assertEqual(w, None)
