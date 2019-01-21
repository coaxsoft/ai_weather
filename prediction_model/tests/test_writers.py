import unittest

import datetime

import settings
from prediction_model import to_list
from prediction_model.converters import MongoIntersectConverter
from prediction_model.estimators import StandardEstimator
from prediction_model.io.writers import MongoWriter
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


class TestMongoWritter(unittest.TestCase):
    def setUp(self):
        self.city = 'Ivano-Frankivsk'
        self.country = 'Ukraine'
        self.distance = 0

        self.converter = MongoIntersectConverter.from_mongo(mongo_uri=MONGO_URI,
                                                            mongo_db=MONGO_DB,
                                                            data_collections=('gismeteo',
                                                                              'meteoprog'),
                                                            real_collection='actual_weather',
                                                            city=self.city,
                                                            country=self.country,
                                                            distance=self.distance,
                                                            limit=200)
        self.est = StandardEstimator.from_converter(self.converter, KEY_MAP)

    def test_write_weights(self):
        writer = MongoWriter(MONGO_URI, MONGO_DB, {'city': self.city,
                                                   'country': self.country,
                                                   'forecast_distance': self.distance})
        self.est.reduce()
        writer.write_weights(self.est)

        today = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        count = writer.mongo[writer.weights_collection].find({'updated': today,
                                                              'forecast_distance': self.distance}).count()
        self.assertTrue(count > 0)

    def test_write_produced(self):
        writer = MongoWriter(MONGO_URI, MONGO_DB, {'city': self.city,
                                                   'forecast_distance': self.distance})
        data = self.converter.predict
        self.est.reduce()
        produced = self.est.produce(data)
        produced['data'] = to_list(produced['data'])
        writer.write_produced(produced)
        today = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        count = writer.mongo[writer.produced_collection].find({'updated': today,
                                                               'label': today,
                                                               'city': self.city,
                                                               'forecast_distance': self.distance}).count()
        self.assertTrue(count >= 0)
        self.assertTrue(len(self.est.y['labels']) != 0)
        res = writer.mongo[writer.produced_collection].find({'updated': today,
                                                             'label': today,
                                                             'city': self.city,
                                                             'forecast_distance': self.distance})
        labels: list = data['labels']
        for v in res:
            self.assertTrue(v['label'] in labels)
        writer.mongo[writer.produced_collection].delete_many({'updated': today,
                                                              'city': self.city,
                                                              'forecast_distance': self.distance})

    def test_write_errors(self):
        writer = MongoWriter(MONGO_URI, MONGO_DB, {'city': self.city,
                                                   'forecast_distance': self.distance})
        data = self.converter.predict
        self.est.reduce()
        produced = self.est.produce(data)
        produced['data'] = to_list(produced['data'])
        writer.write_errors(produced)
        today = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        count = writer.mongo[writer.errors_collection].find({'updated': today,
                                                             'city': self.city,
                                                             'forecast_distance': self.distance}).count()
        self.assertEqual(count, len(self.est.y['labels']))
        res = writer.mongo[writer.errors_collection].find({'updated': today,
                                                           'city': self.city,
                                                           'forecast_distance': self.distance})
        labels: list = data['labels']
        for v in res:
            self.assertTrue(v['label'] in labels)
        writer.mongo[writer.errors_collection].delete_many({'updated': today,
                                                            'city': self.city,
                                                            'forecast_distance': self.distance})
