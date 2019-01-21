import unittest
import pymongo

import settings
from prediction_model.converters import MongoIntersectConverter, \
    CSVDataConverter, MongoUnionConverter
from prediction_model.preprocessors import Word2ClassPreprocessor
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


class TestMongoIntersectConverter(unittest.TestCase):
    def setUp(self):
        self.collections = ['gismeteo', 'meteoprog', 'actual_weather']
        self.mongo = pymongo.MongoClient(MONGO_URI)[MONGO_DB]

    def test_convert(self):
        to_converter = {}
        for col in self.collections[:-1]:
            mc = self.mongo[col].find().sort([('weather_date', -1)]).limit(7)
            to_converter[col] = mc
        converter = MongoIntersectConverter(None, None, ['1', '2', '3'], **to_converter)
        data = converter.convert(KEY_MAP)
        X = data[0]
        X_tmin = X['data']['t_min']
        shape = X_tmin.shape
        self.assertEqual((2, 3), shape)

    def test_convert_from_mongo(self):
        converter = MongoIntersectConverter.from_mongo(mongo_uri=MONGO_URI,
                                                       mongo_db=MONGO_DB,
                                                       data_collections=('gismeteo', 'meteoprog'),
                                                       real_collection='actual_weather',
                                                       city='Ivano-Frankivsk',
                                                       country='Ukraine',
                                                       distance=0,
                                                       limit=7)
        data = converter.convert(KEY_MAP)

        X = data[0]
        X_tmin = X['data']['t_min']
        shape = X_tmin.shape
        self.assertEqual((2, 1), shape)

    def test_data_intersection(self):
        mongo = pymongo.MongoClient(MONGO_URI)[MONGO_DB]
        d_cols = ('gismeteo', 'meteoprog')
        r_col = 'actual_weather'
        city = 'Ivano-Frankivsk'
        country = 'Ukraine'
        distance = 0
        dates = MongoIntersectConverter.intersect_dates(mongo=mongo,
                                                        data_collections=d_cols,
                                                        real_collection=r_col,
                                                        city=city,
                                                        country=country,
                                                        distance=distance,
                                                        limit=7)
        self.assertEqual(len(dates), len(set(dates)))
        rd = mongo[r_col].find({'city': city, 'country': country,
                                'weather_date': {'$in': dates}}).count()
        self.assertEqual(rd, len(dates))
        el = {rd}
        for col in d_cols:
            rd = mongo[col].find({'city': city, 'country': country,
                                  'forecast_distance': distance,
                                  'weather_date': {'$in': dates}}).count()
            self.assertEqual(rd, len(dates))
            el.add(rd)
        self.assertTrue(len(el) <= 1)

    def test_service_names(self):
        services = {
            'one': 1,
            'Two': 1,
            'Three': 1,
        }
        conv = MongoIntersectConverter(None, None, {}, **services)
        self.assertListEqual(list(services.keys()), conv.service_names().tolist())


class TestMongoDataConverter(unittest.TestCase):
    def setUp(self):
        self.collections = ['gismeteo', 'meteoprog', 'actual_weather']
        self.mongo = pymongo.MongoClient(MONGO_URI)[MONGO_DB]

    def test_convert_from_mongo(self):
        converter = MongoUnionConverter.from_mongo(mongo_uri=MONGO_URI,
                                                   mongo_db=MONGO_DB,
                                                   data_collections=('gismeteo',
                                                                     'meteoprog'),
                                                   real_collection=None,
                                                   city='Ivano-Frankivsk',
                                                   country='Ukraine',
                                                   distance=0,
                                                   limit=7)
        X, _ = converter.convert(KEY_MAP)

        X_tmin = X['data']['t_min']
        shape = X_tmin.shape
        self.assertEqual((2, 1), shape)

    def test_data_union(self):
        mongo = pymongo.MongoClient(MONGO_URI)[MONGO_DB]
        d_cols = ('gismeteo', 'meteoprog')
        city = 'Ivano-Frankivsk'
        country = 'Ukraine'
        distance = 1
        dates = MongoUnionConverter.union_dates(mongo=mongo,
                                                data_collections=d_cols,
                                                city=city,
                                                country=country,
                                                distance=distance,
                                                limit=7,
                                                dates=None)
        el = set({})
        for col in d_cols:
            rd = mongo[col].find({'city': city, 'country': country,
                                  'forecast_distance': distance,
                                  'weather_date': {'$in': dates}}).count()
            el.add(rd)
        self.assertTrue(len(el) <= len(dates))

    def test_convert_from_mongo_union(self):
        converter = MongoUnionConverter.from_mongo(mongo_uri=MONGO_URI,
                                                   mongo_db=MONGO_DB,
                                                   data_collections=('gismeteo',
                                                                     'meteoprog'),
                                                   real_collection='actual_weather',
                                                   city='Ivano-Frankivsk',
                                                   country='Ukraine',
                                                   distance=0,
                                                   limit=7)
        data = converter.convert(KEY_MAP)

        X = data[0]
        X_tmin = X['data']['t_min']
        shape = X_tmin.shape
        self.assertEqual((2, 1), shape)

    def test_service_names(self):
        services = {
            'one': 1,
            'Two': 1,
            'Three': 1,
        }
        conv = MongoUnionConverter(None, None, {}, **services)
        self.assertListEqual(list(services.keys()), conv.service_names().tolist())


class TestTestDataConverter(unittest.TestCase):
    def setUp(self):
        self.conv = CSVDataConverter(TEST_WEATHER_PATH)

    def test_convert(self):
        predict, real = self.conv.convert(KEY_MAP)
        self.assertEqual(predict['data']['temperature'].shape, (2, 20))
        self.assertEqual(real['data']['temperature'].shape, (1, 20))

    def test_service_names(self):
        services = {
            'gismeteo': 1,
            'meteoprog': 1
        }
        self.assertListEqual(list(services.keys()), self.conv.service_names().tolist())
