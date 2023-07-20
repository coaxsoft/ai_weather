"""The ``converters`` contains functionality to convert
data from different formats (e. g. MongoDB, JSON) to specific
mathematical model format.

The newly created converters must extend ``AbstractConverter``
class.

Simply saying, the converter should produce two dictionaries:
the first dict containing data for which to calculate weights,
the second one containing real reliable data.

Converted dictionaries must have the following structure::

    {
        'data': {
            'key1': np.ndarray,
            'key2': np.ndarray,
            ...
        },
        'labels': np.ndarray,
        'slots': np.ndarray
    }

Where

* ``data`` - contains data that should be predicted separated by keys.
           Shapes of ``key[i]`` matrices must be equal.
* ``labels`` - simply saying it's names of columns. The len of
             ``labels`` must be equal to amount of columns in ``key[i]``.
* ``slots`` - simply saying it's names of rows. The len of ``slots``
            smust be equal to amount of rows in ``key[i]``

For real reliable data the shape of ``data:key[i]`` will always be
(1, N) and therefore the len of ``slots`` will be equal to 1 and
the len of ``labels`` will be equal to N.
"""
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np

import pandas
import pymongo

from prediction_model.utils import KeyMapMixin, UniquePriorityQueue


class AbstractConverter(ABC):
    """Base class to override for your custom converters"""

    @abstractmethod
    def convert(self, key_map: dict) -> (dict, dict):
        """Convert already got data to models inner format.
        The method must return two converted dictionaries with the
        next structure::

               {
                    'data': {
                        'key1': np.ndarray,
                        'key2': np.ndarray,
                        ...
                    },
                    'labels': np.ndarray,
                    'slots': np.ndarray
                }

        Where

        * ``data`` - contains data that should be predicted separated by keys.
                   Shapes of ``key[i]`` matrices must be equal.
        * ``labels`` - simply saying it's names of columns. The len of
                     ``labels`` must be equal to amount of columns in ``key[i]``.
        * ``slots`` - simply saying it's names of rows. The len of ``slots``
                    must be equal to amount of rows in ``key[i]``
        """

    @abstractmethod
    def get(self) -> (dict, dict):
        """Get converted data. If data was already converted
        there is no need to call heavy ``convert`` method
        once again.

        :return - the first ``dict`` in tuple contains matrices
        of prediction data where key is the name of predicting
        field, the second ``dict`` contains
        """

    @abstractmethod
    def service_names(self) -> np.ndarray:
        """Should return service names, i.e. data slots + real slots"""


class CSVDataConverter(AbstractConverter):
    def __init__(self, path):
        """Converter that processes data loaded from simple file
        containing test data.

        :param path: path to the file
        """
        self.path = path
        self.data: pandas.DataFrame = pandas.read_csv(path, sep=';')

        self.s_names = self.data.columns.values[1:-1].tolist()
        self.slots = np.array(self.s_names)
        self.predict = None
        self.real = None

    def convert(self, key_map):
        """Convert data from ``pandas.DataFrames``, ``pandas.Series`` to
        numpy n-dim arrays.
        """
        self.predict = {
            'data': {
                'temperature': None,
            },

            'slots': self.slots,
            'labels': []
        }

        real_temp = self.data.iloc[:, -1].apply(float)
        self.real = {
            'data': {
                'temperature': np.array(real_temp).reshape(1, len(real_temp)),
            },
            'labels': []
        }
        self.predict['labels'] = self.real['labels'] = \
            [i for i in range(np.size(self.real['data']['temperature']))]
        shape = self.data.shape

        for i in range(1, shape[1] - 1):
            if self.predict['data']['temperature'] is None:
                data = self.data.iloc[:, i].apply(float)
                self.predict['data']['temperature'] = np.array(data).reshape(1, len(data))
            else:
                self.predict['data']['temperature'] = \
                    np.vstack((self.predict['data']['temperature'],
                               self.data.iloc[:, i].apply(float)))
        return self.predict, self.real

    def get(self):
        return self.predict, self.real

    def service_names(self):
        return np.array(self.s_names)


class MongoBaseConverter(AbstractConverter, KeyMapMixin):
    def __init__(self, real, real_name, labels, label_key='weather_date', **queries):
        """Base class for converters that processes data loaded from MongoDB weather
        collections.

        You should not instantiate this class directly; use this in order to inherit
        some common functionality for mongo classes.

        Actually, there is no need to use the constructor explicitly.
        Use instead ``from_mongo`` classmethod.

        :param real: query to DB, key is the name of collection
        :param real_name: the name to be applied to real data set
        :param labels: pass labels (names of columns)
        :param label_key: the key that exist in mongo document, which value will be in labels
        :param queries: queries to DB, key is the name of collection
        value is the very query
        """
        super().__init__()

        self.labels = labels
        self.label_key = label_key
        self.real_query = real
        self.real_name = real_name
        self.queries = queries
        self.names = list(self.queries.keys())
        self.predict = None
        self.real = None

    def convert(self, key_map: dict):
        """Converts data from Mongo queries to model dictionaries"""
        self.init_key_map(key_map)

        self.predict = {
            'data': {k: None for k in self.key_map},

            'slots': np.array(self.names),
            'labels': OrderedDict({k: None for k in self.labels}),
        }
        for q in self.queries.values():
            X = {k: self.predict['labels'].copy() for k in self.key_map}
            for v in q:
                for key in self.key_map:
                    if v[self.label_key] in X[key]:
                        X[key][v[self.label_key]] = self.retrieve(key, v)

            for key in self.key_map:
                if self.predict['data'][key] is None:
                    self.predict['data'][key] = np.array(list(X[key].values()), dtype=float)
                else:
                    try:
                        self.predict['data'][key] = np.vstack((self.predict['data'][key],
                                                               list(X[key].values())))
                    except ValueError:
                        raise ValueError(f"The vector of length {len(X[key])} cannot be appended "
                                         f"to vector of shape {self.predict['data'][key].shape}.\n"
                                         f"Please check that the data homogeneous is in your data set")

        self.predict['labels'] = list(self.predict['labels'])

        if self.real_query:
            self.real = {
                'data': {k: [] for k in self.key_map},

                'slots': np.array([self.real_name]),
                'labels': self.labels,
            }
            for v in self.real_query:
                for key in self.key_map:
                    self.real['data'][key].append(self.retrieve(key, v))

            for key in self.key_map:
                self.real['data'][key] = np.array(self.real['data'][key], dtype=float)\
                    .reshape(1, len(self.real['data'][key]))

        return self.predict, self.real

    def get(self):
        return self.predict, self.real

    def service_names(self):
        return np.array(self.names)


class MongoIntersectConverter(MongoBaseConverter):
    """``MongoIntersectConverter`` looks for data intersection of
    different services.

    Simply saying, this converter look for data that exists in
    all services by some labels
    """
    @classmethod
    def from_mongo(cls, mongo_uri: str, mongo_db: str, data_collections: tuple,
                   real_collection: str, city: str, country: str, distance: int = 0, limit=7):
        """Create ``MongoIntersectConverter`` directly from MongoDB
        data. Use this method to create new class instances.

        :param mongo_uri: mongo uri
        :param mongo_db: mongo db name
        :param data_collections: tuple containing collections names
                                 that predict weather
        :param real_collection: collection name containing actual weather data
        :param city: city for which to calculate weights
        :param country: additional parameter for searching city in certain country
        :param distance: forecast distance
        :param limit: limit extracted data amount to given value
        """
        mongo = pymongo.MongoClient(mongo_uri)[mongo_db]
        dates = cls.intersect_dates(mongo, data_collections, real_collection,
                                    city, country, distance, limit)
        real = mongo[real_collection].find({'city': city, 'country': country,
                                            'forecast_distance': distance,
                                            'weather_date': {'$in': dates}})\
            .sort([('weather_date', -1)])\
            .limit(limit)

        to_converter = {}
        for col in data_collections:
            to_converter[col] = mongo[col]\
                .find({'city': city,
                       'country': country,
                       'forecast_distance': distance,
                       'weather_date': {'$in': dates}})\
                .sort([('weather_date', -1)])

        return cls(real, real_name=real_collection, labels=dates, **to_converter)

    @staticmethod
    def intersect_dates(mongo: pymongo.MongoClient, data_collections: tuple, real_collection: str,
                        city: str, country: str, distance: int = 0, limit=7):
        """Retrieve all related to query dates and find their intersection

        :param mongo: `MongoClient` object
        :param data_collections: tuple containing collections names
                                 that predict weather
        :param real_collection: collection name containing actual weather data
        :param city: city name
        :param country: additional parameter for searching city in certain country
        :param distance:
        :param limit:
        :return:
        """
        dates = mongo[real_collection].find({'city': city, 'country': country, 'forecast_distance': distance},
                                            {'_id': 0, 'weather_date': 1}) \
            .sort([('weather_date', -1)]).limit(limit)
        dates = {v['weather_date'] for v in dates}
        for col in data_collections:
            nd = mongo[col].find({'city': city, 'country': country, 'forecast_distance': distance},
                                 {'_id': 0, 'weather_date': 1})\
                .sort([('weather_date', -1)])\
                .limit(limit)
            dates = dates.intersection({v['weather_date'] for v in nd})
        return list(dates)


class MongoUnionConverter(MongoBaseConverter):
    """``MongoUnionConverter`` looks for data union from all services
    by some labels.

    So, it finds all labels that are at least in one service and union
    the data by them. If some value for some label in some service is
    ``None``, this converter assigns first non-null value for this
    label for from another service.
    """
    @classmethod
    def from_mongo(cls, mongo_uri: str, mongo_db: str, data_collections: tuple,
                   city: str, country: str, real_collection: str = None, distance: int = 0, limit=7):
        """Create ``MongoUnionConverter`` directly from MongoDB
        data. Use this method to create new class instances.

        :param mongo_uri: mongo uri
        :param mongo_db: mongo db name
        :param data_collections: tuple containing collections names
                                 that predict weather
        :param city: city for which to calculate weights
        :param country: additional parameter for searching city in certain country
        :param real_collection: collection name for real weather
                                (if there is not need to use ``real_collection``,
                                pass ``None`` instead)
        :param distance: forecast distance
        :param limit: limit extracted data amount to given value
        """
        mongo = pymongo.MongoClient(mongo_uri)[mongo_db]
        if real_collection:
            dates = cls.real_dates(mongo, real_collection,
                                   city, country, distance, limit)
        else:
            dates = None
        dates = cls.union_dates(mongo, data_collections,
                                city, country, dates, distance, limit)

        if real_collection:
            real = mongo[real_collection].find({'city': city, 'country': country, 'forecast_distance': distance,
                                                'weather_date': {'$in': dates}}) \
                .sort([('weather_date', -1)]) \
                .limit(limit)
        else:
            real = None
        to_converter = {}
        for col in data_collections:
            to_converter[col] = mongo[col] \
                .find({'city': city,
                       'country': country,
                       'forecast_distance': distance,
                       'weather_date': {'$in': dates}}) \
                .sort([('weather_date', -1)])

        return cls(real, real_collection, labels=dates, **to_converter)

    @staticmethod
    def real_dates(mongo: pymongo.MongoClient, real_collection: str,
                   city: str, country: str, distance: int = 0, limit=7):
        real = mongo[real_collection].find({'city': city, 'country': country, 'forecast_distance': distance},
                                           {'_id': 0, 'weather_date': 1}) \
            .sort([('weather_date', -1)]) \
            .limit(limit)
        return [v['weather_date'] for v in real]

    @staticmethod
    def union_dates(mongo: pymongo.MongoClient, data_collections: tuple,
                    city: str, country: str, dates: list = None, distance: int = 0, limit=7):
        """Retrieve all related to query dates and find their intersection

        :param mongo: `MongoClient` object
        :param data_collections: tuple containing collections names
                                 that predict weather
        :param dates: dates list in which to load data for union
        :param city: city for which to filter data
        :param country: additional parameter for searching city in certain country
        :param distance: distance for which to filter data
        :param limit: limiting of results to look for
        :return: list of union-ed data
        """
        heap = UniquePriorityQueue()
        if dates:
            search_query = {
                'city': city, 'country': country, 'forecast_distance': distance,
                'weather_date': {'$in': dates}
            }
        else:
            search_query = {'city': city, 'country': country, 'forecast_distance': distance}
        for col in data_collections:
            nd = mongo[col].find(search_query,
                                 {'_id': 0, 'weather_date': 1}) \
                .sort([('weather_date', -1)]) \
                .limit(limit)
            for v in nd:
                heap.add(v['weather_date'])
        return list(heap.to_list()[-limit:])

    def convert(self, key_map: dict):
        super().convert(key_map)

        for key in self.key_map:
            self._post_process(key)

        return self.predict, self.real

    def _first_not_none(self, col):
        for v in col:
            if v is not None:
                if not np.isnan(v):
                    return v
        return None

    def _post_process(self, key='t_min'):
        for i, row in enumerate(self.predict['data'][key]):
            for j in range(len(row)):
                if self.predict['data'][key][i, j] is None:
                    self.predict['data'][key][i, j] = np.nan
                if isinstance(self.predict['data'][key][i, j], str):
                    self.predict['data'][key][i, j] = float(self.predict['data'][key][i, j])
                if np.isnan(self.predict['data'][key][i, j]):
                    self.predict['data'][key][i, j] = self._first_not_none(self.predict['data'][key][:, j])
