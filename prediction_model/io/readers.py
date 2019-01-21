"""``io.readers`` contains functions and classes that read
data from different places.

All readers must extend ``AbstractReader`` class
"""
from abc import ABC, abstractmethod

import datetime

import numpy as np
import pymongo


class AbstractReader(ABC):
    @abstractmethod
    def read_weights(self):
        """Read weights to convenient to estimators format."""

    @abstractmethod
    def supplement(self) -> dict:
        """Return supplementary information about reader"""


class MongoReader(AbstractReader):
    """Reader that read information from MongoDB

    :param mongo_uri: mongo uri
    :param mongo_db: mongo database name
    :param supplement: supplementary info that should be written
                       to mongo
    :param weights_collection: weights collection name
    :param errors_collection: errors collection name
    """
    def __init__(self, mongo_uri: str, mongo_db: str,
                 weights_collection: str = 'weights',
                 errors_collection: str = 'errors'):
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.weights_collection = weights_collection
        self.errors_collection = errors_collection
        self.mongo = pymongo.MongoClient(mongo_uri)[mongo_db]
        self.date_updated = None
        self.city = None
        self.country = None
        self.forecast_distance = None

    def read_weights(self):
        if self.city is None or self.country is None or self.forecast_distance is None:
            raise TypeError('Filter data is unclear. Please, call search(...) '
                            'method before read_weights(...)')
        w = self.mongo[self.weights_collection].find({'city': self.city,
                                                      'country': self.country,
                                                      'forecast_distance': self.forecast_distance},
                                                     {'weights': 1, '_id': 0})\
            .sort([('updated', -1)])
        try:
            return self._list_to_weights(next(w)['weights'])
        except StopIteration:
            return None

    def search(self, city: str, country: str, forecast_distance: int,
               date: datetime.datetime = None):
        """Specify search data by which to filter documents in Mongo
        collections

        :param date: specified datetime object. You don't need to explicitly
                     set time to 0
        :param city: weights city
        :param country: weights country
        :param forecast_distance: weights forecast distance
        """
        self.date_updated = date
        self.city = city
        self.country = country
        self.forecast_distance = forecast_distance

    def supplement(self):
        return {
            'city': self.city,
            'country': self.country,
            'forecast_distance': self.forecast_distance,
            'date_updated': self.date_updated
        }

    def _list_to_weights(self, weights: dict):
        for k in weights:
            weights[k] = np.array(weights[k]).reshape(len(weights[k]), 1)
        return weights
