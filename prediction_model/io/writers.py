"""``io.writers`` contains classes and functions that write data
in specific formats and places.

All writers must extend ``AbstractWriter`` class.
"""
from abc import ABC, abstractmethod

import datetime
import pymongo

from prediction_model.estimators import AbstractEstimator


class AbstractWriter(ABC):
    @abstractmethod
    def supplement(self, data: dict = None) -> dict:
        """Return supplementary information that should be written
        if ``data`` is ``None``. If ``data`` is not ``None`` it
        sets new supplementary info

        This method serves a guaranty that supplement information
        would be provided.
        """

    @abstractmethod
    def write_weights(self, estimator: AbstractEstimator):
        """Write estimator weights to specified output"""

    @abstractmethod
    def write_errors(self, data: dict):
        """Write estimator errors to specified output"""

    @abstractmethod
    def write_produced(self, data: dict):
        """Write produced by estimator data to specified output"""


class MongoWriter(AbstractWriter):
    """Writer that writes information into MongoDB

    :param mongo_uri: mongo uri
    :param mongo_db: mongo database name
    :param supplement: supplementary info that should be written
                       to mongo. **NOTE** that supplement info would
                       be passed as update filter in
                       ``write_weights`` method
    :param weights_collection: weights collection name
    :param errors_collection: errors collection name
    """
    def __init__(self, mongo_uri: str, mongo_db: str, supplement: dict = None,
                 weights_collection: str = 'weights',
                 errors_collection: str = 'errors',
                 produced_collection: str = 'produced_data'):
        if supplement is None:
            self.supplement_info = {}
        else:
            self.supplement_info = supplement
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.weights_collection = weights_collection
        self.errors_collection = errors_collection
        self.produced_collection = produced_collection
        self.mongo = pymongo.MongoClient(mongo_uri)[mongo_db]

    def supplement(self, data: dict = None):
        if data is not None:
            self.supplement_info = data
        return self.supplement_info

    def write_weights(self, estimator: AbstractEstimator):
        w = self._weights_to_list(estimator.get_weights(), estimator.get_slots())
        today = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        to_mongo = dict(**self.supplement_info,
                        **{
                            'slots': estimator.get_slots(),
                            'updated': today,
                            'weights': w
                        })
        self.mongo[self.weights_collection].update(dict(**self.supplement_info,
                                                        **{'updated': today}),
                                                   to_mongo,
                                                   True)

    def write_errors(self, data: dict):
        for v in self._gen_to_mongo(data, 0):
            self.mongo[self.errors_collection].update(dict(**self.supplement_info,
                                                           **{'label': v['label']}),
                                                      v,
                                                      True)

    def write_produced(self, data: dict):
        for v in self._gen_to_mongo(data):
            self.mongo[self.produced_collection].update(dict(**self.supplement_info,
                                                             **{'label': v['label']}),
                                                        v,
                                                        True)

    def _gen_to_mongo(self, data: dict, hard_index=None):
        today = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        for i, v in enumerate(data['labels']):
            index = hard_index if hard_index is not None else i
            yield dict(**self.supplement_info,
                       **{
                           'updated': today,
                           'data': self._get_data(data['data'], index),
                           'label': v,
                           'slots': list(data['slots'])
                       })

    @staticmethod
    def _get_data(data: dict, index: int):
        return {k: v[index] for k, v in data.items()}

    @staticmethod
    def _weights_to_list(w: dict, slots):
        return {k: v.reshape(len(slots)).tolist() for k, v in w.items()}
