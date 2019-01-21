"""Estimators are the core of whole package.

All estimators must extend ``AbstractEstimator`` class.
The two main methods for estimator are ``reduce``, ``produce``.

* ``reduce`` method calculates slots weights, i. e. `reduce`
  error;
* ``produce`` method calculates new data with respect to calculated
  weights, i. e. `produce` new data.

========
Examples
========

For example usage of ``StandardEstimator``::

    est = StandardEstimator.from_converter(converter)

    # reduce error (calc weights) by Minkovsky metrics
    est.reduce(Reducer.L1)

    # produce data for data in converter
    prod = est.produce(converter.get()[0])

    # and calculate its CV errors
    errors = cv(prod, converter.get()[1])
"""
from abc import ABC, abstractmethod

import numpy as np

from prediction_model import reverse_norma, none_to_zero
from prediction_model.converters import AbstractConverter
from prediction_model.io.readers import AbstractReader
from prediction_model.reducers import AbstractReducer, EuclideanReducer


def produce(reader: AbstractReader, data: dict) -> dict:
    """Use this function when you simply need only to
    calculate result for given data with existing weights and
    there is no need to create ``...Estimator`` class.
    """
    weights = reader.read_weights()
    if weights is None:
        raise TypeError(f"There are no weights for {reader.supplement()} in the database.\nPlease, "
                        f"Honorable Sir, would you be so kind to check whether you reduced any "
                        f"data about this city or forecast distance?")
    estimator = StandardEstimator(data, None)
    estimator.weights = weights
    return estimator.produce(data)


def cv(data: dict, real: dict, reducer: AbstractReducer = EuclideanReducer()) -> dict:
    """Calculate cross validation error between produced ``data``
    and the real one

    :param data: newly produced data
    :param real: actual data
    :param reducer: how to calculate error
    """
    res = {
        'data': {},
        'labels': list(data['labels']),
        'slots': list(data['slots'])
    }
    for k in data['data']:
        X, y = data['data'][k], real['data'][k].T
        row, _ = y.shape
        errors = np.zeros((X.shape[0], 1))

        none_to_zero(X)

        for i, v in enumerate(X):
            v = v.reshape((len(v), 1))
            errors[i] = reducer(v, y) / row
        res['data'][k] = errors
    return res


class AbstractEstimator(ABC):
    @classmethod
    @abstractmethod
    def from_converter(cls, converter: AbstractConverter, key_map: dict):
        """Load data directly from converter"""

    @abstractmethod
    def reduce(self, reducer: AbstractReducer):
        """Calculates the new weights by the given ``reducer``

        :param reducer: enum that specifies the reducer version to use
        """

    @abstractmethod
    def produce(self, data: dict):
        """Calculate the new result value for ``data`` by existing
        weights
        """

    @abstractmethod
    def get_weights(self) -> dict:
        """Return weights"""

    @abstractmethod
    def get_slots(self) -> list:
        """Return slots"""


class StandardEstimator(AbstractEstimator):
    def __init__(self, data, real):
        """``StandardEstimator`` reduce weighted sum of values
        based on their ability to generate right answers.

        :param data: specially formatted input data
        :param real: specially formatted authentic data
        """
        self.data = data
        self.y = real
        try:
            self.weights = self._init_weights(data)
            self.slots = data['slots']
        except KeyError:
            raise KeyError(f"Key 'slots' is not in the given dictionary {self.data}")

    @classmethod
    def from_converter(cls, converter: AbstractConverter, key_map: dict):
        """Create ``StandardEstimator`` instance from presented converter.
        Converter should be the class derived from ``AbstractConverter``
        class.

        :param converter: data converter
        """
        return cls(*converter.convert(key_map))

    def reduce(self, reducer: AbstractReducer = EuclideanReducer()):
        """Calculate weights by distance presented by reducer.
        If the value in predict data is None it will be changed to 0

        :param reducer: metrics by which to calculate weights
        """
        for k in self.data['data']:
            X, y = self.data['data'][k], self.y['data'][k].T
            row, _ = y.shape

            if X.shape[1] == 0:
                raise ValueError(f"Matrix containing predictions for {k}:{self.data['labels']} "
                                 f"is empty {X}.\nPlease, Honorable Sir, check perhaps something "
                                 f"is missed in your data!?")
            if X.shape[1] != len(self.data['labels']):
                raise IndexError(f"Matrix containing predictions for {k} doesn't contains enough "
                                 f"values and its size {X.shape[1]} != {len(self.data['labels'])}."
                                 f"\nPlease, Honorable Sir, check is the data homogeneous for "
                                 f"all slots {self.get_slots()} in your set?")

            errors = np.zeros((X.shape[0], 1))

            none_to_zero(X)

            for i, v in enumerate(X):
                v = v.reshape((len(v), 1))
                errors[i] = reducer(v, y) / row
            self.weights[k] = reverse_norma(errors)

    def produce(self, data: dict):
        """Using reduced weights calculate weighted sum and
        get more precise result

        :param data: dict containing data with the format similar to
                     format returned by converter with ``data``, ``slots``,
                     and ``labels`` keys
        """
        res = {
            'data': {},
            'slots': list(data['slots']),
            'labels': list(data['labels'])
        }
        for k, v in data['data'].items():
            if v.shape[1] != len(res['labels']):
                raise IndexError(f"Matrix containing predictions for {k} doesn't contains enough "
                                 f"values and its size {v.shape[1]} != {len(res['labels'])}.\n"
                                 f"Please, Honorable Sir, check is the data "
                                 f"homogeneous for all slots {res['slots']} in your set?")
            v = none_to_zero(v, should_copy=True)
            try:
                res['data'][k] = self.weights[k].T.dot(v)
            except KeyError:
                pass
        return res

    def get_weights(self):
        return self.weights

    def get_slots(self):
        return list(self.slots)

    @staticmethod
    def _init_weights(data: dict) -> dict:
        return {k: np.zeros(shape=(len(data['slots']), 0)) for k in data['data']}

    def __repr__(self):
        return "If you see this text then I myself, dumb my head, " \
               "forget to write its class representation :/"


class PostProcessEstimator(StandardEstimator):
    """Estimator that post-process resulting data.

    Post-processor dictionary should have the following structure::

        {
            key: (PostProcessor1(), PostProcessor2(), ...),
            ...
        }

    For example::

        {
            'class': (MaxWeightPostprocessor(),)
        }
    """
    def reduce(self, reducer: AbstractReducer = EuclideanReducer(), weights_postprocessors: dict = None):
        super().reduce(reducer)

        if weights_postprocessors:
            for k in self.weights:
                if k in weights_postprocessors:
                    for p in weights_postprocessors[k]:
                        self.weights[k] = p(self.weights[k])

    def produce(self, data: dict, data_postprocessors: dict = None):
        res = super().produce(data)

        if data_postprocessors:
            for k in res:
                if k in data_postprocessors:
                    for p in data_postprocessors[k]:
                        res = p(res[k])
        return res
