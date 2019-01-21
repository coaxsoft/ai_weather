"""This file should contain special classes that pre-process
data before conversion to NumPy matrices
"""
import heapq

from abc import ABC, abstractmethod
from textblob import TextBlob

from prediction_model.utils import levenshtein


class AbstractPreprocessor(ABC):
    @abstractmethod
    def __call__(self, value):
        pass


class Word2ClassPreprocessor(AbstractPreprocessor):
    """Preprocessor that assigns numeric value to string variable.

    Call ``__call__`` method of this class.
    It uses ``textblob`` package to retrieve nouns from sentence.
    Then it calculates Levenshtein distance between retrieved
    nouns and ``check_words`` dict.

    :param check_words: object containing check words and its class
                        number. The object must have the following
                        format::

                            {
                                'sun': 1,
                                'cloud': 2,
                                'rain': 3,
                                'shower': 4,
                                'thunderstorm': 5,
                                'fog': 6,
                                'snow': 7,
                            }
    """
    def __init__(self, check_words: dict):
        self.check_words = check_words
        if not isinstance(self.check_words, dict):
            raise TypeError("check_words must be instance of dict class")
        # pylint: disable=len-as-condition
        if len(check_words) == 0:
            raise IndexError("check_words dict must have length more than 1. len(check_words) >= 1")
        for word, value in self.check_words.items():
            if not isinstance(word, str):
                raise ValueError(f"All keys in check_words must be instance of str class but '{word}' is not")
            if not isinstance(value, int) and not isinstance(value, float):
                raise ValueError(f"All values in check_words must be instance of int or float classes but "
                                 f"'{value}' is not")

    def __call__(self, value):
        tags = [n for n, t in TextBlob(value).lower().tags if t in ('NN', 'NNS', 'NNT')]
        heap = []
        for tag in tags:
            for check_word in self.check_words:
                heapq.heappush(heap, (levenshtein(check_word, tag), check_word))
        if not heap:
            return min(self.check_words.values())
        return self.check_words[heapq.heappop(heap)[1]]


class RadialPreprocessor(AbstractPreprocessor):
    """Pre-processor that converts degree depending on its periodic
    closeness to actual one.

    It pre-processes service generated degree to have minimal error
    with actual value.

    **NOTE** that degree must be valued from 0 to 1. So 1 is equal to 2 PI.

    **NOTE** can't be used yet because of architecture limitations.

    :param value: tuple with containing two values: on first position -
                  actual degree, second one - value to preprocess
    """
    def __call__(self, value: tuple):
        min_, max_ = min(*value), max(*value)
        if max_ - min_ < abs(max_ - min_ - 1):
            return value[1]
        if value[0] > value[1]:
            return value[1] + 1
        return value[1] - 1
