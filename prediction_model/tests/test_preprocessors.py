import unittest

from prediction_model.preprocessors import Word2ClassPreprocessor, RadialPreprocessor


class TestWord2ClassPreprocessor(unittest.TestCase):
    def test_call(self):
        check_words_example = {
            'sun': 1,
            'cloud': 2,
            'rain': 3,
            'shower': 4,
            'thunderstorm': 5,
            'fog': 6,
            'snow': 7,
        }
        test_sentence = "great thunderstorm and clouing wind"
        processor = Word2ClassPreprocessor(check_words_example)
        self.assertEqual(processor(test_sentence), 5)

    def test_wrong_type(self):
        with self.assertRaises(TypeError):
            processor = Word2ClassPreprocessor(None)

    def test_none_len(self):
        with self.assertRaises(IndexError):
            processor = Word2ClassPreprocessor({})

    def test_wrong_word_type(self):
        check_words_example = {
            'sun': 1,
            'cloud': 2,
            1+4j: 3,
            'shower': 4,
            'thunderstorm': 5,
            'fog': 6,
            'snow': 7,
        }
        with self.assertRaises(ValueError):
            processor = Word2ClassPreprocessor(check_words_example)

    def test_wrong_value(self):
        check_words_example = {
            'sun': 1,
            'cloud': 2,
            'fsdf': 'ewerwr',
            'shower': 4,
            'thunderstorm': 5,
            'fog': 6,
            'snow': 7,
        }
        with self.assertRaises(ValueError):
            processor = Word2ClassPreprocessor(check_words_example)


class TestRadialPreprocessor(unittest.TestCase):
    def setUp(self):
        self.processor = RadialPreprocessor()

    def test_close_in_the_middle_of_the_circle(self):
        aw, sw = .54, .45
        value = self.processor((aw, sw))
        self.assertEqual(value, sw)

    def test_adding_one(self):
        aw, sw = .95, .1
        value = self.processor((aw, sw))
        self.assertEqual(value, sw + 1)

    def test_subtract_one(self):
        aw, sw = .1, .95
        value = self.processor((aw, sw))
        self.assertEqual(value, sw - 1)
