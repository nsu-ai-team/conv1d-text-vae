import os
import pickle
import unittest

from conv1d_text_vae.tokenizer import DefaultTokenizer


class TestDefaultTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DefaultTokenizer()
        self.name_of_tokenizer = os.path.join(os.path.dirname(__file__), 'testdata', 'tokenizer.pkl')

    def tearDown(self):
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if os.path.isfile(self.name_of_tokenizer):
            os.remove(self.name_of_tokenizer)

    def test_tokenize_into_words_positive01(self):
        s = ' Мама  мыла23\tраму.'
        true_bounds = [(1, 5), (7, 11), (11, 13), (14, 18), (18, 19)]
        predicted_bounds = self.tokenizer.tokenize_into_words(s)
        self.assertEqual(true_bounds, predicted_bounds)

    def test_tokenize_into_words_positive02(self):
        del self.tokenizer
        self.tokenizer = DefaultTokenizer(special_symbols={'\\n'})
        s = ' Мама  мыла23\\nраму.'
        true_bounds = [(1, 5), (7, 11), (11, 13), (13, 15), (15, 19), (19, 20)]
        predicted_bounds = self.tokenizer.tokenize_into_words(s)
        self.assertEqual(true_bounds, predicted_bounds)

    def test_tokenize_into_characters(self):
        s = ' Мама  мыла23 \n\tраму.'
        bounds_of_words = [(1, 5), (7, 11), (11, 13), (16, 20), (20, 21)]
        true_characters = [' ', 'М', 'а', 'м', 'а', ' ', 'м', 'ы', 'л', 'а', '2', '3', '\n', 'р', 'а', 'м', 'у', '.']
        predicted_characters = self.tokenizer.tokenize_into_characters(s, bounds_of_words)
        self.assertEqual(true_characters, predicted_characters)

    def test_serialize(self):
        with open(self.name_of_tokenizer, 'wb') as fp:
            pickle.dump(self.tokenizer, fp)
        with open(self.name_of_tokenizer, 'rb') as fp:
            res = pickle.load(fp)
        self.assertIsInstance(res, DefaultTokenizer)
        self.assertIsNot(res, self.tokenizer)
        s = ' Мама  мыла23\tраму.'
        self.assertEqual(self.tokenizer.tokenize_into_words(s), res.tokenize_into_words(s))


if __name__ == '__main__':
    unittest.main(verbosity=2)
